"""NSA compressed-branch forward in Triton. Mean-pool K/V into blocks of
BLOCK_C, then FA-2 streaming softmax against the compact sequence of
length ceil(Tk / BLOCK_C). Causally-empty rows emit (0, -inf) so gating
sees finite values.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pool: torch reshape-mean. The compressed branch's pooled tensors are tiny
# (Nc = ceil(Tk / BLOCK_C)) so a Triton pool kernel is unnecessary at this
# stage; the attention kernel below dominates by orders of magnitude. If
# profiling later shows the pool is hot, swap this out for a fused launch.
# ---------------------------------------------------------------------------


def _mean_pool(x: torch.Tensor, block_c: int) -> torch.Tensor:
    """Mean-pool x of shape [B, H, T, D] into [B, H, ceil(T/block_c), D].

    Pads the trailing partial block with zeros, matching the reference's F.pad.
    Note: a trailing partial block is divided by block_c (not the partial count)
    because the reference also pads-then-means by block_c.
    """
    B, H, T, D = x.shape
    Nc = (T + block_c - 1) // block_c
    # Use a reshape-based path on the pooled output tensor for simplicity and to
    # avoid a tricky stride-fused launch. The reference is also reshape based,
    # and this keeps the kernel attention-side honest.
    pad = Nc * block_c - T
    if pad:
        x_p = torch.nn.functional.pad(x, (0, 0, 0, pad))
    else:
        x_p = x
    # Mean over the within-block token axis.
    out = x_p.view(B, H, Nc, block_c, D).mean(dim=3)
    return out.contiguous()


# ---------------------------------------------------------------------------
# Attention kernel: queries x compressed keys, FlashAttention-2 streaming softmax.
# ---------------------------------------------------------------------------


@triton.jit
def _compressed_attn_fwd_kernel(
    Q_ptr, Kc_ptr, Vc_ptr, Out_ptr, Lse_ptr,
    Tq, Nc, Tk_padded,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Bases: caller fused (b, h) so stride_*h covers the linear bh stride.
    q_base = pid_bh * stride_qh
    k_base = pid_bh * stride_kh
    v_base = pid_bh * stride_vh
    o_base = pid_bh * stride_oh
    l_base = pid_bh * stride_lh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                    # [BLOCK_D]
    mask_m = offs_m < Tq
    mask_d = offs_d < BLOCK_D  # BLOCK_D is exact head dim, kept for shape

    # Load Q tile.
    q_ptrs = (
        Q_ptr
        + q_base
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    # FA-2: bake the scale into Q so the inner loop dot is unscaled.
    q = (q.to(tl.float32) * sm_scale).to(q.dtype)

    # Per-query running stats.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Causal upper bound on n (compressed block index) per query row.
    # max_block(q) = floor((q + (Tk_padded - Tq) + 1) / BLOCK_C) - 1.
    # The number of legal blocks for the whole tile is governed by the
    # last query in the tile (largest q, hence largest max_block). We tile
    # across n up to that bound and apply the per-row mask inside.
    if CAUSAL:
        offset = Tk_padded - Tq
        # Largest q in the tile that is in-range:
        last_q = tl.minimum(pid_m * BLOCK_M + BLOCK_M - 1, Tq - 1)
        max_block_tile = (last_q + offset + 1) // BLOCK_C - 1  # scalar
        # Number of n-blocks we need to scan, clamped to >= 0.
        n_end = tl.maximum(max_block_tile + 1, 0)
        n_end = tl.minimum(n_end, Nc)
    else:
        n_end = Nc

    # Loop over compressed-key tiles of width BLOCK_N.
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)         # [BLOCK_N]
        mask_n = offs_n < n_end

        k_ptrs = (
            Kc_ptr
            + k_base
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            Vc_ptr
            + v_base
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # qk: [BLOCK_M, BLOCK_N], fp32 accumulator.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        # Build mask of legal (q, n) entries.
        # Out-of-range n columns -> -inf.
        # Out-of-range q rows -> -inf (their stats stay -inf, output later masked off).
        if CAUSAL:
            offset = Tk_padded - Tq
            row_max_block = (offs_m + offset + 1) // BLOCK_C - 1  # [BLOCK_M]
            legal = (offs_n[None, :] <= row_max_block[:, None]) & mask_n[None, :] & mask_m[:, None]
        else:
            legal = mask_n[None, :] & mask_m[:, None]

        qk = tl.where(legal, qk, float("-inf"))

        # Streaming softmax update.
        m_ij = tl.max(qk, axis=1)                              # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)
        # Guard exp underflow when m_new is -inf (no legal entries seen yet).
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        # For rows where m_new is -inf the whole tile is illegal; p must be 0.
        p = tl.where(
            m_new[:, None] == float("-inf"),
            0.0,
            tl.exp(qk - m_new[:, None]),
        )
        # Drop illegal entries (also -inf turned into 0 by the where above for
        # the all-illegal rows; but for partial rows we still need to zero out
        # masked columns since exp(-inf - m_new) = 0 already on real -inf).
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    # Finalize: out = acc / l_i; lse = m_i + log(l_i).
    # Rows with l_i == 0 (all-illegal): emit zeros for out, -inf for lse.
    safe = l_i > 0
    out = tl.where(safe[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(safe, m_i + tl.log(l_i), float("-inf"))

    out_ptrs = (
        Out_ptr
        + o_base
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    lse_ptrs = Lse_ptr + l_base + offs_m * stride_lm

    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])
    tl.store(lse_ptrs, lse, mask=mask_m)


# ---------------------------------------------------------------------------
# Public wrapper.
# ---------------------------------------------------------------------------


def compressed_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size_c: int = 64,
    pool: str = "mean",
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compressed-branch forward attention via Triton.

    Args:
        Q: [B, H, Tq, D], fp16 or bf16.
        K: [B, H, Tk, D], same dtype as Q.
        V: [B, H, Tk, D], same dtype as Q.
        block_size_c: pool window along Tk.
        pool: only "mean" is supported in this iteration.
        causal: if True, query q can attend to compressed block c only when
            (c + 1) * block_size_c - 1 <= q + (Tk_padded - Tq).
        scale: softmax scale; defaults to 1 / sqrt(D).

    Returns:
        out: [B, H, Tq, D] in the input dtype.
        lse: [B, H, Tq] in fp32. For queries with zero legal blocks (causal
            edge case) lse is -inf and out is zero. The reference produces nan
            in that situation; the test must skip or compare only rows with at
            least one legal block.
    """
    if pool != "mean":
        raise NotImplementedError("compressed_attention: only pool='mean' supported")
    assert Q.dtype in (torch.float16, torch.bfloat16), "Q must be fp16 or bf16"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4
    B, H, Tq, D = Q.shape
    Bk, Hk, Tk, Dk = K.shape
    assert (B, H, D) == (Bk, Hk, Dk), "Q and K must agree on B, H, D"
    assert V.shape == K.shape

    # Triton tl.dot needs the contraction dim to be a power of two and >= 16.
    assert D in (16, 32, 64, 128, 256), f"unsupported head dim D={D}"

    sm_scale = 1.0 / math.sqrt(D) if scale is None else float(scale)

    # 1) Mean-pool K and V along Tk into BLOCK_C blocks.
    Kc = _mean_pool(K, block_size_c)  # [B, H, Nc, D]
    Vc = _mean_pool(V, block_size_c)
    Nc = Kc.shape[2]
    Tk_padded = Nc * block_size_c

    # Make sure Q and pooled tensors are contiguous on the head/batch axes so
    # we can fuse (b, h) into the program-id-1 dimension.
    Q_c = Q.contiguous()
    Kc_c = Kc.contiguous()
    Vc_c = Vc.contiguous()
    out = torch.empty_like(Q_c)
    lse = torch.empty((B, H, Tq), device=Q.device, dtype=torch.float32)

    BLOCK_M = 64
    # Choose BLOCK_N: must be a power of two >= 16 for tl.dot. Cap at 64.
    if Nc <= 16:
        BLOCK_N = 16
    elif Nc <= 32:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    BLOCK_D = D  # head dims supported are exact powers of two
    BLOCK_C = block_size_c

    grid = (triton.cdiv(Tq, BLOCK_M), B * H)

    # Strides: fuse (B, H) into a single linear "bh" axis.
    def fuse_bh_strides(t: torch.Tensor):
        # Returns (stride_h_fused, stride_t, stride_d). stride_h_fused = stride for
        # advancing one head, treating (b, h) as a flat axis.
        sb, sh, st, sd = t.stride()
        # We need bh-linear stride == sh when batch and head are jointly contiguous in
        # standard [B, H, T, D] layout: bh = b * H + h, advancing bh by 1 must advance
        # the pointer by sh when h < H - 1, and by sb - (H - 1) * sh when wrapping.
        # That is only equal to sh if sb == H * sh, which is true for contiguous tensors.
        assert sb == H * sh, "tensor must be contiguous over (B, H)"
        return sh, st, sd

    sQ_h, sQ_m, sQ_d = fuse_bh_strides(Q_c)
    sK_h, sK_n, sK_d = fuse_bh_strides(Kc_c)
    sV_h, sV_n, sV_d = fuse_bh_strides(Vc_c)
    sO_h, sO_m, sO_d = fuse_bh_strides(out)
    # lse is [B, H, Tq]; same fusion.
    sL_b, sL_h, sL_m = lse.stride()
    assert sL_b == H * sL_h

    _compressed_attn_fwd_kernel[grid](
        Q_c, Kc_c, Vc_c, out, lse,
        Tq, Nc, Tk_padded,
        0, sQ_h, sQ_m, sQ_d,
        0, sK_h, sK_n, sK_d,
        0, sV_h, sV_n, sV_d,
        0, sO_h, sO_m, sO_d,
        0, sL_h, sL_m,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_C=BLOCK_C,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return out, lse


__all__ = ["compressed_attention"]
