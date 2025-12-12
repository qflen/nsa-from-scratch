"""NSA selected-branch forward in Triton. Per Q tile of size BLOCK_M,
the caller passes precomputed top-k KV block indices; the kernel gathers
those k blocks and runs FA-2 streaming softmax with causal masking
against the original (pre-gather) token indices. Fully-masked rows write
(0, -inf) so gating stays finite.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# Large negative used in place of -inf inside the kernel; -inf can poison
# fp32 max-reductions in some Triton versions, so we use a big-but-finite
# constant. Anything that ends up as -1e30 will be excluded from the
# softmax via the streaming update math, and rows that remain all
# -1e30 will be detected and zeroed at the end.
# (Inlined as a literal in the kernel because Triton 3.0 does not allow
# referencing module-level Python globals from a @triton.jit function unless
# they are constexpr; literals are simpler and just as fast.)


@triton.jit
def _selected_attn_fwd(
    Q_ptr, K_ptr, V_ptr, IDX_ptr, Out_ptr, Lse_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_ik,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    Tq_p, Tk_p, OFFSET, TOP_K,
    sm_scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_qb = tl.program_id(0)   # which query block
    pid_bh = tl.program_id(1)   # batch * head packed

    # We fold (b, h) into one program-id dimension so the launch grid stays 2D.
    # The strides handle the actual addressing.
    # Caller passes pid_bh = b * H + h via the launch grid shape.
    # We do not need B or H here separately.

    # Compute starting offsets for Q, Out, Lse for this (bh, qb) tile.
    q_block_start = pid_qb * BLOCK_M
    offs_m = q_block_start + tl.arange(0, BLOCK_M)             # [BLOCK_M]
    offs_d = tl.arange(0, HEAD_DIM)                            # [HEAD_DIM]
    offs_n = tl.arange(0, BLOCK_N)                             # [BLOCK_N]

    q_mask_m = offs_m < Tq_p

    # Q tile: [BLOCK_M, HEAD_DIM]
    q_ptrs = (Q_ptr
              + pid_bh * stride_qh
              + offs_m[:, None] * stride_qm
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_mask_m[:, None], other=0.0)

    # Streaming softmax state in fp32.
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal cutoff in original-token coordinates: row m can attend to keys j
    # with j <= q_pos = q_block_start + m + OFFSET.
    q_pos = (q_block_start + tl.arange(0, BLOCK_M)) + OFFSET   # [BLOCK_M]
    # Worst-case last legal key for any row in this block:
    q_block_max_pos = q_block_start + (BLOCK_M - 1) + OFFSET

    # Index pointer base for this (bh, qb): IDX[bh, qb, :TOP_K]
    idx_base = IDX_ptr + pid_bh * stride_ih + pid_qb * stride_iq

    for kk in range(0, TOP_K):
        block_idx = tl.load(idx_base + kk * stride_ik)         # int64/int32
        block_idx = block_idx.to(tl.int32)
        kv_start = block_idx * BLOCK_N

        # Skip blocks that are entirely after the query block's last legal
        # key. This is a coarse causal early-out; per-element masking below
        # still handles partial overlap.
        if (not CAUSAL) or (kv_start <= q_block_max_pos):
            kv_offs = kv_start + offs_n                         # [BLOCK_N]
            kv_mask = kv_offs < Tk_p

            k_ptrs = (K_ptr
                      + pid_bh * stride_kh
                      + kv_offs[:, None] * stride_kn
                      + offs_d[None, :] * stride_kd)
            v_ptrs = (V_ptr
                      + pid_bh * stride_vh
                      + kv_offs[:, None] * stride_vn
                      + offs_d[None, :] * stride_vd)

            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

            # scores: [BLOCK_M, BLOCK_N] = Q @ K^T, scaled.
            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

            # Mask out-of-range KV columns (padding inside the last block).
            scores = tl.where(kv_mask[None, :], scores, -1.0e30)

            if CAUSAL:
                # Mask future tokens using ORIGINAL token indices.
                causal_mask = kv_offs[None, :] <= q_pos[:, None]
                scores = tl.where(causal_mask, scores, -1.0e30)

            # Mask query rows that fall in Q-padding.
            scores = tl.where(q_mask_m[:, None], scores, -1.0e30)

            # Streaming softmax update.
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new

    # Detect rows that never saw a finite key (e.g., everything was causally
    # masked). For those rows l_i stays 0 and m_i stays -1e30; emit zeros
    # for the output and -inf for the lse.
    valid = l_i > 0.0
    out = tl.where(valid[:, None], acc / tl.where(valid, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(valid, m_i + tl.log(tl.where(valid, l_i, 1.0)), float("-inf"))

    # Store output [BLOCK_M, HEAD_DIM].
    out_ptrs = (Out_ptr
                + pid_bh * stride_oh
                + offs_m[:, None] * stride_om
                + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=q_mask_m[:, None])

    lse_ptrs = (Lse_ptr
                + pid_bh * stride_lh
                + offs_m * stride_lm)
    tl.store(lse_ptrs, lse, mask=q_mask_m)


def _scale(D: int, override: Optional[float]) -> float:
    return float(override) if override is not None else 1.0 / math.sqrt(D)


def _topk_indices_from_scores(
    block_scores: torch.Tensor,
    top_k: int,
    n_q_blocks: int,
    n_kv_blocks: int,
    block_size_m: int,
    block_size_n: int,
    Tq: int,
    Tk: int,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    """Apply causal masking to block_scores and return [B, H, n_q, k] int32 indices.

    The masking logic mirrors selected_attention_reference exactly: a KV block
    is legal for a Q block iff its first token index is <= Q block's last
    covered token index plus offset (Tk - Tq).
    """
    if causal:
        offset = Tk - Tq
        q_block_last_token = (
            torch.arange(n_q_blocks, device=device) * block_size_m + (block_size_m - 1)
        )
        kv_block_first_token = torch.arange(n_kv_blocks, device=device) * block_size_n
        legal = kv_block_first_token.view(1, n_kv_blocks) <= (
            q_block_last_token.view(n_q_blocks, 1) + offset
        )
        block_scores = block_scores.masked_fill(
            ~legal.view(1, 1, n_q_blocks, n_kv_blocks), float("-inf")
        )
    k_actual = min(top_k, n_kv_blocks)
    _, top_idx = torch.topk(block_scores, k_actual, dim=-1)
    return top_idx.to(torch.int32).contiguous()


def selected_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size_n: int = 64,
    block_size_m: int = 64,
    top_k: int = 16,
    block_scores: Optional[torch.Tensor] = None,
    block_indices: Optional[torch.Tensor] = None,
    causal: bool = True,
    scale: Optional[float] = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Selected-branch forward in Triton.

    Args:
      Q: [B, H, Tq, D], fp16 or bf16.
      K, V: [B, H, Tk, D], same dtype as Q.
      block_size_n: KV block size.
      block_size_m: Q block size.
      top_k: number of KV blocks to gather per Q block.
      block_scores: [B, H, n_q_blocks, n_kv_blocks] (post-padding Tk). Either
        block_scores or block_indices must be provided.
      block_indices: [B, H, n_q_blocks, k_actual] precomputed top-k indices.
        If given, block_scores is ignored.
      causal: apply per-token causal masking inside each gathered block.
      scale: softmax scale (default 1/sqrt(D)).

    Returns:
      out: same shape and dtype as Q.
      lse: fp32 [B, H, Tq]. Equal to -inf for rows whose causal context is
        empty (no legal keys gathered).
    """
    assert Q.dtype in (torch.float16, torch.bfloat16), "Q must be fp16 or bf16"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    assert K.shape == V.shape == (B, H, Tk, D)

    sm_scale = _scale(D, scale)

    # Pad K/V to multiple of BLOCK_N and Q to multiple of BLOCK_M.
    pad_k = (block_size_n - (Tk % block_size_n)) % block_size_n
    pad_q = (block_size_m - (Tq % block_size_m)) % block_size_m
    if pad_k:
        K = F.pad(K, (0, 0, 0, pad_k))
        V = F.pad(V, (0, 0, 0, pad_k))
    if pad_q:
        Q = F.pad(Q, (0, 0, 0, pad_q))
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    Tq_p = Q.shape[2]
    Tk_p = K.shape[2]
    n_q_blocks = Tq_p // block_size_m
    n_kv_blocks = Tk_p // block_size_n

    # Resolve block_indices.
    if block_indices is None:
        assert block_scores is not None, (
            "selected_attention requires either block_scores or block_indices"
        )
        assert block_scores.shape == (B, H, n_q_blocks, n_kv_blocks), (
            f"block_scores shape {tuple(block_scores.shape)} != "
            f"{(B, H, n_q_blocks, n_kv_blocks)}"
        )
        block_indices = _topk_indices_from_scores(
            block_scores.float(),
            top_k=top_k,
            n_q_blocks=n_q_blocks,
            n_kv_blocks=n_kv_blocks,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            Tq=Tq, Tk=Tk,
            causal=causal,
            device=Q.device,
        )
    else:
        assert block_indices.shape[:3] == (B, H, n_q_blocks)
        block_indices = block_indices.to(torch.int32).contiguous()

    k_actual = block_indices.shape[3]

    out_p = torch.empty((B, H, Tq_p, D), dtype=Q.dtype, device=Q.device)
    lse_p = torch.empty((B, H, Tq_p), dtype=torch.float32, device=Q.device)

    # Fold (B, H) into the second program-id axis. The strides we pass let
    # the kernel address the right slice from a single bh index.
    # We also reshape Q/K/V/out/lse views to be indexed by a flat [BH, T, D]
    # layout, which simplifies the in-kernel arithmetic.
    Qv = Q.view(B * H, Tq_p, D)
    Kv = K.view(B * H, Tk_p, D)
    Vv = V.view(B * H, Tk_p, D)
    Ov = out_p.view(B * H, Tq_p, D)
    Lv = lse_p.view(B * H, Tq_p)
    Iv = block_indices.view(B * H, n_q_blocks, k_actual)

    grid = (n_q_blocks, B * H)

    _selected_attn_fwd[grid](
        Qv, Kv, Vv, Iv, Ov, Lv,
        # Q strides: (bh, m, d). The "stride_qb" slot is unused for the flat
        # layout but kept in the signature for clarity; we pass 0.
        0, Qv.stride(0), Qv.stride(1), Qv.stride(2),
        0, Kv.stride(0), Kv.stride(1), Kv.stride(2),
        0, Vv.stride(0), Vv.stride(1), Vv.stride(2),
        0, Iv.stride(0), Iv.stride(1), Iv.stride(2),
        0, Ov.stride(0), Ov.stride(1), Ov.stride(2),
        0, Lv.stride(0), Lv.stride(1),
        Tq_p, Tk_p, Tk - Tq, k_actual,
        sm_scale,
        CAUSAL=causal,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        HEAD_DIM=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = out_p
    lse = lse_p
    if pad_q:
        out = out[:, :, :Tq, :]
        lse = lse[:, :, :Tq]
    return out, lse


__all__ = ["selected_attention"]
