"""NSA sliding-window forward in Triton. FA-2 streaming softmax with a
per-tile range check: each Q tile streams only the K/V tiles that fall
inside its window. Window is [i + offset - W + 1, i + offset] causal
(or symmetric non-causal), matching sliding_attention_reference.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _sliding_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    so_b, so_h, so_t, so_d,
    sl_b, sl_h, sl_t,
    Tq, Tk, H, scale,
    WINDOW: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q_row_mask = offs_m < Tq

    # Base pointers for this (b, h).
    q_base = Q_ptr + b * sq_b + h * sq_h
    k_base = K_ptr + b * sk_b + h * sk_h
    v_base = V_ptr + b * sv_b + h * sv_h
    o_base = O_ptr + b * so_b + h * so_h
    lse_base = LSE_ptr + b * sl_b + h * sl_h

    # Load Q tile [BLOCK_M, BLOCK_D] and scale once.
    q_ptrs = q_base + offs_m[:, None] * sq_t + offs_d[None, :] * sq_d
    q = tl.load(q_ptrs, mask=q_row_mask[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    # Per-row streaming softmax state in fp32.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    offset = Tk - Tq
    # Range of K columns that any row in this Q tile can possibly attend to.
    # Rows in the tile span q positions [q_start, q_start + BLOCK_M).
    # Their key positions span [q_start + offset, q_start + BLOCK_M - 1 + offset]
    # before the window is applied.
    if CAUSAL:
        start_kv = q_start + offset - WINDOW + 1
        end_kv = q_start + BLOCK_M + offset
    else:
        start_kv = q_start + offset - WINDOW
        end_kv = q_start + BLOCK_M + offset + WINDOW
    # Clamp to [0, Tk).
    if start_kv < 0:
        start_kv = 0
    if end_kv > Tk:
        end_kv = Tk

    # Snap start to a BLOCK_N boundary so the inner loop is aligned and we can
    # rely on tile-uniform pointers. We then mask off out-of-window columns
    # with the per-element mask anyway.
    start_kv_aligned = (start_kv // BLOCK_N) * BLOCK_N

    # Per-query absolute key index (for masking).
    q_pos = offs_m + offset  # [BLOCK_M]

    for kv_off in range(start_kv_aligned, end_kv, BLOCK_N):
        k_col = kv_off + offs_n  # [BLOCK_N]
        col_in_range = k_col < Tk

        k_ptrs = k_base + k_col[:, None] * sk_t + offs_d[None, :] * sk_d
        v_ptrs = v_base + k_col[:, None] * sv_t + offs_d[None, :] * sv_d
        k = tl.load(k_ptrs, mask=col_in_range[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=col_in_range[:, None], other=0.0)

        # qk: [BLOCK_M, BLOCK_N]; q already includes scale.
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32)

        # Build window mask.
        if CAUSAL:
            in_window = (k_col[None, :] <= q_pos[:, None]) & (
                k_col[None, :] >= (q_pos[:, None] - WINDOW + 1)
            )
        else:
            in_window = (k_col[None, :] <= (q_pos[:, None] + WINDOW)) & (
                k_col[None, :] >= (q_pos[:, None] - WINDOW)
            )
        valid = in_window & col_in_range[None, :] & q_row_mask[:, None]
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # Replace -inf with a finite sentinel so the subtraction produces 0
        # weights without poisoning the running max.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        # Rows with no valid columns this tile keep p == 0, alpha handles the
        # accumulator scaling; m_i carries through unchanged via m_safe.
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # Finalize. Rows whose window is entirely empty (should not happen given
    # clamping above when 0 < W and Tk > 0) get lse = -inf and out = 0.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]
    lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))

    o_ptrs = o_base + offs_m[:, None] * so_t + offs_d[None, :] * so_d
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=q_row_mask[:, None])

    lse_ptrs = lse_base + offs_m * sl_t
    tl.store(lse_ptrs, lse, mask=q_row_mask)


def sliding_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int = 512,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton sliding-window attention forward.

    Args:
        Q: [B, H, Tq, D], fp16 or bf16.
        K: [B, H, Tk, D], same dtype as Q.
        V: [B, H, Tk, D], same dtype as Q.
        window_size: positive int, default 512.
        causal: if True use causal sliding window; else symmetric.
        scale: softmax scale; defaults to 1 / sqrt(D).

    Returns:
        (out, lse) where out has shape and dtype of Q and lse is fp32 [B, H, Tq].
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "sliding_attention requires CUDA tensors"
    assert Q.dtype in (torch.float16, torch.bfloat16), "Q dtype must be fp16 or bf16"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype, "K, V dtype must match Q"
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "head dims must match"
    assert K.shape[:2] == Q.shape[:2] and V.shape[:2] == Q.shape[:2]
    assert K.shape[2] == V.shape[2], "Tk must match between K and V"
    assert window_size >= 1, "window_size must be >= 1"

    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(D)

    # Triton's tl.dot needs the head dim to be a power of two and at least 16.
    # Padding the runtime D to the next pow2 lets us keep BLOCK_D as a constexpr
    # while still supporting odd head dims. For NSA we use D in (32, 64, 128).
    assert D in (16, 32, 64, 128, 256), f"unsupported head dim D={D}"

    # Make tensors contiguous to keep the strides predictable.
    Q_c = Q.contiguous()
    K_c = K.contiguous()
    V_c = V.contiguous()

    out = torch.empty_like(Q_c)
    lse = torch.empty((B, H, Tq), device=Q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # head dim is small enough; one tile covers it

    grid = (triton.cdiv(Tq, BLOCK_M), B * H)

    _sliding_fwd[grid](
        Q_c, K_c, V_c, out, lse,
        Q_c.stride(0), Q_c.stride(1), Q_c.stride(2), Q_c.stride(3),
        K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
        V_c.stride(0), V_c.stride(1), V_c.stride(2), V_c.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        Tq, Tk, H, sm_scale,
        WINDOW=int(window_size),
        CAUSAL=bool(causal),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return out, lse


__all__ = ["sliding_attention"]
