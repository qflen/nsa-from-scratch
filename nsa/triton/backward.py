"""Hand-written Triton backward for the NSA selected branch. FA-2 bwd
(Dao 2023) adapted to the top-k gather: dQ local, dK/dV via atomic_add
in fp32 buffers cast back at the end. Pre-step D = (dO*O).sum(-1) in
fp32 lets us compute dS = P*(dP - D) per tile.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bwd_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, LSE_ptr, D_ptr,
    idx_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_idxb, stride_idxh, stride_idxq, stride_idxk,
    H,
    Tq, Tk,
    OFFSET,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOP_K: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H

    q_start = pid_q * BLOCK_M

    q_base = Q_ptr + batch * stride_qb + head * stride_qh
    k_base = K_ptr + batch * stride_kb + head * stride_kh
    v_base = V_ptr + batch * stride_kb + head * stride_kh
    do_base = dO_ptr + batch * stride_qb + head * stride_qh
    dq_base = dQ_ptr + batch * stride_qb + head * stride_qh
    dk_base = dK_ptr + batch * stride_kb + head * stride_kh
    dv_base = dV_ptr + batch * stride_kb + head * stride_kh
    lse_base = LSE_ptr + (batch * H + head) * Tq
    Dvec_base = D_ptr + (batch * H + head) * Tq
    idx_base = idx_ptr + batch * stride_idxb + head * stride_idxh + pid_q * stride_idxq

    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_mask = offs_m < Tq
    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    Q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    do_ptrs = do_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    dO = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)

    lse = tl.load(lse_base + offs_m, mask=q_mask, other=-1.0e30)
    # Clamp -inf rows (no legal context) to a finite floor so exp is finite.
    lse = tl.where(lse > -1.0e30, lse, -1.0e30)
    Dvec = tl.load(Dvec_base + offs_m, mask=q_mask, other=0.0)

    q_pos = offs_m + OFFSET
    q_pos_max = q_start + BLOCK_M - 1 + OFFSET

    dQ_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for kk in range(TOP_K):
        block_idx = tl.load(idx_base + kk * stride_idxk)
        kv_start = block_idx * BLOCK_N
        kv_offs = kv_start + offs_n
        kv_mask = kv_offs < Tk

        # Coarse causal early-out: if entire block lies past q_pos_max.
        if CAUSAL:
            skip = kv_start > q_pos_max
        else:
            skip = False
        if not skip:
            k_ptrs = k_base + kv_offs[:, None] * stride_kt + offs_d[None, :] * stride_kd
            v_ptrs = v_base + kv_offs[:, None] * stride_kt + offs_d[None, :] * stride_kd
            K = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
            V = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

            # S = Q @ K^T * scale (fp32)
            S = tl.dot(Q, tl.trans(K)).to(tl.float32) * sm_scale

            # Per-element causal mask + valid-row + valid-col mask.
            valid = q_mask[:, None] & kv_mask[None, :]
            if CAUSAL:
                causal_mask = kv_offs[None, :] <= q_pos[:, None]
                valid = valid & causal_mask
            S = tl.where(valid, S, -1.0e30)

            P = tl.exp(S - lse[:, None])
            P = tl.where(valid, P, 0.0)

            # dV: P^T @ dO  ([BLOCK_N, HEAD_DIM])
            dV_contrib = tl.dot(tl.trans(P).to(dO.dtype), dO).to(tl.float32)
            dv_ptrs = dv_base + kv_offs[:, None] * stride_kt + offs_d[None, :] * stride_kd
            tl.atomic_add(dv_ptrs, dV_contrib, mask=kv_mask[:, None])

            # dP = dO @ V^T  ([BLOCK_M, BLOCK_N])
            dP = tl.dot(dO, tl.trans(V)).to(tl.float32)

            # dS = P * (dP - D) * scale
            dS = P * (dP - Dvec[:, None]) * sm_scale
            dS = tl.where(valid, dS, 0.0)

            # dQ accumulate
            dQ_acc += tl.dot(dS.to(K.dtype), K).to(tl.float32)

            # dK contrib: dS^T @ Q  ([BLOCK_N, HEAD_DIM])
            dK_contrib = tl.dot(tl.trans(dS).to(Q.dtype), Q).to(tl.float32)
            dk_ptrs = dk_base + kv_offs[:, None] * stride_kt + offs_d[None, :] * stride_kd
            tl.atomic_add(dk_ptrs, dK_contrib, mask=kv_mask[:, None])

    dq_ptrs = dq_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dQ_acc.to(Q.dtype), mask=q_mask[:, None])


def selected_backward(
    dO: torch.Tensor,                # [B, H, Tq_p, D] same dtype as Q
    Q: torch.Tensor,                 # [B, H, Tq_p, D]
    K: torch.Tensor,                 # [B, H, Tk_p, D]
    V: torch.Tensor,                 # [B, H, Tk_p, D]
    O: torch.Tensor,                 # [B, H, Tq_p, D]
    LSE: torch.Tensor,               # [B, H, Tq_p] fp32
    block_indices: torch.Tensor,     # [B, H, n_q_blocks, top_k] int32
    *,
    block_size_m: int,
    block_size_n: int,
    causal: bool,
    scale: float,
    Tq: int,                         # original (pre-pad) Tq
    Tk: int,                         # original (pre-pad) Tk
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-written Triton backward for the selected branch.

    Inputs are PADDED (multiples of block_size_m / block_size_n). Output
    grads are also padded; caller unpads. dQ has the same dtype as Q;
    dK / dV likewise (internally fp32 buffers, cast at the end).
    """
    assert Q.dtype in (torch.float16, torch.bfloat16), "Triton bwd: Q must be fp16/bf16"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype and O.dtype == Q.dtype
    assert dO.dtype == Q.dtype
    assert LSE.dtype == torch.float32
    assert block_indices.dtype == torch.int32
    B, H, Tq_p, D = Q.shape
    Tk_p = K.shape[2]
    n_q_blocks = Tq_p // block_size_m
    top_k = block_indices.shape[-1]

    # Pre-step: D = (dO * O).sum(-1) in fp32.
    D_vec = (dO.float() * O.float()).sum(dim=-1).contiguous()  # [B, H, Tq_p]

    # Allocate gradient buffers in fp32 to keep atomics numerically clean.
    dQ_f = torch.zeros_like(Q, dtype=torch.float32)
    dK_f = torch.zeros_like(K, dtype=torch.float32)
    dV_f = torch.zeros_like(V, dtype=torch.float32)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    dO = dO.contiguous()
    O = O.contiguous()
    LSE = LSE.contiguous()
    block_indices = block_indices.contiguous()

    grid = (n_q_blocks, B * H)
    _bwd_kernel[grid](
        Q, K, V, dO, LSE, D_vec,
        block_indices,
        dQ_f, dK_f, dV_f,
        scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        block_indices.stride(0), block_indices.stride(1),
        block_indices.stride(2), block_indices.stride(3),
        H,
        Tq_p, Tk_p,
        Tk_p - Tq_p,
        HEAD_DIM=D,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        TOP_K=top_k,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return dQ_f.to(Q.dtype), dK_f.to(K.dtype), dV_f.to(V.dtype)


__all__ = ["selected_backward"]
