"""MoBA forward in Triton (Liu et al. 2025, arXiv 2502.13189). Per-query
top-k block routing aggregated to per-query-block union, dispatched
through the NSA selected_attention kernel. Used as a cross-comparison
column in nsa/bench/throughput.py.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from nsa.triton.selected import selected_attention


def _per_query_top_k_indices(
    Q: torch.Tensor, K: torch.Tensor,
    *, block_size_n: int, block_size_m: int, top_k_per_query: int,
    top_k_cap: int, causal: bool, scale: float,
) -> torch.Tensor:
    """Compute MoBA's per-query top-k block selection, then aggregate to
    per-query-block indices for the kernel.

    Returns [B, H, n_q_blocks, k_actual] int32 indices where k_actual is
    the deduplicated union of the BLOCK_M queries' top-k selections,
    padded to top_k_cap.
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_q = (block_size_m - Tq % block_size_m) % block_size_m
    pad_k = (block_size_n - Tk % block_size_n) % block_size_n
    if pad_q:
        Q = F.pad(Q, (0, 0, 0, pad_q))
    if pad_k:
        K = F.pad(K, (0, 0, 0, pad_k))
    n_q_blocks = Q.shape[2] // block_size_m
    n_kv_blocks = K.shape[2] // block_size_n

    K_pooled = K.view(B, H, n_kv_blocks, block_size_n, D).mean(dim=3).float()  # [B,H,nk,D]

    # Per-query block scores
    scores = torch.einsum("bhqd,bhnd->bhqn", Q.float(), K_pooled) * scale
    if causal:
        offset = K.shape[2] - Q.shape[2]
        q_idx = torch.arange(Q.shape[2], device=Q.device)
        kv_first = torch.arange(n_kv_blocks, device=Q.device) * block_size_n
        legal = kv_first.view(1, n_kv_blocks) <= (q_idx.view(Q.shape[2], 1) + offset)
        scores = scores.masked_fill(~legal.view(1, 1, Q.shape[2], n_kv_blocks), float("-inf"))

    k_per_query = min(top_k_per_query, n_kv_blocks)
    _, per_q_idx = torch.topk(scores, k_per_query, dim=-1)  # [B,H,T,k]

    # Aggregate to per-block: union over BLOCK_M queries' picks
    per_q_idx = per_q_idx.view(B, H, n_q_blocks, block_size_m * k_per_query)
    block_idx_list = []
    cap = min(top_k_cap, n_kv_blocks)
    for b in range(B):
        for h in range(H):
            for qb in range(n_q_blocks):
                unique = torch.unique(per_q_idx[b, h, qb])
                if unique.numel() < cap:
                    pad = unique.new_zeros(cap - unique.numel())
                    unique = torch.cat([unique, pad], dim=0)
                else:
                    unique = unique[:cap]
                block_idx_list.append(unique)
    out_idx = torch.stack(block_idx_list, dim=0).view(B, H, n_q_blocks, cap)
    return out_idx.to(torch.int32).contiguous()


def moba_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    block_size_n: int = 64,
    block_size_m: int = 64,
    top_k_per_query: int = 16,
    top_k_cap: int = 64,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MoBA-style attention: per-query top-k block routing, then dense
    attention over the union-gathered blocks.

    top_k_cap upper-bounds the per-block gather slot count so the
    selected-attention kernel stays inside one tile per query block.
    """
    D = Q.shape[-1]
    s = float(scale) if scale is not None else 1.0 / math.sqrt(D)

    block_indices = _per_query_top_k_indices(
        Q, K,
        block_size_n=block_size_n, block_size_m=block_size_m,
        top_k_per_query=top_k_per_query, top_k_cap=top_k_cap,
        causal=causal, scale=s,
    )
    out, lse = selected_attention(
        Q, K, V,
        block_size_n=block_size_n, block_size_m=block_size_m,
        top_k=top_k_cap, block_indices=block_indices,
        causal=causal, scale=scale,
    )
    return out, lse


__all__ = ["moba_attention"]
