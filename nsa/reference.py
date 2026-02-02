"""Reference implementations of NSA's three branches in plain PyTorch.
Correctness ground truth for the Triton and CUDA kernels: clarity over
speed, fp32 accumulators regardless of input dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class NSAConfig:
    block_size_c: int = 64
    block_size_n: int = 64
    block_size_m: int = 64
    top_k: int = 16
    window_size: int = 512
    causal: bool = True
    pool: str = "mean"  # "mean" or "learned"
    gate_activation: str = "sigmoid"  # "sigmoid" or "softmax"
    scale: Optional[float] = None  # defaults to 1/sqrt(D)
    # Held over so the kernel paths can read NSAConfig directly:
    precision: str = "bf16"


def _scale(D: int, override: Optional[float]) -> float:
    return float(override) if override is not None else 1.0 / math.sqrt(D)


def attention_reference(
    Q: Tensor, K: Tensor, V: Tensor, *, causal: bool = True, scale: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    """Naive full-attention reference. fp32 internal regardless of input dtype.

    Returns (out, lse) where out has the input dtype and lse is fp32 with
    shape [B, H, Tq]. lse is the log-sum-exp of the unnormalized scores; it is
    convenient for splitting attention across kernels.
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    s = _scale(D, scale)

    # Use fp32 for fp16/bf16 inputs; preserve fp64 for fp64 inputs (gradcheck).
    internal = torch.float32 if Q.dtype.itemsize < 4 else Q.dtype
    Qf = Q.to(internal)
    Kf = K.to(internal)
    Vf = V.to(internal)

    scores = torch.einsum("bhqd,bhkd->bhqk", Qf, Kf) * s  # [B, H, Tq, Tk]
    if causal:
        # Tq queries align with the last Tq positions of Tk: the key index
        # corresponding to query position q is q + (Tk - Tq).
        offset = Tk - Tq
        i = torch.arange(Tq, device=Q.device).view(1, 1, Tq, 1)
        j = torch.arange(Tk, device=Q.device).view(1, 1, 1, Tk)
        mask = j > (i + offset)
        scores = scores.masked_fill(mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)  # [B, H, Tq]
    weights = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("bhqk,bhkd->bhqd", weights, Vf)
    return out.to(Q.dtype), lse


def compressed_attention_reference(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    block_size_c: int = 64,
    pool: str = "mean",
    pool_proj_k: Optional[Tensor] = None,
    pool_proj_v: Optional[Tensor] = None,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Compressed branch: aggregate K/V into blocks of size block_size_c, then attend.

    With pool="mean", each compressed key is the mean of block_size_c original
    keys; same for V. With pool="learned", the caller passes per-block linear
    projections (pool_proj_k, pool_proj_v) of shape [block_size_c, D] applied
    along the block dimension; the implementation uses einsum so D dims match.

    Causality, when on, requires that a query at position q only sees compressed
    blocks fully landing before or on q. Concretely, the highest legal block
    index for query q is floor((q + (Tk - Tq) + 1) / block_size_c) - 1.
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    if Tk % block_size_c != 0:
        pad = block_size_c - (Tk % block_size_c)
        K = F.pad(K, (0, 0, 0, pad))
        V = F.pad(V, (0, 0, 0, pad))
        Tk = K.shape[2]

    n_blocks = Tk // block_size_c
    K_blocks = K.view(B, H, n_blocks, block_size_c, D)
    V_blocks = V.view(B, H, n_blocks, block_size_c, D)

    if pool == "mean":
        K_c = K_blocks.mean(dim=3)  # [B, H, n_blocks, D]
        V_c = V_blocks.mean(dim=3)
    elif pool == "learned":
        assert pool_proj_k is not None and pool_proj_v is not None
        # pool_proj_*: [block_size_c, D] applied along the block-token dim.
        K_c = torch.einsum("bhnsd,sd->bhnd", K_blocks, pool_proj_k)
        V_c = torch.einsum("bhnsd,sd->bhnd", V_blocks, pool_proj_v)
    else:
        raise ValueError(f"unknown pool: {pool}")

    s = _scale(D, scale)
    internal = torch.float32 if Q.dtype.itemsize < 4 else Q.dtype
    scores = torch.einsum("bhqd,bhnd->bhqn", Q.to(internal), K_c.to(internal)) * s  # [B, H, Tq, n_blocks]

    if causal:
        offset = Tk - Tq
        q_idx = torch.arange(Tq, device=Q.device)
        # Highest block index legal for query q: each block covers indices
        # [c*block_size_c, (c+1)*block_size_c - 1]; require (c+1)*block_size_c - 1 <= q + offset.
        max_block = ((q_idx + offset + 1) // block_size_c) - 1  # [Tq]
        b_idx = torch.arange(n_blocks, device=Q.device).view(1, 1, 1, n_blocks)
        mask = b_idx > max_block.view(1, 1, Tq, 1)
        scores = scores.masked_fill(mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)  # [B, H, Tq]
    weights = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("bhqn,bhnd->bhqd", weights, V_c.to(internal))
    return out.to(Q.dtype), lse


def selected_attention_reference(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    block_size_n: int = 64,
    block_size_m: int = 64,
    top_k: int = 16,
    block_scores: Optional[Tensor] = None,
    block_indices: Optional[Tensor] = None,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Selected branch: per query block, pick top_k key blocks and attend over them.

    block_scores: [B, H, num_q_blocks, num_kv_blocks] importance scores. If None,
    falls back to using the compressed branch's score matrix derived from a
    mean-pooled K. Causality, when on, restricts the candidate set so a query
    block can only select KV blocks that lie at or before it.

    block_indices: [B, H, num_q_blocks, k] precomputed top-k. If supplied,
    block_scores is ignored and these indices are used directly. Useful when
    the reference is called inside autograd from a kernel test that wants the
    exact same gather pattern as the kernel saw.
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]

    # Pad K/V so Tk divides block_size_n, and Q so Tq divides block_size_m.
    pad_k = (block_size_n - (Tk % block_size_n)) % block_size_n
    pad_q = (block_size_m - (Tq % block_size_m)) % block_size_m
    if pad_k:
        K = F.pad(K, (0, 0, 0, pad_k))
        V = F.pad(V, (0, 0, 0, pad_k))
    if pad_q:
        Q = F.pad(Q, (0, 0, 0, pad_q))
    Tk_p = K.shape[2]
    Tq_p = Q.shape[2]
    n_q_blocks = Tq_p // block_size_m
    n_kv_blocks = Tk_p // block_size_n

    if block_indices is not None:
        # Caller-supplied indices override scoring entirely.
        top_idx = block_indices.to(torch.long)
        k_actual = top_idx.shape[-1]
        assert top_idx.shape == (B, H, n_q_blocks, k_actual), (
            f"block_indices shape {tuple(top_idx.shape)} != "
            f"{(B, H, n_q_blocks, k_actual)}"
        )
    else:
        if block_scores is None:
            # Default scorer: mean-pool K to block granularity, dot Q against it,
            # mean over the query block dim.
            K_blocks = K.view(B, H, n_kv_blocks, block_size_n, D).mean(dim=3)  # [B, H, n_kv, D]
            Q_blocks = Q.view(B, H, n_q_blocks, block_size_m, D).mean(dim=3)
            block_scores = torch.einsum("bhqd,bhkd->bhqk", Q_blocks.float(), K_blocks.float())

        if causal:
            # KV block c is allowed for Q block q iff c covers indices entirely
            # at or before Q block q's last covered index, accounting for offset.
            offset = Tk - Tq
            q_block_last_token = (
                torch.arange(n_q_blocks, device=Q.device) * block_size_m + (block_size_m - 1)
            )
            kv_block_first_token = torch.arange(n_kv_blocks, device=Q.device) * block_size_n
            # legal[q, c] = kv_block_first_token[c] <= q_block_last_token[q] + offset.
            legal = kv_block_first_token.view(1, n_kv_blocks) <= (
                q_block_last_token.view(n_q_blocks, 1) + offset
            )
            block_scores = block_scores.masked_fill(~legal.view(1, 1, n_q_blocks, n_kv_blocks), float("-inf"))

        k_actual = min(top_k, n_kv_blocks)
        _, top_idx = torch.topk(block_scores, k_actual, dim=-1)  # [B, H, n_q, k]

    # Per-query-block dense attention over the gathered KV blocks. Build a
    # gather index of shape [B, H, n_q, k * block_size_n] and use it to fetch.
    s = _scale(D, scale)
    internal = torch.float32 if Q.dtype.itemsize < 4 else Q.dtype
    out = torch.zeros((B, H, Tq_p, D), device=Q.device, dtype=internal)
    lse = torch.full((B, H, Tq_p), float("-inf"), device=Q.device, dtype=internal)

    for qb in range(n_q_blocks):
        q_slice = Q[:, :, qb * block_size_m : (qb + 1) * block_size_m, :].to(internal)  # [B, H, M, D]
        idx = top_idx[:, :, qb, :]  # [B, H, k]
        # Build per-batch-per-head gather: token indices = idx[..., :, None] * block_size_n + arange(block_size_n).
        token_arange = torch.arange(block_size_n, device=Q.device)
        token_idx = idx.unsqueeze(-1) * block_size_n + token_arange  # [B, H, k, N]
        token_idx = token_idx.view(B, H, k_actual * block_size_n)  # [B, H, k*N]

        gather_index = token_idx.unsqueeze(-1).expand(B, H, k_actual * block_size_n, D)
        K_gather = torch.gather(K, dim=2, index=gather_index).to(internal)
        V_gather = torch.gather(V, dim=2, index=gather_index).to(internal)

        scores = torch.einsum("bhmd,bhkd->bhmk", q_slice, K_gather) * s  # [B, H, M, k*N]

        if causal:
            offset = Tk - Tq
            q_pos = qb * block_size_m + torch.arange(block_size_m, device=Q.device) + offset
            # Re-derive each gathered token's original position to mask future tokens.
            scores = scores.masked_fill(token_idx.unsqueeze(-2) > q_pos.view(1, 1, block_size_m, 1), float("-inf"))

        lse_block = torch.logsumexp(scores, dim=-1)  # [B, H, M]
        weights = torch.exp(scores - lse_block.unsqueeze(-1))
        out_block = torch.einsum("bhmk,bhkd->bhmd", weights, V_gather)
        out[:, :, qb * block_size_m : (qb + 1) * block_size_m, :] = out_block
        lse[:, :, qb * block_size_m : (qb + 1) * block_size_m] = lse_block

    if pad_q:
        out = out[:, :, :Tq, :]
        lse = lse[:, :, :Tq]
    return out.to(Q.dtype), lse


def sliding_attention_reference(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    window_size: int = 512,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Sliding window: each query attends to the last window_size tokens (causal)
    or the symmetric window of size 2*window_size+1 (non-causal).
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    s = _scale(D, scale)

    # Use fp32 for fp16/bf16 inputs; preserve fp64 for fp64 inputs (gradcheck).
    internal = torch.float32 if Q.dtype.itemsize < 4 else Q.dtype
    Qf = Q.to(internal)
    Kf = K.to(internal)
    Vf = V.to(internal)
    scores = torch.einsum("bhqd,bhkd->bhqk", Qf, Kf) * s

    offset = Tk - Tq
    i = torch.arange(Tq, device=Q.device).view(1, 1, Tq, 1) + offset
    j = torch.arange(Tk, device=Q.device).view(1, 1, 1, Tk)
    if causal:
        mask = (j > i) | (j < (i - window_size + 1))
    else:
        mask = (j > (i + window_size)) | (j < (i - window_size))
    scores = scores.masked_fill(mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse.unsqueeze(-1))
    out = torch.einsum("bhqk,bhkd->bhqd", weights, Vf)
    return out.to(Q.dtype), lse


def nsa_attention_reference(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    cfg: NSAConfig,
    gates: Optional[Tensor] = None,
    *,
    pool_proj_k: Optional[Tensor] = None,
    pool_proj_v: Optional[Tensor] = None,
    block_scores: Optional[Tensor] = None,
) -> Tensor:
    """Combined NSA reference: run all three branches and combine via per-head gates.

    gates: [B, H, Tq, 3] in branch order (compressed, selected, sliding) or None
    (uniform 1/3 gating). gate_activation in cfg controls how gates are
    normalized: "sigmoid" applies sigmoid elementwise (independent branches),
    "softmax" applies softmax across the 3 branches (mutually exclusive).
    """
    out_c, _ = compressed_attention_reference(
        Q, K, V,
        block_size_c=cfg.block_size_c,
        pool=cfg.pool,
        pool_proj_k=pool_proj_k,
        pool_proj_v=pool_proj_v,
        causal=cfg.causal,
        scale=cfg.scale,
    )
    out_s, _ = selected_attention_reference(
        Q, K, V,
        block_size_n=cfg.block_size_n,
        block_size_m=cfg.block_size_m,
        top_k=cfg.top_k,
        block_scores=block_scores,
        causal=cfg.causal,
        scale=cfg.scale,
    )
    out_w, _ = sliding_attention_reference(
        Q, K, V,
        window_size=cfg.window_size,
        causal=cfg.causal,
        scale=cfg.scale,
    )

    B, H, Tq, _ = Q.shape
    if gates is None:
        gates = Q.new_full((B, H, Tq, 3), 1.0 / 3.0)
    if cfg.gate_activation == "sigmoid":
        g = torch.sigmoid(gates)
    elif cfg.gate_activation == "softmax":
        g = torch.softmax(gates, dim=-1)
    else:
        raise ValueError(f"unknown gate_activation: {cfg.gate_activation}")

    # gates: [B, H, Tq, 3]; outputs: [B, H, Tq, D].
    out = (
        g[..., 0:1] * out_c
        + g[..., 1:2] * out_s
        + g[..., 2:3] * out_w
    )
    return out


__all__ = [
    "NSAConfig",
    "attention_reference",
    "compressed_attention_reference",
    "selected_attention_reference",
    "sliding_attention_reference",
    "nsa_attention_reference",
]
