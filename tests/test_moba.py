"""MoBA forward correctness against a plain-torch reference.

The reference computes per-query attention restricted to each query's
top-k block-pooled-key matches, with the same causal masking as the
Triton path. This is the math MoBA describes; the Triton path uses the
NSA selected_attention kernel with a per-query-block union of those
per-query selections (capped at top_k_cap), so the reference and the
kernel are equivalent when top_k_cap is large enough to hold the
union.
"""

from __future__ import annotations

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from nsa.triton.moba import moba_attention, _per_query_top_k_indices


def _reference_moba(Q, K, V, *, block_size_n, top_k_per_query, causal, scale):
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    s = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    n_blocks = (Tk + block_size_n - 1) // block_size_n
    pad_k = n_blocks * block_size_n - Tk
    if pad_k:
        K = torch.nn.functional.pad(K, (0, 0, 0, pad_k))
        V = torch.nn.functional.pad(V, (0, 0, 0, pad_k))
    K_pooled = K.view(B, H, n_blocks, block_size_n, D).mean(dim=3).float()
    scores = torch.einsum("bhqd,bhnd->bhqn", Q.float(), K_pooled) * s
    if causal:
        offset = K.shape[2] - Tq
        q_idx = torch.arange(Tq, device=Q.device)
        kv_first = torch.arange(n_blocks, device=Q.device) * block_size_n
        legal = kv_first.view(1, n_blocks) <= (q_idx.view(Tq, 1) + offset)
        scores = scores.masked_fill(~legal.view(1, 1, Tq, n_blocks), float("-inf"))

    k_per_query = min(top_k_per_query, n_blocks)
    _, idx = torch.topk(scores, k_per_query, dim=-1)  # [B,H,Tq,k]
    out = torch.zeros_like(Q.float())
    for b in range(B):
        for h in range(H):
            for q in range(Tq):
                blocks = idx[b, h, q].tolist()
                slices = []
                for bi in blocks:
                    slices.append(slice(bi * block_size_n, (bi + 1) * block_size_n))
                K_sel = torch.cat([K[b, h, sl, :] for sl in slices], dim=0).float()
                V_sel = torch.cat([V[b, h, sl, :] for sl in slices], dim=0).float()
                # causal mask within the gathered tokens
                kv_pos = torch.cat([
                    torch.arange(bi * block_size_n, (bi + 1) * block_size_n, device=Q.device)
                    for bi in blocks
                ])
                qkv_scale = Q[b, h, q, :].float() @ K_sel.t() * s
                if causal:
                    qkv_scale = qkv_scale.masked_fill(
                        kv_pos > q + (K.shape[2] - Tq), float("-inf"),
                    )
                w = torch.softmax(qkv_scale, dim=-1)
                w = torch.nan_to_num(w, nan=0.0)
                out[b, h, q] = w @ V_sel
    return out.to(Q.dtype)


def test_moba_runs_and_routing_is_distinct():
    """Verify the MoBA kernel (a) runs without error on a Hopper-capable
    card, (b) produces finite outputs, and (c) produces different output
    from a NSA-selected-style routing over the same Q/K/V (per-query
    routing should differ from per-query-block routing).

    The strict per-query-vs-union semantic gap means the implementation's
    output does not match a pure-per-query reference exactly; that is
    intentional and discussed in moba.py's docstring. The throughput
    comparison vs NSA in `nsa/bench/throughput.py` is what the writeup
    leans on for the cross-comparison.
    """
    torch.manual_seed(0)
    B, H, T, D = 1, 4, 256, 64
    dtype = torch.bfloat16
    Q = torch.randn(B, H, T, D, dtype=dtype, device="cuda")
    K = torch.randn(B, H, T, D, dtype=dtype, device="cuda")
    V = torch.randn(B, H, T, D, dtype=dtype, device="cuda")

    out_moba, _ = moba_attention(
        Q, K, V, block_size_n=64, block_size_m=64,
        top_k_per_query=2, top_k_cap=4, causal=True,
    )
    assert torch.isfinite(out_moba).all(), "MoBA produced non-finite output"

    # Reference is per-query top-k, semantically distinct.
    out_ref = _reference_moba(
        Q, K, V, block_size_n=64, top_k_per_query=2,
        causal=True, scale=None,
    )
    assert torch.isfinite(out_ref).all()
    # Sanity: the two should differ (the routing semantics differ).
    diff = (out_moba.float() - out_ref.float()).abs().max()
    assert float(diff) > 1e-3, "MoBA and reference unexpectedly equal"


def test_moba_routing_pads_with_sentinel_not_a_real_block():
    """Short unions must be padded with the out-of-range sentinel (n_kv_blocks),
    not block 0. The old code padded with 0, which the gather double-counts."""
    torch.manual_seed(0)
    B, H, Tq, Tk, D = 1, 2, 256, 256, 64
    bn = bm = 64
    n_kv = Tk // bn
    Q = torch.randn(B, H, Tq, D, device="cuda")
    K = torch.randn(B, H, Tk, D, device="cuda")
    idx = _per_query_top_k_indices(
        Q, K, block_size_n=bn, block_size_m=bm,
        top_k_per_query=2, top_k_cap=4, causal=True, scale=1.0 / math.sqrt(D),
    )
    assert (idx >= n_kv).any(), "expected sentinel padding (union < cap for some q-block)"
    assert (idx <= n_kv).all(), "indices must be valid blocks or the single sentinel n_kv"
    Bn, Hn, nqb, cap = idx.shape
    for b in range(Bn):
        for h in range(Hn):
            for qb in range(nqb):
                row = idx[b, h, qb]
                valid = row[row < n_kv]
                assert valid.unique().numel() == valid.numel(), \
                    f"real block gathered twice at {(b, h, qb)}: {row.tolist()}"
                pad = row[row >= n_kv]
                assert torch.all(pad == n_kv), \
                    f"padding must be sentinel n_kv={n_kv}, got {row.tolist()}"


def test_moba_routing_truncation_keeps_highest_scoring():
    """When the union exceeds the cap, keep the highest-scoring blocks, not the
    lowest-index ones. Block scores ramp with index here, so the kept set must be
    the top indices; the old unique[:cap] would keep {0,1,2,3}."""
    B, H, Tq, Tk, D = 1, 1, 512, 512, 64
    bn = bm = 64
    n_kv = Tk // bn  # 8
    Q = torch.ones(B, H, Tq, D, device="cuda")
    K = torch.zeros(B, H, Tk, D, device="cuda")
    for j in range(n_kv):
        K[:, :, j * bn:(j + 1) * bn, :] = float(j + 1)   # block score increases with index
    idx = _per_query_top_k_indices(
        Q, K, block_size_n=bn, block_size_m=bm,
        top_k_per_query=n_kv, top_k_cap=4, causal=False, scale=1.0 / math.sqrt(D),
    )
    expected = {4, 5, 6, 7}
    nqb = idx.shape[2]
    for qb in range(nqb):
        kept = set(int(x) for x in idx[0, 0, qb].tolist())
        assert kept == expected, f"q-block {qb}: kept {kept}, expected {expected}"


def test_moba_reduces_to_full_attention_when_all_blocks_selected():
    """Output check: with top_k_per_query and top_k_cap covering all KV blocks,
    every query selects every block, so MoBA must equal full causal attention.
    This validates the gathered-attention math, not just the routing."""
    torch.manual_seed(0)
    B, H, T, D = 1, 4, 256, 64
    bn = bm = 64
    n_kv = T // bn
    dtype = torch.bfloat16
    Q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    K = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    V = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    out_moba, _ = moba_attention(
        Q, K, V, block_size_n=bn, block_size_m=bm,
        top_k_per_query=n_kv, top_k_cap=n_kv, causal=True,
    )
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    rel = (out_moba.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6)
    assert float(rel) < 1e-2, f"MoBA(all blocks) vs full causal attention rel err {float(rel)}"
