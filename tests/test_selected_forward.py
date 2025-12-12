"""Correctness for the Triton selected-branch forward against the
reference. block_indices is precomputed and shared so we are not
chasing topk tie-breaking. Tol: 1e-2 rel for out, abs for lse.
"""

from __future__ import annotations

import math

import pytest
import torch

# Skip the whole module if CUDA is unavailable. Triton requires a GPU.
if not torch.cuda.is_available():
    pytest.skip("CUDA required for selected-branch kernel tests", allow_module_level=True)

from nsa.reference import (
    attention_reference,
    selected_attention_reference,
)
from nsa.triton.selected import selected_attention


def _make_qkv(B, H, Tq, Tk, D, dtype, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(B, H, Tq, D, dtype=dtype, device=device, generator=g)
    K = torch.randn(B, H, Tk, D, dtype=dtype, device=device, generator=g)
    V = torch.randn(B, H, Tk, D, dtype=dtype, device=device, generator=g)
    return Q, K, V


def _make_block_scores(B, H, n_q, n_kv, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed + 1)
    return torch.randn(B, H, n_q, n_kv, dtype=torch.float32, device=device, generator=g)


def _max_rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (out.float() - ref.float()).abs().max()
    denom = ref.float().abs().max() + 1e-6
    return float(diff / denom)


def _causal_block_legal_mask(n_q, n_kv, block_size_m, block_size_n, Tq, Tk, device):
    offset = Tk - Tq
    q_block_last_token = (
        torch.arange(n_q, device=device) * block_size_m + (block_size_m - 1)
    )
    kv_block_first_token = torch.arange(n_kv, device=device) * block_size_n
    legal = kv_block_first_token.view(1, n_kv) <= (
        q_block_last_token.view(n_q, 1) + offset
    )
    return legal  # [n_q, n_kv]


def _shared_topk_indices(block_scores, top_k, block_size_m, block_size_n, Tq, Tk, causal):
    """Compute the same top-k indices the kernel uses, with causal masking applied,
    so we can hand identical block_indices to both the reference and the kernel.

    Returns int64 indices on the same device.
    """
    B, H, n_q, n_kv = block_scores.shape
    bs = block_scores.float().clone()
    if causal:
        legal = _causal_block_legal_mask(
            n_q, n_kv, block_size_m, block_size_n, Tq, Tk, block_scores.device
        )
        bs = bs.masked_fill(~legal.view(1, 1, n_q, n_kv), float("-inf"))
    k_actual = min(top_k, n_kv)
    _, idx = torch.topk(bs, k_actual, dim=-1)
    return idx  # int64


def _reference_with_fixed_indices(
    Q, K, V, *, block_indices, block_size_n, block_size_m, causal, scale=None
):
    """A direct port of selected_attention_reference but accepting block_indices
    instead of recomputing them. This eliminates any disagreement caused by
    tie-breaking inside torch.topk between the reference's masking path and
    the kernel wrapper's masking path. They use the same input here, but we
    play it safe.
    """
    import torch.nn.functional as F

    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_k = (block_size_n - (Tk % block_size_n)) % block_size_n
    pad_q = (block_size_m - (Tq % block_size_m)) % block_size_m
    if pad_k:
        K = F.pad(K, (0, 0, 0, pad_k))
        V = F.pad(V, (0, 0, 0, pad_k))
    if pad_q:
        Q = F.pad(Q, (0, 0, 0, pad_q))
    Tk_p = K.shape[2]
    Tq_p = Q.shape[2]
    n_q = Tq_p // block_size_m

    s = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    out = torch.zeros((B, H, Tq_p, D), device=Q.device, dtype=torch.float32)
    lse = torch.full((B, H, Tq_p), float("-inf"), device=Q.device, dtype=torch.float32)
    k_actual = block_indices.shape[3]

    for qb in range(n_q):
        q_slice = Q[:, :, qb * block_size_m : (qb + 1) * block_size_m, :].float()
        idx = block_indices[:, :, qb, :].to(torch.long)
        token_arange = torch.arange(block_size_n, device=Q.device)
        token_idx = idx.unsqueeze(-1) * block_size_n + token_arange
        token_idx = token_idx.view(B, H, k_actual * block_size_n)
        gather_index = token_idx.unsqueeze(-1).expand(B, H, k_actual * block_size_n, D)
        K_gather = torch.gather(K, dim=2, index=gather_index).float()
        V_gather = torch.gather(V, dim=2, index=gather_index).float()
        scores = torch.einsum("bhmd,bhkd->bhmk", q_slice, K_gather) * s
        if causal:
            offset = Tk - Tq
            q_pos = qb * block_size_m + torch.arange(block_size_m, device=Q.device) + offset
            scores = scores.masked_fill(
                token_idx.unsqueeze(-2) > q_pos.view(1, 1, block_size_m, 1),
                float("-inf"),
            )
        # Handle rows that are entirely -inf: lse = -inf, weights = 0.
        max_per_row = scores.amax(dim=-1, keepdim=True)
        finite = torch.isfinite(max_per_row)
        safe_max = torch.where(finite, max_per_row, torch.zeros_like(max_per_row))
        ex = torch.exp(scores - safe_max)
        ex = torch.where(torch.isfinite(scores), ex, torch.zeros_like(ex))
        denom = ex.sum(dim=-1, keepdim=True)
        weights = torch.where(denom > 0, ex / denom, torch.zeros_like(ex))
        out_block = torch.einsum("bhmk,bhkd->bhmd", weights, V_gather)
        lse_block = torch.where(
            finite.squeeze(-1),
            (max_per_row.squeeze(-1) + torch.log(denom.squeeze(-1).clamp_min(1e-30))),
            torch.full_like(max_per_row.squeeze(-1), float("-inf")),
        )
        out[:, :, qb * block_size_m : (qb + 1) * block_size_m, :] = out_block
        lse[:, :, qb * block_size_m : (qb + 1) * block_size_m] = lse_block

    if pad_q:
        out = out[:, :, :Tq, :]
        lse = lse[:, :, :Tq]
    return out.to(Q.dtype), lse


SHAPES = [
    # (B, H, Tq, Tk, D, top_k, BLOCK_M, BLOCK_N)
    (1, 4, 128, 128, 64, 2, 32, 32),
    (1, 8, 1024, 2048, 64, 8, 64, 64),
    (2, 8, 512, 1024, 64, 4, 64, 64),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_selected_kernel_matches_reference(shape, dtype):
    B, H, Tq, Tk, D, top_k, BM, BN = shape
    torch.manual_seed(0)
    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=0)

    n_q_blocks = (Tq + BM - 1) // BM
    n_kv_blocks = (Tk + BN - 1) // BN
    block_scores = _make_block_scores(B, H, n_q_blocks, n_kv_blocks, seed=1)

    # Compute one shared set of indices, hand it to both paths.
    indices = _shared_topk_indices(
        block_scores, top_k=top_k,
        block_size_m=BM, block_size_n=BN,
        Tq=Tq, Tk=Tk, causal=True,
    )

    out_ref, lse_ref = _reference_with_fixed_indices(
        Q, K, V, block_indices=indices, block_size_n=BN, block_size_m=BM, causal=True
    )
    out_k, lse_k = selected_attention(
        Q, K, V,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_indices=indices, causal=True,
    )

    assert out_k.shape == (B, H, Tq, D)
    assert lse_k.shape == (B, H, Tq)
    assert out_k.dtype == Q.dtype

    rel = _max_rel(out_k, out_ref)
    assert rel < 1e-2, f"out rel err {rel} too large for {shape}, dtype={dtype}"

    # LSE: only compare rows where both are finite. Rows with no causally-legal
    # key produce -inf in both implementations; subtracting them gives NaN.
    finite = torch.isfinite(lse_ref) & torch.isfinite(lse_k)
    diff = (lse_k.float() - lse_ref.float()).abs()
    diff = diff[finite]
    if diff.numel() > 0:
        max_diff = float(diff.max())
        assert max_diff < 1e-2, f"lse diff {max_diff} too large for {shape}, dtype={dtype}"
    # Where one is finite and the other is not, that is a real disagreement.
    mismatch = torch.isfinite(lse_ref) ^ torch.isfinite(lse_k)
    assert not mismatch.any(), "lse finiteness pattern differs from reference"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_selected_topk_full_matches_full_attention(dtype):
    """When every KV block is selected, selected attention must equal full causal."""
    B, H, Tq, Tk, D = 1, 4, 256, 256, 64
    BM = BN = 64
    n_q_blocks = Tq // BM
    n_kv_blocks = Tk // BN

    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=2)

    # Make block_scores favor in-order picking; with top_k == n_kv_blocks the
    # exact ranking does not matter, every block is selected.
    block_scores = torch.zeros(B, H, n_q_blocks, n_kv_blocks, device="cuda")
    block_scores += torch.arange(n_kv_blocks, device="cuda").float().view(1, 1, 1, -1)

    indices = _shared_topk_indices(
        block_scores, top_k=n_kv_blocks,
        block_size_m=BM, block_size_n=BN,
        Tq=Tq, Tk=Tk, causal=True,
    )

    out_k, lse_k = selected_attention(
        Q, K, V,
        block_size_n=BN, block_size_m=BM, top_k=n_kv_blocks,
        block_indices=indices, causal=True,
    )

    out_full, lse_full = attention_reference(Q, K, V, causal=True)

    rel = _max_rel(out_k, out_full)
    assert rel < 1e-2, f"saturated-top-k vs full-attention rel err {rel}"
    assert (lse_k - lse_full).abs().max() < 1e-2


def test_selected_kernel_matches_reference_via_block_scores():
    """End-to-end path: pass block_scores, let the wrapper do top-k. Make sure
    we still match the original selected_attention_reference (which does the
    same top-k). This is the realistic call path from the NSA forward.
    """
    B, H, Tq, Tk, D = 1, 8, 1024, 2048, 64
    BM = BN = 64
    top_k = 8
    dtype = torch.float16

    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=3)
    n_q_blocks = Tq // BM
    n_kv_blocks = Tk // BN
    block_scores = _make_block_scores(B, H, n_q_blocks, n_kv_blocks, seed=4)

    out_ref, lse_ref = selected_attention_reference(
        Q, K, V,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_scores=block_scores, causal=True,
    )
    out_k, lse_k = selected_attention(
        Q, K, V,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_scores=block_scores, causal=True,
    )

    rel = _max_rel(out_k, out_ref)
    assert rel < 1e-2, f"end-to-end rel err {rel}"
    finite = torch.isfinite(lse_ref) & torch.isfinite(lse_k)
    diff = (lse_k.float() - lse_ref.float()).abs()[finite]
    if diff.numel() > 0:
        assert float(diff.max()) < 1e-2
