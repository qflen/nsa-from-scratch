"""Correctness for the Hopper WGMMA selected-branch forward against
the Triton kernel. Skipped off Hopper. Headline shape asserts the
tighter 1e-3 rel error; sweep uses 1e-2 (bf16 streaming softmax).
"""

from __future__ import annotations


import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

# Hopper-only.
_cap = torch.cuda.get_device_capability(0)
if _cap[0] < 9:
    pytest.skip(f"Hopper sm_90 required (got sm_{_cap[0]}{_cap[1]})", allow_module_level=True)

from nsa.triton.selected import selected_attention as triton_selected
from nsa.cuda import selected_attention_cuda


def _make_qkv(B, H, Tq, Tk, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, Tq, D, dtype=dtype, device="cuda", generator=g)
    K = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g)
    V = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g)
    return Q, K, V


def _make_block_scores(B, H, n_q, n_kv, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    return torch.randn(B, H, n_q, n_kv, dtype=torch.float32, device="cuda", generator=g)


def _causal_block_legal_mask(n_q, n_kv, BM, BN, Tq, Tk, device):
    offset = Tk - Tq
    q_block_last_token = (
        torch.arange(n_q, device=device) * BM + (BM - 1)
    )
    kv_block_first_token = torch.arange(n_kv, device=device) * BN
    legal = kv_block_first_token.view(1, n_kv) <= (
        q_block_last_token.view(n_q, 1) + offset
    )
    return legal


def _shared_topk_indices(block_scores, top_k, BM, BN, Tq, Tk, causal):
    B, H, n_q, n_kv = block_scores.shape
    bs = block_scores.float().clone()
    if causal:
        legal = _causal_block_legal_mask(n_q, n_kv, BM, BN, Tq, Tk, block_scores.device)
        bs = bs.masked_fill(~legal.view(1, 1, n_q, n_kv), float("-inf"))
    k_actual = min(top_k, n_kv)
    _, idx = torch.topk(bs, k_actual, dim=-1)
    return idx.to(torch.int32).contiguous()


def _max_rel(out, ref):
    diff = (out.float() - ref.float()).abs().max()
    denom = ref.float().abs().max() + 1e-6
    return float(diff / denom)


def _pad_qkv(Q, K, V, BM, BN):
    """Pad Q to multiple of BM and K/V to multiple of BN (the CUDA wrapper
    requires the caller to pad; the Triton wrapper does it internally
    so the same shape is mimicked here for the CUDA call)."""
    import torch.nn.functional as F

    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_q = (BM - (Tq % BM)) % BM
    pad_k = (BN - (Tk % BN)) % BN
    if pad_q:
        Q = F.pad(Q, (0, 0, 0, pad_q))
    if pad_k:
        K = F.pad(K, (0, 0, 0, pad_k))
        V = F.pad(V, (0, 0, 0, pad_k))
    return Q.contiguous(), K.contiguous(), V.contiguous(), Tq, Tk, pad_q, pad_k


# ---------------------------------------------------------------------------
# Build sanity check.
# ---------------------------------------------------------------------------
def test_extension_loads():
    # Importing the package triggers JIT compile lazily; force the
    # extension to load to surface compile errors cleanly.
    from nsa.cuda import _get_ext
    ext = _get_ext()
    assert hasattr(ext, "selected_attention_fwd_cuda")


# ---------------------------------------------------------------------------
# Headline shape: (B=1, H=8, Tq=1024, Tk=2048, D=64, top_k=8, BM=BN=64).
# ---------------------------------------------------------------------------
def test_cuda_matches_triton_headline():
    B, H, Tq, Tk, D, top_k = 1, 8, 1024, 2048, 64, 8
    BM, BN = 64, 64
    dtype = torch.bfloat16

    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=0)
    n_q = Tq // BM
    n_kv = Tk // BN
    bs = _make_block_scores(B, H, n_q, n_kv, seed=1)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq, Tk=Tk, causal=True)

    # Triton path uses unpadded shapes (it pads internally).
    out_t, lse_t = triton_selected(
        Q, K, V, block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_indices=indices, causal=True,
    )

    # CUDA path: caller pads. Tq, Tk are already multiples of BM, BN here.
    Qp, Kp, Vp, Tq_orig, Tk_orig, pad_q, pad_k = _pad_qkv(Q, K, V, BM, BN)
    out_c, lse_c = selected_attention_cuda(
        Qp, Kp, Vp, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=True, scale=None,
    )
    if pad_q:
        out_c = out_c[:, :, :Tq_orig, :]
        lse_c = lse_c[:, :, :Tq_orig]

    assert out_c.shape == out_t.shape
    assert lse_c.shape == lse_t.shape

    rel = _max_rel(out_c, out_t)
    # Brief asks for 1e-3 at this shape. Keep that as the headline bar.
    assert rel < 1e-3, f"out rel err {rel} above the 1e-3 headline bar"

    finite = torch.isfinite(lse_c) & torch.isfinite(lse_t)
    diff = (lse_c.float() - lse_t.float()).abs()[finite]
    if diff.numel() > 0:
        max_diff = float(diff.max())
        assert max_diff < 1e-2, f"lse abs err {max_diff} too large"

    mismatch = torch.isfinite(lse_c) ^ torch.isfinite(lse_t)
    assert not mismatch.any(), "lse finiteness pattern differs from Triton"

    print(f"headline shape: rel(out)={rel:.4e}, "
          f"max|lse_diff|={float(diff.max()):.4e}")


# ---------------------------------------------------------------------------
# Smaller shape sweep.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [
    (2, 4, 512, 1024, 64, 4, 64, 64),
    # D=128 case included; the kernel has a templated specialization.
    (1, 8, 2048, 4096, 128, 8, 64, 64),
])
def test_cuda_matches_triton_sweep(shape):
    B, H, Tq, Tk, D, top_k, BM, BN = shape
    dtype = torch.bfloat16

    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=2)
    n_q = Tq // BM
    n_kv = Tk // BN
    bs = _make_block_scores(B, H, n_q, n_kv, seed=3)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq, Tk=Tk, causal=True)

    out_t, lse_t = triton_selected(
        Q, K, V, block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_indices=indices, causal=True,
    )

    Qp, Kp, Vp, Tq_orig, _, pad_q, _ = _pad_qkv(Q, K, V, BM, BN)
    out_c, lse_c = selected_attention_cuda(
        Qp, Kp, Vp, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=True, scale=None,
    )
    if pad_q:
        out_c = out_c[:, :, :Tq_orig, :]
        lse_c = lse_c[:, :, :Tq_orig]

    rel = _max_rel(out_c, out_t)
    assert rel < 1e-2, f"shape {shape}: out rel err {rel}"

    finite = torch.isfinite(lse_c) & torch.isfinite(lse_t)
    diff = (lse_c.float() - lse_t.float()).abs()[finite]
    if diff.numel() > 0:
        assert float(diff.max()) < 1e-2, f"shape {shape}: lse diff {float(diff.max())}"


# ---------------------------------------------------------------------------
# Causal off / on toggle: verify no leak in either path.
# ---------------------------------------------------------------------------
def test_cuda_causal_off_matches_triton():
    B, H, Tq, Tk, D, top_k = 1, 4, 512, 1024, 64, 4
    BM, BN = 64, 64
    dtype = torch.bfloat16

    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=4)
    n_q = Tq // BM
    n_kv = Tk // BN
    bs = _make_block_scores(B, H, n_q, n_kv, seed=5)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq, Tk=Tk, causal=False)

    out_t, lse_t = triton_selected(
        Q, K, V, block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_indices=indices, causal=False,
    )

    Qp, Kp, Vp, Tq_orig, _, pad_q, _ = _pad_qkv(Q, K, V, BM, BN)
    out_c, lse_c = selected_attention_cuda(
        Qp, Kp, Vp, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=False, scale=None,
    )
    if pad_q:
        out_c = out_c[:, :, :Tq_orig, :]
        lse_c = lse_c[:, :, :Tq_orig]

    rel = _max_rel(out_c, out_t)
    assert rel < 1e-2, f"causal=False: out rel err {rel}"
    diff = (lse_c.float() - lse_t.float()).abs()
    assert float(diff.max()) < 1e-2
