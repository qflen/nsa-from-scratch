"""Correctness for the Hopper WGMMA selected-branch backward vs the
Triton bwd at the same inputs. Skipped off Hopper. Headline shape asserts
1e-3 rel; sweep uses 1e-2 (bwd amplifies atomic-add nondeterminism).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

_cap = torch.cuda.get_device_capability(0)
if _cap[0] < 9:
    pytest.skip(f"Hopper sm_90 required (got sm_{_cap[0]}{_cap[1]})", allow_module_level=True)

from nsa.triton.selected import selected_attention as triton_selected
from nsa.triton.backward import selected_backward as triton_selected_bwd
from nsa.cuda import selected_backward_cuda


def _make_qkv_do(B, H, Tq, Tk, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q  = torch.randn(B, H, Tq, D, dtype=dtype, device="cuda", generator=g)
    K  = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g)
    V  = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g)
    dO = torch.randn(B, H, Tq, D, dtype=dtype, device="cuda", generator=g)
    return Q, K, V, dO


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


def _pad(Q, K, V, dO, BM, BN):
    """Pad Q/dO to multiple of BM, K/V to multiple of BN.

    The CUDA backward (and Triton backward) take already-padded inputs;
    this helper performs that padding so we can drive both with the
    same tensors. We zero-pad: the kernels mask Q-padding via the
    Q_rows_valid check inside the WGMMA loops.
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_q = (BM - (Tq % BM)) % BM
    pad_k = (BN - (Tk % BN)) % BN
    if pad_q:
        Q  = F.pad(Q,  (0, 0, 0, pad_q))
        dO = F.pad(dO, (0, 0, 0, pad_q))
    if pad_k:
        K = F.pad(K, (0, 0, 0, pad_k))
        V = F.pad(V, (0, 0, 0, pad_k))
    return Q.contiguous(), K.contiguous(), V.contiguous(), dO.contiguous(), Tq, Tk, pad_q, pad_k


def _run_fwd_for_O_LSE(Qp, Kp, Vp, indices, BM, BN, top_k, causal):
    """Run the Triton forward on padded inputs to get O and LSE that
    feed both backward implementations. The forward returns unpadded
    (out, lse) shapes; the backward needs padded versions.
    """
    B, H, Tq_p, D = Qp.shape
    out, lse = triton_selected(
        Qp, Kp, Vp, block_size_n=BN, block_size_m=BM, top_k=top_k,
        block_indices=indices, causal=causal,
    )
    # Triton fwd unpads when input is padded; here Tq_p == Tq_p (already padded)
    # so out and lse are at padded length already.
    assert out.shape[2] == Tq_p, f"expected padded fwd out: {out.shape[2]} vs {Tq_p}"
    return out.contiguous(), lse.contiguous()


# ---------------------------------------------------------------------------
# Build sanity check.
# ---------------------------------------------------------------------------
def test_bwd_extension_loads():
    from nsa.cuda import _get_ext
    ext = _get_ext()
    assert hasattr(ext, "selected_attention_bwd_cuda")


# ---------------------------------------------------------------------------
# Headline shape.
# ---------------------------------------------------------------------------
def test_cuda_bwd_matches_triton_headline():
    B, H, Tq, Tk, D, top_k = 1, 8, 1024, 2048, 64, 8
    BM, BN = 64, 64
    dtype = torch.bfloat16

    Q, K, V, dO = _make_qkv_do(B, H, Tq, Tk, D, dtype, seed=0)
    n_q = Tq // BM
    n_kv = Tk // BN
    bs = _make_block_scores(B, H, n_q, n_kv, seed=1)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq, Tk=Tk, causal=True)

    Qp, Kp, Vp, dOp, Tq0, Tk0, pad_q, pad_k = _pad(Q, K, V, dO, BM, BN)
    O_p, LSE_p = _run_fwd_for_O_LSE(Qp, Kp, Vp, indices, BM, BN, top_k, causal=True)
    scale = 1.0 / math.sqrt(D)

    dQt, dKt, dVt = triton_selected_bwd(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_m=BM, block_size_n=BN,
        causal=True, scale=scale,
        Tq=Tq0, Tk=Tk0,
    )

    dQc, dKc, dVc = selected_backward_cuda(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=True, scale=scale,
    )

    assert dQc.shape == dQt.shape
    assert dKc.shape == dKt.shape
    assert dVc.shape == dVt.shape

    rq = _max_rel(dQc, dQt)
    rk = _max_rel(dKc, dKt)
    rv = _max_rel(dVc, dVt)

    print(f"headline: rel(dQ)={rq:.4e}, rel(dK)={rk:.4e}, rel(dV)={rv:.4e}")
    assert rq < 1e-3, f"dQ rel err {rq} above 1e-3 gate"
    assert rk < 1e-3, f"dK rel err {rk} above 1e-3 gate"
    assert rv < 1e-3, f"dV rel err {rv} above 1e-3 gate"


# ---------------------------------------------------------------------------
# Smaller shape sweep, including D=128.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [
    (2, 4, 512, 1024, 64, 4, 64, 64),
    (1, 8, 1024, 2048, 128, 8, 64, 64),
])
def test_cuda_bwd_matches_triton_sweep(shape):
    B, H, Tq, Tk, D, top_k, BM, BN = shape
    dtype = torch.bfloat16

    Q, K, V, dO = _make_qkv_do(B, H, Tq, Tk, D, dtype, seed=2)
    n_q = Tq // BM
    n_kv = Tk // BN
    bs = _make_block_scores(B, H, n_q, n_kv, seed=3)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq, Tk=Tk, causal=True)

    Qp, Kp, Vp, dOp, Tq0, Tk0, _, _ = _pad(Q, K, V, dO, BM, BN)
    O_p, LSE_p = _run_fwd_for_O_LSE(Qp, Kp, Vp, indices, BM, BN, top_k, causal=True)
    scale = 1.0 / math.sqrt(D)

    dQt, dKt, dVt = triton_selected_bwd(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_m=BM, block_size_n=BN,
        causal=True, scale=scale,
        Tq=Tq0, Tk=Tk0,
    )
    dQc, dKc, dVc = selected_backward_cuda(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=True, scale=scale,
    )

    for name, c, t in (("dQ", dQc, dQt), ("dK", dKc, dKt), ("dV", dVc, dVt)):
        rel = _max_rel(c, t)
        assert rel < 1e-2, f"shape {shape}: {name} rel err {rel}"


# ---------------------------------------------------------------------------
# causal=False path.
# ---------------------------------------------------------------------------
def test_cuda_bwd_causal_off_matches_triton():
    B, H, Tq, Tk, D, top_k = 1, 4, 512, 1024, 64, 4
    BM, BN = 64, 64
    dtype = torch.bfloat16

    Q, K, V, dO = _make_qkv_do(B, H, Tq, Tk, D, dtype, seed=4)
    n_q = Tq // BM
    n_kv = Tk // BN
    bs = _make_block_scores(B, H, n_q, n_kv, seed=5)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq, Tk=Tk, causal=False)

    Qp, Kp, Vp, dOp, Tq0, Tk0, _, _ = _pad(Q, K, V, dO, BM, BN)
    O_p, LSE_p = _run_fwd_for_O_LSE(Qp, Kp, Vp, indices, BM, BN, top_k, causal=False)
    scale = 1.0 / math.sqrt(D)

    dQt, dKt, dVt = triton_selected_bwd(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_m=BM, block_size_n=BN,
        causal=False, scale=scale,
        Tq=Tq0, Tk=Tk0,
    )
    dQc, dKc, dVc = selected_backward_cuda(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=False, scale=scale,
    )

    for name, c, t in (("dQ", dQc, dQt), ("dK", dKc, dKt), ("dV", dVc, dVt)):
        rel = _max_rel(c, t)
        assert rel < 1e-2, f"causal=False {name} rel err {rel}"


# ---------------------------------------------------------------------------
# Padded shape: Tq, Tk that are not multiples of BLOCK_M, BLOCK_N. The
# Python wrapper of the bwd accepts already-padded inputs, so this test
# performs the padding itself (the Triton bwd does the same).
# ---------------------------------------------------------------------------
def test_cuda_bwd_padded_shapes():
    B, H, Tq, Tk, D, top_k = 1, 4, 100, 200, 64, 4
    BM, BN = 64, 64
    dtype = torch.bfloat16

    Q, K, V, dO = _make_qkv_do(B, H, Tq, Tk, D, dtype, seed=6)

    Qp, Kp, Vp, dOp, Tq0, Tk0, pad_q, pad_k = _pad(Q, K, V, dO, BM, BN)
    Tq_p = Qp.shape[2]
    Tk_p = Kp.shape[2]
    n_q = Tq_p // BM
    n_kv = Tk_p // BN

    bs = _make_block_scores(B, H, n_q, n_kv, seed=7)
    indices = _shared_topk_indices(bs, top_k=top_k, BM=BM, BN=BN, Tq=Tq0, Tk=Tk0, causal=True)
    O_p, LSE_p = _run_fwd_for_O_LSE(Qp, Kp, Vp, indices, BM, BN, top_k, causal=True)
    scale = 1.0 / math.sqrt(D)

    dQt, dKt, dVt = triton_selected_bwd(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_m=BM, block_size_n=BN,
        causal=True, scale=scale,
        Tq=Tq0, Tk=Tk0,
    )
    dQc, dKc, dVc = selected_backward_cuda(
        dOp, Qp, Kp, Vp, O_p, LSE_p, indices,
        block_size_n=BN, block_size_m=BM, top_k=top_k,
        causal=True, scale=scale,
    )

    for name, c, t in (("dQ", dQc, dQt), ("dK", dKc, dKt), ("dV", dVc, dVt)):
        rel = _max_rel(c, t)
        assert rel < 1e-2, f"padded {name} rel err {rel}"
