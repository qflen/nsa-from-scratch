"""Backward tests for the selected branch: (1) gradcheck through the
fp64 reference path validates the autograd.Function plumbing, (2) the
hand-written Triton backward at bf16 vs autograd through the reference
at rtol=1e-2 / atol=5e-2 (standard FA-2 bwd bf16 tolerance).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")


def _block_scores_pooled(Q, K, *, bs_m, bs_n):
    """Mirror nsa.triton.forward._resolve_block_indices but return raw scores
    so callers can also use them with the reference path."""
    import torch.nn.functional as F
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_q = (bs_m - Tq % bs_m) % bs_m
    pad_k = (bs_n - Tk % bs_n) % bs_n
    Qp = F.pad(Q, (0, 0, 0, pad_q)) if pad_q else Q
    Kp = F.pad(K, (0, 0, 0, pad_k)) if pad_k else K
    n_q = Qp.shape[2] // bs_m
    n_k = Kp.shape[2] // bs_n
    Qb = Qp.detach().float().view(B, H, n_q, bs_m, D).mean(dim=3)
    Kb = Kp.detach().float().view(B, H, n_k, bs_n, D).mean(dim=3)
    return torch.einsum("bhqd,bhkd->bhqk", Qb, Kb) / (D ** 0.5)


def test_gradcheck_fp64():
    """torch.autograd.gradcheck on the autograd.Function plumbing.

    Tiny shape so finite-diff is fast. fp64 is rerouted to the reference
    path inside _SelectedFn (the Triton kernels are bf16/fp16 only); this
    test is therefore not exercising the kernel directly, just the
    autograd contract. The kernel is exercised in test_triton_backward_*.
    """
    from nsa.triton.forward import _SelectedFn

    g = torch.Generator(device="cuda").manual_seed(0)
    B, H, Tq, Tk, D = 1, 1, 32, 64, 16
    bs_m = bs_n = 16
    top_k = 2

    Q = torch.randn(B, H, Tq, D, device="cuda", dtype=torch.float64, generator=g, requires_grad=True)
    K = torch.randn(B, H, Tk, D, device="cuda", dtype=torch.float64, generator=g, requires_grad=True)
    V = torch.randn(B, H, Tk, D, device="cuda", dtype=torch.float64, generator=g, requires_grad=True)

    # Precompute block_indices once so gradcheck has a stable gather pattern.
    bs = _block_scores_pooled(Q, K, bs_m=bs_m, bs_n=bs_n)
    n_q_blocks = Tq // bs_m
    n_kv_blocks = Tk // bs_n
    # Apply causal mask before topk.
    q_block_last = torch.arange(n_q_blocks, device="cuda") * bs_m + (bs_m - 1)
    kv_block_first = torch.arange(n_kv_blocks, device="cuda") * bs_n
    legal = kv_block_first.view(1, n_kv_blocks) <= q_block_last.view(n_q_blocks, 1)
    bs = bs.masked_fill(~legal.view(1, 1, n_q_blocks, n_kv_blocks), float("-inf"))
    _, indices = torch.topk(bs, top_k, dim=-1)
    block_indices = indices.to(torch.int32).contiguous()

    def fn(q, k, v):
        out = _SelectedFn.apply(q, k, v, bs_n, bs_m, top_k, block_indices, True, None)
        return out.sum()

    ok = torch.autograd.gradcheck(
        fn, (Q, K, V), eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True,
    )
    assert ok


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_backward_matches_reference(dtype):
    """Compare hand-written Triton backward dQ / dK / dV against reference autograd."""
    from nsa.reference import selected_attention_reference
    from nsa.triton.forward import _SelectedFn, _resolve_block_indices

    g = torch.Generator(device="cuda").manual_seed(1)
    B, H, Tq, Tk, D = 1, 4, 128, 256, 64
    bs_m = bs_n = 64
    top_k = 4

    Q = torch.randn(B, H, Tq, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, Tk, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, Tk, D, device="cuda", dtype=dtype, generator=g)

    block_indices = _resolve_block_indices(
        Q, K, block_size_m=bs_m, block_size_n=bs_n, top_k=top_k,
        causal=True, scale=1.0 / (D ** 0.5),
    )

    # Triton path
    Qt = Q.detach().clone().requires_grad_(True)
    Kt = K.detach().clone().requires_grad_(True)
    Vt = V.detach().clone().requires_grad_(True)
    out_t = _SelectedFn.apply(Qt, Kt, Vt, bs_n, bs_m, top_k, block_indices, True, None)
    grad_out = torch.randn_like(out_t)
    out_t.backward(grad_out)
    dQ_t, dK_t, dV_t = Qt.grad, Kt.grad, Vt.grad

    # Reference path: autograd through reference forward with the same indices.
    Qr = Q.detach().clone().requires_grad_(True)
    Kr = K.detach().clone().requires_grad_(True)
    Vr = V.detach().clone().requires_grad_(True)
    out_r, _ = selected_attention_reference(
        Qr, Kr, Vr,
        block_size_n=bs_n, block_size_m=bs_m, top_k=top_k,
        block_indices=block_indices, causal=True,
    )
    out_r = torch.nan_to_num(out_r, nan=0.0, posinf=0.0, neginf=0.0)
    out_r.backward(grad_out)
    dQ_r = torch.nan_to_num(Qr.grad, nan=0.0, posinf=0.0, neginf=0.0)
    dK_r = torch.nan_to_num(Kr.grad, nan=0.0, posinf=0.0, neginf=0.0)
    dV_r = torch.nan_to_num(Vr.grad, nan=0.0, posinf=0.0, neginf=0.0)

    rtol, atol = 1e-2, 5e-2
    for name, dT, dR in [("dQ", dQ_t, dQ_r), ("dK", dK_t, dK_r), ("dV", dV_t, dV_r)]:
        diff = (dT.float() - dR.float()).abs()
        ref = dR.float().abs()
        bad = (diff > atol + rtol * ref).sum().item()
        assert bad == 0, (
            f"{name}: {bad}/{diff.numel()} elements exceed allclose "
            f"(rtol={rtol}, atol={atol}); max_abs={diff.max().item():.3e} "
            f"max_ref={ref.max().item():.3e}"
        )


def test_backward_through_combined_forward():
    """End to end: nsa_forward backward must produce finite, non-zero grads on
    Q, K, V even at the realistic block sizes (top_k=8, BLOCK_M=BLOCK_N=64)."""
    from nsa.reference import NSAConfig
    from nsa.triton.forward import nsa_forward

    cfg = NSAConfig(
        block_size_c=64, block_size_n=64, block_size_m=64,
        top_k=4, window_size=128, causal=True,
    )
    g = torch.Generator(device="cuda").manual_seed(2)
    Q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    K = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    V = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)

    out = nsa_forward(Q, K, V, cfg)
    loss = out.float().pow(2).mean()
    loss.backward()

    for name, t in [("Q", Q), ("K", K), ("V", V)]:
        assert t.grad is not None, f"{name} got no grad"
        assert torch.isfinite(t.grad).all(), f"{name} grad not finite"
        assert t.grad.abs().max() > 1e-6, f"{name} grad is zero"
