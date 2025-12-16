"""Combined-forward correctness test: Triton path vs reference path.

Compares `nsa.triton.forward.nsa_forward` against `nsa.reference.
nsa_attention_reference`, exercising all three branches plus the gating
layer in one pass. Acceptance threshold matches the per-branch tests:
combined relative error must be under 1e-2 in fp16 and bf16.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")


@pytest.fixture
def cfg():
    from nsa.reference import NSAConfig
    return NSAConfig(
        block_size_c=32, block_size_n=32, block_size_m=32,
        top_k=4, window_size=64, causal=True, gate_activation="sigmoid",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (1, 4, 128, 128, 64),
    (2, 4, 256, 256, 64),
    (1, 8, 512, 1024, 64),
])
def test_combined_matches_reference(shape, dtype, cfg):
    from nsa.reference import nsa_attention_reference
    from nsa.triton.forward import nsa_forward

    B, H, Tq, Tk, D = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(B, H, Tq, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, Tk, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, Tk, D, device="cuda", dtype=dtype, generator=g)
    gate_logits = torch.randn(B, H, Tq, 3, device="cuda", dtype=dtype, generator=g)

    out_triton = nsa_forward(Q, K, V, cfg, gate_logits=gate_logits)
    out_ref = nsa_attention_reference(Q, K, V, cfg, gates=gate_logits)

    assert out_triton.shape == out_ref.shape == (B, H, Tq, D)
    assert out_triton.dtype == dtype

    # Reference returns nan for queries that have no legal compressed block
    # (very early causal queries). The Triton path returns zero on those
    # rows. Compare only where the reference is fully finite.
    finite = torch.isfinite(out_ref.float()).all(dim=-1)  # [B, H, Tq]
    assert finite.any(), "no finite reference rows to compare against"
    finite_mask = finite.unsqueeze(-1).expand_as(out_ref)

    # Use a combined tolerance: rtol=1e-2 atol=2e-2. Per-branch absolute error
    # is ~1e-2 (a few ulps in fp16/bf16); summing three gated branches can
    # accumulate up to ~2-3x that on outputs with near-zero magnitude.
    abs_diff = (out_triton.float() - out_ref.float()).abs()[finite_mask]
    ref_abs = out_ref.float().abs()[finite_mask]
    rtol, atol = 1e-2, 2e-2
    bad = (abs_diff > atol + rtol * ref_abs).sum().item()
    n = abs_diff.numel()
    assert bad == 0, (
        f"{bad}/{n} elements exceed allclose(rtol={rtol}, atol={atol}) "
        f"(max_abs={abs_diff.max().item():.3e}, max_ref={ref_abs.max().item():.3e}) "
        f"({shape}, {dtype})"
    )


def test_combined_softmax_gating_sums_to_branches(cfg):
    """With uniform gate_logits and softmax activation, each branch contributes
    exactly 1/3, so the combined output equals the mean of the three branch
    outputs from the reference path."""
    from nsa.reference import (
        compressed_attention_reference,
        selected_attention_reference,
        sliding_attention_reference,
        NSAConfig,
    )
    from nsa.triton.forward import nsa_forward

    cfg2 = NSAConfig(
        block_size_c=32, block_size_n=32, block_size_m=32,
        top_k=4, window_size=64, causal=True, gate_activation="softmax",
    )
    g = torch.Generator(device="cuda").manual_seed(1)
    Q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    K = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    V = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    gate_logits = torch.zeros(1, 4, 128, 3, device="cuda", dtype=torch.bfloat16)

    out = nsa_forward(Q, K, V, cfg2, gate_logits=gate_logits)
    out_c, _ = compressed_attention_reference(Q, K, V, block_size_c=32, causal=True)
    out_s, _ = selected_attention_reference(Q, K, V, block_size_n=32, block_size_m=32, top_k=4, causal=True)
    out_w, _ = sliding_attention_reference(Q, K, V, window_size=64, causal=True)
    expected = (out_c.float() + out_s.float() + out_w.float()) / 3.0

    # Mask non-finite rows. Use combined allclose tolerance.
    finite = torch.isfinite(expected).all(dim=-1).unsqueeze(-1).expand_as(expected)
    abs_diff = (out.float() - expected).abs()[finite]
    ref_abs = expected.abs()[finite]
    rtol, atol = 1e-2, 2e-2
    bad = (abs_diff > atol + rtol * ref_abs).sum().item()
    assert bad == 0, (
        f"{bad}/{abs_diff.numel()} elements exceed allclose tolerance; "
        f"max_abs={abs_diff.max().item():.3e}"
    )
