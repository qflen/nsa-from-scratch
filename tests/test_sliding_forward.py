"""Correctness tests for the Triton sliding-window forward kernel.

Compares the kernel output against sliding_attention_reference for a sweep of
shapes, dtypes (fp16, bf16), and the causal flag. Also includes a sanity check
that, with window_size >= Tk, the causal sliding kernel reproduces full causal
attention (attention_reference). Tests skip when CUDA is unavailable so the
file remains importable on CPU-only hosts.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from nsa.reference import attention_reference, sliding_attention_reference

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_qkv(
    B: int, H: int, Tq: int, Tk: int, D: int, dtype: torch.dtype, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, Tq, D, dtype=dtype, device="cuda", generator=g)
    K = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g)
    V = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g)
    return Q, K, V


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6)).item()


def _abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


SHAPES = [
    (1, 4, 128, 128, 64),
    (2, 8, 256, 256, 64),
    (1, 4, 1024, 1024, 64),
    (1, 4, 2048, 2048, 64),
]
DTYPES = [torch.float16, torch.bfloat16]
CAUSAL = [True, False]


@cuda_only
@pytest.mark.parametrize("B,H,Tq,Tk,D", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", CAUSAL)
def test_sliding_forward_matches_reference(B, H, Tq, Tk, D, dtype, causal):
    from nsa.triton.sliding import sliding_attention

    # Window smaller than Tk to exercise actual sparsity.
    window_size = 128 if Tk >= 256 else 64

    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=0)

    out_ref, lse_ref = sliding_attention_reference(
        Q, K, V, window_size=window_size, causal=causal
    )
    out_kernel, lse_kernel = sliding_attention(
        Q, K, V, window_size=window_size, causal=causal
    )

    assert out_kernel.shape == (B, H, Tq, D)
    assert lse_kernel.shape == (B, H, Tq)
    assert out_kernel.dtype == Q.dtype
    assert lse_kernel.dtype == torch.float32

    rel = _rel_err(out_kernel, out_ref)
    assert rel < 1e-2, f"out rel err {rel:.3e} exceeds 1e-2 ({B},{H},{Tq},{Tk},{D},{dtype},causal={causal})"

    # Compare LSE only on rows that have any in-window keys; the reference
    # produces -inf elsewhere, and -inf compares fine but max over -inf is
    # undefined. Filter both sides identically.
    finite_mask = torch.isfinite(lse_ref)
    if finite_mask.any():
        lse_abs = (lse_kernel[finite_mask].float() - lse_ref[finite_mask].float()).abs().max().item()
        assert lse_abs < 1e-2, f"lse abs err {lse_abs:.3e} exceeds 1e-2"


@cuda_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_sliding_window_ge_tk_matches_full_causal(dtype):
    """When window_size >= Tk, sliding causal must equal full causal attention."""
    from nsa.triton.sliding import sliding_attention

    B, H, Tq, Tk, D = 1, 4, 256, 256, 64
    Q, K, V = _make_qkv(B, H, Tq, Tk, D, dtype, seed=1)

    out_full, lse_full = attention_reference(Q, K, V, causal=True)
    out_slide, lse_slide = sliding_attention(
        Q, K, V, window_size=Tk, causal=True
    )

    rel = _rel_err(out_slide, out_full)
    assert rel < 1e-2, f"sliding(W>=Tk) vs full causal rel err {rel:.3e}"
    abs_lse = _abs_err(lse_slide, lse_full)
    assert abs_lse < 1e-2, f"lse abs err {abs_lse:.3e}"


@cuda_only
def test_sliding_output_dtype_and_shape_fp16():
    from nsa.triton.sliding import sliding_attention

    Q, K, V = _make_qkv(1, 2, 64, 64, 64, torch.float16, seed=2)
    out, lse = sliding_attention(Q, K, V, window_size=32, causal=True)
    assert out.shape == Q.shape
    assert out.dtype == torch.float16
    assert lse.shape == (1, 2, 64)
    assert lse.dtype == torch.float32


@cuda_only
def test_sliding_handles_tq_lt_tk_offset():
    """When Tq < Tk, queries align with the tail of K (offset = Tk - Tq).

    Verifies the kernel honors the offset, matching the reference path.
    """
    from nsa.triton.sliding import sliding_attention

    B, H, Tq, Tk, D = 1, 2, 64, 192, 64
    Q, K, V = _make_qkv(B, H, Tq, Tk, D, torch.bfloat16, seed=3)
    out_ref, lse_ref = sliding_attention_reference(Q, K, V, window_size=48, causal=True)
    out_kernel, lse_kernel = sliding_attention(Q, K, V, window_size=48, causal=True)
    assert _rel_err(out_kernel, out_ref) < 1e-2
    finite = torch.isfinite(lse_ref)
    assert (lse_kernel[finite].float() - lse_ref[finite].float()).abs().max().item() < 1e-2
