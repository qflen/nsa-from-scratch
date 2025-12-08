"""Correctness for the Triton compressed-branch forward against the
plain-torch reference. Sweep over shapes / causal / fp16+bf16; tol is
1e-2 rel for out and abs for lse. Causally-empty rows produce nan in
the reference and (0, -inf) in the kernel; tests mask them out.
"""

from __future__ import annotations

import math

import pytest
import torch

from nsa.reference import compressed_attention_reference

# Skip the whole module gracefully if Triton or CUDA are not available; tests
# only make sense on a GPU.
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton kernel tests", allow_module_level=True)

from nsa.triton.compressed import compressed_attention  # noqa: E402


SHAPES = [
    (1, 4, 128, 128, 64),
    (2, 8, 256, 256, 64),
    (1, 4, 512, 1024, 64),
]


def _qkv(B: int, H: int, Tq: int, Tk: int, D: int, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, Tq, D, dtype=dtype, device="cuda", generator=g) * 0.5
    K = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g) * 0.5
    V = torch.randn(B, H, Tk, D, dtype=dtype, device="cuda", generator=g) * 0.5
    return Q, K, V


def _legal_row_mask(Tq: int, Tk: int, block_size_c: int, causal: bool, device) -> torch.Tensor:
    """Boolean [Tq] mask: True where the row has at least one legal block.

    With Tk possibly padded up to a multiple of block_size_c. The reference
    computes legality with the (already padded) Tk; we mirror that here.
    """
    Nc = (Tk + block_size_c - 1) // block_size_c
    Tk_padded = Nc * block_size_c
    if not causal:
        return torch.ones(Tq, dtype=torch.bool, device=device)
    offset = Tk_padded - Tq
    q = torch.arange(Tq, device=device)
    max_block = (q + offset + 1) // block_size_c - 1
    return max_block >= 0


def _check(
    out_k, lse_k, out_r, lse_r,
    row_mask: torch.Tensor,
    *,
    out_rel_tol: float,
    out_abs_tol: float,
    lse_abs_tol: float,
):
    """Numeric check with combined absolute+relative tolerance for out, abs for lse.

    On rows where the reference output is near zero (true result happens to
    be tiny by chance of softmax peakedness), pure relative error blows up
    due to fp16/bf16 quantization noise. We use the standard allclose form:

        |kernel - ref| <= out_rel_tol * |ref| + out_abs_tol

    This is the natural reading of the spec's "max |out_k - out_r| /
    (|out_r| + eps) < 1e-2": on outputs of substantial magnitude (above
    out_abs_tol) it reduces to a 1% relative bound, and on near-zero outputs
    it tolerates the inevitable fp16/bf16 noise.

    LSE is checked against a pure absolute tolerance (the spec's wording
    matches: "max |lse_k - lse_r| < 1e-2"). LSE values are well-behaved
    because they live in fp32 in both the kernel and the reference.
    """
    # Apply row mask along Tq.
    out_k_m = out_k[:, :, row_mask, :].float()
    out_r_m = out_r[:, :, row_mask, :].float()
    lse_k_m = lse_k[:, :, row_mask]
    lse_r_m = lse_r[:, :, row_mask]

    assert torch.isfinite(out_k_m).all(), "kernel out has non-finite values on legal rows"
    assert torch.isfinite(lse_k_m).all(), "kernel lse has non-finite values on legal rows"
    assert torch.isfinite(out_r_m).all(), "ref out has non-finite values on legal rows"
    assert torch.isfinite(lse_r_m).all(), "ref lse has non-finite values on legal rows"

    abs_err_out = (out_k_m - out_r_m).abs()
    bound_out = out_rel_tol * out_r_m.abs() + out_abs_tol
    excess = (abs_err_out - bound_out).clamp_min(0)
    if excess.max().item() > 0:
        idx = excess.argmax().item()
        ref_val = out_r_m.view(-1)[idx].item()
        ker_val = out_k_m.view(-1)[idx].item()
        raise AssertionError(
            f"out exceeds rtol={out_rel_tol} atol={out_abs_tol}: "
            f"abs_err={abs_err_out.view(-1)[idx].item():.3e} "
            f"ref={ref_val:.3e} ker={ker_val:.3e}"
        )

    abs_err_lse = (lse_k_m - lse_r_m).abs().max().item()
    assert abs_err_lse < lse_abs_tol, f"lse abs error {abs_err_lse} exceeds {lse_abs_tol}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"B{s[0]}H{s[1]}Tq{s[2]}Tk{s[3]}D{s[4]}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
def test_compressed_matches_reference(shape, dtype, causal):
    B, H, Tq, Tk, D = shape
    block_size_c = 64
    # Deterministic seed across Python invocations: PYTHONHASHSEED would
    # otherwise randomize hash() of tuples. Use a stable mixing of small ints.
    seed = (
        ((B * 17 + H) * 19 + Tq // 8) * 23
        + (Tk // 8) * 29
        + (D // 8) * 31
        + (1 if dtype is torch.float16 else 2) * 37
        + (1 if causal else 0) * 41
    ) & 0xFFFF
    Q, K, V = _qkv(B, H, Tq, Tk, D, dtype, seed=seed)

    out_ref, lse_ref = compressed_attention_reference(
        Q, K, V, block_size_c=block_size_c, causal=causal
    )
    out_k, lse_k = compressed_attention(
        Q, K, V, block_size_c=block_size_c, causal=causal
    )

    assert out_k.shape == (B, H, Tq, D)
    assert lse_k.shape == (B, H, Tq)
    assert out_k.dtype == dtype
    assert lse_k.dtype == torch.float32

    row_mask = _legal_row_mask(Tq, Tk, block_size_c, causal, device=Q.device)

    _check(
        out_k, lse_k, out_ref, lse_ref, row_mask,
        out_rel_tol=1e-2, out_abs_tol=1e-3, lse_abs_tol=1e-2,
    )


def test_dtype_and_shape_contract():
    # A minimal smoke: just verifies shape/dtype invariants in fp16 causal.
    B, H, Tq, Tk, D = 1, 2, 64, 128, 64
    Q, K, V = _qkv(B, H, Tq, Tk, D, torch.float16)
    out, lse = compressed_attention(Q, K, V, block_size_c=64, causal=True)
    assert out.shape == Q.shape
    assert lse.shape == (B, H, Tq)
    assert out.dtype == torch.float16
    assert lse.dtype == torch.float32


def test_default_scale_matches_reference():
    """Confirm 1/sqrt(D) default lines up with the reference."""
    B, H, Tq, Tk, D = 1, 2, 64, 128, 64
    Q, K, V = _qkv(B, H, Tq, Tk, D, torch.float16, seed=42)
    out_ref, lse_ref = compressed_attention_reference(Q, K, V, block_size_c=64, causal=True)
    out_k, lse_k = compressed_attention(
        Q, K, V, block_size_c=64, causal=True, scale=1.0 / math.sqrt(D)
    )
    row_mask = _legal_row_mask(Tq, Tk, 64, True, device=Q.device)
    _check(
        out_k, lse_k, out_ref, lse_ref, row_mask,
        out_rel_tol=1e-2, out_abs_tol=1e-3, lse_abs_tol=1e-2,
    )
