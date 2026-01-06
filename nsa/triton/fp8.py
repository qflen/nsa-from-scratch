"""FP8 storage path for the Triton selected branch (FA-3 style: FP8
storage, bf16 compute). Dequantizes Q/K/V on load and hands off to the
existing bf16 selected_attention. FP8-operand WGMMA compute lives in
nsa/cuda/selected_fwd.cu if needed separately.
"""

from __future__ import annotations

from typing import Optional

import torch

from nsa.triton.selected import selected_attention


_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0


def quantize_to_fp8(
    x: torch.Tensor,
    *,
    fmt: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float]:
    """Per-tensor quantize a fp32/bf16 tensor to fp8 with absmax scaling.

    Returns (x_fp8, scale) such that `dequantize(x_fp8) = x_fp8.to(bf16) * scale`
    approximates `x`.
    """
    if fmt == torch.float8_e4m3fn:
        fmt_max = _E4M3_MAX
    elif fmt == torch.float8_e5m2:
        fmt_max = _E5M2_MAX
    else:
        raise ValueError(f"unsupported fp8 format: {fmt}")
    absmax = x.detach().abs().max().clamp(min=eps).item()
    scale = absmax / fmt_max
    x_scaled = (x / scale).clamp(-fmt_max, fmt_max)
    return x_scaled.to(fmt), scale


def selected_attention_fp8(
    Q: torch.Tensor,                 # fp8e4m3 or fp8e5m2 [B, H, Tq, D]
    K: torch.Tensor,
    V: torch.Tensor,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    *,
    block_size_n: int = 64,
    block_size_m: int = 64,
    top_k: int = 16,
    block_scores: Optional[torch.Tensor] = None,
    block_indices: Optional[torch.Tensor] = None,
    causal: bool = True,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected-branch forward with FP8 storage. Compute is bf16.

    The dequant scales are required (per-tensor absmax convention). Returns
    (out, lse) in bf16 and fp32 respectively, same shape as `selected_attention`.
    """
    assert Q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), \
        f"Q must be fp8; got {Q.dtype}"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype
    Q_bf16 = Q.to(torch.bfloat16) * scale_q
    K_bf16 = K.to(torch.bfloat16) * scale_k
    V_bf16 = V.to(torch.bfloat16) * scale_v
    return selected_attention(
        Q_bf16, K_bf16, V_bf16,
        block_size_n=block_size_n, block_size_m=block_size_m, top_k=top_k,
        block_scores=block_scores, block_indices=block_indices,
        causal=causal, scale=scale,
    )


__all__ = ["selected_attention_fp8", "quantize_to_fp8"]
