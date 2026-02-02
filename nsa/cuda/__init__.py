"""NSA CUDA extension on Hopper (sm_90a): selected forward (WGMMA) and
backward (in-progress, dispatched via Triton for now). The Python
wrappers mirror nsa.triton.selected's API and require pre-padded inputs.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_extension():
    """JIT-compile the extension on first call. TORCH_CUDA_ARCH_LIST is
    forced to "9.0a" so ptxas accepts WGMMA / async-barrier ops."""
    from torch.utils.cpp_extension import load

    if "a" not in os.environ.get("TORCH_CUDA_ARCH_LIST", ""):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    sources = [
        os.path.join(_HERE, "selected_fwd.cu"),
        os.path.join(_HERE, "selected_bwd.cu"),
        os.path.join(_HERE, "bindings.cpp"),
    ]
    ext = load(
        name="nsa_cuda_selected",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-arch=sm_90a",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )
    return ext


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_extension()
    return _ext


def selected_attention_cuda(
    Q: torch.Tensor,                    # [B, H, Tq_p, D] bf16, Tq_p multiple of block_size_m
    K: torch.Tensor,
    V: torch.Tensor,
    block_indices: torch.Tensor,        # [B, H, n_q_blocks, top_k] int32
    block_size_n: int,
    block_size_m: int,
    top_k: int,
    causal: bool,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NSA selected-branch forward, Hopper CUDA path. Caller must pre-pad
    Q to a multiple of block_size_m and K/V to a multiple of block_size_n.
    Returns (out [B, H, Tq_p, D] bf16, lse [B, H, Tq_p] fp32)."""
    assert Q.dtype == torch.bfloat16, "this kernel expects bf16 inputs"
    assert K.dtype == torch.bfloat16
    assert V.dtype == torch.bfloat16
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert block_indices.is_cuda
    assert block_indices.dtype == torch.int32
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
    assert block_indices.is_contiguous()

    B, H, Tq_p, D = Q.shape
    Tk_p = K.shape[2]
    assert K.shape == V.shape == (B, H, Tk_p, D)
    assert Tq_p % block_size_m == 0
    assert Tk_p % block_size_n == 0
    n_q_blocks = Tq_p // block_size_m
    assert block_indices.shape == (B, H, n_q_blocks, top_k)

    offset = Tk_p - Tq_p
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(D)

    out, lse = _get_ext().selected_attention_fwd_cuda(
        Q, K, V, block_indices,
        block_size_m, block_size_n, top_k,
        offset, bool(causal), sm_scale,
    )
    return out, lse


def selected_backward_cuda(
    dO: torch.Tensor,                    # [B, H, Tq_p, D] bf16
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,                     # [B, H, Tq_p, D] bf16, used to compute D_vec
    LSE: torch.Tensor,                   # [B, H, Tq_p] fp32
    block_indices: torch.Tensor,         # [B, H, n_q_blocks, top_k] int32
    block_size_n: int,
    block_size_m: int,
    top_k: int,
    causal: bool,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NSA selected-branch backward. Dispatches to the Triton bwd while
    the CUDA tnsp=1 WGMMA value-correctness debug continues; the native
    kernel is exposed via _selected_backward_cuda_native for diagnostics.
    """
    from nsa.triton.backward import selected_backward as _triton_bwd

    return _triton_bwd(
        dO, Q, K, V, O, LSE, block_indices,
        block_size_m=block_size_m, block_size_n=block_size_n,
        causal=causal, scale=scale,
        Tq=Q.shape[2], Tk=K.shape[2],
    )


def _selected_backward_cuda_native(
    dO: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    O: torch.Tensor, LSE: torch.Tensor, block_indices: torch.Tensor,
    *, block_size_n: int, block_size_m: int, top_k: int,
    causal: bool, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-written CUDA backward, in-tree for diagnostics only: the WGMMA tnsp=1 descriptor encoding diverges from the Triton bwd until verified against CUTLASS source."""
    assert Q.dtype == torch.bfloat16
    assert Q.is_cuda
    B, H, Tq_p, D = Q.shape
    Tk_p = K.shape[2]
    assert Tq_p % block_size_m == 0
    assert Tk_p % block_size_n == 0

    Dvec = (dO.float() * O.float()).sum(dim=-1).contiguous()
    offset = Tk_p - Tq_p
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(D)

    dQ, dK_f, dV_f = _get_ext().selected_attention_bwd_cuda(
        dO, Q, K, V, LSE, Dvec, block_indices,
        block_size_m, block_size_n, top_k,
        offset, bool(causal), sm_scale,
    )
    return dQ, dK_f.to(K.dtype), dV_f.to(V.dtype)


__all__ = ["selected_attention_cuda", "selected_backward_cuda"]
