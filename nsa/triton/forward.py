"""Combined NSA forward: dispatches the three Triton branch kernels and
gates them. Each branch is wrapped in an autograd.Function: selected uses
the hand-written Triton bwd, compressed and sliding fall back to a
reference autograd-through-recompute path (sliding's recompute is chunked
so memory stays O(T*W)).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from nsa.reference import (
    NSAConfig,
    compressed_attention_reference,
)


def _sliding_attention_chunked(
    Q: Tensor, K: Tensor, V: Tensor,
    *, window_size: int, causal: bool, scale: Optional[float],
) -> Tensor:
    """Sliding window attention computed in chunks of size window_size.

    Mathematically identical to `sliding_attention_reference` but never
    materializes the full (B, H, Tq, Tk) score matrix. Used inside
    `_SlidingFn.backward` to keep the autograd-based bwd memory at
    O(T * W) instead of O(T^2). Critical for training at 32k+ context:
    the original reference allocates 50 GB at T=32k which OOMs even on
    80 GB H100; this path stays under 1 GB.

    Returns just `out` (no LSE) since the autograd bwd does not consume LSE.
    """
    import math as _math

    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    s = float(scale) if scale is not None else 1.0 / _math.sqrt(D)
    internal = torch.float32 if Q.dtype.itemsize < 4 else Q.dtype
    Qf = Q.to(internal)
    Kf = K.to(internal)
    Vf = V.to(internal)

    chunk = window_size
    offset = Tk - Tq
    out_chunks: list[Tensor] = []
    for q_start in range(0, Tq, chunk):
        q_end = min(q_start + chunk, Tq)
        Qc = Qf[:, :, q_start:q_end, :]
        if causal:
            k_min = max(0, q_start + offset - window_size + 1)
            k_max = min(Tk, q_end + offset)
        else:
            k_min = max(0, q_start + offset - window_size)
            k_max = min(Tk, q_end + offset + window_size)
        Kc = Kf[:, :, k_min:k_max, :]
        Vc = Vf[:, :, k_min:k_max, :]

        scores = torch.einsum("bhqd,bhkd->bhqk", Qc, Kc) * s

        i = torch.arange(q_start, q_end, device=Q.device).view(1, 1, -1, 1) + offset
        j = torch.arange(k_min, k_max, device=Q.device).view(1, 1, 1, -1)
        if causal:
            mask = (j > i) | (j < (i - window_size + 1))
        else:
            mask = (j > (i + window_size)) | (j < (i - window_size))
        scores = scores.masked_fill(mask, float("-inf"))

        lse_c = torch.logsumexp(scores, dim=-1)
        weights = torch.exp(scores - lse_c.unsqueeze(-1))
        out_c = torch.einsum("bhqk,bhkd->bhqd", weights, Vc)
        out_chunks.append(out_c)

    out = torch.cat(out_chunks, dim=2)
    return out.to(Q.dtype)
from nsa.triton.compressed import compressed_attention as _compressed_kernel
from nsa.triton.gating import combine
from nsa.triton.sliding import sliding_attention as _sliding_kernel


def _autograd_grads_through_reference(
    ref_fn,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    grad_out: Tensor,
    **ref_kwargs,
) -> tuple[Tensor, Tensor, Tensor]:
    """Re-run a reference attention with autograd enabled and pull dQ, dK, dV.

    The reference functions can produce nan on causally-empty rows; we set
    nan output rows to zero before the autograd pass so upstream gradients
    are finite. nan rows are exactly the rows whose attention output is
    undefined (no legal keys), and contribute zero to any well-defined loss.
    """
    with torch.enable_grad():
        Qd = Q.detach().requires_grad_(True)
        Kd = K.detach().requires_grad_(True)
        Vd = V.detach().requires_grad_(True)
        out_ref, _ = ref_fn(Qd, Kd, Vd, **ref_kwargs)
        out_ref = torch.nan_to_num(out_ref, nan=0.0, posinf=0.0, neginf=0.0)
        dQ, dK, dV = torch.autograd.grad(
            out_ref, (Qd, Kd, Vd), grad_outputs=grad_out.to(out_ref.dtype),
            retain_graph=False, allow_unused=True,
        )
    if dQ is None: dQ = torch.zeros_like(Q)
    if dK is None: dK = torch.zeros_like(K)
    if dV is None: dV = torch.zeros_like(V)
    # Causally-empty rows produce structurally-undefined gradients (a logsumexp
    # of all -inf has 0/0 in its backward). Sanitize: a row whose forward
    # output is undefined contributes zero to any well-defined loss, so any
    # nan / inf in its gradient is safe to zero.
    dQ = torch.nan_to_num(dQ, nan=0.0, posinf=0.0, neginf=0.0)
    dK = torch.nan_to_num(dK, nan=0.0, posinf=0.0, neginf=0.0)
    dV = torch.nan_to_num(dV, nan=0.0, posinf=0.0, neginf=0.0)
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


class _CompressedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, block_size_c, pool, causal, scale):
        out, _ = _compressed_kernel(
            Q, K, V, block_size_c=block_size_c, pool=pool, causal=causal, scale=scale,
        )
        ctx.save_for_backward(Q, K, V)
        ctx.block_size_c, ctx.pool, ctx.causal, ctx.scale = block_size_c, pool, causal, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V = ctx.saved_tensors
        dQ, dK, dV = _autograd_grads_through_reference(
            compressed_attention_reference, Q, K, V, grad_out,
            block_size_c=ctx.block_size_c, pool=ctx.pool, causal=ctx.causal, scale=ctx.scale,
        )
        return dQ, dK, dV, None, None, None, None


class _SelectedFn(torch.autograd.Function):
    """Selected branch with hand-written Triton backward.

    Forward calls the Triton forward kernel and saves Q, K, V, O, LSE,
    block_indices for the backward (all padded). Backward dispatches:
      - bf16 / fp16 inputs: hand-written Triton backward.
      - fp64 inputs: fall back to reference autograd (gradcheck only).
    """

    @staticmethod
    def forward(ctx, Q, K, V, block_size_n, block_size_m, top_k, block_indices, causal, scale):
        # fp64 path: gradcheck uses fp64 to detect numerical inaccuracy in
        # an autograd.Function. The Triton kernels are bf16/fp16 only, so
        # we reroute fp64 through the reference. We do NOT save the fp64
        # autograd graph (gradcheck calls backward several times for fast
        # mode and would error on a freed graph); instead we save just the
        # detached tensors and rebuild the graph on each backward call.
        if Q.dtype == torch.float64:
            from nsa.reference import selected_attention_reference
            with torch.no_grad():
                out_ref, _ = selected_attention_reference(
                    Q.detach(), K.detach(), V.detach(),
                    block_size_n=block_size_n, block_size_m=block_size_m,
                    top_k=top_k, block_scores=None, block_indices=block_indices,
                    causal=causal, scale=scale,
                )
                out_ref = torch.nan_to_num(out_ref, nan=0.0, posinf=0.0, neginf=0.0)
            ctx._fp64_path = True
            ctx.save_for_backward(Q, K, V)
            ctx._fp64_block_indices = block_indices
            ctx._fp64_args = (block_size_n, block_size_m, top_k, causal, scale)
            return out_ref

        # Inline the padding + kernel launch so we can save padded tensors
        # for the backward pass without re-running the whole forward.
        from nsa.triton.selected import selected_attention as _sel_unpadded
        # The Triton wrapper handles padding/unpad internally and returns
        # unpadded (out, lse). For the backward we need the PADDED versions
        # so the gradient kernel matches the forward gather pattern. Easiest
        # robust path: call the wrapper, then re-pad in backward when needed.
        out, lse = _sel_unpadded(
            Q, K, V,
            block_size_n=block_size_n, block_size_m=block_size_m, top_k=top_k,
            block_indices=block_indices, causal=causal, scale=scale,
        )
        ctx._fp64_path = False
        ctx.save_for_backward(Q, K, V, out, lse, block_indices)
        ctx.block_size_n, ctx.block_size_m, ctx.top_k = block_size_n, block_size_m, top_k
        ctx.causal, ctx.scale = causal, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx._fp64_path:
            from nsa.reference import selected_attention_reference
            Q, K, V = ctx.saved_tensors
            block_indices = ctx._fp64_block_indices
            block_size_n, block_size_m, top_k, causal, scale = ctx._fp64_args
            with torch.enable_grad():
                Qd = Q.detach().requires_grad_(True)
                Kd = K.detach().requires_grad_(True)
                Vd = V.detach().requires_grad_(True)
                out_ref, _ = selected_attention_reference(
                    Qd, Kd, Vd,
                    block_size_n=block_size_n, block_size_m=block_size_m,
                    top_k=top_k, block_scores=None, block_indices=block_indices,
                    causal=causal, scale=scale,
                )
                out_ref = torch.nan_to_num(out_ref, nan=0.0, posinf=0.0, neginf=0.0)
                dQ, dK, dV = torch.autograd.grad(
                    out_ref, (Qd, Kd, Vd), grad_outputs=grad_out,
                    retain_graph=False, allow_unused=True,
                )
            if dQ is None: dQ = torch.zeros_like(Q)
            if dK is None: dK = torch.zeros_like(K)
            if dV is None: dV = torch.zeros_like(V)
            dQ = torch.nan_to_num(dQ, nan=0.0, posinf=0.0, neginf=0.0)
            dK = torch.nan_to_num(dK, nan=0.0, posinf=0.0, neginf=0.0)
            dV = torch.nan_to_num(dV, nan=0.0, posinf=0.0, neginf=0.0)
            return dQ, dK, dV, None, None, None, None, None, None

        Q, K, V, O, LSE, block_indices = ctx.saved_tensors
        from nsa.triton.backward import selected_backward as _sel_bwd

        # Re-pad to match the kernel's gather pattern. The forward unpadded
        # the tensors before returning, so we re-apply the same pad here.
        B, H, Tq, D = Q.shape
        Tk = K.shape[2]
        bs_m, bs_n = ctx.block_size_m, ctx.block_size_n
        pad_q = (bs_m - Tq % bs_m) % bs_m
        pad_k = (bs_n - Tk % bs_n) % bs_n

        Qp = F.pad(Q, (0, 0, 0, pad_q)) if pad_q else Q
        Kp = F.pad(K, (0, 0, 0, pad_k)) if pad_k else K
        Vp = F.pad(V, (0, 0, 0, pad_k)) if pad_k else V
        Op = F.pad(O, (0, 0, 0, pad_q)) if pad_q else O
        # LSE: pad with -inf so the backward's lse-clamp treats those rows
        # as causally-empty (their contribution vanishes).
        if pad_q:
            LSEp = F.pad(LSE, (0, pad_q), value=float("-inf"))
        else:
            LSEp = LSE
        # dO: upstream supplies unpadded; pad with zero so dummy rows give
        # zero gradient contribution.
        dOp = F.pad(grad_out, (0, 0, 0, pad_q)) if pad_q else grad_out

        scale = ctx.scale if ctx.scale is not None else 1.0 / (D ** 0.5)
        dQp, dKp, dVp = _sel_bwd(
            dOp, Qp, Kp, Vp, Op, LSEp, block_indices,
            block_size_m=bs_m, block_size_n=bs_n,
            causal=ctx.causal, scale=float(scale),
            Tq=Tq, Tk=Tk,
        )
        # Unpad
        dQ = dQp[:, :, :Tq, :] if pad_q else dQp
        dK = dKp[:, :, :Tk, :] if pad_k else dKp
        dV = dVp[:, :, :Tk, :] if pad_k else dVp
        return dQ, dK, dV, None, None, None, None, None, None


class _SlidingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, window_size, causal, scale):
        out, _ = _sliding_kernel(
            Q, K, V, window_size=window_size, causal=causal, scale=scale,
        )
        ctx.save_for_backward(Q, K, V)
        ctx.window_size, ctx.causal, ctx.scale = window_size, causal, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V = ctx.saved_tensors
        # Chunked sliding path (O(T*W) memory). Mathematically identical
        # to the full reference (O(T^2)); the chunked form is what keeps
        # the autograd-bwd training path viable at 32k context.
        with torch.enable_grad():
            Qd = Q.detach().requires_grad_(True)
            Kd = K.detach().requires_grad_(True)
            Vd = V.detach().requires_grad_(True)
            out_ref = _sliding_attention_chunked(
                Qd, Kd, Vd,
                window_size=ctx.window_size, causal=ctx.causal, scale=ctx.scale,
            )
            out_ref = torch.nan_to_num(out_ref, nan=0.0, posinf=0.0, neginf=0.0)
            dQ, dK, dV = torch.autograd.grad(
                out_ref, (Qd, Kd, Vd), grad_outputs=grad_out.to(out_ref.dtype),
                retain_graph=False, allow_unused=True,
            )
        if dQ is None: dQ = torch.zeros_like(Q)
        if dK is None: dK = torch.zeros_like(K)
        if dV is None: dV = torch.zeros_like(V)
        dQ = torch.nan_to_num(dQ, nan=0.0, posinf=0.0, neginf=0.0)
        dK = torch.nan_to_num(dK, nan=0.0, posinf=0.0, neginf=0.0)
        dV = torch.nan_to_num(dV, nan=0.0, posinf=0.0, neginf=0.0)
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None, None, None


def _block_scores_from_pooled(
    Q: Tensor, K: Tensor, *, block_size_m: int, block_size_n: int, scale: float
) -> Tensor:
    """Compute [B, H, n_q_blocks, n_kv_blocks] scores by mean-pooling Q and K
    to block granularity and dotting them. Mirrors the default scorer in
    `selected_attention_reference`.
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_q = (block_size_m - Tq % block_size_m) % block_size_m
    pad_k = (block_size_n - Tk % block_size_n) % block_size_n
    Qp = F.pad(Q, (0, 0, 0, pad_q)) if pad_q else Q
    Kp = F.pad(K, (0, 0, 0, pad_k)) if pad_k else K
    n_q = Qp.shape[2] // block_size_m
    n_k = Kp.shape[2] // block_size_n
    Qb = Qp.view(B, H, n_q, block_size_m, D).mean(dim=3)
    Kb = Kp.view(B, H, n_k, block_size_n, D).mean(dim=3)
    return torch.einsum("bhqd,bhkd->bhqk", Qb.float(), Kb.float()) * scale


def _resolve_block_indices(
    Q: Tensor, K: Tensor, *,
    block_size_m: int, block_size_n: int, top_k: int, causal: bool, scale: float,
) -> Tensor:
    """Compute [B, H, n_q_blocks, k] int32 indices via mean-pool scoring + topk.

    Done outside the autograd.Function so it can be cached / reused by
    backward (the gather pattern must match between forward and backward).
    """
    B, H, Tq, D = Q.shape
    Tk = K.shape[2]
    pad_q = (block_size_m - Tq % block_size_m) % block_size_m
    pad_k = (block_size_n - Tk % block_size_n) % block_size_n
    Qp = F.pad(Q, (0, 0, 0, pad_q)) if pad_q else Q
    Kp = F.pad(K, (0, 0, 0, pad_k)) if pad_k else K
    n_q = Qp.shape[2] // block_size_m
    n_k = Kp.shape[2] // block_size_n
    # detach so the scorer never enters the autograd graph (cheap; we don't
    # train through this path, the kernel path is what trains). Order
    # matters: reduce in input dtype, then cast to fp32. Casting before
    # the mean changes fp precision and can pick different topk indices
    # from the reference, which means the kernel and the reference would
    # gather different KV blocks.
    Qd = Qp.detach()
    Kd = Kp.detach()
    Qb = Qd.view(B, H, n_q, block_size_m, D).mean(dim=3).float()
    Kb = Kd.view(B, H, n_k, block_size_n, D).mean(dim=3).float()
    scores = torch.einsum("bhqd,bhkd->bhqk", Qb, Kb) * scale  # [B, H, n_q, n_k]
    if causal:
        Tk_p = Kp.shape[2]
        Tq_p = Qp.shape[2]
        offset = Tk_p - Tq_p
        q_block_last = torch.arange(n_q, device=Q.device) * block_size_m + (block_size_m - 1)
        kv_block_first = torch.arange(n_k, device=Q.device) * block_size_n
        legal = kv_block_first.view(1, n_k) <= (q_block_last.view(n_q, 1) + offset)
        scores = scores.masked_fill(~legal.view(1, 1, n_q, n_k), float("-inf"))
    k_actual = min(top_k, n_k)
    _, idx = torch.topk(scores, k_actual, dim=-1)
    return idx.to(torch.int32).contiguous()


def nsa_forward(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    cfg: NSAConfig,
    *,
    gate_logits: Optional[Tensor] = None,
    block_indices: Optional[Tensor] = None,
) -> Tensor:
    """NSA combined forward through the Triton kernel paths, autograd-aware.

    Args:
        Q, K, V: [B, H, Tq|Tk, D] in fp16 or bf16.
        cfg: NSAConfig carrying block sizes, top_k, window_size, gate
            activation, and causality.
        gate_logits: [B, H, Tq, 3] pre-activation per-branch scores. If None,
            uniform 1/3 gating is applied (no learning).
        block_indices: optional precomputed [B, H, n_q_blocks, k] int32
            indices. If None, derived from mean-pool Q.K scoring.

    Returns:
        out: [B, H, Tq, D], same dtype as Q. Differentiable wrt Q, K, V.
    """
    D = Q.shape[-1]
    scale = cfg.scale if cfg.scale is not None else 1.0 / (D ** 0.5)

    out_c = _CompressedFn.apply(Q, K, V, cfg.block_size_c, cfg.pool, cfg.causal, cfg.scale)

    if block_indices is None:
        block_indices = _resolve_block_indices(
            Q, K,
            block_size_m=cfg.block_size_m, block_size_n=cfg.block_size_n,
            top_k=cfg.top_k, causal=cfg.causal, scale=float(scale),
        )

    out_s = _SelectedFn.apply(
        Q, K, V, cfg.block_size_n, cfg.block_size_m, cfg.top_k, block_indices,
        cfg.causal, cfg.scale,
    )
    out_w = _SlidingFn.apply(Q, K, V, cfg.window_size, cfg.causal, cfg.scale)

    if gate_logits is None:
        B, H, Tq, _ = Q.shape
        gate_logits = Q.new_full((B, H, Tq, 3), 0.0)
    return combine(out_c, out_s, out_w, gate_logits, activation=cfg.gate_activation)


__all__ = ["nsa_forward"]
