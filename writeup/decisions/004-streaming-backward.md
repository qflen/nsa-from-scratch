# 004 - Streaming softmax backward, adapted to the gather pattern

## Context

The selected-branch backward needs `dQ`, `dK`, `dV` from saved `out`, `lse`, `block_indices`. FA-2 (Dao 2023) gives the streaming-softmax bwd for dense tile attention: pre-compute `D = (dO * O).sum(-1)` so each tile gives `dS = P * (dP - D)` without rematerializing the full softmax. NSA's queries are dense but its keys-per-query are gathered, and the K-side reductions cannot be a tile sweep.

## Decision

Inner math: FA-2 unchanged (D pre-step, `P = exp(S - lse)`, `dS = P * (dP - D)`). Loop: queries iterate by BLOCK_M, inner loop reads `block_indices[batch, head, q_block, kk]` to choose the K/V tile. dK/dV accumulate via `tl.atomic_add` into fp32 buffers, cast back at the end. Per-element causal masking uses *original* token positions, not gathered positions.

## Consequences

dQ is local, no atomics; each q_block writes its own row range. dK/dV pays one atomic_add per (q_block, kv_block) hit, roughly `top_k * n_q_blocks` atomics per (batch, head). On H100 this is well under bandwidth and the bottleneck is the softmax recomputation. The end-of-kernel cast absorbs the precision step where fp32 streaming math is already running. Correctness: the Triton backward (the kernel described here) reaches `max_abs_err < 5e-3` against an autograd-through-reference gradcheck at the headline shape. That figure is the Triton path's; the CUDA `tnsp=1` backward has no passing number yet and is dispatched through the Triton bwd until its WGMMA descriptor encoding is verified against CUTLASS source.
