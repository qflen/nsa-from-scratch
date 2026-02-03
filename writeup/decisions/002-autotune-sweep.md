# 002 - Autotune sweep space and pruning

## Context

The selected-branch Triton forward has four knobs that change generated PTX: BLOCK_M (queries per CTA), BLOCK_N (keys per gather tile), num_warps, num_stages. A naive sweep over BLOCK_M in {32,64,128}, BLOCK_N in {16,32,64,128}, num_warps in {2,4,8}, num_stages in {2,3,4} is 144 configs, of which many fail to compile (smem overflow, register spill) and most are dominated.

## Decision

Sweep 48 configs at the headline shape (B=1, H=8, T_q=4096, T_k=8192, D=64, top_k=16, bf16): BLOCK_M in {32,64,128}, BLOCK_N in {16,32,64,128}, num_warps in {2,4}, num_stages in {2,3}. num_warps=8 pruned (forces smem layouts that fight the gather); num_stages=4 pruned (inner loop body is too short to benefit). Recompute `block_indices` per (BLOCK_M, BLOCK_N) so the gather geometry matches the tile.

## Consequences

Winner: `BLOCK_M=64, BLOCK_N=16, num_warps=4, num_stages=2` at 0.063 ms median. The BLOCK_N=16 result is mildly surprising and feeds the "Surprises" section: at this kernel's per-tile workload, atomic shuffle latency in the streaming softmax dominates and a smaller tile lets more queries' partial reductions overlap. Failed configs stay in `runs/autotune.json` so the smem/register failure modes are legible to anyone extending the grid.
