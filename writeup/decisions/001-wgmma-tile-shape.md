# 001 - WGMMA tile shape: m64n64k16 at D=64, m64n128k16 at D=128

## Context

The selected-branch CUDA C++ forward issues two WGMMA matmuls per tile: `S = Q @ K^T` then `O += P @ V`. Hopper supports m64n{8,16,32,64,128,256}k16 for bf16. The right tile fits one warpgroup's register budget, aligns with the K-major smem layout, and does not leave compute idle waiting on K/V gather loads. Q tiles are 64 queries per CTA (one warpgroup); K/V tiles are 64 keys per top-k iteration.

## Decision

`m64n64k16` for both QK^T and PV at D=64; `m64n128k16` for PV at D=128. Both are the SS (smem-smem) variant with `tnspA = tnspB = 0` (K-major). The output fragment is f32 with bf16 inputs.

## Consequences

At D=64, n=64 means PV reads exactly the 64x64 V tile staged in smem, no cross-tile dance. At D=128, n=128 lets one PV MMA consume the full V tile instead of two, at the cost of growing the output accumulator to 64 fp32 values per thread (vs 32 at D=64). The register pressure that creates is already tight enough to forbid adding a third pipeline stage in the inner top-k loop. Forward correctness at the headline shape: `rel(out) = 6.07e-4` vs Triton, comfortably below the 1e-3 gate.
