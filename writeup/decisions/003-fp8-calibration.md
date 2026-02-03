# 003 - FP8 calibration: per-tensor absmax, bf16 dequant in the wrapper

## Context

The FP8 selected path stores Q, K, V as `float8_e4m3fn` or `float8_e5m2` and runs the existing bf16 kernel after dequant. The calibration choice (per-tensor vs per-channel scale, where to dequant) is the difference between an FP8 path that matches bf16 and one that silently degrades. FA-3 uses per-channel because their kernel reads the scale inside the inner loop; we dequant in the wrapper, so the cost model differs.

## Decision

Per-tensor absmax: one fp32 scalar per tensor (Q, K, V independently) computed as `absmax / fmt_max` where `fmt_max` is 448.0 (E4M3) or 57344.0 (E5M2). Dequant in Python before launching the bf16 kernel: `x_bf16 = x_fp8.to(bf16) * scale`. No per-channel, no exponent clipping. Outliers saturate to `+/- fmt_max`.

## Consequences

The FP8 path is a 10-line wrapper over the existing kernel; the kernel stays bf16, FP8 is purely storage. The cost: per-tensor absmax leaves precision on the table when a single outlier sets the scale for every other value. Measured residual: `1e-2` to `5e-2` relative error vs bf16 forward at the headline shape, within the writeup's gate but visibly worse than bf16's `6e-4`. The natural follow-up is per-channel calibration for K and V (Q is short enough not to matter); the kernel does not change.
