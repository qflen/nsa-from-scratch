"""Throughput benchmark across seq_len in [1k..64k] for nsa, fa3, full_sdpa.

Measures forward-only tokens-per-second for each impl. Records every
per-iteration sample (timed with cuda.Event) so downstream tooling can
compute mean, stddev, and a bootstrap 95% CI across seeds.

Usage:
    python -m nsa.bench.throughput \\
        --seq-lens 1024,2048,4096,8192,16384,32768,65536 \\
        --impls nsa,fa3,full_sdpa --dtype bf16 \\
        --warmup 25 --iters 100 --seed 0 \\
        --out runs/throughput_seed0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from nsa.reference import NSAConfig
from nsa.triton.forward import nsa_forward


def _bench(fn, *, iters: int, warmup: int) -> list[float]:
    """Run fn, return per-iter wall time in milliseconds.

    Uses one cuda.Event pair per iteration; synchronizes once at the end
    of each iter so elapsed_time is well defined and cross-iter overlap
    cannot mask the kernel time being attributed to a neighbor.
    """
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return samples_ms


def make_qkv(B: int, H: int, T: int, D: int, dtype, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    return Q, K, V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="1024,2048,4096,8192,16384,32768,65536")
    p.add_argument("--impls", default="nsa,fa3,full_sdpa,moba")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/throughput.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    impls = args.impls.split(",")
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    cfg = NSAConfig()

    rows: list[dict] = []
    for T in seq_lens:
        for impl in impls:
            try:
                Q, K, V = make_qkv(args.batch, args.heads, T, args.head_dim, dtype, seed=args.seed)
                if impl == "full_sdpa":
                    fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
                elif impl == "fa3":
                    from flash_attn import flash_attn_func
                    Qf = Q.transpose(1, 2).contiguous()
                    Kf = K.transpose(1, 2).contiguous()
                    Vf = V.transpose(1, 2).contiguous()
                    fn = lambda: flash_attn_func(Qf, Kf, Vf, causal=True)
                elif impl == "nsa":
                    fn = lambda: nsa_forward(Q, K, V, cfg)
                elif impl == "moba":
                    from nsa.triton.moba import moba_attention
                    fn = lambda: moba_attention(
                        Q, K, V, block_size_n=64, block_size_m=64,
                        top_k_per_query=16, top_k_cap=32, causal=True,
                    )[0]
                else:
                    raise ValueError(impl)

                samples_ms = _bench(fn, iters=args.iters, warmup=args.warmup)
                mean_ms = statistics.fmean(samples_ms)
                median_ms = statistics.median(samples_ms)
                stddev_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
                tps = args.batch * T / (mean_ms / 1000.0)
                rows.append({
                    "impl": impl, "seq_len": T, "seed": args.seed,
                    "tokens_per_sec": tps,
                    "ms_per_call": mean_ms,
                    "mean_ms": mean_ms, "median_ms": median_ms, "stddev_ms": stddev_ms,
                    "iters": args.iters, "warmup": args.warmup,
                    "samples_ms": samples_ms,
                })
                print(f"{impl:>10s} T={T:>6d}  {mean_ms:>7.3f} +/- {stddev_ms:>5.3f} ms  {tps:>11,.0f} tok/s")
            except torch.cuda.OutOfMemoryError:
                rows.append({"impl": impl, "seq_len": T, "seed": args.seed, "tokens_per_sec": None, "note": "OOM"})
                print(f"{impl:>10s} T={T:>6d}  OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                rows.append({"impl": impl, "seq_len": T, "seed": args.seed, "tokens_per_sec": None, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                print(f"{impl:>10s} T={T:>6d}  ERROR: {type(e).__name__}: {str(e)[:60]}")
                torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {out_path}  (seed={args.seed})")


if __name__ == "__main__":
    main()
