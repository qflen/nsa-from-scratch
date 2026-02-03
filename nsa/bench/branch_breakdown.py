"""Per-branch latency breakdown of the NSA forward.

Brackets each of the five compute stages (score, compressed, selected,
sliding, combine) with `torch.cuda.synchronize() + time.perf_counter()`
and records per-iteration samples so the writeup can show which branch
dominates as context grows.

Usage:
    python -m nsa.bench.branch_breakdown \\
        --seq-lens 4096,16384,65536 --seed 0 \\
        --warmup 10 --iters 50 \\
        --out runs/branch_breakdown_seed0.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from nsa.reference import NSAConfig
from nsa.triton.forward import (
    _CompressedFn,
    _SelectedFn,
    _SlidingFn,
    _resolve_block_indices,
)
from nsa.triton.gating import combine


_STAGES = ("score", "compressed", "selected", "sliding", "combine", "total")


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "stddev": 0.0}
    return {
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "stddev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def _bracket(fn):
    """Run fn under cuda sync + perf_counter. Returns (output, elapsed_ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def make_qkv(B, H, T, D, dtype, *, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    return Q, K, V


def _one_forward(Q, K, V, cfg: NSAConfig, *, samples: dict[str, list[float]]):
    """One end-to-end NSA forward, timed by stage. Mirrors nsa_forward exactly
    so the per-stage costs add to a credible "total"."""
    D = Q.shape[-1]
    scale = cfg.scale if cfg.scale is not None else 1.0 / (D ** 0.5)

    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    out_c, dt_c = _bracket(lambda: _CompressedFn.apply(
        Q, K, V, cfg.block_size_c, cfg.pool, cfg.causal, cfg.scale,
    ))

    block_indices, dt_score = _bracket(lambda: _resolve_block_indices(
        Q, K,
        block_size_m=cfg.block_size_m, block_size_n=cfg.block_size_n,
        top_k=cfg.top_k, causal=cfg.causal, scale=float(scale),
    ))

    out_s, dt_sel = _bracket(lambda: _SelectedFn.apply(
        Q, K, V, cfg.block_size_n, cfg.block_size_m, cfg.top_k,
        block_indices, cfg.causal, cfg.scale,
    ))

    out_w, dt_slid = _bracket(lambda: _SlidingFn.apply(
        Q, K, V, cfg.window_size, cfg.causal, cfg.scale,
    ))

    B, H, Tq, _ = Q.shape
    gate_logits = Q.new_full((B, H, Tq, 3), 0.0)
    _, dt_comb = _bracket(lambda: combine(
        out_c, out_s, out_w, gate_logits, activation=cfg.gate_activation,
    ))

    torch.cuda.synchronize()
    dt_total = (time.perf_counter() - t_total_start) * 1000.0

    samples["score"].append(dt_score)
    samples["compressed"].append(dt_c)
    samples["selected"].append(dt_sel)
    samples["sliding"].append(dt_slid)
    samples["combine"].append(dt_comb)
    samples["total"].append(dt_total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="4096,16384,65536")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/branch_breakdown.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    cfg = NSAConfig()

    rows: list[dict] = []
    for T in seq_lens:
        try:
            Q, K, V = make_qkv(args.batch, args.heads, T, args.head_dim, dtype, seed=args.seed)
            # warmup
            for _ in range(args.warmup):
                warm: dict[str, list[float]] = {s: [] for s in _STAGES}
                _one_forward(Q, K, V, cfg, samples=warm)

            samples: dict[str, list[float]] = {s: [] for s in _STAGES}
            for _ in range(args.iters):
                _one_forward(Q, K, V, cfg, samples=samples)

            summary = {s: _summary(samples[s]) for s in _STAGES}
            rows.append({
                "seq_len": T, "seed": args.seed,
                "iters": args.iters, "warmup": args.warmup,
                "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
                "samples_ms": samples,
                "summary_ms": summary,
            })
            mean_total = summary["total"]["mean"]
            print(
                f"T={T:>6d}  total {mean_total:>6.3f} ms  "
                f"score {summary['score']['mean']:>5.3f}  "
                f"compressed {summary['compressed']['mean']:>5.3f}  "
                f"selected {summary['selected']['mean']:>6.3f}  "
                f"sliding {summary['sliding']['mean']:>5.3f}  "
                f"combine {summary['combine']['mean']:>5.3f}"
            )
            del Q, K, V
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            rows.append({"seq_len": T, "seed": args.seed, "note": "OOM"})
            print(f"T={T:>6d}  OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append({"seq_len": T, "seed": args.seed, "error": f"{type(e).__name__}: {str(e)[:80]}"})
            print(f"T={T:>6d}  ERROR: {type(e).__name__}: {str(e)[:60]}")
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {out_path}  (seed={args.seed})")


if __name__ == "__main__":
    main()
