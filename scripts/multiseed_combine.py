"""Combine per-seed throughput JSONs into a single multi-seed file with
mean, stddev across seeds, and a 95% bootstrap CI from pooled per-iter
samples.

Usage:
    python scripts/multiseed_combine.py \\
        --seeds runs/throughput_seed0.json,runs/throughput_seed1.json,runs/throughput_seed2.json \\
        --out runs/throughput_multiseed.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def _bootstrap_ci_mean(samples: np.ndarray, *, n_boot: int = 10_000, alpha: float = 0.05, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean of `samples`."""
    n = samples.shape[0]
    if n < 2:
        v = float(samples.mean()) if n else 0.0
        return v, v
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = samples[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True,
                   help="comma-separated list of per-seed throughput JSON paths")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=12345)
    p.add_argument("--out", default="runs/throughput_multiseed.json")
    args = p.parse_args()

    seed_paths = [Path(x) for x in args.seeds.split(",")]
    if len(seed_paths) < 2:
        raise SystemExit("need at least 2 seed files for CI aggregation")

    per_seed = [json.loads(path.read_text()) for path in seed_paths]

    # Index each seed by (impl, seq_len).
    by_cell: dict[tuple[str, int], list[dict]] = {}
    for rows in per_seed:
        for r in rows:
            key = (r["impl"], r["seq_len"])
            by_cell.setdefault(key, []).append(r)

    rng = np.random.default_rng(args.rng_seed)
    out_cells: list[dict] = []

    for (impl, T), records in sorted(by_cell.items()):
        # Drop any seed that errored / OOMed for this cell.
        ok = [r for r in records if r.get("samples_ms")]
        if not ok:
            out_cells.append({
                "impl": impl, "seq_len": T,
                "ok": False,
                "n_seeds": 0, "n_iters_per_seed": None,
                "note": "all seeds failed for this cell",
            })
            continue

        seed_means = [statistics.fmean(r["samples_ms"]) for r in ok]
        seeds = [r.get("seed") for r in ok]

        pooled = np.array(
            [s for r in ok for s in r["samples_ms"]], dtype=np.float64,
        )
        mean_ms = float(pooled.mean())
        # Stddev across seed means: how much the per-seed estimate of the
        # mean wiggles run to run. Population stddev because the seeds are
        # a small sample.
        stddev_across_seeds_ms = (
            statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0
        )
        ci_lo_ms, ci_hi_ms = _bootstrap_ci_mean(
            pooled, n_boot=args.n_boot, alpha=args.alpha, rng=rng,
        )

        # tokens-per-second mirrors: a tighter ms is faster, so the upper
        # bound on ms maps to the lower bound on tps and vice versa.
        # Batch / heads / head_dim are constant across seeds for a cell:
        batch = ok[0].get("batch", 1) if "batch" in ok[0] else 1
        tps_mean = batch * T / (mean_ms / 1000.0)
        tps_lo = batch * T / (ci_hi_ms / 1000.0)
        tps_hi = batch * T / (ci_lo_ms / 1000.0)

        out_cells.append({
            "impl": impl, "seq_len": T, "ok": True,
            "n_seeds": len(ok), "n_iters_per_seed": len(ok[0]["samples_ms"]),
            "seeds": seeds,
            "seed_means_ms": seed_means,
            "mean_ms": mean_ms,
            "stddev_across_seeds_ms": stddev_across_seeds_ms,
            "ci_lo_ms": ci_lo_ms, "ci_hi_ms": ci_hi_ms,
            "tokens_per_sec": tps_mean,
            "tps_ci_lo": tps_lo, "tps_ci_hi": tps_hi,
            "ci_method": "percentile bootstrap on pooled per-iter samples",
            "alpha": args.alpha, "n_boot": args.n_boot,
        })

    # Crossover analysis: for each pair (nsa vs other), find the smallest
    # seq_len where nsa's tps_ci_lo >= other's tps_ci_hi (NSA strictly faster
    # at 95% confidence) and where other's tps_ci_lo >= nsa's tps_ci_hi
    # (other strictly faster).
    crossovers = []
    impls_in_data = sorted({c["impl"] for c in out_cells if c.get("ok")})
    for other in impls_in_data:
        if other == "nsa":
            continue
        seq_lens = sorted({c["seq_len"] for c in out_cells if c.get("ok") and c["impl"] in ("nsa", other)})
        nsa_by_T = {c["seq_len"]: c for c in out_cells if c.get("ok") and c["impl"] == "nsa"}
        other_by_T = {c["seq_len"]: c for c in out_cells if c.get("ok") and c["impl"] == other}

        regimes: list[dict] = []
        for T in seq_lens:
            n = nsa_by_T.get(T)
            o = other_by_T.get(T)
            if n is None or o is None:
                continue
            if n["tps_ci_lo"] >= o["tps_ci_hi"]:
                verdict = "nsa wins"
            elif o["tps_ci_lo"] >= n["tps_ci_hi"]:
                verdict = f"{other} wins"
            else:
                verdict = "tie (CIs overlap)"
            regimes.append({"seq_len": T, "verdict": verdict,
                            "nsa_tps_ci": [n["tps_ci_lo"], n["tps_ci_hi"]],
                            f"{other}_tps_ci": [o["tps_ci_lo"], o["tps_ci_hi"]]})
        crossovers.append({"pair": f"nsa vs {other}", "regimes": regimes})

    out_payload = {
        "cells": out_cells,
        "crossovers": crossovers,
        "n_seeds": len(seed_paths),
        "seed_sources": [str(p) for p in seed_paths],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {len(out_cells)} cells to {out_path}")
    for c in out_cells:
        if c.get("ok"):
            print(
                f"  {c['impl']:>10s} T={c['seq_len']:>6d}  "
                f"{c['mean_ms']:>7.3f} ms  "
                f"CI [{c['ci_lo_ms']:>6.3f}, {c['ci_hi_ms']:>6.3f}]  "
                f"tps {c['tokens_per_sec']:>11,.0f}"
            )


if __name__ == "__main__":
    main()
