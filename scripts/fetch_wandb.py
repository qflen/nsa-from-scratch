"""Pull training history (loss, lr, grad_norm, tokens_seen) for the
NSA-100M, NSA-150M, dense-100M, and 64k extension runs from wandb.
Output drives the loss curve and scaling-trend plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

RUN_IDS = [
    "nsa-100m-1b-32k",
    "nsa-150m-1b-32k",
    "nsa-300m-500m-32k",
    "dense-100m-1b-8k",
    "nsa-100m-500m-64k",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="nkimon00-maastricht-university/nsa-from-scratch")
    p.add_argument("--out", default="writeup/figures/data/")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(args.project)
    by_name = {r.name: r for r in runs}

    summary = {}
    for name in RUN_IDS:
        runs_with_name = [r for r in by_name.values() if r.name == name]
        if not runs_with_name:
            print(f"  ! no wandb run found with name '{name}'")
            continue
        # Multiple runs may have the same name (re-runs after restart).
        # Pick the longest one (most steps).
        chosen = max(runs_with_name, key=lambda r: r.summary.get("train/step", 0) or 0)
        history = chosen.history(
            keys=["train/loss", "train/lr", "train/grad_norm", "train/tokens_seen", "train/peak_memory_gb"],
            samples=100000,
        )
        history_path = out_dir / f"{name}.csv"
        history.to_csv(history_path, index=False)

        cfg = dict(chosen.config) if chosen.config else {}
        summary[name] = {
            "run_id": chosen.id,
            "state": chosen.state,
            "n_steps": int(chosen.summary.get("train/step", 0) or 0),
            "tokens_seen": int(chosen.summary.get("train/tokens_seen", 0) or 0),
            "final_loss": float(chosen.summary.get("train/loss", float("nan"))),
            "final_lr": float(chosen.summary.get("train/lr", float("nan"))),
            "peak_memory_gb": float(chosen.summary.get("train/peak_memory_gb", float("nan"))),
            "n_params_estimate": cfg.get("n_params_estimate"),
            "history_csv": str(history_path),
        }
        print(f"  {name}: {summary[name]['n_steps']} steps, final loss {summary[name]['final_loss']:.4f}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary to {summary_path}")


if __name__ == "__main__":
    main()
