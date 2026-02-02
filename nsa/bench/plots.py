"""Render plot 1 (throughput vs seq_len) and plot 2 (memory vs seq_len) into
writeup/figures/, from the JSON files written by throughput.py and memory.py.

Usage:
    python -m nsa.bench.plots \\
        --throughput-json runs/throughput.json \\
        --memory-json runs/memory.json \\
        --out-dir writeup/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_IMPL_LABEL = {
    "nsa": "NSA (Triton)",
    "fa3": "FlashAttention-3",
    "full_sdpa": "torch SDPA",
    "naive": "naive O(T^2)",
}
_IMPL_COLOR = {
    "nsa": "#2E7D32",       # green
    "fa3": "#1565C0",       # blue
    "full_sdpa": "#C62828", # red
    "naive": "#6A1B9A",     # purple
}


def _group_by_impl(rows: list[dict], y_key: str) -> dict[str, list[tuple[int, float | None]]]:
    out: dict[str, list[tuple[int, float | None]]] = {}
    for r in rows:
        out.setdefault(r["impl"], []).append((r["seq_len"], r.get(y_key)))
    for k in out:
        out[k].sort()
    return out


def plot_throughput(rows: list[dict], out_path: Path):
    grouped = _group_by_impl(rows, "tokens_per_sec")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for impl, points in grouped.items():
        xs = [p[0] for p in points if p[1] is not None]
        ys = [p[1] for p in points if p[1] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, "o-", label=_IMPL_LABEL.get(impl, impl),
                color=_IMPL_COLOR.get(impl), linewidth=2, markersize=6)
        # Mark OOM with X at the seq_len boundary.
        for x, y in points:
            if y is None:
                ax.plot([x], [ys[-1] if ys else 1], "x", color=_IMPL_COLOR.get(impl, "k"),
                        markersize=10, markeredgewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("forward tokens/sec")
    ax.set_title("NSA forward throughput vs full attention (H100)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_memory(rows: list[dict], out_path: Path):
    grouped = _group_by_impl(rows, "peak_gb")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for impl, points in grouped.items():
        xs = [p[0] for p in points if p[1] is not None]
        ys = [p[1] for p in points if p[1] is not None]
        oom_x = [p[0] for p in points if p[1] is None]
        color = _IMPL_COLOR.get(impl, "k")
        if xs:
            ax.plot(xs, ys, "o-", label=_IMPL_LABEL.get(impl, impl),
                    color=color, linewidth=2, markersize=6)
        for x in oom_x:
            # Plot the OOM marker as a single X above the y limit will be
            # set later. Use the H100 NVL boundary (80 GB) as a visible y.
            ax.plot([x], [80], marker="x", color=color, markersize=14, markeredgewidth=2.5,
                    label=f"{_IMPL_LABEL.get(impl, impl)} OOM" if x == oom_x[0] else None)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("peak GPU memory (GB), log scale")
    ax.set_title("NSA forward memory vs full attention (H100, B=1, H=16, D=64)")
    ax.axhline(80, color="k", linestyle="--", alpha=0.3, linewidth=1, zorder=1)
    ax.text(2**13, 80, "H100 NVL 80 GB", color="k", alpha=0.6, fontsize=9,
            ha="left", va="bottom")
    leg = ax.legend(loc="upper left", frameon=True, framealpha=1.0,
                    edgecolor="none", facecolor="white")
    leg.set_zorder(5)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--throughput-json", default="runs/throughput.json")
    p.add_argument("--memory-json", default="runs/memory.json")
    p.add_argument("--out-dir", default="writeup/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.throughput_json).exists():
        rows = json.loads(Path(args.throughput_json).read_text())
        plot_throughput(rows, out_dir / "01_throughput.png")
    if Path(args.memory_json).exists():
        rows = json.loads(Path(args.memory_json).read_text())
        plot_memory(rows, out_dir / "02_memory.png")


if __name__ == "__main__":
    main()
