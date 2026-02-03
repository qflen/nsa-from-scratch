"""Render plot 1 (throughput vs seq_len), plot 2 (memory vs seq_len),
plot 7 (per-branch latency breakdown), and plot 8 (max-batch frontier)
into writeup/figures/, from the JSON files written by the bench modules.

Usage:
    python -m nsa.bench.plots \\
        --throughput-json runs/throughput.json \\
        --multiseed-json runs/throughput_multiseed.json \\
        --memory-json runs/memory.json \\
        --memory-sweep-json runs/memory_sweep.json \\
        --branch-breakdown-json runs/branch_breakdown.json \\
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
_BRANCH_LABEL = {
    "compressed": "compressed",
    "selected": "selected",
    "sliding": "sliding",
    "score": "score (Q.K topk)",
    "combine": "gate combine",
}
_BRANCH_COLOR = {
    "compressed": "#1565C0",
    "selected": "#C62828",
    "sliding": "#2E7D32",
    "score": "#F9A825",
    "combine": "#6A1B9A",
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


def plot_throughput_ci(payload: dict, out_path: Path):
    """Plot 1 with 95% CI bands on each impl. Consumes the multiseed payload
    written by scripts/multiseed_combine.py."""
    cells = [c for c in payload["cells"] if c.get("ok")]
    n_seeds = payload.get("n_seeds", "?")
    by_impl: dict[str, list[dict]] = {}
    for c in cells:
        by_impl.setdefault(c["impl"], []).append(c)
    for impl in by_impl:
        by_impl[impl].sort(key=lambda c: c["seq_len"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for impl, cs in by_impl.items():
        xs = [c["seq_len"] for c in cs]
        ys = [c["tokens_per_sec"] for c in cs]
        lo = [c["tps_ci_lo"] for c in cs]
        hi = [c["tps_ci_hi"] for c in cs]
        color = _IMPL_COLOR.get(impl, "k")
        ax.plot(xs, ys, "o-", label=_IMPL_LABEL.get(impl, impl),
                color=color, linewidth=2, markersize=6)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("forward tokens/sec  (mean and 95% CI)")
    ax.set_title(f"NSA forward throughput vs full attention (H100 NVL, {n_seeds} seeds)")
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
            ax.plot([x], [94], marker="x", color=color, markersize=14, markeredgewidth=2.5,
                    label=f"{_IMPL_LABEL.get(impl, impl)} OOM" if x == oom_x[0] else None)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("peak GPU memory (GB), log scale")
    ax.set_title("NSA forward memory vs full attention (H100, B=1, H=16, D=64)")
    ax.axhline(94, color="k", linestyle="--", alpha=0.3, linewidth=1, zorder=1)
    ax.text(2**13, 94, "H100 NVL 94 GB", color="k", alpha=0.6, fontsize=9,
            ha="left", va="bottom")
    leg = ax.legend(loc="upper left", frameon=True, framealpha=1.0,
                    edgecolor="none", facecolor="white")
    leg.set_zorder(5)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_branch_breakdown(rows: list[dict], out_path: Path):
    """Stacked bar of {score, compressed, selected, sliding, combine} ms per
    seq_len. Aggregates across seeds (mean of per-seed means)."""
    by_T: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if "summary_ms" not in r:
            continue
        T = r["seq_len"]
        by_T.setdefault(T, {})
        for stage, st in r["summary_ms"].items():
            by_T[T].setdefault(stage, []).append(st["mean"])

    seq_lens = sorted(by_T.keys())
    stages = ["compressed", "score", "selected", "sliding", "combine"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottoms = [0.0] * len(seq_lens)
    for stage in stages:
        ys = []
        for T in seq_lens:
            samples = by_T.get(T, {}).get(stage, [])
            ys.append(sum(samples) / len(samples) if samples else 0.0)
        ax.bar(
            [str(T) if T < 1024 else f"{T // 1024}k" for T in seq_lens],
            ys, bottom=bottoms,
            color=_BRANCH_COLOR.get(stage), label=_BRANCH_LABEL.get(stage, stage),
            edgecolor="white", linewidth=0.5,
        )
        bottoms = [b + y for b, y in zip(bottoms, ys)]

    for i, T in enumerate(seq_lens):
        total_mean = sum(
            sum(by_T[T][s]) / len(by_T[T][s]) for s in stages if s in by_T[T]
        )
        ax.annotate(f"{total_mean:.2f} ms",
                    xy=(i, total_mean), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("forward latency (ms)")
    ax.set_title("NSA forward per-branch latency (H100 NVL, B=1, H=16, D=64)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_memory_sweep(rows: list[dict], out_path: Path):
    """Max-batch frontier across seq_lens for each impl."""
    by_impl: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        impl = r["impl"]
        by_impl.setdefault(impl, []).append((r["seq_len"], r["max_batch_ok"]))
    for impl in by_impl:
        by_impl[impl].sort()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for impl, points in by_impl.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = _IMPL_COLOR.get(impl, "k")
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=7,
                label=_IMPL_LABEL.get(impl, impl))
        for x, y in points:
            ax.annotate(str(y), xy=(x, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=color)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("max batch_size at peak < 94 GB")
    ax.set_title("Max usable batch_size at H100 NVL (94 GB)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--throughput-json", default="runs/throughput.json")
    p.add_argument("--multiseed-json", default="runs/throughput_multiseed.json")
    p.add_argument("--memory-json", default="runs/memory.json")
    p.add_argument("--memory-sweep-json", default="runs/memory_sweep.json")
    p.add_argument("--branch-breakdown-json", default="runs/branch_breakdown.json")
    p.add_argument("--out-dir", default="writeup/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Throughput: prefer the multiseed file with CI bands; fall back to
    # single-shot if not present.
    if Path(args.multiseed_json).exists():
        payload = json.loads(Path(args.multiseed_json).read_text())
        plot_throughput_ci(payload, out_dir / "01_throughput.png")
    elif Path(args.throughput_json).exists():
        rows = json.loads(Path(args.throughput_json).read_text())
        plot_throughput(rows, out_dir / "01_throughput.png")

    if Path(args.memory_json).exists():
        rows = json.loads(Path(args.memory_json).read_text())
        plot_memory(rows, out_dir / "02_memory.png")

    if Path(args.branch_breakdown_json).exists():
        rows = json.loads(Path(args.branch_breakdown_json).read_text())
        plot_branch_breakdown(rows, out_dir / "07_branch_breakdown.png")

    if Path(args.memory_sweep_json).exists():
        rows = json.loads(Path(args.memory_sweep_json).read_text())
        plot_memory_sweep(rows, out_dir / "08_memory_sweep.png")


if __name__ == "__main__":
    main()
