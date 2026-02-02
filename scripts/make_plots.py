"""Loss curve, perplexity, and scaling-trend plots from the wandb
history CSVs in writeup/figures/data/ and the NSA-100M perplexity JSON.
Output: writeup/figures/{03_loss_curves,04_perplexity_vs_ctx,05_scaling_trend}.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

DATA_DIR = Path("writeup/figures/data")
FIG_DIR = Path("writeup/figures")
RUNS_DIR = Path("runs")

RUNS = {
    "nsa-100m-1b-32k": ("NSA-100M, 32k, 1B tok", "tab:blue", "-"),
    "nsa-150m-1b-32k": ("NSA-150M, 32k, 1B tok", "tab:orange", "-"),
    "nsa-300m-500m-32k": ("NSA-300M, 32k, 500M tok", "tab:green", "-"),
    "dense-100m-1b-8k": ("dense-100M, 8k, 1B tok", "tab:red", "--"),
    "nsa-100m-500m-64k": ("NSA-100M, 64k, 500M tok", "tab:purple", "-."),
}


def _ema(s: pd.Series, alpha: float = 0.05) -> pd.Series:
    return s.ewm(alpha=alpha).mean()


def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (label, color, ls) in RUNS.items():
        csv = DATA_DIR / f"{name}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv).dropna(subset=["train/loss", "train/tokens_seen"])
        df = df[df["train/loss"] < 50]  # clip the random-init spike
        ax.plot(
            df["train/tokens_seen"] / 1e9,
            _ema(df["train/loss"]),
            color=color, linestyle=ls, linewidth=1.5, label=label,
        )
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("training cross-entropy")
    ax.set_title("Training loss curves")
    ax.set_ylim(3.5, 12)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_loss_curves.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_DIR / '03_loss_curves.png'}")


def plot_perplexity_vs_ctx():
    ppl_path = DATA_DIR / "perplexity_nsa-100m-1b-32k.json"
    if not ppl_path.exists():
        ppl_path = RUNS_DIR / "nsa-100m-1b-32k" / "perplexity.json"
    if not ppl_path.exists():
        print(f"  skip plot 4: {ppl_path} not found")
        return
    data = json.loads(ppl_path.read_text())
    in_range = [(r["seq_len"], r["perplexity"]) for r in data["results"] if "perplexity" in r]

    extended = []
    ext_path = DATA_DIR / "long_probe_nsa-100m.json"
    if ext_path.exists():
        ext_data = json.loads(ext_path.read_text())
        # prefer "straight" mode (no rope rescaling) so the curve is the
        # honest degradation
        for r in ext_data["results"]:
            if r.get("mode") == "straight" and "perplexity" in r:
                extended.append((r["seq_len"], r["perplexity"]))

    combined = sorted(set(in_range + extended))
    seq_lens, ppl = zip(*combined)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    train_max = 32768
    in_train = [(x, y) for x, y in combined if x <= train_max]
    out_train = [(x, y) for x, y in combined if x > train_max]
    if in_train:
        xs, ys = zip(*in_train)
        ax.plot(xs, ys, "o-", color="tab:blue", linewidth=1.8, markersize=8,
                label="NSA-100M, at or within training context (32k)")
    if out_train:
        xs, ys = zip(*out_train)
        # Connect with the last in-training point
        xs_connect = [in_train[-1][0]] + list(xs)
        ys_connect = [in_train[-1][1]] + list(ys)
        ax.plot(xs_connect, ys_connect, "o--", color="tab:orange",
                linewidth=1.8, markersize=8,
                label="NSA-100M, extrapolated beyond training context")
    ax.axvline(train_max, color="gray", linestyle=":", alpha=0.6)
    ax.text(train_max, max(ppl) * 1.04, " 32k = training context",
            color="gray", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("evaluation context length (tokens, log scale)")
    ax.set_ylabel("perplexity (FineWeb-Edu held-out)")
    ax.set_title("NSA-100M perplexity vs evaluation context length")
    ax.set_xticks(list(seq_lens))
    ax.set_xticklabels([str(s) if s < 1024 else f"{s // 1024}k" for s in seq_lens])
    for x, y in combined:
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylim(min(ppl) * 0.92, max(ppl) * 1.10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_perplexity_vs_ctx.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_DIR / '04_perplexity_vs_ctx.png'}")


def plot_longbench():
    lb_path = DATA_DIR / "longbench_nsa-100m.json"
    if not lb_path.exists():
        print(f"  skip plot 6: {lb_path} not found")
        return
    data = json.loads(lb_path.read_text())
    rows = [r for r in data["results"] if "mean_nll" in r]
    if not rows:
        return
    tasks = [r["task"] for r in rows]
    nlls = [r["mean_nll"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = ["tab:blue"] * len(tasks)
    bars = ax.bar(range(len(tasks)), nlls, color=colors, alpha=0.85)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_ylabel("answer-token NLL  (lower = better)")
    ax.set_title("NSA-100M on LongBench v2 subset (gold-answer likelihood)")
    for bar, nll in zip(bars, nlls):
        ax.annotate(f"{nll:.2f}", xy=(bar.get_x() + bar.get_width() / 2, nll),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_longbench.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_DIR / '06_longbench.png'}")


def plot_scaling_trend():
    summary_path = DATA_DIR / "summary.json"
    if not summary_path.exists():
        print(f"  skip plot 5: {summary_path} not found")
        return
    summary = json.loads(summary_path.read_text())

    points = []
    for name, info in summary.items():
        n_params = info.get("n_params_estimate")
        final_loss = info.get("final_loss")
        if n_params is None or final_loss is None:
            continue
        label = name
        points.append((n_params, final_loss, label))

    nsa_pts = [p for p in points if p[2].startswith("nsa-") and "64k" not in p[2]]
    dense_pts = [p for p in points if p[2].startswith("dense-")]
    long_pts = [p for p in points if "64k" in p[2]]

    nsa_pts.sort()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if nsa_pts:
        xs, ys, names = zip(*nsa_pts)
        ax.plot([x / 1e6 for x in xs], ys, "o-", color="tab:blue",
                linewidth=1.5, markersize=9, label="NSA, 32k context",
                zorder=2)
        for x, y, n in nsa_pts:
            ax.annotate(n.split("-")[1].upper(), xy=(x / 1e6, y),
                        xytext=(10, -3), textcoords="offset points", fontsize=9)
    if dense_pts:
        xs, ys, names = zip(*dense_pts)
        ax.scatter([x / 1e6 for x in xs], ys, color="tab:red", marker="s",
                   s=80, label="dense-100M, 8k context", zorder=3)
    if long_pts:
        xs, ys, names = zip(*long_pts)
        ax.scatter([x / 1e6 for x in xs], ys, color="tab:purple", marker="D",
                   s=80, label="NSA-100M, 64k context", zorder=3)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([100, 150, 200, 300]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(90, 330)
    all_ys = [p[1] for p in nsa_pts + dense_pts + long_pts]
    ax.set_ylim(min(all_ys) * 0.97, max(all_ys) * 1.04)
    ax.set_xlabel("parameters (M, log scale)")
    ax.set_ylabel("final training cross-entropy")
    ax.set_title("Scaling trend: final train loss vs model size")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_scaling_trend.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_DIR / '05_scaling_trend.png'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves()
    plot_perplexity_vs_ctx()
    plot_scaling_trend()
    plot_longbench()


if __name__ == "__main__":
    main()
