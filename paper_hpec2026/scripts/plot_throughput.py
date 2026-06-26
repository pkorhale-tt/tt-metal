"""
plot_throughput.py — Generate throughput vs N figure for HPEC 2026 paper.

Usage:
    python plot_throughput.py

Output:
    throughput.pdf  (and throughput.png) in the same directory as this script.

Data is hardcoded from the benchmark run on yyzc-wh-03 (Wormhole n150).
Edit the DATA list below if you re-run benchmarks.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Benchmark data (from yyzc-wh-03, Wormhole n150, 50-run median) ──────────
# Each entry: (N, tier, dtype, gflops)
DATA = [
    # Stockham fp32 (B=64)
    (32,          "Stockham",   "fp32", 0.30),
    (64,          "Stockham",   "fp32", 0.64),
    (128,         "Stockham",   "fp32", 1.32),
    (256,         "Stockham",   "fp32", 2.58),
    (512,         "Stockham",   "fp32", 4.24),
    (1024,        "Stockham",   "fp32", 6.22),

    # Stockham bf16 (B=64)
    (32,          "Stockham",   "bf16", 0.21),
    (64,          "Stockham",   "bf16", 0.54),
    (128,         "Stockham",   "bf16", 1.08),
    (256,         "Stockham",   "bf16", 2.29),
    (512,         "Stockham",   "bf16", 3.70),
    (1024,        "Stockham",   "bf16", 5.46),

    # Two-pass fp32 (B=1)
    (2048,        "Two-pass",   "fp32", 0.24),
    (4096,        "Two-pass",   "fp32", 0.44),
    (8192,        "Two-pass",   "fp32", 0.74),
    (65536,       "Two-pass",   "fp32", 2.21),
    (131072,      "Two-pass",   "fp32", 2.60),
    (1048576,     "Two-pass",   "fp32", 3.61),

    # Two-pass bf16 (B=1)
    (2048,        "Two-pass",   "bf16", 0.23),
    (4096,        "Two-pass",   "bf16", 0.43),
    (8192,        "Two-pass",   "bf16", 0.72),
    (65536,       "Two-pass",   "bf16", 2.09),
    (131072,      "Two-pass",   "bf16", 2.46),
    (1048576,     "Two-pass",   "bf16", 3.44),

    # Three-pass fp32 (B=1)
    (2097152,     "Three-pass", "fp32", 1.29),
    (8388608,     "Three-pass", "fp32", 1.76),
    (16777216,    "Three-pass", "fp32", 1.99),

    # Three-pass bf16 (B=1)
    (2097152,     "Three-pass", "bf16", 1.24),
    (8388608,     "Three-pass", "bf16", 1.67),
    (16777216,    "Three-pass", "bf16", 1.89),

    # Bluestein fp32 (B=1)
    (97,          "Bluestein",  "fp32", 0.01),
    (127,         "Bluestein",  "fp32", 0.01),
    (257,         "Bluestein",  "fp32", 0.01),
    (509,         "Bluestein",  "fp32", 0.02),
    (1000,        "Bluestein",  "fp32", 0.03),
    (3000,        "Bluestein",  "fp32", 0.07),
    (9999,        "Bluestein",  "fp32", 0.14),
    (64512,       "Bluestein",  "fp32", 0.44),
    (525312,      "Bluestein",  "fp32", 0.14),
    (786432,      "Bluestein",  "fp32", 0.21),
]

TIER_COLORS = {
    "Stockham":   "#1f77b4",   # blue
    "Two-pass":   "#2ca02c",   # green
    "Three-pass": "#ff7f0e",   # orange
    "Bluestein":  "#9467bd",   # purple
}

TIER_MARKERS = {
    "Stockham":   "o",
    "Two-pass":   "s",
    "Three-pass": "^",
    "Bluestein":  "D",
}

def split_by_tier_dtype(data):
    """Return dict keyed by (tier, dtype) → sorted list of (N, gflops)."""
    out = {}
    for N, tier, dtype, gf in data:
        key = (tier, dtype)
        out.setdefault(key, []).append((N, gf))
    for k in out:
        out[k].sort()
    return out

def main():
    grouped = split_by_tier_dtype(DATA)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    tiers_plotted = set()
    for (tier, dtype), points in sorted(grouped.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color  = TIER_COLORS[tier]
        marker = TIER_MARKERS[tier]
        ls     = "-" if dtype == "fp32" else "--"
        lw     = 1.8 if dtype == "fp32" else 1.4
        label  = f"{tier} ({dtype})"

        ax.plot(xs, ys,
                color=color, marker=marker, markersize=5,
                linestyle=ls, linewidth=lw, label=label)
        tiers_plotted.add(tier)

    # ── Tier boundary vertical lines ─────────────────────────────────────────
    boundaries = [
        (1024,    "Stockham\n→ Two-pass"),
        (1048576, "Two-pass\n→ Three-pass"),
    ]
    for bx, blabel in boundaries:
        ax.axvline(bx, color="gray", linestyle=":", linewidth=1.0)
        ax.text(bx * 1.05, 0.05, blabel,
                fontsize=6.5, color="gray", va="bottom", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_yscale("linear")

    ax.set_xlabel("Transform length $N$", fontsize=11)
    ax.set_ylabel("Throughput (GFLOPs/s)", fontsize=11)
    ax.set_title("FFT Throughput vs. Transform Length on Wormhole B0 n150",
                 fontsize=11)

    ax.set_xlim(16, 2**25)
    ax.set_ylim(0, 7.5)

    # X-axis ticks at powers of 2 (and a few non-pow2 Bluestein points)
    pow2_ticks = [2**k for k in range(5, 25)]
    ax.set_xticks(pow2_ticks)
    ax.set_xticklabels(
        [f"$2^{{{k}}}$" for k in range(5, 25)],
        fontsize=7, rotation=45, ha="right"
    )

    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    # Legend — two columns
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7.5, ncol=2,
              loc="upper left", framealpha=0.9)

    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"throughput.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")

if __name__ == "__main__":
    import matplotlib.ticker
    main()
