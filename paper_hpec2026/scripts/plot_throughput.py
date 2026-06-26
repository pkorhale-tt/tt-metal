"""
plot_throughput.py — Generate throughput vs N figure for HiPC 2026 paper.

Usage:
    python plot_throughput.py

Output:
    throughput.pdf  (vector, use this in LaTeX)
    throughput.png  (dpi=300, for preview only)
"""

import os
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Benchmark data (Wormhole B0 n150, 50-run median) ─────────────────────────
# Bluestein excluded — non-pow2 N cannot be plotted on pow2 x-axis cleanly.
# Bluestein results reported in Table III / Table IV.
DATA = [
    # Stockham fp32 (B=64)
    (32,       "Stockham",   "fp32", 0.30),
    (64,       "Stockham",   "fp32", 0.64),
    (128,      "Stockham",   "fp32", 1.32),
    (256,      "Stockham",   "fp32", 2.58),
    (512,      "Stockham",   "fp32", 4.24),
    (1024,     "Stockham",   "fp32", 6.22),

    # Stockham bf16 (B=64)
    (32,       "Stockham",   "bf16", 0.21),
    (64,       "Stockham",   "bf16", 0.54),
    (128,      "Stockham",   "bf16", 1.08),
    (256,      "Stockham",   "bf16", 2.29),
    (512,      "Stockham",   "bf16", 3.70),
    (1024,     "Stockham",   "bf16", 5.46),

    # Two-pass fp32 (B=1)
    (2048,     "Two-pass",   "fp32", 0.24),
    (4096,     "Two-pass",   "fp32", 0.44),
    (8192,     "Two-pass",   "fp32", 0.74),
    (65536,    "Two-pass",   "fp32", 2.21),
    (131072,   "Two-pass",   "fp32", 2.60),
    (1048576,  "Two-pass",   "fp32", 3.61),

    # Two-pass bf16 (B=1)
    (2048,     "Two-pass",   "bf16", 0.23),
    (4096,     "Two-pass",   "bf16", 0.43),
    (8192,     "Two-pass",   "bf16", 0.72),
    (65536,    "Two-pass",   "bf16", 2.09),
    (131072,   "Two-pass",   "bf16", 2.46),
    (1048576,  "Two-pass",   "bf16", 3.44),

    # Three-pass fp32 (B=1)
    (2097152,  "Three-pass", "fp32", 1.29),
    (8388608,  "Three-pass", "fp32", 1.76),
    (16777216, "Three-pass", "fp32", 1.99),

    # Three-pass bf16 (B=1)
    (2097152,  "Three-pass", "bf16", 1.24),
    (8388608,  "Three-pass", "bf16", 1.67),
    (16777216, "Three-pass", "bf16", 1.89),
]

TIER_COLORS = {
    "Stockham":   "#1f77b4",   # blue
    "Two-pass":   "#2ca02c",   # green
    "Three-pass": "#ff7f0e",   # orange
}

TIER_MARKERS = {
    "Stockham":   "o",
    "Two-pass":   "s",
    "Three-pass": "^",
}


def split_by_tier_dtype(data):
    out = {}
    for N, tier, dtype, gf in data:
        key = (tier, dtype)
        out.setdefault(key, []).append((N, gf))
    for k in out:
        out[k].sort()
    return out


def main():
    grouped = split_by_tier_dtype(DATA)

    # ── Figure — larger size + higher DPI for sharpness ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for (tier, dtype), points in sorted(grouped.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color  = TIER_COLORS[tier]
        marker = TIER_MARKERS[tier]
        ls     = "-"  if dtype == "fp32" else "--"
        lw     = 2.0  if dtype == "fp32" else 1.5
        ms     = 6    if dtype == "fp32" else 5
        label  = f"{tier} ({'fp32' if dtype == 'fp32' else 'bfloat16'})"

        ax.plot(xs, ys,
                color=color, marker=marker, markersize=ms,
                linestyle=ls, linewidth=lw, label=label,
                zorder=3)

    # ── Tier boundary vertical lines ─────────────────────────────────────────
    boundaries = [
        (1024,    "Stockham\n→ Two-pass"),
        (1048576, "Two-pass\n→ Three-pass"),
    ]
    for bx, blabel in boundaries:
        ax.axvline(bx, color="gray", linestyle=":", linewidth=1.0, zorder=2)
        ax.text(bx * 1.08, 0.15, blabel,
                fontsize=7.5, color="gray", va="bottom", ha="left")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log", base=2)
    ax.set_xlim(2**4, 2**25)
    ax.set_ylim(0, 7.5)

    pow2_ticks = [2**k for k in range(5, 25)]
    ax.set_xticks(pow2_ticks)
    ax.set_xticklabels(
        [f"$2^{{{k}}}$" for k in range(5, 25)],
        fontsize=8.5, rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(0, 8, 1))
    ax.tick_params(axis="y", labelsize=9)

    ax.set_xlabel("Transform length $N$", fontsize=12)
    ax.set_ylabel("Throughput (GFLOPs/s)", fontsize=12)

    # ── Grid ─────────────────────────────────────────────────────────────────
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.5, zorder=1)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.3, zorder=1)

    # ── Legend — placed outside plot area to avoid overlapping curves ────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              fontsize=8.5, ncol=1,
              loc="upper left",
              bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0,
              framealpha=0.95,
              edgecolor="gray",
              borderpad=0.8)

    # ── Note about Bluestein ─────────────────────────────────────────────────
    ax.text(0.99, 0.04,
            "Bluestein (non-pow-2 $N$): see Table III",
            transform=ax.transAxes,
            fontsize=7.5, color="gray", ha="right", va="bottom")

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(right=0.78)  # make room for legend on the right

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # PDF — vector, always sharp, use this in LaTeX
    pdf_path = os.path.join(out_dir, "throughput.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved (vector): {pdf_path}")

    # PNG — raster preview at 300 dpi for sharpness
    png_path = os.path.join(out_dir, "throughput.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved (raster): {png_path}")


if __name__ == "__main__":
    main()