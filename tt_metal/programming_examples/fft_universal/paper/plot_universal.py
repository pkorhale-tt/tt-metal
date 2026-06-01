#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate HPEC 2026 paper figures from the universal_sweep.csv.

Figures produced (saved as PDF + PNG):

  fig1_latency_vs_N.{pdf,png}         — Cached median latency vs N, log-log,
                                        coloured by dispatch path. Headline plot.
  fig2_gflops_vs_N.{pdf,png}          — Achieved GFLOP/s = 5·N·log2(N) / time.
  fig3_host_pct_vs_N.{pdf,png}        — % of wall in host glue vs % on device.
                                        Validates the "publishable" host story.
  fig4_path_distribution.{pdf,png}    — Stacked bar of which path each N took.

Usage:
  python plot_universal.py --csv paper_results/universal_sweep.csv \
                           --out  paper_results/figs/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("This script needs matplotlib. pip install matplotlib\n")
    raise

# ──────────────────────────────────────────────────────────────────────
# Style. Greyscale-safe palette, IEEE conference-ish.
# ──────────────────────────────────────────────────────────────────────
PATH_COLOR = {
    "packed_dft":    "#2e7d32",   # green
    "pow2_stockham": "#1565c0",   # blue
    "bluestein":     "#c62828",   # red
    "cooley_tukey":  "#6a1b9a",   # purple
    "identity":      "#9e9e9e",   # grey
}
PATH_MARKER = {
    "packed_dft":    "o",
    "pow2_stockham": "s",
    "bluestein":     "^",
    "cooley_tukey":  "D",
    "identity":      "x",
}
PATH_LABEL = {
    "packed_dft":    "Packed DFT (N ≤ 32)",
    "pow2_stockham": "Stockham (pow-2)",
    "bluestein":     "Bluestein (prime)",
    "cooley_tukey":  "Cooley–Tukey (composite)",
    "identity":      "identity",
}


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "N":             int(r["N"]),
                    "path":          r["path"],
                    "cached_ms":     float(r["cached_median_ms"]),
                    "p05_ms":        float(r["cached_p05_ms"]),
                    "p95_ms":        float(r["cached_p95_ms"]),
                    "gflops":        float(r["gflops_median"]),
                    "msamps_s":      float(r["msamples_per_sec_median"]),
                    "host_pct":      float(r.get("host_pct_median", "nan")),
                    "device_pct":    float(r.get("device_pct_median", "nan")),
                    "ndisp":         float(r.get("dispatches_per_call", "nan")),
                    "cold_ms":       float(r["cold_ms"]),
                })
            except (ValueError, KeyError) as e:
                sys.stderr.write(f"skip bad row: {r} ({e})\n")
    return rows


def by_path(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["path"]].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["N"])
    return g


def savefig(fig, out_dir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
def fig1_latency_vs_N(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    g = by_path(rows)
    for path, items in g.items():
        if not items:
            continue
        Ns       = [r["N"] for r in items]
        med      = [r["cached_ms"] for r in items]
        p05      = [r["p05_ms"] for r in items]
        p95      = [r["p95_ms"] for r in items]
        yerr_lo  = [m - lo for m, lo in zip(med, p05)]
        yerr_hi  = [hi - m for m, hi in zip(med, p95)]
        ax.errorbar(
            Ns, med, yerr=[yerr_lo, yerr_hi],
            fmt=PATH_MARKER.get(path, "."),
            color=PATH_COLOR.get(path, "k"),
            linestyle="-", linewidth=1.0, markersize=5,
            capsize=2, alpha=0.85,
            label=PATH_LABEL.get(path, path),
        )

    # 5N log N reference slope, anchored at the median pow2 point.
    pow2 = [r for r in rows if r["path"] == "pow2_stockham" and r["N"] >= 64]
    if pow2:
        anchor = pow2[len(pow2) // 2]
        N0, T0 = anchor["N"], anchor["cached_ms"]
        ref_N  = sorted({r["N"] for r in rows})
        ref_T  = [T0 * (N * math.log2(N)) / (N0 * math.log2(N0))
                  if N > 1 else None for N in ref_N]
        ax.plot(ref_N, ref_T, ls="--", color="grey", lw=0.8, alpha=0.7,
                label="O(N log N) reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("Cached latency (ms, median)")
    ax.set_title("fft_universal end-to-end latency vs N (Wormhole n300)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    savefig(fig, out_dir, "fig1_latency_vs_N")


# ──────────────────────────────────────────────────────────────────────
def fig2_gflops_vs_N(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    g = by_path(rows)
    for path, items in g.items():
        items = [r for r in items if r["N"] > 1]
        if not items:
            continue
        Ns = [r["N"] for r in items]
        gf = [r["gflops"] for r in items]
        ax.plot(Ns, gf,
                marker=PATH_MARKER.get(path, "."),
                color=PATH_COLOR.get(path, "k"),
                linestyle="-", linewidth=1.0, markersize=5, alpha=0.85,
                label=PATH_LABEL.get(path, path))
    ax.set_xscale("log")
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("Sustained GFLOP/s  (5·N·log₂N / latency)")
    ax.set_title("fft_universal throughput vs N (Wormhole n300)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    savefig(fig, out_dir, "fig2_gflops_vs_N")


# ──────────────────────────────────────────────────────────────────────
def fig3_host_pct_vs_N(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = [r for r in rows if not math.isnan(r["host_pct"])]
    g = by_path(rows)
    for path, items in g.items():
        if not items:
            continue
        Ns       = [r["N"] for r in items]
        host_pct = [r["host_pct"] for r in items]
        ax.plot(Ns, host_pct,
                marker=PATH_MARKER.get(path, "."),
                color=PATH_COLOR.get(path, "k"),
                linestyle="-", linewidth=1.0, markersize=5, alpha=0.85,
                label=PATH_LABEL.get(path, path))
    ax.axhline(50, ls=":", color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("% of wall time spent in host glue")
    ax.set_title("Host overhead vs N (lower is better; large-N is device-bound)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    savefig(fig, out_dir, "fig3_host_pct_vs_N")


# ──────────────────────────────────────────────────────────────────────
def fig4_path_distribution(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    g = by_path(rows)
    paths = [p for p in ("packed_dft", "pow2_stockham",
                         "bluestein", "cooley_tukey", "identity")
             if g.get(p)]
    counts = [len(g[p]) for p in paths]
    bars = ax.bar(
        [PATH_LABEL.get(p, p) for p in paths],
        counts,
        color=[PATH_COLOR.get(p, "k") for p in paths],
    )
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                str(c), ha="center", fontsize=9)
    ax.set_ylabel("# of N values in sweep")
    ax.set_title("Sweep coverage by dispatch path")
    plt.xticks(rotation=15, ha="right")
    savefig(fig, out_dir, "fig4_path_distribution")


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="paper_results/figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load(args.csv)
    if not rows:
        sys.exit(f"no rows loaded from {args.csv}")
    print(f"loaded {len(rows)} rows from {args.csv}")

    fig1_latency_vs_N(rows, args.out)
    fig2_gflops_vs_N(rows, args.out)
    fig3_host_pct_vs_N(rows, args.out)
    fig4_path_distribution(rows, args.out)

    print("\nDone. Drop these into the LaTeX paper:")
    for name in ("fig1_latency_vs_N", "fig2_gflops_vs_N",
                 "fig3_host_pct_vs_N", "fig4_path_distribution"):
        print(f"  {os.path.join(args.out, name)}.pdf")


if __name__ == "__main__":
    main()
