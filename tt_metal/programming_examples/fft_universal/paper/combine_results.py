#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Merge Wormhole, FFTW, and ablation CSVs into the paper's joint figures.

Inputs are produced by:
  * metal_example_fft_universal_sweep            -> universal_sweep.csv
  * metal_example_fft_universal_sweep --disable-packed-dft
                                                 -> universal_sweep_noPDFT.csv
  * fftw_baseline                                -> fftw_baseline.csv
  * metal_example_fft_universal_sweep --batch 64 -> universal_sweep_B64.csv

Outputs (PDF + PNG into --out):
  fig5_wh_vs_fftw_latency.{pdf,png}      Wormhole vs FFTW vs CPU, log-log
  fig6_wh_vs_fftw_gflops.{pdf,png}       Sustained GFLOPs head-to-head
  fig7_ablation_packed_dft.{pdf,png}     With / without packed-DFT
  fig8_batch_scaling.{pdf,png}           Per-call latency vs batch size

Usage:
  python combine_results.py \
      --wh paper_results/universal_sweep.csv \
      --fftw paper_results/fftw_baseline.csv \
      --no-pdft paper_results/universal_sweep_noPDFT.csv \
      --batched paper_results/universal_sweep_B64.csv \
      --out paper_results/figs/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("pip install matplotlib\n"); raise


def load(path, time_key, label_key="path"):
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "N":      int(r["N"]),
                    "path":   r.get(label_key, ""),
                    "t_ms":   float(r[time_key]),
                    "gflops": float(r["gflops_median"]),
                })
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda x: x["N"])
    return rows


def savefig(fig, out_dir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
def fig5_wh_vs_fftw_latency(wh, fftw, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if wh:
        ax.plot([r["N"] for r in wh], [r["t_ms"] for r in wh],
                "o-", color="#1565c0", lw=1.2, ms=4, label="Wormhole fft_universal")
    if fftw:
        ax.plot([r["N"] for r in fftw], [r["t_ms"] for r in fftw],
                "s-", color="#c62828", lw=1.2, ms=4, label="FFTW3 (host CPU, fp32)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("Cached / executed latency (ms, median)")
    ax.set_title("Wormhole vs FFTW3 — end-to-end latency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    savefig(fig, out_dir, "fig5_wh_vs_fftw_latency")


def fig6_wh_vs_fftw_gflops(wh, fftw, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if wh:
        ax.plot([r["N"] for r in wh if r["N"] > 1],
                [r["gflops"] for r in wh if r["N"] > 1],
                "o-", color="#1565c0", lw=1.2, ms=4, label="Wormhole fft_universal")
    if fftw:
        ax.plot([r["N"] for r in fftw if r["N"] > 1],
                [r["gflops"] for r in fftw if r["N"] > 1],
                "s-", color="#c62828", lw=1.2, ms=4, label="FFTW3 (host CPU, fp32)")
    ax.set_xscale("log")
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("Sustained GFLOP/s  (5·N·log₂N / latency)")
    ax.set_title("Throughput head-to-head")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    savefig(fig, out_dir, "fig6_wh_vs_fftw_gflops")


def fig7_ablation_packed_dft(wh_on, wh_off, out_dir):
    if not wh_on or not wh_off:
        print("  skipping fig7 (need both --wh and --no-pdft)")
        return
    # Pair by N
    off_by_N = {r["N"]: r for r in wh_off}
    paired = [(r, off_by_N[r["N"]]) for r in wh_on if r["N"] in off_by_N]
    paired = [(on, off) for on, off in paired if on["N"] <= 1024]  # focus on small-N regime
    if not paired:
        print("  no paired rows for fig7")
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    Ns       = [on["N"] for on, _ in paired]
    on_med   = [on["t_ms"] for on, _ in paired]
    off_med  = [off["t_ms"] for _, off in paired]
    speedup  = [off / on for on, off in zip(on_med, off_med)]

    ax2 = ax.twinx()
    ax.plot(Ns, on_med,  "o-",  color="#1565c0", lw=1.2, ms=4, label="with packed DFT")
    ax.plot(Ns, off_med, "s--", color="#c62828", lw=1.2, ms=4, label="without packed DFT (zero-padded pow-2)")
    ax2.plot(Ns, speedup, "^-", color="#2e7d32", lw=1.0, ms=3, alpha=0.7,
             label="speedup (right axis)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("Latency (ms, median)")
    ax2.set_ylabel("Speedup from packed DFT")
    ax.set_title("Ablation: packed direct-DFT kernel (small-N regime)")
    ax.grid(True, which="both", alpha=0.3)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)
    savefig(fig, out_dir, "fig7_ablation_packed_dft")


def fig8_batch_scaling(wh, wh_b64, out_dir):
    if not wh or not wh_b64:
        print("  skipping fig8 (need both --wh and --batched)")
        return
    by_b1  = {r["N"]: r for r in wh}
    by_b64 = {r["N"]: r for r in wh_b64}
    Ns = sorted(set(by_b1) & set(by_b64))
    if not Ns: return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(Ns, [by_b1[N]["t_ms"]  for N in Ns],
            "o-", color="#1565c0", lw=1.2, ms=4, label="B = 1")
    ax.plot(Ns, [by_b64[N]["t_ms"] for N in Ns],
            "s-", color="#2e7d32", lw=1.2, ms=4, label="B = 64 (per-call)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Transform length N")
    ax.set_ylabel("Per-call latency (ms, median)")
    ax.set_title("Batched throughput: B=1 vs B=64 back-to-back calls")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    savefig(fig, out_dir, "fig8_batch_scaling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh",       help="universal_sweep.csv")
    ap.add_argument("--fftw",     help="fftw_baseline.csv")
    ap.add_argument("--no-pdft",  dest="no_pdft", help="universal_sweep_noPDFT.csv")
    ap.add_argument("--batched",  help="universal_sweep_B64.csv")
    ap.add_argument("--out",      default="paper_results/figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wh      = load(args.wh,      "cached_median_ms")
    no_pdft = load(args.no_pdft, "cached_median_ms")
    batched = load(args.batched, "cached_median_ms")
    fftw    = load(args.fftw,    "median_ms")

    fig5_wh_vs_fftw_latency(wh, fftw, args.out)
    fig6_wh_vs_fftw_gflops(wh, fftw, args.out)
    fig7_ablation_packed_dft(wh, no_pdft, args.out)
    fig8_batch_scaling(wh, batched, args.out)


if __name__ == "__main__":
    main()
