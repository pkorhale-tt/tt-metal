#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
plot_results.py — turn paper_results/csv/*.csv into paper_results/figs/*.pdf.

Pure post-processing: never opens the device. Idempotent: rerun safely.

Usage:
    python plot_results.py                # generate every figure
    python plot_results.py --only latency # only fig_latency.pdf
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required for plot_results.py. "
        "pip install matplotlib"
    )


CSV_DIR = C.DEFAULT_RESULTS_DIR / "csv"
FIG_DIR = C.DEFAULT_RESULTS_DIR / "figs"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── individual figures ───────────────────────────

def fig_latency() -> None:
    rows: list[dict] = []
    for name in ("latency.csv", "latency_fp32.csv", "latency_bf16.csv"):
        rows += _read_csv(CSV_DIR / name)
    if not rows:
        C.log("fig_latency: no input rows; skipping.")
        return

    # group by (dtype, precision, batch)
    series: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        try:
            key = (r["dtype"], r["precision"], int(r["batch"]))
            series[key].append((int(r["N"]), float(r["median_us"])))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (dtype, prec, B), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        label = f"{dtype}/{prec} B={B}"
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Median latency (µs)")
    ax.set_title("ttnn.experimental.fft: median latency vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_latency.pdf")
    fig.savefig(FIG_DIR / "fig_latency.png", dpi=160)
    plt.close(fig)
    C.log(f"Wrote {FIG_DIR/'fig_latency.pdf'}")


def fig_throughput() -> None:
    rows: list[dict] = []
    for name in ("throughput.csv", "throughput_fp32.csv", "throughput_bf16.csv"):
        rows += _read_csv(CSV_DIR / name)
    if not rows:
        C.log("fig_throughput: no input rows; skipping.")
        return

    series: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        try:
            key = (r["dtype"], r["precision"], int(r["batch"]))
            series[key].append((int(r["N"]), float(r["gflops"])))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (dtype, prec, B), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        label = f"{dtype}/{prec} B={B}"
        ax.plot(xs, ys, marker="s", linewidth=1.2, markersize=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Sustained GFLOP/s  (5·N·log₂N model)")
    ax.set_title("ttnn.experimental.fft: throughput vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_throughput.pdf")
    fig.savefig(FIG_DIR / "fig_throughput.png", dpi=160)
    plt.close(fig)
    C.log(f"Wrote {FIG_DIR/'fig_throughput.pdf'}")


def fig_accuracy() -> None:
    rows: list[dict] = []
    for name in ("accuracy.csv", "accuracy_fp32.csv", "accuracy_bf16.csv"):
        rows += _read_csv(CSV_DIR / name)
    if not rows:
        C.log("fig_accuracy: no input rows; skipping.")
        return

    series: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        try:
            key = (r["dtype"], r["precision"])
            series[key].append((int(r["N"]), float(r["rel_err"])))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (dtype, prec), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="^", linewidth=1.0, markersize=3,
                label=f"{dtype}/{prec}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("L2 relative error  vs  torch.fft (fp64)")
    ax.set_title("Numerical precision")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_accuracy.pdf")
    fig.savefig(FIG_DIR / "fig_accuracy.png", dpi=160)
    plt.close(fig)
    C.log(f"Wrote {FIG_DIR/'fig_accuracy.pdf'}")


def fig_program_cache() -> None:
    rows = _read_csv(CSV_DIR / "program_cache.csv")
    if not rows:
        C.log("fig_program_cache: no input rows; skipping.")
        return
    series: dict[tuple, list[tuple[int, float, float]]] = defaultdict(list)
    for r in rows:
        try:
            key = (r["dtype"], r["precision"])
            series[key].append((int(r["N"]),
                                float(r["cold_us"]),
                                float(r["warm_median_us"])))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (dtype, prec), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        speedups = [p[1]/p[2] if p[2] > 0 else 0 for p in pts]
        ax.plot(xs, speedups, marker="d", linewidth=1.2, markersize=3,
                label=f"{dtype}/{prec}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Speedup  cold / warm")
    ax.set_title("Program-cache amortisation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_program_cache.pdf")
    fig.savefig(FIG_DIR / "fig_program_cache.png", dpi=160)
    plt.close(fig)
    C.log(f"Wrote {FIG_DIR/'fig_program_cache.pdf'}")


def fig_metal_trace() -> None:
    rows = _read_csv(CSV_DIR / "metal_trace.csv")
    if not rows:
        C.log("fig_metal_trace: no input rows; skipping.")
        return

    series: dict[tuple, list[tuple[int, float, float]]] = defaultdict(list)
    for r in rows:
        if r.get("traced_us") in ("", "unsupported"):
            continue
        try:
            key = (r["dtype"], r["precision"])
            series[key].append((int(r["N"]),
                                float(r["untraced_us"]),
                                float(r["traced_us"])))
        except Exception:
            continue

    if not series:
        C.log("fig_metal_trace: trace API unsupported in this build; "
              "skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (dtype, prec), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        speedups = [p[1]/p[2] if p[2] > 0 else 0 for p in pts]
        ax.plot(xs, speedups, marker="*", linewidth=1.2, markersize=4,
                label=f"{dtype}/{prec}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Speedup  untraced / traced")
    ax.set_title("Metal Trace replay vs untraced dispatch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_metal_trace.pdf")
    fig.savefig(FIG_DIR / "fig_metal_trace.png", dpi=160)
    plt.close(fig)
    C.log(f"Wrote {FIG_DIR/'fig_metal_trace.pdf'}")


def fig_host_device_split() -> None:
    rows = _read_csv(CSV_DIR / "host_device_split.csv")
    if not rows:
        C.log("fig_host_device_split: no input rows; skipping.")
        return

    series: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        try:
            key = (r["dtype"], r["precision"])
            series[key].append((int(r["N"]), float(r["host_io_pct"])))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (dtype, prec), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="x", linewidth=1.2, markersize=4,
                label=f"{dtype}/{prec}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("% of e2e time spent on host I/O")
    ax.set_title("Host round-trip share of wall-clock time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_host_device_split.pdf")
    fig.savefig(FIG_DIR / "fig_host_device_split.png", dpi=160)
    plt.close(fig)
    C.log(f"Wrote {FIG_DIR/'fig_host_device_split.pdf'}")


# ─────────────────────────── dispatcher ────────────────────────────────────

ALL_FIGURES = {
    "latency":            fig_latency,
    "throughput":         fig_throughput,
    "accuracy":           fig_accuracy,
    "program_cache":      fig_program_cache,
    "metal_trace":        fig_metal_trace,
    "host_device_split":  fig_host_device_split,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="", help="csv list of figure stems")
    args = p.parse_args()
    _ensure_fig_dir()

    if args.only:
        wanted = set(s.strip() for s in args.only.split(",") if s.strip())
        figs = {k: v for k, v in ALL_FIGURES.items() if k in wanted}
        if not figs:
            C.log(f"WARN: --only={args.only} matched no known figure.")
    else:
        figs = ALL_FIGURES

    for name, fn in figs.items():
        try:
            fn()
        except Exception as e:
            C.log(f"ERROR generating {name}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
