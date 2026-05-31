# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench/throughput.py — FFT throughput vs N, vs B  (HPEC 2026, results R4 + R8)

Reports the two metrics reviewers care about for a Fourier-transform
paper:

  * **GFLOPs**  — uses the standard 5·N·log2(N) FLOP count per single FFT
                  (this is the same count Brown et al. 2025 and cuFFT
                  report against, so the numbers are directly comparable).
  * **Samples/sec** — total throughput in input samples processed per
                  second.  Dominant metric when batching matters.

Outputs
-------
  <out>/throughput.csv           per-(N,B,dtype,trace) GFLOPs + samples/sec
  <out>/throughput_vs_N.png      log-log, lines = dtype × trace, B=64 (or max-B)
  <out>/scaling_vs_B.png         linear, lines = N ∈ {1K, 16K}, fp32, trace=1
                                 — shows multi-core scaling story (R8)

Usage
-----
  TT_FFT_NATIVE=1 python tests/ttnn/unit_tests/operations/experimental/fft/bench/throughput.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ttnn

# allow `python bench/throughput.py` from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (                                          # noqa: E402
    config_supported, make_op, time_eager, time_trace, stats,
    gflops, samples_per_sec, open_device, write_csv,
    TWO_PASS_MAX_N, THREE_PASS_MIN_N,
)


# ─── sweep config ──────────────────────────────────────────────────────
# Throughput cares about steady-state numbers, so we iterate more and
# sweep a finer B grid than latency.py.
N_SWEEP = [
    1024, 2048, 4096, 16384,                     # small / single-tile + 2-pass
    32 * 1024, 64 * 1024, 256 * 1024,            # three-pass
    1 * 1024 * 1024, 2 * 1024 * 1024,            # three-pass large
]
B_SWEEP = [1, 4, 16, 64, 256]
DTYPES  = [("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)]
TRACE   = [False, True]


def _run_one(device, N, B, dtype, use_trace, iters):
    try:
        inputs, op = make_op(B, N, dtype, device)
    except Exception as e:
        return {"status": f"make_input_failed: {type(e).__name__}", "err": str(e)}

    try:
        if use_trace:
            lats = time_trace(device, op, inputs, iters)
        else:
            lats = time_eager(device, op, inputs, iters)
    except Exception as e:
        return {"status": f"run_failed: {type(e).__name__}", "err": str(e)}

    st = stats(lats)
    med = st["median_us"]
    st["gflops"]         = gflops(N, B, med)
    st["samples_per_sec"] = samples_per_sec(N, B, med)
    st["status"]         = "ok"
    return st


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
        default="tests/ttnn/unit_tests/operations/experimental/fft/bench/results",
        type=Path)
    parser.add_argument("--iters",       default=50, type=int)
    parser.add_argument("--max-n",       default=2 * 1024 * 1024, type=int)
    parser.add_argument("--max-b",       default=256, type=int)
    parser.add_argument("--trace-region", default=2 * 1024 * 1024, type=int)
    parser.add_argument("--device-id",   default=0, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = open_device(args.device_id, args.trace_region)
    rows = []
    try:
        for N in N_SWEEP:
            if N > args.max_n:
                continue
            for B in B_SWEEP:
                if B > args.max_b:
                    continue
                for dtype_label, dtype in DTYPES:
                    if not config_supported(N, B, dtype_label):
                        continue
                    for use_trace in TRACE:
                        tag = (f"N={N:>8d}  B={B:>4d}  "
                               f"{dtype_label:>4s}  trace={int(use_trace)}")
                        print(f"[bench] {tag}", end="  ", flush=True)
                        r = _run_one(device, N, B, dtype, use_trace,
                                     args.iters)
                        row = {"N": N, "B": B, "dtype": dtype_label,
                               "trace": int(use_trace), **r}
                        rows.append(row)
                        if r["status"] == "ok":
                            print(f"-> {r['gflops']:7.2f} GFLOPs  "
                                  f"{r['samples_per_sec']/1e6:8.2f} Msamp/s  "
                                  f"(med {r['median_us']:8.1f} us)")
                        else:
                            print(f"-> SKIP ({r['status']})")
    finally:
        ttnn.close_device(device)

    fieldnames = ["N", "B", "dtype", "trace", "status",
                  "median_us", "p05_us", "p95_us", "min_us", "max_us",
                  "n_iters", "gflops", "samples_per_sec", "err"]
    write_csv(rows, args.out / "throughput.csv", fieldnames)

    try:
        _plot_throughput_vs_N(rows, args.out / "throughput_vs_N.png")
        _plot_scaling_vs_B(rows, args.out / "scaling_vs_B.png")
    except ImportError:
        print("[bench] matplotlib not available, skipping plots")


# ─── plots ─────────────────────────────────────────────────────────────
def _plot_throughput_vs_N(rows, png_path):
    """R4: GFLOPs vs N, lines = dtype × trace, picks the max-B available
    per row so we show the device's best throughput at each N."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    style = {
        ("fp32", 0): dict(linestyle="--", marker="o", color="C0",
                          label="fp32  eager (best B)"),
        ("fp32", 1): dict(linestyle="-",  marker="o", color="C0",
                          label="fp32  trace (best B)"),
        ("bf16", 0): dict(linestyle="--", marker="s", color="C1",
                          label="bf16  eager (best B)"),
        ("bf16", 1): dict(linestyle="-",  marker="s", color="C1",
                          label="bf16  trace (best B)"),
    }

    # For each (dtype, trace, N) keep the row with max GFLOPs (best B).
    best = defaultdict(lambda: (-1.0, None))
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["dtype"], r["trace"], r["N"])
        if r["gflops"] > best[key][0]:
            best[key] = (r["gflops"], r)

    fig, ax = plt.subplots(figsize=(9, 6))
    for (dt, tr), sty in style.items():
        pts = [(N, gflops_) for (d, t, N), (gflops_, _) in best.items()
               if d == dt and t == tr]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, **sty)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("FFT length N")
    ax.set_ylabel("Throughput  (GFLOPs, 5·N·log₂N model)")
    ax.set_title("ttnn.experimental.fft — peak throughput vs N (n300)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"[bench] wrote {png_path}")


def _plot_scaling_vs_B(rows, png_path):
    """R8: samples/sec vs B, lines = N, fp32 trace=1.
    Saturation = memory-bandwidth-bound (the standard FFT story)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    pick_N = [1024, 4096, 16384]
    colors = ["C0", "C1", "C2"]
    for N, color in zip(pick_N, colors):
        pts = [(r["B"], r["samples_per_sec"] / 1e6) for r in rows
               if r.get("status") == "ok"
               and r["dtype"] == "fp32" and r["trace"] == 1
               and r["N"] == N]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", color=color, label=f"N={N}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size  B")
    ax.set_ylabel("Throughput  (Msamples / sec)")
    ax.set_title("Batched scaling — fp32, trace replay (n300)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"[bench] wrote {png_path}")


if __name__ == "__main__":
    main()
