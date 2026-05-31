# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench/latency.py — FFT latency vs N  (HPEC 2026 paper, results R3 + R11)

Measures end-to-end host-visible latency of `ttnn.experimental.fft` (and
`fft_three_pass` for N > 1 M) over the full supported N range, for both
fp32 and bf16, eager vs Metal-Trace, batched B in {1, 8, 64}.

Outputs
-------
  <out>/latency.csv   raw per-configuration measurements
  <out>/latency.png   log-log plot, B=1, 4 lines: {fp32,bf16} × {eager,trace}

Usage
-----
  TT_FFT_NATIVE=1 python tests/ttnn/unit_tests/operations/experimental/fft/bench/latency.py \\
      --out tests/ttnn/unit_tests/operations/experimental/fft/bench/results \\
      --iters 100

Notes
-----
* Each (N, B, dtype, trace) point is run for `--iters` measurements,
  preceded by 3 warmup calls.  Median, p05, p95, min, max reported.
* Synchronization is done with `ttnn.synchronize_device` between calls
  for the eager path; trace replay is blocking.
* For N > 1 M we explicitly call `fft_three_pass` with the canonical
  factorization the unit tests use (since `ttnn.experimental.fft` only
  auto-routes through 1 M today).
* Three-pass is fp32-only and B=1 only in the current build (matches
  the unit-test gating).  Those cells are skipped in the sweep.
* The IFFT N <= 1024 legacy carve-out documented in the trace sweep
  test does NOT apply here because we measure FORWARD FFT.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import ttnn


# ───────────────────────── routing helpers ──────────────────────────────
_THREE_PASS_FACTOR = {
    1 << 21: (64, 32, 1024),   # 2 M
    1 << 22: (64, 64, 1024),   # 4 M
    1 << 24: (128, 128, 1024), # 16 M
}


def _make_input_rm(B: int, N: int, dtype, device):
    torch.manual_seed(0xA11CE)
    x = torch.randn(B, N, dtype=torch.float32)
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _make_input_three_pass(N: int, dtype, device):
    if N not in _THREE_PASS_FACTOR:
        raise ValueError(f"No three-pass factorization tabulated for N={N}")
    N1, N2, N3 = _THREE_PASS_FACTOR[N]
    torch.manual_seed(0xA11CE)
    x = torch.randn(N1 * N2, N3, dtype=torch.float32)
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _make_op(B: int, N: int, dtype, device) -> tuple[tuple, Callable]:
    """Return (input_tensors, op_callable). op_callable(*input_tensors)."""
    if N > (1 << 20):
        tt_x = _make_input_three_pass(N, dtype, device)
        return (tt_x,), (lambda x: ttnn.experimental.fft_three_pass(x, full_N=N))
    tt_x = _make_input_rm(B, N, dtype, device)
    return (tt_x,), (lambda x: ttnn.experimental.fft(x))


# ───────────────────────── timing core ──────────────────────────────────
def _percentile(sorted_lats, p):
    idx = int(round((p / 100.0) * (len(sorted_lats) - 1)))
    return sorted_lats[idx]


def _time_eager(device, op, inputs, iters):
    for _ in range(3):
        op(*inputs)
    ttnn.synchronize_device(device)

    lats_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        op(*inputs)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        lats_us.append((t1 - t0) * 1e6)
    return lats_us


def _time_trace(device, op, inputs, iters):
    # Warm program cache against the SAME persistent input addresses
    # the trace will reference.
    op(*inputs)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    op(*inputs)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    lats_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t1 = time.perf_counter()
        lats_us.append((t1 - t0) * 1e6)

    ttnn.release_trace(device, tid)
    return lats_us


def _stats(lats_us):
    s = sorted(lats_us)
    return {
        "median_us": statistics.median(s),
        "min_us":    s[0],
        "max_us":    s[-1],
        "p05_us":    _percentile(s, 5),
        "p95_us":    _percentile(s, 95),
        "n_iters":   len(s),
    }


# ───────────────────────── sweep config ─────────────────────────────────
N_SWEEP = [16, 64, 256, 1024, 2048, 4096, 16384, 65536, 262144, 1048576,
           1 << 21, 1 << 22]
B_SWEEP = [1, 8, 64]
DTYPES  = [("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)]
TRACE   = [False, True]


def _config_supported(N, B, dtype_label):
    if N > (1 << 20):
        # three-pass: fp32 only, B=1 only.
        if dtype_label != "fp32" or B != 1:
            return False
    return True


def _run_one(device, N, B, dtype_label, dtype, use_trace, iters):
    try:
        inputs, op = _make_op(B, N, dtype, device)
    except Exception as e:
        return {"status": f"make_input_failed: {type(e).__name__}", "err": str(e)}

    try:
        if use_trace:
            lats = _time_trace(device, op, inputs, iters)
        else:
            lats = _time_eager(device, op, inputs, iters)
    except Exception as e:
        return {"status": f"run_failed: {type(e).__name__}", "err": str(e)}

    return {"status": "ok", **_stats(lats)}


# ───────────────────────── main ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="tests/ttnn/unit_tests/operations/experimental/fft/bench/results",
        type=Path, help="output directory for csv/png")
    parser.add_argument("--iters", default=100, type=int,
                        help="measurement iterations per config")
    parser.add_argument("--max-n", default=1 << 22, type=int,
                        help="cap on N (default 4M; raise to 16M with care)")
    parser.add_argument("--trace-region", default=2 * 1024 * 1024, type=int,
                        help="trace region size in bytes")
    parser.add_argument("--device-id", default=0, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if os.environ.get("TT_FFT_NATIVE", "0") != "1":
        os.environ["TT_FFT_NATIVE"] = "1"
        print("[bench] forcing TT_FFT_NATIVE=1")

    device = ttnn.open_device(device_id=args.device_id,
                              trace_region_size=args.trace_region)
    rows = []
    try:
        for N in N_SWEEP:
            if N > args.max_n:
                continue
            for B in B_SWEEP:
                for dtype_label, dtype in DTYPES:
                    if not _config_supported(N, B, dtype_label):
                        continue
                    for use_trace in TRACE:
                        tag = f"N={N:>8d} B={B:>3d} {dtype_label:>4s} trace={int(use_trace)}"
                        print(f"[bench] {tag}", end="  ", flush=True)
                        stats = _run_one(device, N, B, dtype_label, dtype,
                                         use_trace, args.iters)
                        row = {
                            "N": N, "B": B, "dtype": dtype_label,
                            "trace": int(use_trace),
                            **stats,
                        }
                        rows.append(row)
                        if stats["status"] == "ok":
                            print(f"-> med={stats['median_us']:8.2f} us  "
                                  f"p05={stats['p05_us']:8.2f}  "
                                  f"p95={stats['p95_us']:8.2f}")
                        else:
                            print(f"-> SKIP ({stats['status']})")
    finally:
        ttnn.close_device(device)

    csv_path = args.out / "latency.csv"
    fieldnames = ["N", "B", "dtype", "trace", "status",
                  "median_us", "min_us", "max_us",
                  "p05_us", "p95_us", "n_iters", "err"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[bench] wrote {csv_path} ({len(rows)} rows)")

    png_path = args.out / "latency.png"
    try:
        _plot_latency(rows, png_path)
    except ImportError:
        print("[bench] matplotlib not available, skipping plot")


def _plot_latency(rows, png_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    style = {
        ("fp32", 0): {"linestyle": "--", "marker": "o", "color": "C0",
                     "label": "fp32  eager"},
        ("fp32", 1): {"linestyle": "-",  "marker": "o", "color": "C0",
                     "label": "fp32  trace"},
        ("bf16", 0): {"linestyle": "--", "marker": "s", "color": "C1",
                     "label": "bf16  eager"},
        ("bf16", 1): {"linestyle": "-",  "marker": "s", "color": "C1",
                     "label": "bf16  trace"},
    }

    for key, sty in style.items():
        dt, tr = key
        pts = [(r["N"], r["median_us"]) for r in rows
               if r["dtype"] == dt and r["trace"] == tr and r["B"] == 1
               and r.get("status") == "ok"]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, **sty)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("FFT length N")
    ax.set_ylabel("Latency  (μs per call)")
    ax.set_title("ttnn.experimental.fft  —  latency vs N  (B=1, n300)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"[bench] wrote {png_path}")


if __name__ == "__main__":
    main()
