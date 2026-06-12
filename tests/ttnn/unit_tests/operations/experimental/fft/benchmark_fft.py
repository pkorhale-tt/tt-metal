#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
FFT benchmark for HPEC 2026 paper evaluation.

Usage:
    python benchmark_fft.py                    # all N, fp32+bf16, 10 warm + 20 timed
    python benchmark_fft.py --dtype fp32       # fp32 only
    python benchmark_fft.py --warmup 5 --runs 50
    python benchmark_fft.py --csv results.csv  # save CSV for plotting

Metrics reported per (N, dtype, algorithm):
  - Wormhole device time  (ms)   — median of timed runs
  - CPU time              (ms)   — NumPy on host (single-threaded)
  - GFLOPs/s (device)            — 5 N log2(N) / time_s
  - Speedup vs CPU               — cpu_time / device_time  (<1 = WH faster)
  - Estimated energy ratio       — (cpu_time × cpu_TDP) / (wh_time × wh_TDP)
                                    default: cpu_TDP=240W, wh_TDP=75W (n300 PCIe)

FFT FLOP count convention (matches cuFFT docs):
  real FFT of length N:  5 N log2(N)  FLOPs
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch
import ttnn

# ── Configuration ─────────────────────────────────────────────────────────────

# Wormhole n300 PCIe TDP in watts (measured ~75 W under load)
WH_TDP_W = 75.0
# Xeon Platinum 8260 (24-core) TDP in watts — matches Brown et al. ISC 2025
CPU_TDP_W = 240.0


@dataclass
class BenchResult:
    N: int
    dtype: str          # "fp32" or "bf16"
    algorithm: str      # "stockham" | "two_pass" | "three_pass" | "bluestein"
    wh_median_ms: float
    wh_p25_ms: float
    wh_p75_ms: float
    cpu_median_ms: float
    gflops_s: float
    speedup_vs_cpu: float   # > 1 means WH is faster
    energy_ratio: float     # > 1 means WH uses less energy
    rel_err: float


# ── N list per algorithm tier ─────────────────────────────────────────────────

POW2_STOCKHAM = [32, 64, 128, 256, 512, 1024]
POW2_TWO_PASS = [2048, 4096, 8192, 65536, 1 << 17, 1 << 20]
POW2_THREE_PASS = [1 << 21, 1 << 23, 1 << 24, 1 << 26]   # ≤ 2^26 fp32-safe
BLUESTEIN_N = [
    # small primes and composites
    97, 127, 257, 509,
    # medium composites (M = 2-pass)
    1000, 3000, 9999, 65535,
    # large 1024-aligned (M = 2-pass inner)
    64512,          # 63 × 1024, M = 2^17
    # XL (M = 3-pass inner)
    525312,         # 513 × 1024
    786432,         # 768 × 1024
]


def _algorithm_label(N: int) -> str:
    if N & (N - 1) == 0:          # power of 2
        if N <= 1024:
            return "stockham"
        if N <= (1 << 20):
            return "two_pass"
        return "three_pass"
    return "bluestein"


def _fft_flops(N: int) -> float:
    """5 N log2(N) — standard complex FFT FLOP count."""
    return 5.0 * N * math.log2(max(N, 2))


# ── Device helpers ────────────────────────────────────────────────────────────

def _open_device():
    """Open the first available Wormhole device."""
    device = ttnn.open_device(device_id=0)
    ttnn.enable_program_cache(device)
    return device


def _upload(x_np: np.ndarray, device, tt_dtype) -> ttnn.Tensor:
    t = torch.from_numpy(x_np).reshape(1, -1)
    return ttnn.from_torch(t, dtype=tt_dtype,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _run_wh_fft(tt_in: ttnn.Tensor):
    re, im = ttnn.experimental.fft(tt_in)
    return re, im


def _download(re: ttnn.Tensor, im: ttnn.Tensor, N: int) -> np.ndarray:
    r = ttnn.to_torch(re).reshape(N).to(torch.float32).numpy()
    i = ttnn.to_torch(im).reshape(N).to(torch.float32).numpy()
    return r + 1j * i


def _wh_fft_timed(tt_in: ttnn.Tensor, N: int, n_runs: int) -> List[float]:
    """Return list of per-run wall-clock times in ms (device synced)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        re, im = _run_wh_fft(tt_in)
        # Force completion — to_torch blocks until kernel finishes
        ttnn.to_torch(re)
        ttnn.to_torch(im)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


def _cpu_fft_timed(x_np: np.ndarray, n_runs: int) -> List[float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = np.fft.fft(x_np)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


# ── Benchmark one (N, dtype) point ───────────────────────────────────────────

def benchmark_one(
    N: int,
    dtype_str: str,
    device,
    warmup: int,
    runs: int,
) -> Optional[BenchResult]:
    tt_dtype = ttnn.float32 if dtype_str == "fp32" else ttnn.bfloat16

    # Skip N values that require more DRAM than safe for this dtype
    # (fp32 N=2^27 input alone = 512 MB → OOM on 1 GB WH B0)
    max_fp32_n = 1 << 26
    max_bf16_n = 1 << 26   # conservative; 2^27 bf16 intermediates also OOM
    if dtype_str == "fp32" and N > max_fp32_n:
        print(f"  SKIP N={N:>10,} fp32 (DRAM OOM on WH B0)")
        return None
    if dtype_str == "bf16" and N > max_bf16_n:
        print(f"  SKIP N={N:>10,} bf16 (DRAM OOM on WH B0)")
        return None

    algo = _algorithm_label(N)
    np.random.seed(N % (1 << 20))
    x_np = np.random.randn(N).astype(np.float32)
    if dtype_str == "bf16":
        x_np = x_np.astype(np.float16).astype(np.float32)  # quantise to bf16 range

    print(f"  bench N={N:>10,}  dtype={dtype_str}  algo={algo:<12}", end="", flush=True)

    # Upload to device
    try:
        tt_in = _upload(x_np, device, tt_dtype)
    except RuntimeError as e:
        print(f"  → SKIP (device alloc failed: {e})")
        return None

    # Warmup — fills program cache
    try:
        for _ in range(warmup):
            re, im = _run_wh_fft(tt_in)
            ttnn.to_torch(re); ttnn.to_torch(im)
    except RuntimeError as e:
        print(f"  → SKIP (warmup failed: {e})")
        return None

    # Accuracy check
    ref = np.fft.fft(x_np)
    re, im = _run_wh_fft(tt_in)
    got = _download(re, im, N)
    rel_err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))

    # Timed runs — device
    wh_times = _wh_fft_timed(tt_in, N, runs)
    wh_times.sort()
    wh_med = float(np.median(wh_times))
    wh_p25 = wh_times[runs // 4]
    wh_p75 = wh_times[3 * runs // 4]

    # CPU baseline (NumPy single-thread)
    cpu_times = _cpu_fft_timed(x_np, max(runs, 10))
    cpu_times.sort()
    cpu_med = float(np.median(cpu_times))

    flops = _fft_flops(N)
    gflops_s = flops / (wh_med * 1e-3) / 1e9
    speedup = cpu_med / wh_med
    energy_ratio = speedup * (CPU_TDP_W / WH_TDP_W)  # > 1 = WH more energy efficient

    print(f"  WH={wh_med:7.2f}ms  CPU={cpu_med:7.2f}ms  "
          f"{gflops_s:6.2f}GFlops/s  "
          f"speedup={speedup:.2f}×  energy={energy_ratio:.1f}×  "
          f"err={rel_err:.1e}")

    return BenchResult(
        N=N, dtype=dtype_str, algorithm=algo,
        wh_median_ms=wh_med, wh_p25_ms=wh_p25, wh_p75_ms=wh_p75,
        cpu_median_ms=cpu_med,
        gflops_s=gflops_s,
        speedup_vs_cpu=speedup,
        energy_ratio=energy_ratio,
        rel_err=rel_err,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FFT benchmark for HPEC 2026")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup iterations per (N, dtype)")
    parser.add_argument("--runs", type=int, default=20,
                        help="timed iterations per (N, dtype)")
    parser.add_argument("--csv", default="fft_benchmark.csv",
                        help="output CSV path")
    parser.add_argument("--stockham",    action="store_true", help="only Stockham N")
    parser.add_argument("--two-pass",    action="store_true", help="only two-pass N")
    parser.add_argument("--three-pass",  action="store_true", help="only three-pass N")
    parser.add_argument("--bluestein",   action="store_true", help="only Bluestein N")
    args = parser.parse_args()

    dtypes = (["fp32", "bf16"] if args.dtype == "both"
              else [args.dtype])

    # Build N list
    any_selected = args.stockham or args.two_pass or args.three_pass or args.bluestein
    n_list = []
    if not any_selected or args.stockham:
        n_list += POW2_STOCKHAM
    if not any_selected or args.two_pass:
        n_list += POW2_TWO_PASS
    if not any_selected or args.three_pass:
        n_list += POW2_THREE_PASS
    if not any_selected or args.bluestein:
        n_list += BLUESTEIN_N
    n_list = sorted(set(n_list))

    print("=" * 80)
    print("FFT Benchmark — Tenstorrent Wormhole B0  (HPEC 2026)")
    print(f"  dtypes  : {dtypes}")
    print(f"  warmup  : {args.warmup}   runs: {args.runs}")
    print(f"  WH TDP  : {WH_TDP_W} W   CPU TDP: {CPU_TDP_W} W (Xeon Platinum)")
    print(f"  N count : {len(n_list)} sizes × {len(dtypes)} dtypes = "
          f"{len(n_list)*len(dtypes)} benchmarks")
    print("=" * 80)

    device = _open_device()
    results: List[BenchResult] = []

    try:
        for dtype_str in dtypes:
            print(f"\n── dtype = {dtype_str} ──")
            for N in n_list:
                r = benchmark_one(N, dtype_str, device,
                                  warmup=args.warmup, runs=args.runs)
                if r is not None:
                    results.append(r)
    finally:
        ttnn.close_device(device)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'N':>10}  {'dtype':<5}  {'algo':<12}  "
          f"{'WH(ms)':>8}  {'CPU(ms)':>8}  "
          f"{'GFlops/s':>9}  {'Speedup':>8}  {'Energy×':>8}  {'RelErr':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r.N:>10,}  {r.dtype:<5}  {r.algorithm:<12}  "
              f"{r.wh_median_ms:>8.2f}  {r.cpu_median_ms:>8.2f}  "
              f"{r.gflops_s:>9.2f}  {r.speedup_vs_cpu:>8.2f}×  "
              f"{r.energy_ratio:>8.1f}×  {r.rel_err:>8.1e}")

    # ── CSV output ─────────────────────────────────────────────────────────────
    if results and args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"\nCSV saved to: {args.csv}")

    # ── Paper-ready summary ────────────────────────────────────────────────────
    if results:
        print("\n── Paper highlights (median across N per algorithm tier) ──")
        from collections import defaultdict
        by_algo = defaultdict(list)
        for r in results:
            if r.dtype == "fp32":
                by_algo[r.algorithm].append(r)
        for algo, rs in sorted(by_algo.items()):
            med_energy = sorted([r.energy_ratio for r in rs])[len(rs)//2]
            med_gflops = sorted([r.gflops_s for r in rs])[len(rs)//2]
            print(f"  {algo:<12}:  {med_gflops:5.2f} GFlops/s  "
                  f"energy efficiency {med_energy:.1f}× vs Xeon Platinum")


if __name__ == "__main__":
    main()
