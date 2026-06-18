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
    python benchmark_fft.py --wh-power 42 --cpu-power 353  # paper-accurate energy

Metrics reported per (N, dtype, algorithm):
  - Wormhole device time (ms) — median of timed runs, kernel execution only
    (excludes host-device transfer, matching Brown et al. ISC 2025 methodology)
  - CPU time (ms) — NumPy on host (all available cores)
  - GFLOPs/s (device) — 5 N log2(N) / time_s  (complex FFT convention)
  - Speedup vs CPU — cpu_time / device_time (>1 = WH faster)
  - Energy ratio — (cpu_time × cpu_power) / (wh_time × wh_power)
    Pass --wh-power and --cpu-power for paper-accurate measured values.
    Falls back to TDP estimates: cpu_TDP=240W, wh_TDP=75W (conservative).
    Formula: Brown et al. ISC 2025, Table 3 — E = P × t, ratio = E_cpu / E_wh

FFT FLOP count convention (matches cuFFT docs):
  complex FFT of length N:  5 N log2(N)  FLOPs
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch
import ttnn

# ── Configuration ─────────────────────────────────────────────────────────────

# TDP fallback values (used if --wh-power / --cpu-power not supplied).
# For paper-accurate results, measure with:
#   WH  power: tt_power_sidecar.py --backend sysfs → avg_power_W
#   CPU power: RAPL via /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
# Paper (Brown et al. ISC 2025) measured: CPU=353W, WH=42W for 2D FFT 1024×1024
WH_TDP_W  = 75.0
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
POW2_THREE_PASS = [1 << 21, 1 << 23, 1 << 24]  # 2^26 OOMs on 1 GB WH B0
BLUESTEIN_N = [
    # small primes and composites
    97, 127, 257, 509,
    # medium composites (M = 2-pass)
    1000, 3000, 9999,
    # large 1024-aligned (M = 2-pass inner, fp32 hw limit: M ≤ 2^17)
    64512,          # 63 × 1024, M = 2^17
    # XL (M = 3-pass inner); non-1024-aligned (65535) is L1-unsafe → excluded
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


def _fft_flops(N: int, B: int = 1) -> float:
    """5 B N log2(N) — standard complex FFT FLOP count convention (matches cuFFT docs).
    Input to the benchmark is real-valued, but we use the complex convention
    (5 N log2 N rather than 2.5 N log2 N) for comparability with FFT literature.
    Multiplied by B (batch size) to reflect total device work per call."""
    return 5.0 * B * N * math.log2(max(N, 2))


# Stockham batch size — uses B=64 so all 64 Tensix cores are active.
# Two-pass / three-pass / Bluestein distribute work across cores internally
# regardless of batch size, so they use B=1.
STOCKHAM_BATCH = 64


# ── Device helpers ────────────────────────────────────────────────────────────

def _open_device():
    """Open the first available Wormhole device."""
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    return device


def _upload(x_np: np.ndarray, device, tt_dtype, B: int = 1) -> ttnn.Tensor:
    # x_np is 1-D shape (N,); tile to (B, N) so each row is an independent FFT.
    t = torch.from_numpy(np.tile(x_np.reshape(1, -1), (B, 1)))
    return ttnn.from_torch(t, dtype=tt_dtype,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _run_wh_fft(tt_in: ttnn.Tensor):
    re, im = ttnn.experimental.fft(tt_in)
    return re, im


def _download(re: ttnn.Tensor, im: ttnn.Tensor, N: int, B: int = 1) -> np.ndarray:
    """Download first row only for accuracy check."""
    r = ttnn.to_torch(re).reshape(B, N)[0].to(torch.float32).numpy()
    i = ttnn.to_torch(im).reshape(B, N)[0].to(torch.float32).numpy()
    return r + 1j * i


def _wh_fft_timed(tt_in: ttnn.Tensor, N: int, n_runs: int, device) -> List[float]:
    """Return list of per-run wall-clock times in ms.
    Times kernel execution only — excludes D2H download, matching
    Brown et al. ISC 2025: 'performance numbers for WH are execution time only.'
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        re, im = _run_wh_fft(tt_in)
        ttnn.synchronize_device(device)  # sync without downloading
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


def _cpu_fft_timed(x_np: np.ndarray, n_runs: int) -> List[float]:
    """Return list of per-run wall-clock times in ms (all available CPU cores).
    x_np may be 1-D (B=1) or 2-D (B>1); np.fft.fft operates on last axis."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = np.fft.fft(x_np, axis=-1)
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
    wh_power_w: float,
    cpu_power_w: float,
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
    # Stockham (N≤1024) runs one transform per core → use B=STOCKHAM_BATCH
    # to fill all 64 cores and measure peak device throughput fairly.
    # All other algorithms distribute work across cores internally (B=1).
    B = STOCKHAM_BATCH if algo == "stockham" else 1

    np.random.seed(N % (1 << 20))
    x_np = np.random.randn(N).astype(np.float32)
    if dtype_str == "bf16":
        # Correct bf16 quantisation via PyTorch (np.float16 is fp16, not bf16)
        x_np = torch.tensor(x_np).to(torch.bfloat16).float().numpy()

    print(f"  bench N={N:>10,}  B={B:<2}  dtype={dtype_str}  algo={algo:<12}", end="", flush=True)

    # Upload to device — shape (B, N)
    try:
        tt_in = _upload(x_np, device, tt_dtype, B=B)
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

    # Accuracy check — compare first row against NumPy reference
    ref = np.fft.fft(x_np)
    re, im = _run_wh_fft(tt_in)
    got = _download(re, im, N, B=B)
    rel_err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))

    # Timed runs — device (kernel only, no D2H)
    wh_times = _wh_fft_timed(tt_in, N, runs, device)
    wh_times.sort()
    wh_med = float(np.median(wh_times))
    wh_p25 = float(np.percentile(wh_times, 25))
    wh_p75 = float(np.percentile(wh_times, 75))

    # CPU baseline: B independent FFTs of length N (matches WH total work)
    x_batch_np = np.tile(x_np, (B, 1))   # shape (B, N)
    cpu_times = _cpu_fft_timed(x_batch_np, max(runs, 10))
    cpu_times.sort()
    cpu_med = float(np.median(cpu_times))

    flops = _fft_flops(N, B=B)           # 5 * B * N * log2(N)
    gflops_s = flops / (wh_med * 1e-3) / 1e9
    speedup = cpu_med / wh_med
    energy_ratio = speedup * (cpu_power_w / wh_power_w)

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
    parser.add_argument("--stockham",   action="store_true", help="only Stockham N")
    parser.add_argument("--two-pass",   action="store_true", help="only two-pass N")
    parser.add_argument("--three-pass", action="store_true", help="only three-pass N")
    parser.add_argument("--bluestein",  action="store_true", help="only Bluestein N")
    parser.add_argument("--wh-power",  type=float, default=None,
                        help="Measured WH avg power in watts (from TT-SMI). "
                             "Overrides WH_TDP_W for energy calculation.")
    parser.add_argument("--cpu-power", type=float, default=None,
                        help="Measured CPU avg power in watts (from RAPL). "
                             "Overrides CPU_TDP_W for energy calculation.")
    args = parser.parse_args()

    wh_power_w  = args.wh_power  if args.wh_power  is not None else WH_TDP_W
    cpu_power_w = args.cpu_power if args.cpu_power is not None else CPU_TDP_W

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
    print(f"  dtypes   : {dtypes}")
    print(f"  warmup   : {args.warmup}   runs: {args.runs}")
    print(f"  WH power : {wh_power_w} W   CPU power: {cpu_power_w} W")
    print(f"  N count  : {len(n_list)} sizes × {len(dtypes)} dtypes = "
          f"{len(n_list)*len(dtypes)} benchmarks")
    print("=" * 80)

    device = _open_device()
    results: List[BenchResult] = []

    try:
        for dtype_str in dtypes:
            print(f"\n── dtype = {dtype_str} ──")
            for N in n_list:
                r = benchmark_one(N, dtype_str, device,
                                  warmup=args.warmup, runs=args.runs,
                                  wh_power_w=wh_power_w, cpu_power_w=cpu_power_w)
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
        print("\n── Paper highlights (median across N per algorithm tier, fp32) ──")
        from collections import defaultdict
        by_algo = defaultdict(list)
        for r in results:
            if r.dtype == "fp32":
                by_algo[r.algorithm].append(r)
        for algo, rs in sorted(by_algo.items()):
            med_energy = sorted([r.energy_ratio for r in rs])[len(rs) // 2]
            med_gflops = sorted([r.gflops_s for r in rs])[len(rs) // 2]
            print(f"  {algo:<12}:  {med_gflops:5.2f} GFlops/s  "
                  f"energy efficiency {med_energy:.1f}× vs CPU")


if __name__ == "__main__":
    main()
