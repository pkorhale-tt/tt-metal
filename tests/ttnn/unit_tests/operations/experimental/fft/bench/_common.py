# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for FFT benchmarks (latency, throughput, accuracy, ...).

Single source of truth for:
  * routing / factorization (matches `fft.cpp::pick_three_factorization`)
  * input-tensor construction
  * timing (eager + trace) with proper warmup + sync
  * percentile stats
  * CSV / PNG writers

Each `bench/*.py` script imports from here so the routing logic is
identical across every paper figure.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import torch
import ttnn


# ─── routing ───────────────────────────────────────────────────────────
# Practical L1 cap: fft_two_pass keeps both passes' working sets resident
# in 1.49 MB of L1.  Empirically that's good for N <= 16 K (fp32); beyond
# that the static CB allocation overflows.  Bench scripts explicitly hand
# off to fft_three_pass at N > 16 K.
TWO_PASS_MAX_N    = 16 * 1024
# fft_three_pass requires log2N in [15, 30] → N in [32 K, 1 G].
THREE_PASS_MIN_N  = 32 * 1024


def pick_three_factorization(N: int) -> tuple[int, int, int]:
    """Mirror of ttnn::operations::experimental::pick_three_factorization
    in fft.cpp.  (N1, N2, N3) all in [32, 1024], product = N.
    """
    log2N = N.bit_length() - 1
    assert (1 << log2N) == N, f"N must be a power of 2 (got {N})"
    assert 15 <= log2N <= 30, f"three-pass needs log2N in [15, 30] (got {log2N})"

    log2_N3 = 10
    if log2N - log2_N3 < 10:
        log2_N3 = log2N - 10
    log2_rest = log2N - log2_N3
    log2_N1 = (log2_rest + 1) // 2
    log2_N2 = log2_rest - log2_N1
    return (1 << log2_N1, 1 << log2_N2, 1 << log2_N3)


def config_supported(N: int, B: int, dtype_label: str) -> bool:
    """Apply the three-pass restrictions: fp32-only, B=1-only, N >= 32K."""
    if N > TWO_PASS_MAX_N:
        if N < THREE_PASS_MIN_N:
            return False
        if dtype_label != "fp32" or B != 1:
            return False
    return True


# ─── input construction ────────────────────────────────────────────────
def make_input_rm(B: int, N: int, dtype, device, seed: int = 0xA11CE):
    torch.manual_seed(seed)
    x = torch.randn(B, N, dtype=torch.float32)
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device)


def make_input_three_pass(N: int, dtype, device, seed: int = 0xA11CE):
    N1, N2, N3 = pick_three_factorization(N)
    torch.manual_seed(seed)
    x = torch.randn(N1 * N2, N3, dtype=torch.float32)
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device)


def make_op(B: int, N: int, dtype, device, seed: int = 0xA11CE
            ) -> tuple[tuple, Callable]:
    """Return (input_tensors, op_callable).  op_callable(*input_tensors).

    Routing:
      N <= TWO_PASS_MAX_N (16 K) → ttnn.experimental.fft  (auto-routes
                                   to single-tile or two-pass)
      N >= THREE_PASS_MIN_N (32 K) → ttnn.experimental.fft_three_pass
                                     with pre-shaped (N1·N2, N3) input
    """
    if N > TWO_PASS_MAX_N:
        tt_x = make_input_three_pass(N, dtype, device, seed=seed)
        return (tt_x,), (lambda x: ttnn.experimental.fft_three_pass(x, full_N=N))
    tt_x = make_input_rm(B, N, dtype, device, seed=seed)
    return (tt_x,), (lambda x: ttnn.experimental.fft(x))


# ─── timing primitives ─────────────────────────────────────────────────
def time_eager(device, op, inputs, iters: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
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


def time_trace(device, op, inputs, iters: int, warmup: int = 3) -> list[float]:
    op(*inputs)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    op(*inputs)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for _ in range(warmup):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    lats_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t1 = time.perf_counter()
        lats_us.append((t1 - t0) * 1e6)

    ttnn.release_trace(device, tid)
    return lats_us


# ─── stats / derived metrics ───────────────────────────────────────────
def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def stats(lats_us: list[float]) -> dict:
    s = sorted(lats_us)
    return {
        "median_us": statistics.median(s),
        "min_us":    s[0],
        "max_us":    s[-1],
        "p05_us":    percentile(s, 5),
        "p95_us":    percentile(s, 95),
        "n_iters":   len(s),
    }


def fft_flops(N: int, B: int) -> int:
    """Standard FFT FLOP count: 5·N·log2(N) per single FFT, ×B for batch.
    Matches Brown et al. 2025 and the cuFFT convention.
    """
    log2N = N.bit_length() - 1
    return 5 * N * log2N * B


def samples_per_sec(N: int, B: int, latency_us: float) -> float:
    if latency_us <= 0:
        return float("nan")
    return (N * B) / (latency_us * 1e-6)


def gflops(N: int, B: int, latency_us: float) -> float:
    if latency_us <= 0:
        return float("nan")
    return fft_flops(N, B) / (latency_us * 1e-6) / 1e9


# ─── device setup ──────────────────────────────────────────────────────
def open_device(device_id: int = 0, trace_region: int = 2 * 1024 * 1024):
    if os.environ.get("TT_FFT_NATIVE", "0") != "1":
        os.environ["TT_FFT_NATIVE"] = "1"
        print("[bench] forcing TT_FFT_NATIVE=1")
    return ttnn.open_device(device_id=device_id, trace_region_size=trace_region)


# ─── CSV writer ────────────────────────────────────────────────────────
def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[bench] wrote {path} ({len(rows)} rows)")
