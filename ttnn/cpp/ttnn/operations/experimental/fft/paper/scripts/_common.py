# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the ttnn.experimental.fft paper-kit benchmarks.

Every bench_*.py script in this folder imports from here. Keep this
module dependency-free beyond stdlib + torch + ttnn so the scripts
remain runnable inside the standard tt-metal dev container.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch

try:
    import ttnn
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import ttnn. Build tt-metal with `./build_metal.sh -b Release`"
        " and `pip install -e ttnn`, then re-run."
    ) from e


# ───────────────────────── Defaults / N lists ──────────────────────────────

DEFAULT_RESULTS_DIR = Path(
    os.environ.get("TT_FFT_PAPER_RESULTS_DIR")
    or (Path(os.environ.get("TT_METAL_HOME", ".")).resolve() / "paper_results")
)

DEFAULT_N_STOCKHAM_FP32 = [
    32, 64, 128, 256, 512, 1024,
    4096, 16384, 65536, 262144, 1048576,
]
DEFAULT_N_UNIVERSAL_FP32 = [
    5, 7, 11, 17, 25, 49, 96, 100, 360, 729, 1000,
    4095, 8191, 65521, 524287,
]
DEFAULT_N_UNIVERSAL_BF16 = DEFAULT_N_UNIVERSAL_FP32 + [32768, 131072]
DEFAULT_N_XL = [2097152, 4194304, 8388608, 16777216]

DEFAULT_N_BY_BACKEND = {
    "stockham":      DEFAULT_N_STOCKHAM_FP32,
    "universal":     DEFAULT_N_UNIVERSAL_FP32,
    "universal_bf16": DEFAULT_N_UNIVERSAL_BF16,
    "universal_xl":  DEFAULT_N_XL,
}

# A compact "sanity" N list used by the smoke runs at the top of run_all.sh
SMOKE_N = [32, 1024, 4096, 16384]


# ───────────────────────── dtype / precision mapping ──────────────────────

DTYPE_MAP = {
    "fp32": (ttnn.float32, torch.float32),
    "bf16": (ttnn.bfloat16, torch.bfloat16),
}


def parse_dtype_list(arg: str) -> list[str]:
    if arg == "both":
        return ["fp32", "bf16"]
    return [s.strip() for s in arg.split(",") if s.strip()]


def parse_precision_list(arg: str, dtype: str) -> list[str]:
    """bf16 only has the FPU 'fast' path; clip it accordingly."""
    if dtype == "bf16":
        return ["fast"]
    if arg == "both":
        return ["precise", "fast"]
    return [s.strip() for s in arg.split(",") if s.strip()]


def parse_int_list(arg: str) -> list[int]:
    return [int(s) for s in arg.split(",") if s.strip()]


# ───────────────────────── tensor plumbing ─────────────────────────────────

def make_input(N: int, batch: int, dtype: str, device, *,
               seed: int | None = None) -> tuple["ttnn.Tensor", torch.Tensor]:
    """Return (tt_input, torch_input_fp32) drawn from a fixed seed.

    The torch fp32 view is what we feed to torch.fft.fft for the
    accuracy reference, regardless of the device dtype.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    torch_in = torch.randn(batch, N, generator=g, dtype=torch.float32) \
        if batch > 1 else torch.randn(N, generator=g, dtype=torch.float32)

    ttnn_dtype, torch_dev_dtype = DTYPE_MAP[dtype]
    feed = torch_in.to(torch_dev_dtype)

    tt_in = ttnn.from_torch(
        feed,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return tt_in, torch_in


def call_fft(tt_in, *, inverse: bool = False, precision: str = "precise",
             tt_in_imag=None):
    """Single-spot wrapper so we can swap the kw-name if the binding changes."""
    if inverse:
        return ttnn.experimental.ifft(
            tt_in, tt_in_imag, precision=precision)
    return ttnn.experimental.fft(tt_in, precision=precision)


# ───────────────────────── timing primitives ───────────────────────────────

def synchronize(device) -> None:
    """Best-effort cross-API sync; works on both Wormhole and Blackhole."""
    if hasattr(ttnn, "synchronize_device"):
        ttnn.synchronize_device(device)
    else:                            # pragma: no cover  (older bindings)
        ttnn.synchronize_devices([device])


def time_call_us(fn, device, *, warmup: int, iters: int) -> dict[str, float]:
    """Run `fn()` warmup+iters times and return latency statistics in us.

    Returns dict with keys: first_call_us, median_us, p05_us, p95_us, mean_us.
    """
    samples: list[float] = []

    # Warmup runs absorb plan / JIT / program-cache miss
    first = None
    for i in range(max(1, warmup + 1)):
        t0 = time.perf_counter_ns()
        fn()
        synchronize(device)
        t1 = time.perf_counter_ns()
        if first is None:
            first = (t1 - t0) / 1e3

    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        synchronize(device)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e3)

    samples.sort()
    return {
        "first_call_us": float(first or float("nan")),
        "median_us": statistics.median(samples),
        "p05_us":    samples[max(0, int(0.05 * len(samples)))],
        "p95_us":    samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "mean_us":   statistics.fmean(samples),
    }


def gflops(N: int, batch: int, seconds: float) -> float:
    """Conventional radix-2 FFT cost model: 5 N log2 N flops per length-N FFT."""
    if seconds <= 0.0 or N < 2:
        return 0.0
    return 5.0 * batch * N * math.log2(N) / seconds / 1e9


# ───────────────────────── accuracy ────────────────────────────────────────

def rel_err_complex(got_complex: torch.Tensor,
                    ref_complex: torch.Tensor) -> float:
    return (
        torch.linalg.norm(got_complex - ref_complex)
        / torch.linalg.norm(ref_complex).clamp_min(1e-12)
    ).item()


def torch_ref_fft(torch_in_fp32: torch.Tensor, *, inverse: bool = False
                  ) -> torch.Tensor:
    """fp64 reference via torch, cast back to fp32 complex for fair comparison."""
    x = torch_in_fp32.to(torch.float64)
    cplx = torch.complex(x, torch.zeros_like(x))
    if inverse:
        return torch.fft.ifft(cplx, dim=-1).to(torch.complex64)
    return torch.fft.fft(cplx, dim=-1).to(torch.complex64)


def tt_output_as_complex(re_tt, im_tt) -> torch.Tensor:
    re = ttnn.to_torch(re_tt).reshape(re_tt.shape).to(torch.float32)
    im = ttnn.to_torch(im_tt).reshape(im_tt.shape).to(torch.float32)
    return torch.complex(re, im)


# ───────────────────────── CSV writer ──────────────────────────────────────

class CsvWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fieldnames = list(fieldnames)
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        self._w.writeheader()

    def write(self, row: dict) -> None:
        # Fill any missing columns with empty string so partial rows are OK.
        full = {k: row.get(k, "") for k in self.fieldnames}
        self._w.writerow(full)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ───────────────────────── argparse boilerplate ────────────────────────────

def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dtype",     default="both",
                   help="fp32 | bf16 | both | csv list")
    p.add_argument("--precision", default="both",
                   help="precise | fast | both | csv list (fp32 only)")
    p.add_argument("--N", default="",
                   help="csv N list. Empty = use the per-backend default lists.")
    p.add_argument("--batch", default="1",
                   help="csv list of batch sizes")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=50)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    default="",
                   help="Output CSV path. Empty = inferred under paper_results/csv/.")
    return p


def default_N_list_for(dtype: str) -> list[int]:
    if dtype == "bf16":
        return sorted(set(DEFAULT_N_BY_BACKEND["universal_bf16"]
                          + DEFAULT_N_STOCKHAM_FP32))
    return sorted(set(
        DEFAULT_N_STOCKHAM_FP32
        + DEFAULT_N_UNIVERSAL_FP32
        + DEFAULT_N_XL
    ))


def resolve_N_list(arg: str, dtype: str) -> list[int]:
    return parse_int_list(arg) if arg else default_N_list_for(dtype)


# ───────────────────────── device context ──────────────────────────────────

@contextlib.contextmanager
def open_device():
    """Open device 0 and tear it down cleanly."""
    device = ttnn.open_device(device_id=0)
    try:
        ttnn.enable_program_cache(device)
    except AttributeError:
        # On newer ttnn builds program cache is on by default.
        pass
    try:
        yield device
    finally:
        with contextlib.suppress(Exception):
            ttnn.close_device(device)


def default_out_path(stem: str, dtype: str | None = None) -> Path:
    parts = [stem]
    if dtype:
        parts.append(dtype)
    name = "_".join(parts) + ".csv"
    return DEFAULT_RESULTS_DIR / "csv" / name


# ───────────────────────── pretty-print row ────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cleanup() -> None:
    """Best-effort GC of CPU caches between sweep points."""
    import gc
    gc.collect()
