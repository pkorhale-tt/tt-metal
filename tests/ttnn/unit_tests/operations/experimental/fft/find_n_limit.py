#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
find_n_limit.py — Determines the maximum supported FFT length on this WH device.

Measures:
  1. Maximum pow-2 N (fp32 and bf16) via three-pass auto-route
  2. Maximum non-pow-2 N (Bluestein XL) via bluestein_dispatch
  3. Peak verified DRAM usage at the limit

Run from the tt-metal root:
    python tests/ttnn/unit_tests/operations/experimental/fft/find_n_limit.py

Prints a table suitable for copy-paste into a paper.
"""

import math
import sys
import time
import traceback

import torch
import ttnn

# ── Device init ──────────────────────────────────────────────────────────────

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return (torch.abs(got - ref).max() / (torch.abs(ref).max() + 1e-9)).item()

def _run_fft(N: int, tt_dtype, torch_dtype) -> float:
    """Returns relative error vs numpy DFT, or float('inf') on OOM/error."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    try:
        tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        re, im = ttnn.experimental.fft(tt_x)
        got_r = ttnn.to_torch(re).reshape(1, N).to(torch.float32)
        got_i = ttnn.to_torch(im).reshape(1, N).to(torch.float32)
        return _rel_err(torch.complex(got_r, got_i), ref)
    except Exception:
        return float('inf')

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _bluestein_M(N: int) -> int:
    return _next_pow2(2 * N - 1)

# ── Section 1: pow-2 limit ────────────────────────────────────────────────────

print("\n" + "="*70)
print("  POW-2 N LIMIT  (fp32 and bf16)")
print("="*70)
print(f"{'N':>15}  {'log2N':>6}  {'DRAM input':>12}  {'fp32 err':>10}  {'bf16 err':>10}  {'status'}")
print("-"*70)

pow2_last_pass = {ttnn.float32: None, ttnn.bfloat16: None}
tols = {ttnn.float32: 5e-4, ttnn.bfloat16: 5e-2}
labels = {ttnn.float32: "fp32", ttnn.bfloat16: "bf16"}
torch_dtypes = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

# Candidate pow-2 values (from 2^10 up to 2^30)
pow2_candidates = [1 << k for k in range(10, 31)]

for N in pow2_candidates:
    log2N = int(math.log2(N))
    input_mb = N * 4 / (1 << 20)
    results = {}
    for tt_dtype in [ttnn.float32, ttnn.bfloat16]:
        t0 = time.time()
        err = _run_fft(N, tt_dtype, torch_dtypes[tt_dtype])
        dt = time.time() - t0
        results[tt_dtype] = (err, dt)

    fp32_err, fp32_t = results[ttnn.float32]
    bf16_err, bf16_t = results[ttnn.bfloat16]

    fp32_ok = math.isfinite(fp32_err) and fp32_err < tols[ttnn.float32]
    bf16_ok = math.isfinite(bf16_err) and bf16_err < tols[ttnn.bfloat16]

    if fp32_ok:
        pow2_last_pass[ttnn.float32] = N
    if bf16_ok:
        pow2_last_pass[ttnn.bfloat16] = N

    fp32_str = f"{fp32_err:.2e}" if math.isfinite(fp32_err) else "  OOM/ERR"
    bf16_str = f"{bf16_err:.2e}" if math.isfinite(bf16_err) else "  OOM/ERR"
    status = "PASS" if (fp32_ok and bf16_ok) else ("fp32-only" if fp32_ok else "FAIL/OOM")

    print(f"{N:>15,}  {log2N:>6}  {input_mb:>10.1f}M  {fp32_str:>10}  {bf16_str:>10}  {status}")
    sys.stdout.flush()

    # Stop after both fail (OOM reached)
    if not fp32_ok and not bf16_ok:
        print("  → OOM boundary reached, stopping.")
        break

# ── Section 2: non-pow-2 Bluestein limit ─────────────────────────────────────

print("\n" + "="*70)
print("  NON-POW-2 N LIMIT  (Bluestein, fp32 only)")
print("="*70)
print(f"{'N':>15}  {'M=nxp2(2N-1)':>14}  {'log2M':>6}  {'DRAM input':>12}  {'fp32 err':>10}  {'status'}")
print("-"*70)

# Candidate non-pow-2 values: primes / composites near each doubling boundary
bluestein_candidates = [
    # M in range (2^17, 2^18)  → two-pass inner
    100_003,
    # M in range (2^18, 2^19)
    200_003,
    # M up to 2^20 inner FFT boundary
    500_009,
    # XL Bluestein: M just above 2^20
    524_289,
    600_000,
    1_000_003,
    # M in (2^21, 2^22)
    2_000_003,
    # M in (2^22, 2^23)
    4_000_037,
    # M in (2^23, 2^24)
    8_000_011,
    # M in (2^24, 2^25)
    16_000_057,
    # M in (2^25, 2^26)
    32_000_011,
    # M in (2^26, 2^27)
    64_000_037,
]

bluestein_last_pass = None
TOL_BLUESTEIN = 1e-3  # fp32, multi-stage accumulation

for N in bluestein_candidates:
    M = _bluestein_M(N)
    log2M = math.log2(M)
    input_mb = N * 4 / (1 << 20)
    err = _run_fft(N, ttnn.float32, torch.float32)
    ok = math.isfinite(err) and err < TOL_BLUESTEIN
    if ok:
        bluestein_last_pass = N
    err_str = f"{err:.2e}" if math.isfinite(err) else "  OOM/ERR"
    status = "PASS" if ok else "FAIL/OOM"
    print(f"{N:>15,}  {M:>14,}  {log2M:>6.1f}  {input_mb:>10.1f}M  {err_str:>10}  {status}")
    sys.stdout.flush()
    if not ok:
        print("  → limit reached, stopping.")
        break

# ── Summary for paper ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  SUMMARY — Verified N limits on this WH device")
print("="*70)

if pow2_last_pass[ttnn.float32]:
    N = pow2_last_pass[ttnn.float32]
    dram = N * 4 / (1 << 20)
    print(f"  Pow-2  fp32 :  N = {N:,}  (2^{int(math.log2(N))})   input = {dram:.0f} MB")
if pow2_last_pass[ttnn.bfloat16]:
    N = pow2_last_pass[ttnn.bfloat16]
    dram = N * 2 / (1 << 20)
    print(f"  Pow-2  bf16 :  N = {N:,}  (2^{int(math.log2(N))})   input = {dram:.0f} MB")
if bluestein_last_pass:
    N = bluestein_last_pass
    M = _bluestein_M(N)
    dram_in = N * 4 / (1 << 20)
    dram_m  = M * 4 / (1 << 20)
    print(f"  Non-pow-2 fp32 :  N = {N:,}   M = {M:,}  (2^{math.log2(M):.1f})   input = {dram_in:.0f} MB  padded = {dram_m:.0f} MB")

print("="*70)

ttnn.close_device(device)
