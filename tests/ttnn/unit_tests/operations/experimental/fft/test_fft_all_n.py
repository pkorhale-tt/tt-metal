# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Unified accuracy sweep — ttnn.experimental.fft / ifft — all N ranges.

This is the single canonical test for the PR.  It exercises EVERY routing
branch of the unified C++ router in fft.cpp through ONE public API call
(`ttnn.experimental.fft`), matching the cuFFT usage pattern.

Coverage:
  ┌──────────────────────────────────┬────────────────────────────────────┐
  │ N range                          │ Internal path                      │
  ├──────────────────────────────────┼────────────────────────────────────┤
  │ pow-2, N ≤ 1024                  │ SingleTile/BatchedStockhamFactory  │
  │ pow-2, 1024 < N ≤ 2^20          │ fft_two_pass composite             │
  │ pow-2, 2^20 < N ≤ 2^30          │ fft_three_pass_auto composite      │
  │ non-pow-2, M ≤ 2^20             │ bluestein_dispatch (two-pass inner)│
  │ non-pow-2, 2^20 < M ≤ 2^30      │ bluestein_dispatch (3-pass inner)  │
  └──────────────────────────────────┴────────────────────────────────────┘

Aggressive (large N) cases are gated behind TT_FFT_AGGRESSIVE=1 to keep
the default CI run fast.  Run:
    TT_FFT_AGGRESSIVE=1 pytest test_fft_all_n.py -v
to verify the full N envelope.

Tolerances:
  fp32 : 5e-4 (two-stage Bluestein can accumulate ≈1e-5 per op)
  bf16 : 5e-2 (two quantisation steps dominate)
  bf16 Bluestein : 1.5e-1 (three cmul + two FFT stages add rounding)
"""

import os
import math
import pytest
import torch
import ttnn

# ─── helpers ────────────────────────────────────────────────────────────────

_AGGRESSIVE = os.environ.get("TT_FFT_AGGRESSIVE", "0") == "1"


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float(
        (got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30)
    )


def _run_fft(device, x_re: torch.Tensor, tt_dtype, *, N: int, B: int = 1):
    """Upload (B, N) tensor, call ttnn.experimental.fft, return complex torch."""
    tt_x = ttnn.from_torch(
        x_re.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    return torch.complex(got_r, got_i)


def _run_ifft(device, x_re: torch.Tensor, x_im: torch.Tensor,
              tt_dtype, *, N: int, B: int = 1):
    """Upload (B, N) complex spectrum, call ttnn.experimental.ifft."""
    tt_r = ttnn.from_torch(
        x_re.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_i = ttnn.from_torch(
        x_im.reshape(B, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.ifft(tt_r, tt_i)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    return torch.complex(got_r, got_i)


# ─── dtype fixtures ──────────────────────────────────────────────────────────

_DTYPES_POW2 = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]
_DTYPES_BLUESTEIN = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 1.5e-1),
]


# ════════════════════════════════════════════════════════════════════════════
# 1.  Stockham — pow-2, N ≤ 1024  (SingleTile / BatchedStockhamFactory)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("N", [2, 4, 32, 64, 256, 512, 1024])
def test_stockham_fft(device, N, B, tt_dtype, torch_dtype, label, tol):
    """Stockham pow-2 N ≤ 1024 via SingleTile/BatchedStockhamFactory."""
    torch.manual_seed(N + B)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=B)
    assert _rel_err(got, ref) < tol, \
        f"Stockham N={N} B={B} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [2, 32, 256, 1024])
def test_stockham_ifft_roundtrip(device, N, tt_dtype, torch_dtype, label, tol):
    """Forward → Inverse roundtrip for Stockham path."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = x.to(torch.float32)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    tt_dtype, N=N)
    assert _rel_err(got.real, ref) < tol * 4, \
        f"Stockham IFFT roundtrip N={N} {label}: rel_err={_rel_err(got.real, ref):.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 2.  Two-pass — pow-2, 1024 < N ≤ 2^20
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("N", [2048, 4096, 8192, 65536,
                                pytest.param(1 << 20, marks=pytest.mark.skipif(
                                    not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set"))])
def test_two_pass_fft(device, N, B, tt_dtype, torch_dtype, label, tol):
    """Two-pass composite pow-2 N in (1024, 1M]."""
    torch.manual_seed(N + B)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=B)
    assert _rel_err(got, ref) < tol, \
        f"TwoPass N={N} B={B} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [2048, 8192])
def test_two_pass_ifft_roundtrip(device, N, tt_dtype, torch_dtype, label, tol):
    """Forward → Inverse roundtrip for two-pass path."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = x.to(torch.float32)
    tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    tt_dtype, N=N)
    assert _rel_err(got.real, ref) < tol * 4, \
        f"TwoPass IFFT roundtrip N={N} {label}: rel_err={_rel_err(got.real, ref):.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 3.  Three-pass auto-route — pow-2, 2^20 < N ≤ 2^30
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set")
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_POW2,
                         ids=[d[2] for d in _DTYPES_POW2])
@pytest.mark.parametrize("N", [1 << 21, 1 << 24, 1 << 27])
def test_three_pass_fft(device, N, tt_dtype, torch_dtype, label, tol):
    """Three-pass auto-routed composite for very large pow-2 N."""
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=1)
    assert _rel_err(got, ref) < tol, \
        f"ThreePass N={N} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 4.  Bluestein — non-pow-2 N, M ≤ 2^20 (two-pass inner FFTs)
# ════════════════════════════════════════════════════════════════════════════

_BLUESTEIN_SMALL = [
    # primes
    3, 5, 7, 11, 13, 17, 31, 97, 127, 257, 509,
    # composites
    6, 12, 100, 200, 384, 500,
    # just above / below a pow-2 boundary
    33, 63, 65, 128, 129, 255,
]

@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_BLUESTEIN,
                         ids=[d[2] for d in _DTYPES_BLUESTEIN])
@pytest.mark.parametrize("N", _BLUESTEIN_SMALL)
def test_bluestein_fft_small(device, N, tt_dtype, torch_dtype, label, tol):
    """Bluestein non-pow-2 N with M ≤ 2^20 via ttnn.experimental.fft."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, tt_dtype, N=N, B=1)
    assert _rel_err(got, ref) < tol, \
        f"Bluestein N={N} {label}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"


@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES_BLUESTEIN,
                         ids=[d[2] for d in _DTYPES_BLUESTEIN])
@pytest.mark.parametrize("N", [7, 97, 383, 997])
def test_bluestein_ifft_roundtrip(device, N, tt_dtype, torch_dtype, label, tol):
    """Bluestein forward → inverse roundtrip via unified API."""
    torch.manual_seed(N)
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = x.to(torch.float32)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re_fft, im_fft = ttnn.experimental.fft(tt_x)
    got = _run_ifft(device,
                    ttnn.to_torch(re_fft).to(torch.float32),
                    ttnn.to_torch(im_fft).to(torch.float32),
                    tt_dtype, N=N)
    assert _rel_err(got.real, ref) < tol * 4, \
        f"Bluestein IFFT roundtrip N={N} {label}: rel_err={_rel_err(got.real, ref):.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 5.  XL Bluestein — non-pow-2, M > 2^20 (three-pass inner FFTs)
# ════════════════════════════════════════════════════════════════════════════

def _bluestein_M(N: int) -> int:
    v = 2 * N - 1
    p = 1
    while p < v:
        p <<= 1
    return p


_BLUESTEIN_XL = [
    # M just above 2^20 = 1M
    524_289,     # M = 2^21
    600_000,     # M ~ 1.2M
    1_000_003,   # prime, M ~ 2M
]

@pytest.mark.skipif(not _AGGRESSIVE, reason="TT_FFT_AGGRESSIVE not set")
@pytest.mark.parametrize("N", _BLUESTEIN_XL)
def test_bluestein_xl_fft(device, N):
    """XL Bluestein: M > 2^20 — inner FFTs use fft_three_pass_auto."""
    M = _bluestein_M(N)
    assert M > (1 << 20), f"Expected M > 1M for XL case, got M={M}"
    torch.manual_seed(N % (1 << 20))
    x = torch.randn(1, N, dtype=torch.float32)
    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(1, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(1, N).to(torch.float32)
    got = torch.complex(got_r, got_i)
    err = _rel_err(got, ref)
    assert err < 1e-3, f"XL Bluestein N={N} M={M}: rel_err={err:.2e} > 1e-3"


# ════════════════════════════════════════════════════════════════════════════
# 6.  Program cache hit — second call should NOT recompile
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N", [256, 4096, 97])
def test_program_cache_hit(device, N):
    """Second call with same (N, dtype) must reuse the cached program."""
    x = torch.randn(1, N, dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    device.enable_program_cache()
    ttnn.experimental.fft(tt_x)
    num_after_first = device.num_program_cache_entries()
    ttnn.experimental.fft(tt_x)
    num_after_second = device.num_program_cache_entries()
    device.disable_and_clear_program_cache()

    assert num_after_second == num_after_first, (
        f"Program cache grew on second fft call for N={N}: "
        f"{num_after_first} → {num_after_second}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 7.  Single-call sanity: the public API entry point dispatches correctly
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N", [
    # One value from each routing bucket
    512,       # Stockham
    4096,      # Two-pass
    7,         # Bluestein (prime)
    100,       # Bluestein (composite)
])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16],
                         ids=["fp32", "bf16"])
def test_unified_api_dispatch(device, N, dtype):
    """Smoke: ttnn.experimental.fft(x) produces correct DFT for every bucket."""
    tol = 5e-4 if dtype == ttnn.float32 else (5e-2 if N & (N - 1) == 0 else 1.5e-1)
    torch.manual_seed(N)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = torch.randn(1, N, dtype=torch.float32).to(torch_dtype)
    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)
    got = _run_fft(device, x, dtype, N=N, B=1)
    assert _rel_err(got, ref) < tol, \
        f"Unified API N={N} dtype={dtype}: rel_err={_rel_err(got, ref):.2e} > tol={tol:.2e}"
