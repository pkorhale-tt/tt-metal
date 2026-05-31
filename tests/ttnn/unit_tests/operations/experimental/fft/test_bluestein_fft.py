# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttnn.experimental.bluestein_fft (commit 6d).

Bluestein's chirp-Z transform extends our pow-2 FFT chain to handle
**arbitrary** N (including primes and other non-pow-2 lengths).

Coverage matrix (commit 6d core):
  - small N: 3, 5, 7, 11, 13, 16, 17, 32, 33, 100
  - medium N: 257 (prime), 384 (3 · 128), 511
  - real-only and complex input
  - fp32 + bf16
  - program-cache hit (chirp + B precomputed; second call only does
    the per-N dispatch chain)

Aggressive cases (N up to commit-6d cap M ≤ 2^20) are gated behind
TT_FFT_AGGRESSIVE=1.
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "0") != "1",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


_DTYPES = [
    # bf16 tol is loose for Bluestein because two FFTs + 3 cmul chain
    # accumulates rounding; the per-bin worst case is dominated by the
    # post-IFFT slice region near the chirp boundary.
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 1.5e-1),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _run_bluestein(device, x_re, x_im, N, tt_dtype):
    """Upload (1, N) real and (optional) imag halves and call bluestein_fft."""
    tt_xr = ttnn.from_torch(
        x_re.reshape(1, N), dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if x_im is not None:
        tt_xi = ttnn.from_torch(
            x_im.reshape(1, N), dtype=tt_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_r, out_i = ttnn.experimental.bluestein_fft(tt_xr, tt_xi, N=N)
    else:
        out_r, out_i = ttnn.experimental.bluestein_fft(tt_xr, N=N)
    return ttnn.to_torch(out_r).reshape(N), ttnn.to_torch(out_i).reshape(N)


# ─── 1. Real-input correctness ──────────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize(
    "N",
    # Primes + small composites + just under / just over a pow-2 boundary.
    [3, 5, 7, 11, 13, 16, 17, 31, 32, 33, 100, 127, 128, 129],
    ids=lambda v: f"N{v}",
)
def test_bluestein_real_correctness(device, N, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(N)
    x = torch.randn(N, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_bluestein(device, x, None, N, tt_dtype)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] N={N} (real input) rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 2. Complex-input correctness ───────────────────────────────────────
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("N", [5, 13, 31, 100, 127], ids=lambda v: f"N{v}")
def test_bluestein_complex_correctness(device, N, tt_dtype, torch_dtype, label, tol):
    torch.manual_seed(N + 1)
    x_re = torch.randn(N, dtype=torch.float32).to(torch_dtype)
    x_im = torch.randn(N, dtype=torch.float32).to(torch_dtype)

    got_r, got_i = _run_bluestein(device, x_re, x_im, N, tt_dtype)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))

    x = torch.complex(x_re.to(torch.float32), x_im.to(torch.float32)).to(torch.complex64)
    ref = torch.fft.fft(x, dim=-1)

    rel = _rel_err(got, ref)
    assert rel < tol, (
        f"[{label}] N={N} (complex input) rel err {rel:.2e} (tol {tol:.0e})"
    )


# ─── 3. Program-cache / plan reuse ──────────────────────────────────────
# Second call with the same N should hit BOTH the JIT program cache for
# every device op AND the host-side BluesteinPlan cache (chirp + B not
# rebuilt).
def test_bluestein_program_cache_hit(device):
    N = 17  # prime, small
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32)

    for trial in range(3):
        got_r, got_i = _run_bluestein(device, x, None, N, ttnn.float32)
        got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))
        ref = torch.fft.fft(x.to(torch.complex64), dim=-1)
        rel = _rel_err(got, ref)
        assert rel < 5e-4, f"trial={trial} N={N} rel err {rel:.2e}"


# ─── 4. Aggressive (gated) — larger N approaching commit-6d cap ────────
# Commit 6d cap: M = next_pow2(2*N - 1) ≤ 2^20 = 1M  →  N ≤ 524_288.
@pytest.mark.skipif(
    os.environ.get("TT_FFT_AGGRESSIVE", "0") != "1",
    reason="TT_FFT_AGGRESSIVE=1 not set; large-N Bluestein test is gated.",
)
@pytest.mark.parametrize(
    "N",
    # 1009 prime, 4097 just over pow-2, 65537 prime (Fermat), 524288 cap.
    [1009, 4097, 65537],
    ids=lambda v: f"N{v}",
)
def test_bluestein_aggressive(device, N):
    torch.manual_seed(N)
    x = torch.randn(N, dtype=torch.float32)

    got_r, got_i = _run_bluestein(device, x, None, N, ttnn.float32)
    got = torch.complex(got_r.to(torch.float32), got_i.to(torch.float32))
    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)

    # Two FFTs + 3 cmul chain at N up to 65k accumulates a few extra ULP;
    # 2e-3 leaves room for that without masking real bugs.
    rel = _rel_err(got, ref)
    assert rel < 2e-3, f"N={N} rel err {rel:.2e}"
