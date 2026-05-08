# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Phase-1 tests for ttnn.experimental.fft.
#
# Phase 1 ("Path A — host pass-through"): the program factory runs an
# iterative radix-2 Cooley–Tukey FFT on the host CPU and writes the
# spectrum back to two device tensors. This means the dispatch wiring
# AND the math are both verifiable end-to-end here. Phase 2 will swap
# the host kernel for the on-device Stockham program with no change to
# this test file.

import pytest
import torch
import ttnn


# ── Shape / dtype plumbing ──────────────────────────────────────────────────
@pytest.mark.parametrize("N", [1024, 4096])
def test_fft_returns_correct_shape_and_dtype(device, N):
    torch_in = torch.randn(N, dtype=torch.float32)

    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    real, imag = ttnn.fft(tt_in)

    assert real.shape == tt_in.shape, "real spectrum shape must match input"
    assert imag.shape == tt_in.shape, "imag spectrum shape must match input"
    assert real.dtype == ttnn.float32
    assert imag.dtype == ttnn.float32


# ── Math correctness (forward FFT) ──────────────────────────────────────────
# Tolerance scales as O(sqrt(log2 N)) for fp32 radix-2 with std::cos/std::sin
# twiddles — so N=2 is essentially exact, N=64K loses ~16 stages of precision.
# We pick a per-N tolerance that's tight enough to catch wiring bugs but
# loose enough to avoid false fails on the fp32 noise floor.
@pytest.mark.parametrize(
    "N, tol",
    [
        (2,     1e-6),
        (8,     1e-5),
        (64,    5e-5),
        (1024,  1e-4),
        (4096,  1e-4),
        (65536, 5e-4),
    ],
)
def test_fft_matches_torch(device, N, tol):
    """
    Compares ttnn.fft against torch.fft.fft on a 1D Float32 input.
    """
    torch_in = torch.randn(N, dtype=torch.float32)
    torch_X = torch.fft.fft(torch_in)

    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_re, tt_im = ttnn.fft(tt_in)

    got_re = ttnn.to_torch(tt_re).reshape(-1)
    got_im = ttnn.to_torch(tt_im).reshape(-1)

    rel = torch.linalg.norm(
        torch.complex(got_re, got_im) - torch_X
    ) / torch.linalg.norm(torch_X).clamp_min(1e-12)
    assert rel < tol, f"ttnn.fft N={N} rel err {rel.item():.2e} exceeds {tol:.0e}"


# ── Roundtrip (FFT → IFFT) ─────────────────────────────────────────────────
@pytest.mark.parametrize("N", [8, 1024, 4096])
def test_fft_ifft_roundtrip(device, N):
    """
    IFFT(FFT(x)) should reproduce x. Verifies (a) the inverse path is
    wired and (b) the 1/N scale is applied exactly once.
    """
    torch_in = torch.randn(N, dtype=torch.float32)

    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    spec_re, spec_im = ttnn.fft(tt_in)
    rec_re, rec_im = ttnn.ifft(spec_re, spec_im)

    got = ttnn.to_torch(rec_re).reshape(-1)
    err_imag = ttnn.to_torch(rec_im).reshape(-1).abs().max().item()

    rel = torch.linalg.norm(got - torch_in) / torch.linalg.norm(torch_in)
    assert rel < 1e-5, f"roundtrip rel err {rel.item():.2e} too high"
    assert err_imag < 1e-4, f"reconstructed imag part should be ~0 (got {err_imag:.2e})"


# ── Out-of-support guard ────────────────────────────────────────────────────
def test_fft_rejects_non_pow2_phase1(device):
    """
    Phase 1 only wires Float32 + pow2 N <= 1M (Stockham backend). Other
    combos are validated and rejected with a clear error.
    """
    torch_in = torch.randn(1000, dtype=torch.float32)  # 1000 is not pow2
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    with pytest.raises(RuntimeError):
        ttnn.fft(tt_in)
