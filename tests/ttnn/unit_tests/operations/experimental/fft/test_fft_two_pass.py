# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the two-pass Cooley–Tukey composite FFT path (commit 3c).

For pow-2 N with 1024 < N ≤ 1M, ttnn.experimental.fft factors N = N1*N2
(balanced, both pow-2 in [32, 1024]) and runs a 6-op device-side chain:

    Pass-1 batched FFT  →  apply_twiddles  →  transpose_rm  →
    Pass-2 batched complex FFT  →  transpose_rm

Activated by TT_FFT_NATIVE=1.

Coverage:
  - correctness vs torch.fft, fp32 and bf16, various N and batch dims
  - program-cache hit on repeat (six entries cache, then stay flat)
  - Metal-Trace replay on a single (B, N) shape (all work device-side)
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "0") != "1",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


# (ttnn dtype, torch dtype, dtype label, rel-err tolerance)
# Two-pass uses fp32 internal compute; bf16 only at DRAM I/O boundary
# so the tolerance stays tight at ~5e-2.
_DTYPES = [
    (ttnn.float32,  torch.float32,  "fp32", 5e-4),
    (ttnn.bfloat16, torch.bfloat16, "bf16", 5e-2),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _expected_factorization(N: int) -> tuple[int, int]:
    """Mirror C++ pick_factorization for sanity."""
    log2N = N.bit_length() - 1
    log2N2 = log2N // 2
    log2N1 = log2N - log2N2
    return (1 << log2N1, 1 << log2N2)


# ─── 1. Correctness — flat (B=1) and small batched ─────────────────────────
# Each (B, N) here exercises a different factorization and at least one
# multi-pass dispatch chain.  N covers the boundary just above the
# single-tile cutoff (2048 → N1=64,N2=32) up through 8192 (128,64).
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B", [1, 2, 4])
@pytest.mark.parametrize("N", [2048, 4096, 8192])
def test_two_pass_correctness(device, B, N, tt_dtype, torch_dtype, label, tol):
    N1, N2 = _expected_factorization(N)
    assert N1 * N2 == N
    assert N1 >= 32 and N2 >= 32  # transpose_rm constraint

    torch.manual_seed(7)
    x_fp32 = torch.randn(B, N, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(x.to(torch.float32).to(torch.complex64), dim=-1)

    # Per-row rel-err — every row must individually satisfy the bound.
    for b in range(B):
        rel = _rel_err(got[b], ref[b])
        assert rel < tol, (
            f"[{label}] B={B} N={N} (N1={N1},N2={N2}) row={b} "
            f"rel err {rel:.2e} (tol {tol:.0e})"
        )


# ─── 2. Program cache hit ──────────────────────────────────────────────────
# Two-pass dispatches six ops (with two transpose_rm at different shapes,
# so all six are distinct program cache entries).  After warmup the
# entry count must not grow on a repeat call with the same shape/dtype.
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_two_pass_program_cache_hit(device, tt_dtype, torch_dtype, label, tol):
    B, N = 2, 2048
    torch.manual_seed(0)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)

    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn.experimental.fft(tt_x)
    n_after_warmup = device.num_program_cache_entries()

    tt_x2 = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn.experimental.fft(tt_x2)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"[{label}] two-pass program cache regression: "
        f"{n_after_warmup} → {n_after_repeat}"
    )


# ─── 3. Metal Trace replay ─────────────────────────────────────────────────
# NOTE: marked xfail intentionally.  The two-pass composite is a sequence
# of six dispatches + intermediate-tensor allocations + ttnn.reshape calls.
# Metal Trace today does not support intermediate device-tensor allocation
# inside the captured region (the reshape path triggers a synchronous
# allocator/page-size query, which surfaces as
#   "TT_FATAL: Reads are not supported during trace capture."
# ).  This test will start passing automatically once commit 4
# (ttnn::prim::fft_radix_pass standalone device op) folds the composite
# into a single device dispatch with pre-allocated intermediates.
@pytest.mark.xfail(
    reason="two-pass composite uses host-side ttnn.reshape + intermediate "
           "tensor allocations; not trace-replayable until commit 4/5 folds "
           "the chain into a single device op (fft_radix_pass / "
           "fft_universal_xl). The single-tile and batched trace tests "
           "(test_fft_native.py::test_singletile_metal_trace_replay) "
           "already verify the underlying ProgramDescriptor path is "
           "trace-safe.",
    strict=True,
)
@pytest.mark.parametrize("tt_dtype,torch_dtype,label,tol", _DTYPES,
                         ids=[d[2] for d in _DTYPES])
def test_two_pass_metal_trace_replay(device, tt_dtype, torch_dtype, label, tol):
    B, N = 1, 2048
    torch.manual_seed(1)
    x = torch.randn(B, N, dtype=torch.float32).to(torch_dtype)

    tt_x = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re_e, im_e = ttnn.experimental.fft(tt_x)
    eager_r = ttnn.to_torch(re_e).reshape(-1).to(torch.float32).clone()
    eager_i = ttnn.to_torch(im_e).reshape(-1).to(torch.float32).clone()

    tt_x_w = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn.experimental.fft(tt_x_w)

    tt_x_t = ttnn.from_torch(
        x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    re_t, im_t = ttnn.experimental.fft(tt_x_t)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    try:
        for i in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            replay_r = ttnn.to_torch(re_t).reshape(-1).to(torch.float32)
            replay_i = ttnn.to_torch(im_t).reshape(-1).to(torch.float32)
            assert torch.allclose(replay_r, eager_r, rtol=tol, atol=tol)
            assert torch.allclose(replay_i, eager_i, rtol=tol, atol=tol)
    finally:
        ttnn.release_trace(device, tid)
