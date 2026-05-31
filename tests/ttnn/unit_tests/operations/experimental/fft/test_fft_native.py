# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the new ProgramDescriptor-based SingleTileStockhamFactory path
(commit 1 of the host-to-device refactor).

Activated by env var TT_FFT_NATIVE=1.

Scope (commit 1):
  - fp32, real input, forward FFT, pow-2 N in [2, 1024]
  - correctness vs torch.fft
  - program-cache-hit verification (no re-compile on repeat call)
  - Metal-Trace replay verification (all work device-side, replayable)

Out of scope for commit 1 (future commits will add):
  - bf16 (commit 2)
  - N > 1024 (commits 3-5)
  - non-pow-2 N (commit 5)
  - IFFT (commit 6)
"""

import os
import pytest
import torch
import ttnn


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_FFT_NATIVE", "0") != "1",
    reason="TT_FFT_NATIVE=1 not set; new ProgramDescriptor path is gated.",
)


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _run_fft(device, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    re, im = ttnn.experimental.fft(tt_x)
    return ttnn.to_torch(re).reshape(-1).to(torch.float32), ttnn.to_torch(im).reshape(-1).to(torch.float32)


# ─── 1. Correctness ────────────────────────────────────────────────────────
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_singletile_correctness_fp32(device, N):
    torch.manual_seed(42)
    x = torch.randn(N, dtype=torch.float32)

    got_r, got_i = _run_fft(device, x)
    got = torch.complex(got_r, got_i)
    ref = torch.fft.fft(x.to(torch.complex64))

    rel = _rel_err(got, ref)
    assert rel < 1e-4, f"N={N} rel err {rel:.2e}"


# ─── 2. Program cache hit ──────────────────────────────────────────────────
def test_singletile_program_cache_hit(device):
    """First call may compile (cache miss); second call MUST be a cache hit
    (no new program entry). This is what the reviewer's review specifically
    requires per adding_new_ttnn_operation.html."""
    N = 1024
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32)

    # Warmup: first call compiles
    _run_fft(device, x)
    n_after_warmup = device.num_program_cache_entries()

    # Second call: must hit cache, no new entry
    _run_fft(device, x)
    n_after_repeat = device.num_program_cache_entries()

    assert n_after_repeat == n_after_warmup, (
        f"Program cache regression: {n_after_warmup} entries after warmup, "
        f"{n_after_repeat} after repeat. New ProgramDescriptor path is not "
        f"cacheable."
    )


# ─── 3. Metal Trace replay ─────────────────────────────────────────────────
def test_singletile_metal_trace_replay(device):
    """Capture a trace around an FFT call, replay 10x, every replay must
    produce a result that matches the original eager call within fp tolerance.
    If any host work leaks into the op path, the trace can't reproduce it
    and this test fails."""
    N = 1024
    torch.manual_seed(1)
    x = torch.randn(N, dtype=torch.float32)

    # Warmup: program cache must be populated before trace capture.
    eager_r, eager_i = _run_fft(device, x)

    # Capture the trace.
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    re_t, im_t = ttnn.experimental.fft(tt_x)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    try:
        for i in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            replay_r = ttnn.to_torch(re_t).reshape(-1).to(torch.float32)
            replay_i = ttnn.to_torch(im_t).reshape(-1).to(torch.float32)

            assert torch.allclose(replay_r, eager_r, rtol=1e-4, atol=1e-4), (
                f"trace replay {i} real mismatch: max abs diff "
                f"{(replay_r - eager_r).abs().max().item():.2e}"
            )
            assert torch.allclose(replay_i, eager_i, rtol=1e-4, atol=1e-4), (
                f"trace replay {i} imag mismatch: max abs diff "
                f"{(replay_i - eager_i).abs().max().item():.2e}"
            )
    finally:
        ttnn.release_trace(device, tid)
