# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic for the fft_two_pass Cooley–Tukey composite.

Static math analysis suggests the algorithm has a missing initial-transpose
bug and the twiddle is applied to the wrong pass — symptom would be the
output being uncorrelated with torch.fft.fft (rel err ≈ √2).

This script bypasses pytest's all-or-nothing assert and PRINTS the actual
got / expected values + per-bin error for the smallest 2-pass-eligible N
so we can directly verify the bug (or refute it).

Run with: TT_FFT_NATIVE=1 python tests/ttnn/unit_tests/operations/experimental/fft/test_two_pass_diagnostic.py
"""

import os
import sys

import torch
import ttnn


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _max_per_bin(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _run_one(device, N: int, B: int = 1) -> None:
    print(f"\n========== N={N}, B={B} ==========")
    torch.manual_seed(7)
    x = torch.randn(B, N, dtype=torch.float32)

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re, im = ttnn.experimental.fft(tt_x)
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)

    rel = _rel_err(got[0], ref[0])
    mx  = _max_per_bin(got[0], ref[0])
    print(f"aggregate rel err = {rel:.4e}")
    print(f"max per-bin rel err = {mx:.4e}")

    # Print first 4 and last 4 bins side-by-side.
    print("\n  k        got                                ref                                |diff|")
    for k in list(range(0, 4)) + list(range(N - 4, N)):
        g = complex(got[0, k].item())
        r = complex(ref[0, k].item())
        d = abs(g - r)
        print(f"  {k:6d}  ({g.real:+.4e},{g.imag:+.4e})  ({r.real:+.4e},{r.imag:+.4e})  {d:.4e}")

    # Two diagnostic hypotheses:
    #   (a) Output is in a permuted order (e.g. K = k2*N1+k1 instead of k1*N2+k2).
    #       Test by checking |got[k] - ref[swap_factor_positions(k)]|.
    #   (b) Output is scaled by some constant.
    #       Test via least-squares fit.
    log2N = N.bit_length() - 1
    log2N2 = log2N // 2
    log2N1 = log2N - log2N2
    N1, N2 = 1 << log2N1, 1 << log2N2
    print(f"\n  factorization N1={N1}, N2={N2}")

    perm = torch.tensor([(k % N1) * N2 + (k // N1) for k in range(N)], dtype=torch.long)
    rel_perm = _rel_err(got[0], ref[0, perm])
    print(f"  rel err if output is permuted as K=k2*N1+k1 (swap factor positions): {rel_perm:.4e}")

    perm2 = torch.tensor([(k % N2) * N1 + (k // N2) for k in range(N)], dtype=torch.long)
    rel_perm2 = _rel_err(got[0], ref[0, perm2])
    print(f"  rel err if output is permuted as K=k1*N2+k2 (bit-reverse on (N1, N2)): {rel_perm2:.4e}")

    # Scaling
    g_flat = got[0]
    r_flat = ref[0]
    # least-squares scale (complex):
    if r_flat.abs().sum() > 0:
        scale = (g_flat * r_flat.conj()).sum() / (r_flat.abs().pow(2).sum() + 1e-30)
        rel_scaled = _rel_err(g_flat, scale * r_flat)
        print(f"  best-fit scale = {complex(scale.item()):.4e}; residual rel err: {rel_scaled:.4e}")


def main():
    if os.environ.get("TT_FFT_NATIVE", "0") != "1":
        print("ERROR: set TT_FFT_NATIVE=1 to enable the native 2-pass path.")
        sys.exit(1)

    device = ttnn.open_device(device_id=0)
    try:
        # Smallest 2-pass-eligible N: 2048 (N1=64, N2=32, asymmetric).
        # Square factorization: 4096 (N1=N2=64).
        # If both show rel err ≈ √2, the algorithm is broken in all cases.
        for N in (2048, 4096):
            _run_one(device, N=N, B=1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
