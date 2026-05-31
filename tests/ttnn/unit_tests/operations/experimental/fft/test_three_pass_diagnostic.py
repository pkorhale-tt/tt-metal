# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic for the corrected fft_three_pass Cooley–Tukey composite.

Mirrors test_two_pass_diagnostic.py: PRINTS the actual vs reference DFT
for the smallest 3-pass-eligible N so we can directly verify correctness
(or see which permutation hypothesis matches if it's still wrong).

Run with:
  TT_FFT_NATIVE=1 python tests/ttnn/unit_tests/operations/experimental/fft/test_three_pass_diagnostic.py
"""

import os
import sys

import torch
import ttnn


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().norm() / ref.abs().norm().clamp_min(1e-30))


def _expected_three_factorization(N: int) -> tuple[int, int, int]:
    log2N = N.bit_length() - 1
    log2_N3 = 10 if (log2N - 10) >= 10 else max(5, log2N - 10)
    log2_rest = log2N - log2_N3
    log2_N1 = (log2_rest + 1) // 2
    log2_N2 = log2_rest - log2_N1
    return (1 << log2_N1, 1 << log2_N2, 1 << log2_N3)


def _run_one(device, N: int, B: int = 1) -> None:
    N1, N2, N3 = _expected_three_factorization(N)
    print(f"\n========== N={N}, B={B}, (N1,N2,N3)=({N1},{N2},{N3}) ==========")
    torch.manual_seed(7)
    x = torch.randn(B, N, dtype=torch.float32)

    x_preshaped = x.reshape(B * N1 * N2, N3).contiguous()
    tt_x = ttnn.from_torch(
        x_preshaped, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re, im = ttnn.experimental.fft_three_pass(tt_x, full_N=N)

    print(f"  device output real shape = {tuple(re.padded_shape)}")
    got_r = ttnn.to_torch(re).reshape(B, N).to(torch.float32)
    got_i = ttnn.to_torch(im).reshape(B, N).to(torch.float32)
    got = torch.complex(got_r, got_i)

    ref = torch.fft.fft(x.to(torch.complex64), dim=-1)

    rel = _rel_err(got[0], ref[0])
    mx  = float((got[0] - ref[0]).abs().max() / ref[0].abs().max().clamp_min(1e-30))
    print(f"  aggregate rel err   = {rel:.4e}")
    print(f"  max per-bin rel err = {mx:.4e}")

    print("\n    k        got                                ref                                |diff|")
    for k in list(range(0, 4)) + list(range(N - 4, N)):
        g = complex(got[0, k].item())
        r = complex(ref[0, k].item())
        d = abs(g - r)
        print(f"    {k:8d}  ({g.real:+.4e},{g.imag:+.4e})  ({r.real:+.4e},{r.imag:+.4e})  {d:.4e}")

    # Permutation diagnostics: if the algorithm is wrong, what packing would match?
    print()
    perms = [
        ("k1·N2·N3 + k2·N3 + k3 (OLD claimed packing)",
         [(k // (N2 * N3)) + N1 * ((k // N3) % N2) + N1 * N2 * (k % N3) for k in range(N)]),
        ("k3·N1·N2 + k2·N1 + k1 (NEW claimed packing, identity if correct)",
         list(range(N))),
    ]
    for label, perm_list in perms:
        perm = torch.tensor(perm_list, dtype=torch.long)
        rel_perm = _rel_err(got[0], ref[0, perm])
        marker = "  ← MATCH" if rel_perm < 1e-3 else ""
        print(f"  rel err under permutation {label}: {rel_perm:.4e}{marker}")


def main():
    if os.environ.get("TT_FFT_NATIVE", "0") != "1":
        print("ERROR: set TT_FFT_NATIVE=1 to enable the native 3-pass path.")
        sys.exit(1)

    device = ttnn.open_device(device_id=0)
    try:
        # Smallest 3-pass-eligible N: 2^21 (N1=64, N2=32, N3=1024).
        # Also test 2^22 (N1=N2=64, N3=1024) — symmetric N1,N2.
        for N in (1 << 21, 1 << 22):
            _run_one(device, N=N, B=1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
