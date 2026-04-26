#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
compare_with_torch.py — run fft_universal on a random complex signal of
length N (ANY N >= 2) and compare against torch.fft.fft.

Usage:
    python compare_with_torch.py                       # default N=1000
    python compare_with_torch.py --N 33                # composite non-pow2
    python compare_with_torch.py --N 97                # prime -> Bluestein
    python compare_with_torch.py --N 65536             # pow2 pass-through
    python compare_with_torch.py --N 1000 --seed 7
    python compare_with_torch.py --N 1000 --bin /path/to/binary

Environment variables:
    FFT_BIN   : path to metal_example_fft_universal_vs_torch (overridden by --bin).
                Default is
                "./build/programming_examples/fft_universal/metal_example_fft_universal_vs_torch".

The script:
  1. Invokes the C++ binary with (N, seed); it writes fft_input.txt and
     fft_output.txt in a tmp dir.
  2. Loads the input signal and runs torch.fft.fft on it (complex128 for
     the reference), compares in complex64 space.
  3. Prints max/mean/rel error, SNR, and the top-5 worst bins.
  4. Returns exit code 0 on pass, 2 on fail.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch


DEFAULT_BIN = "./build/programming_examples/fft_universal/metal_example_fft_universal_vs_torch"


def load_complex_file(path: str) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"{path}: expected N rows of 'real imag', got shape {data.shape}")
    return data[:, 0] + 1j * data[:, 1]


def describe_path(N: int) -> str:
    if N == 1:
        return "identity"
    if N >= 2 and (N & (N - 1)) == 0:
        return "pow2 pass-through (fft_stockham)"
    # crude primality check, fine for the sizes this script is used for
    if N >= 3:
        p = 2
        while p * p <= N:
            if N % p == 0:
                break
            p += 1
        else:
            return "Bluestein (prime)"
    return "Cooley-Tukey split (composite non-pow2)"


def default_tolerance(N: int) -> float:
    """Looser budget when the device path is Bluestein or deep Cooley-Tukey."""
    if N >= 2 and (N & (N - 1)) == 0:
        return 2e-3 if N <= 65536 else 5e-3
    return 1e-2  # prime or composite non-pow2


def run_tt_metal(binary: str, N: int, seed: int, in_path: str, out_path: str) -> None:
    if not os.path.exists(binary):
        sys.exit(
            f"Binary not found: {binary}\n"
            f"Build it with:\n"
            f"    cmake --build build --target metal_example_fft_universal_vs_torch -j\n"
            f"Or override with --bin /path/to/binary")
    cmd = [binary, str(N), str(seed), in_path, out_path]
    env = os.environ.copy()
    env.setdefault("ARCH_NAME", "wormhole_b0")
    print(f"[compare] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"tt-metal binary failed with exit code {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--N", type=int, default=1000,
                    help="FFT length (any integer >= 2)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--bin", dest="binary",
                    default=os.environ.get("FFT_BIN", DEFAULT_BIN),
                    help="path to metal_example_fft_universal_vs_torch")
    ap.add_argument("--tol", type=float, default=None,
                    help="relative-error pass threshold "
                         "(default: 2e-3 pow2 <=65K, 5e-3 pow2 >65K, 1e-2 otherwise)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary input/output text files")
    args = ap.parse_args()

    if args.N < 2:
        sys.exit(f"N must be >= 2, got {args.N}")

    tol = args.tol if args.tol is not None else default_tolerance(args.N)

    tmpdir = tempfile.mkdtemp(prefix="fft_universal_vs_torch_")
    in_path  = os.path.join(tmpdir, "fft_input.txt")
    out_path = os.path.join(tmpdir, "fft_output.txt")

    try:
        run_tt_metal(args.binary, args.N, args.seed, in_path, out_path)

        x     = load_complex_file(in_path)
        X_tt  = load_complex_file(out_path)
        if x.shape != (args.N,) or X_tt.shape != (args.N,):
            sys.exit(
                f"shape mismatch: x={x.shape}, X_tt={X_tt.shape}, "
                f"expected ({args.N},)")

        x_t     = torch.from_numpy(x).to(torch.complex128)
        X_torch = torch.fft.fft(x_t).numpy()

        diff     = X_tt - X_torch
        abs_err  = np.max(np.abs(diff))
        mean_err = np.mean(np.abs(diff))
        ref_max  = float(np.max(np.abs(X_torch))) or 1.0
        rel_err  = abs_err / ref_max
        num      = float(np.sum(np.abs(X_torch) ** 2))
        den      = float(np.sum(np.abs(diff)    ** 2)) or 1e-300
        snr_db   = 10.0 * np.log10(num / den)

        worst_idx = np.argsort(np.abs(diff))[-5:][::-1]

        pass_  = rel_err < tol
        status = "PASS" if pass_ else "FAIL"

        print()
        print("=" * 72)
        print(f"  fft_universal  vs  torch.fft.fft   "
              f"(N={args.N}, seed={args.seed})")
        print(f"  dispatch path      : {describe_path(args.N)}")
        print("=" * 72)
        print(f"  reference max |X|  : {ref_max:.6e}")
        print(f"  max abs error      : {abs_err:.6e}")
        print(f"  max rel error      : {rel_err:.6e}   (threshold {tol:g})")
        print(f"  mean abs error     : {mean_err:.6e}")
        print(f"  SNR                : {snr_db:.2f} dB")
        print()
        print(f"  worst 5 bins (k, |err|, X_tt, X_torch):")
        for k in worst_idx:
            print(f"    k={int(k):6d}  |err|={np.abs(diff[k]):.3e}  "
                  f"tt=({X_tt[k].real:+.4e}{X_tt[k].imag:+.4e}j)  "
                  f"torch=({X_torch[k].real:+.4e}{X_torch[k].imag:+.4e}j)")
        print()
        if pass_:
            print(f"  [{status}]  fft_universal matches torch.fft.fft "
                  f"within tolerance {tol:g}")
        else:
            print(f"  [{status}]  fft_universal DOES NOT match torch.fft.fft "
                  f"within tolerance {tol:g}")
        print("=" * 72)

        sys.exit(0 if pass_ else 2)

    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[compare] kept temp files in {tmpdir}")


if __name__ == "__main__":
    main()
