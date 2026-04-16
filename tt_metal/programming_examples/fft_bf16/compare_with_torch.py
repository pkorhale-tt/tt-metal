#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
compare_with_torch.py (bf16) — run the tt-metal bf16 FFT on a random complex
signal of length N, run torch.fft.fft on the same signal, and report
accuracy.

Usage:
    python compare_with_torch.py                         # default N=1024
    python compare_with_torch.py --N 65536
    python compare_with_torch.py --N 4096 --seed 7
    python compare_with_torch.py --N 4096 --bin /path/to/metal_example_fft_bf16_vs_torch

Environment variables:
    FFT_BF16_BIN  : path to the bf16 dumper binary (overridden by --bin).
                    Default is "./build/programming_examples/fft_bf16/metal_example_fft_bf16_vs_torch".

Note: bf16 has only ~8 mantissa bits, so the reasonable pass threshold for
the relative error is around 5e-2 to 2e-1 (vs ~2e-3 for fp32). The default
here is 2e-1.
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


DEFAULT_BIN = "./build/programming_examples/fft_bf16/metal_example_fft_bf16_vs_torch"


def load_complex_file(path: str) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path}: expected N rows of 'real imag', got shape {data.shape}")
    return data[:, 0] + 1j * data[:, 1]


def run_tt_metal(binary: str, N: int, seed: int, in_path: str, out_path: str) -> None:
    if not os.path.exists(binary):
        sys.exit(f"Binary not found: {binary}\n"
                 f"Build it with:\n"
                 f"    cmake --build build --target metal_example_fft_bf16_vs_torch -j\n"
                 f"Or override with --bin /path/to/binary")
    cmd = [binary, str(N), str(seed), in_path, out_path]
    env = os.environ.copy()
    env.setdefault("ARCH_NAME", "wormhole_b0")
    print(f"[compare_bf16] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"tt-metal binary failed with exit code {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--N", type=int, default=1024, help="FFT length (power of 2, 2..65536)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--bin", dest="binary",
                    default=os.environ.get("FFT_BF16_BIN", DEFAULT_BIN),
                    help="path to metal_example_fft_bf16_vs_torch")
    ap.add_argument("--tol", type=float, default=2e-1,
                    help="relative-error pass threshold (default 2e-1 for bf16)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary input/output text files")
    args = ap.parse_args()

    if args.N < 2 or (args.N & (args.N - 1)) != 0 or args.N > 65536:
        sys.exit(f"N must be a power of 2 in [2, 65536], got {args.N}")

    tmpdir = tempfile.mkdtemp(prefix="fft_bf16_vs_torch_")
    in_path  = os.path.join(tmpdir, "fft_input.txt")
    out_path = os.path.join(tmpdir, "fft_output.txt")

    try:
        run_tt_metal(args.binary, args.N, args.seed, in_path, out_path)

        x     = load_complex_file(in_path)
        X_tt  = load_complex_file(out_path)
        if x.shape != (args.N,) or X_tt.shape != (args.N,):
            sys.exit(f"shape mismatch: x={x.shape}, X_tt={X_tt.shape}, expected ({args.N},)")

        x_t      = torch.from_numpy(x).to(torch.complex128)
        X_torch  = torch.fft.fft(x_t).numpy()

        diff      = X_tt - X_torch
        abs_err   = np.max(np.abs(diff))
        mean_err  = np.mean(np.abs(diff))
        ref_max   = float(np.max(np.abs(X_torch))) or 1.0
        rel_err   = abs_err / ref_max
        num       = float(np.sum(np.abs(X_torch) ** 2))
        den       = float(np.sum(np.abs(diff) ** 2)) or 1e-300
        snr_db    = 10.0 * np.log10(num / den)

        worst_idx = np.argsort(np.abs(diff))[-5:][::-1]

        pass_ = rel_err < args.tol
        status = "PASS" if pass_ else "FAIL"

        print()
        print("=" * 72)
        print(f"  tt-metal FFT (bf16) vs torch.fft.fft  (N={args.N}, seed={args.seed})")
        print("=" * 72)
        print(f"  reference max |X|   : {ref_max:.6e}")
        print(f"  max abs error       : {abs_err:.6e}")
        print(f"  max rel error       : {rel_err:.6e}   (threshold {args.tol:g})")
        print(f"  mean abs error      : {mean_err:.6e}")
        print(f"  SNR                 : {snr_db:.2f} dB")
        print()
        print(f"  worst 5 bins (k, |err|, X_tt, X_torch):")
        for k in worst_idx:
            print(f"    k={int(k):6d}  |err|={np.abs(diff[k]):.3e}  "
                  f"tt=({X_tt[k].real:+.4e}{X_tt[k].imag:+.4e}j)  "
                  f"torch=({X_torch[k].real:+.4e}{X_torch[k].imag:+.4e}j)")
        print()
        print(f"  [{status}]  tt-metal bf16 FFT matches torch.fft.fft within tolerance "
              f"{args.tol:g}" if pass_ else
              f"  [{status}]  tt-metal bf16 FFT DOES NOT match torch.fft.fft within "
              f"tolerance {args.tol:g}")
        print("=" * 72)

        sys.exit(0 if pass_ else 2)

    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[compare_bf16] kept temp files in {tmpdir}")


if __name__ == "__main__":
    main()
