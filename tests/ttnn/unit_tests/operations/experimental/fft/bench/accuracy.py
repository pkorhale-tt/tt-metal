# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench/accuracy.py — relative error vs torch.fft fp64 reference  (HPEC R2)

Produces the paper's accuracy table: aggregate rel-err for every
supported (op, N, dtype) configuration, computed against a
torch.fft complex128 reference (treated as ground truth).

Coverage
--------
  * forward FFT  : N in {16, 64, ..., 4M}  × {fp32, bf16}
  * IFFT         : same N range (uses swap-trick / fft_two_pass / etc)
  * Bluestein    : a tight prime/non-pow2 sweep (covered already by
                   test_bluestein_fft.py's sweep but reproduced here
                   so the accuracy CSV is self-contained for the paper)

Outputs
-------
  <out>/accuracy.csv      one row per (op, N, B, dtype, input_kind)
  <out>/accuracy.png      rel-err vs N, lines = dtype × op (forward/inverse)

Usage
-----
  TT_FFT_NATIVE=1 python tests/ttnn/unit_tests/operations/experimental/fft/bench/accuracy.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (                                            # noqa: E402
    pick_three_factorization, config_supported, open_device, write_csv,
    TWO_PASS_MAX_N, THREE_PASS_MIN_N,
)


# ─── reference + error helpers ─────────────────────────────────────────
def _to_torch_complex64(real, imag):
    if isinstance(real, ttnn.Tensor):
        real = ttnn.to_torch(real)
    if isinstance(imag, ttnn.Tensor):
        imag = ttnn.to_torch(imag)
    return torch.complex(real.to(torch.float32), imag.to(torch.float32))


def _rel_err_fp64(got_complex64: torch.Tensor, ref_complex128: torch.Tensor) -> float:
    """L2 relative error against an fp64 reference."""
    got = got_complex64.to(torch.complex128)
    diff = got - ref_complex128
    num = torch.linalg.vector_norm(diff.reshape(-1))
    den = torch.linalg.vector_norm(ref_complex128.reshape(-1))
    if den.item() == 0.0:
        return float("nan")
    return float((num / den).item())


# ─── per-op runners ────────────────────────────────────────────────────
def _run_forward_fft(device, N, B, dtype, input_kind):
    """One forward FFT: build input, run, compute rel-err vs torch.fft fp64."""
    torch.manual_seed(0xACC0 + N + B + (1 if input_kind == "complex" else 0))

    if N > TWO_PASS_MAX_N:
        # three-pass path: pre-shape input as (N1·N2, N3) and only B=1 fp32
        N1, N2, N3 = pick_three_factorization(N)
        xr_natural = torch.randn(N, dtype=torch.float32)
        xr_shaped  = xr_natural.view(N1 * N2, N3)
        tt_x = ttnn.from_torch(xr_shaped, dtype=dtype,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_r, out_i = ttnn.experimental.fft_three_pass(tt_x, full_N=N)
        # output is (N3, N2, N1) which after .reshape(N) is natural-order
        got = _to_torch_complex64(out_r, out_i).reshape(N)
        ref = torch.fft.fft(xr_natural.to(torch.complex128))
        return _rel_err_fp64(got, ref)

    # single-tile / two-pass: (B, N)
    if input_kind == "real":
        xr = torch.randn(B, N, dtype=torch.float32)
        tt_xr = ttnn.from_torch(xr, dtype=dtype,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_r, out_i = ttnn.experimental.fft(tt_xr)
        got = _to_torch_complex64(out_r, out_i)
        ref = torch.fft.fft(xr.to(torch.complex128), dim=-1)
        return _rel_err_fp64(got, ref)

    # complex input
    xr = torch.randn(B, N, dtype=torch.float32)
    xi = torch.randn(B, N, dtype=torch.float32)
    tt_xr = ttnn.from_torch(xr, dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_xi = ttnn.from_torch(xi, dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out_r, out_i = ttnn.experimental.fft(tt_xr, tt_xi)
    got = _to_torch_complex64(out_r, out_i)
    ref = torch.fft.fft(torch.complex(xr, xi).to(torch.complex128), dim=-1)
    return _rel_err_fp64(got, ref)


def _run_ifft(device, N, B, dtype):
    """Round-trip IFFT(FFT(x)) → rel-err vs original x.  Tests the IFFT
    path AND the 1/N scaling correctness in one shot."""
    torch.manual_seed(0xDEC0DE + N + B)
    xr = torch.randn(B, N, dtype=torch.float32)
    tt_xr = ttnn.from_torch(xr, dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    spec_r, spec_i = ttnn.experimental.fft(tt_xr)
    rec_r, rec_i   = ttnn.experimental.ifft(spec_r, spec_i)
    rec = _to_torch_complex64(rec_r, rec_i)
    ref = xr.to(torch.complex128)
    return _rel_err_fp64(rec, ref)


def _run_bluestein(device, N, B, dtype, input_kind):
    """Bluestein for arbitrary (non-pow-2) N."""
    torch.manual_seed(0xB1E5 + N)
    if input_kind == "real":
        xr = torch.randn(B, N, dtype=torch.float32)
        tt_xr = ttnn.from_torch(xr, dtype=dtype,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_r, out_i = ttnn.experimental.bluestein_fft(tt_xr, N=N)
        ref = torch.fft.fft(xr.to(torch.complex128), dim=-1)
    else:
        xr = torch.randn(B, N, dtype=torch.float32)
        xi = torch.randn(B, N, dtype=torch.float32)
        tt_xr = ttnn.from_torch(xr, dtype=dtype,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tt_xi = ttnn.from_torch(xi, dtype=dtype,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out_r, out_i = ttnn.experimental.bluestein_fft(tt_xr, tt_xi, N=N)
        ref = torch.fft.fft(torch.complex(xr, xi).to(torch.complex128), dim=-1)
    got = _to_torch_complex64(out_r, out_i)
    return _rel_err_fp64(got, ref)


# ─── sweep config ──────────────────────────────────────────────────────
N_SWEEP_POW2 = [
    16, 64, 256, 1024, 2048, 4096, 16384,            # single-tile + two-pass
    32 * 1024, 64 * 1024, 256 * 1024,                # three-pass small
    1 * 1024 * 1024, 2 * 1024 * 1024,                # three-pass large
]
B_SWEEP   = [1, 8]                          # one small, one batched
DTYPES    = [("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)]

# Bluestein arbitrary-N sweep.  Current-build cap: complex_mul rejects
# last-dim > 1024, so M = next_pow2(2N-1) ≤ 1024 → **N ≤ 512**.  We
# document this honestly in the paper as a current-build limit (lifting
# the cmul cap is future work).  Within that cap we still cover the
# DSP-relevant cases: small primes, just-around-pow2, mid-range primes.
N_SWEEP_BLUESTEIN = [3, 5, 7, 11, 13, 17, 23, 31,
                     100, 127, 129, 257, 384, 511]
B_SWEEP_BLUESTEIN = [1, 4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
        default="tests/ttnn/unit_tests/operations/experimental/fft/bench/results",
        type=Path)
    parser.add_argument("--max-n",       default=2 * 1024 * 1024, type=int)
    parser.add_argument("--trace-region", default=2 * 1024 * 1024, type=int)
    parser.add_argument("--device-id",   default=0, type=int)
    parser.add_argument("--skip-bluestein", action="store_true",
                        help="skip the Bluestein arbitrary-N sweep")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = open_device(args.device_id, args.trace_region)
    rows = []
    try:
        # forward FFT (real input)
        for N in N_SWEEP_POW2:
            if N > args.max_n:
                continue
            for B in B_SWEEP:
                for dtype_label, dtype in DTYPES:
                    if not config_supported(N, B, dtype_label):
                        continue
                    tag = (f"FFT  fwd  real     N={N:>8d}  B={B:>3d}  "
                           f"{dtype_label}")
                    print(f"[bench] {tag}", end="  ", flush=True)
                    try:
                        err = _run_forward_fft(device, N, B, dtype, "real")
                        print(f"-> rel_err = {err:.3e}")
                        status = "ok"
                    except Exception as e:
                        err = float("nan")
                        status = f"err: {type(e).__name__}"
                        print(f"-> SKIP ({status})")
                    rows.append({"op": "fft", "input_kind": "real",
                                 "N": N, "B": B, "dtype": dtype_label,
                                 "rel_err": err, "status": status})

        # forward FFT (complex input) — only single-tile + two-pass
        for N in N_SWEEP_POW2:
            if N > TWO_PASS_MAX_N:
                continue
            for B in B_SWEEP:
                for dtype_label, dtype in DTYPES:
                    tag = (f"FFT  fwd  complex  N={N:>8d}  B={B:>3d}  "
                           f"{dtype_label}")
                    print(f"[bench] {tag}", end="  ", flush=True)
                    try:
                        err = _run_forward_fft(device, N, B, dtype, "complex")
                        print(f"-> rel_err = {err:.3e}")
                        status = "ok"
                    except Exception as e:
                        err = float("nan")
                        status = f"err: {type(e).__name__}"
                        print(f"-> SKIP ({status})")
                    rows.append({"op": "fft", "input_kind": "complex",
                                 "N": N, "B": B, "dtype": dtype_label,
                                 "rel_err": err, "status": status})

        # IFFT (round-trip FFT→IFFT against original input)
        # legacy single-tile IFFT path is NOT trace-safe but DOES produce
        # correct numerics, so we still measure rel-err here.
        for N in N_SWEEP_POW2:
            if N > TWO_PASS_MAX_N:
                continue  # three-pass IFFT not in current build
            for B in B_SWEEP:
                for dtype_label, dtype in DTYPES:
                    tag = (f"IFFT round-trip   N={N:>8d}  B={B:>3d}  "
                           f"{dtype_label}")
                    print(f"[bench] {tag}", end="  ", flush=True)
                    try:
                        err = _run_ifft(device, N, B, dtype)
                        print(f"-> rel_err = {err:.3e}")
                        status = "ok"
                    except Exception as e:
                        err = float("nan")
                        status = f"err: {type(e).__name__}"
                        print(f"-> SKIP ({status})")
                    rows.append({"op": "ifft_roundtrip", "input_kind": "real",
                                 "N": N, "B": B, "dtype": dtype_label,
                                 "rel_err": err, "status": status})

        # Bluestein arbitrary-N
        if not args.skip_bluestein:
            for N in N_SWEEP_BLUESTEIN:
                for B in B_SWEEP_BLUESTEIN:
                    for dtype_label, dtype in DTYPES:
                        for input_kind in ("real", "complex"):
                            tag = (f"Bluestein {input_kind:7s}  N={N:>6d}  "
                                   f"B={B:>2d}  {dtype_label}")
                            print(f"[bench] {tag}", end="  ", flush=True)
                            try:
                                err = _run_bluestein(device, N, B, dtype,
                                                     input_kind)
                                print(f"-> rel_err = {err:.3e}")
                                status = "ok"
                            except Exception as e:
                                err = float("nan")
                                status = f"err: {type(e).__name__}"
                                print(f"-> SKIP ({status})")
                            rows.append({"op": "bluestein",
                                         "input_kind": input_kind,
                                         "N": N, "B": B, "dtype": dtype_label,
                                         "rel_err": err, "status": status})
    finally:
        ttnn.close_device(device)

    fieldnames = ["op", "input_kind", "N", "B", "dtype", "rel_err", "status"]
    write_csv(rows, args.out / "accuracy.csv", fieldnames)

    try:
        _plot_accuracy(rows, args.out / "accuracy.png")
    except ImportError:
        print("[bench] matplotlib not available, skipping plot")


def _plot_accuracy(rows, png_path):
    """Rel-err vs N for forward FFT (real input, B=1), one line per dtype.
    Also overlays a horizontal band marking the expected fp32 ε (~1e-6)
    and bf16 ε (~1e-2) so reviewers can see we're within precision."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    style = {
        "fp32": dict(linestyle="-", marker="o", color="C0",
                     label="fp32  (fwd FFT, real, B=1)"),
        "bf16": dict(linestyle="-", marker="s", color="C1",
                     label="bf16  (fwd FFT, real, B=1)"),
    }
    for dt, sty in style.items():
        pts = [(r["N"], r["rel_err"]) for r in rows
               if r["op"] == "fft" and r["input_kind"] == "real"
               and r["B"] == 1 and r["dtype"] == dt
               and r.get("status") == "ok"]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, **sty)

    # epsilon reference bands
    ax.axhline(1e-6, color="C0", linestyle=":", alpha=0.5,
               label="fp32 ε ≈ 1e-6")
    ax.axhline(1e-2, color="C1", linestyle=":", alpha=0.5,
               label="bf16 ε ≈ 1e-2")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("FFT length N")
    ax.set_ylabel("L2 relative error  (vs torch.fft fp64)")
    ax.set_title("ttnn.experimental.fft — numerical accuracy vs N (n300)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"[bench] wrote {png_path}")


if __name__ == "__main__":
    main()
