#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench_ifft_roundtrip.py — correctness of the conjugate-trick IFFT.

For each N we compute:
    re, im   = fft(x)            on device
    x', _    = ifft(re, im)      on device
and report the L2 relative error of x' vs x.

CSV columns:
    N, dtype, precision, batch, fwd_then_inv_rel_err
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
import ttnn


def main() -> int:
    p = C.base_argparser(__doc__ or "")
    args = p.parse_args()

    dtypes = C.parse_dtype_list(args.dtype)
    batches = C.parse_int_list(args.batch)

    out_path = Path(args.out) if args.out else C.default_out_path("ifft_roundtrip")
    writer = C.CsvWriter(out_path, [
        "N", "dtype", "precision", "batch", "fwd_then_inv_rel_err",
    ])

    with C.open_device() as device:
        for dtype in dtypes:
            precisions = C.parse_precision_list(args.precision, dtype)
            N_list = C.resolve_N_list(args.N, dtype)
            for N in N_list:
                for B in batches:
                    for prec in precisions:
                        try:
                            tt_in, torch_in = C.make_input(
                                N, B, dtype, device, seed=args.seed)
                        except Exception as e:
                            C.log(f"[skip alloc] N={N} dtype={dtype}: {e}")
                            continue

                        try:
                            re_tt, im_tt = ttnn.experimental.fft(
                                tt_in, precision=prec)
                            x_back_re, x_back_im = ttnn.experimental.ifft(
                                re_tt, im_tt, precision=prec)
                        except Exception as e:
                            C.log(f"[skip run] N={N} dtype={dtype}"
                                  f" prec={prec}: {e}")
                            continue

                        got_re = ttnn.to_torch(x_back_re).reshape(
                            torch_in.shape).to(torch.float32)
                        # imag should be ~0 for real input
                        err = (torch.linalg.norm(got_re - torch_in)
                               / torch.linalg.norm(torch_in)
                               .clamp_min(1e-12)).item()

                        writer.write({
                            "N": N, "dtype": dtype, "precision": prec,
                            "batch": B,
                            "fwd_then_inv_rel_err": f"{err:.3e}",
                        })
                        C.log(
                            f"N={N:>8} B={B:>3} {dtype}/{prec:>7}: "
                            f"roundtrip_rel_err={err:.2e}")
                        C.cleanup()

    writer.close()
    C.log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
