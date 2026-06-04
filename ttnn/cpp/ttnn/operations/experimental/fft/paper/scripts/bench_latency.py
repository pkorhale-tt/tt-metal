#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench_latency.py — wall-clock latency sweep of ttnn.experimental.fft.

For every (N, dtype, precision, batch) combo:
    * 5 warmup calls (absorb program-cache miss + first-call JIT)
    * 50 measured calls (default), report median / p05 / p95 / first-call

CSV columns:
    N, dtype, precision, batch, first_call_us, median_us, p05_us, p95_us, mean_us
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C


def main() -> int:
    p = C.base_argparser(__doc__ or "")
    args = p.parse_args()

    dtypes = C.parse_dtype_list(args.dtype)
    batches = C.parse_int_list(args.batch)

    out_path = Path(args.out) if args.out else C.default_out_path("latency")
    writer = C.CsvWriter(out_path, [
        "N", "dtype", "precision", "batch",
        "first_call_us", "median_us", "p05_us", "p95_us", "mean_us",
    ])

    with C.open_device() as device:
        for dtype in dtypes:
            precisions = C.parse_precision_list(args.precision, dtype)
            N_list = C.resolve_N_list(args.N, dtype)
            for N in N_list:
                for B in batches:
                    for prec in precisions:
                        try:
                            tt_in, _ = C.make_input(
                                N, B, dtype, device, seed=args.seed)
                        except Exception as e:
                            C.log(f"[skip alloc] N={N} B={B} dtype={dtype}: {e}")
                            continue

                        def call():
                            C.call_fft(tt_in, precision=prec)

                        try:
                            stats = C.time_call_us(
                                call, device,
                                warmup=args.warmup, iters=args.iters)
                        except Exception as e:
                            C.log(f"[skip run] N={N} B={B} dtype={dtype}"
                                  f" prec={prec}: {e}")
                            C.cleanup()
                            continue

                        row = {
                            "N": N, "dtype": dtype,
                            "precision": prec, "batch": B,
                            **{k: f"{v:.3f}" for k, v in stats.items()},
                        }
                        writer.write(row)
                        C.log(
                            f"N={N:>8} B={B:>3} {dtype}/{prec:>7}: "
                            f"first={stats['first_call_us']:.1f}us  "
                            f"med={stats['median_us']:.1f}us  "
                            f"p95={stats['p95_us']:.1f}us")
                        C.cleanup()

    writer.close()
    C.log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
