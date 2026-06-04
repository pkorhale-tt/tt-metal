#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench_program_cache.py — quantify the program-cache speedup.

For every (N, dtype, precision) combo:

    * call the op once on a fresh (cleared) cache  → cold_us
    * call the op `--iters` times once cached      → warm_us = median

Reports speedup = cold_us / warm_us.

CSV columns:
    N, dtype, precision, cold_us, warm_median_us, speedup
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
import ttnn


def _clear_cache(device) -> None:
    if hasattr(ttnn, "clear_program_cache"):
        try:
            ttnn.clear_program_cache(device)
            return
        except Exception:
            pass
    if hasattr(ttnn, "disable_and_clear_program_cache"):
        try:
            ttnn.disable_and_clear_program_cache(device)
            ttnn.enable_program_cache(device)
            return
        except Exception:
            pass
    C.log("WARN: could not clear program cache; cold sample may be hot.")


def main() -> int:
    p = C.base_argparser(__doc__ or "")
    args = p.parse_args()

    dtypes = C.parse_dtype_list(args.dtype)
    batch = 1
    out_path = Path(args.out) if args.out else C.default_out_path("program_cache")
    writer = C.CsvWriter(out_path, [
        "N", "dtype", "precision", "cold_us", "warm_median_us", "speedup",
    ])

    with C.open_device() as device:
        for dtype in dtypes:
            precisions = C.parse_precision_list(args.precision, dtype)
            N_list = C.resolve_N_list(args.N, dtype)
            for N in N_list:
                for prec in precisions:
                    try:
                        tt_in, _ = C.make_input(
                            N, batch, dtype, device, seed=args.seed)
                    except Exception as e:
                        C.log(f"[skip alloc] N={N} dtype={dtype}: {e}")
                        continue

                    _clear_cache(device)

                    # Cold sample
                    t0 = time.perf_counter_ns()
                    try:
                        C.call_fft(tt_in, precision=prec)
                        C.synchronize(device)
                    except Exception as e:
                        C.log(f"[skip cold] N={N} dtype={dtype}"
                              f" prec={prec}: {e}")
                        continue
                    cold_us = (time.perf_counter_ns() - t0) / 1e3

                    # Warm samples
                    def call():
                        C.call_fft(tt_in, precision=prec)

                    warm = C.time_call_us(
                        call, device, warmup=0, iters=args.iters)

                    speedup = cold_us / warm["median_us"] \
                        if warm["median_us"] > 0 else 0.0
                    writer.write({
                        "N": N, "dtype": dtype, "precision": prec,
                        "cold_us": f"{cold_us:.3f}",
                        "warm_median_us": f"{warm['median_us']:.3f}",
                        "speedup": f"{speedup:.2f}",
                    })
                    C.log(
                        f"N={N:>8} {dtype}/{prec:>7}: "
                        f"cold={cold_us:.1f}us  "
                        f"warm={warm['median_us']:.1f}us  "
                        f"speedup={speedup:.1f}x")
                    C.cleanup()

    writer.close()
    C.log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
