#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench_host_device_split.py — quantify the host round-trip cost.

Because fft_program_factory.cpp currently materialises the input
tensor on the host (read_real_as_fp32) and writes the output back, every
ttnn.experimental.fft call includes a host memcpy + cast on both ends
(see paper/HOST_VS_DEVICE.md).

This script measures:

    e2e_us       — total wall time of the ttnn.experimental.fft call
                   *with* the host round-trip (the user-visible number)

    no_io_us     — wall time of the same call *after* we have lifted the
                   input to device-resident form, by allocating the input
                   ONCE outside the timing loop and only timing the op
                   invocation itself.

    host_io_us   — e2e_us − no_io_us  (best estimate of host overhead)
    host_io_pct  — 100 · host_io_us / e2e_us

CSV columns:
    N, dtype, precision, e2e_us, no_io_us, host_io_us, host_io_pct
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
import torch
import ttnn


def main() -> int:
    p = C.base_argparser(__doc__ or "")
    args = p.parse_args()

    dtypes = C.parse_dtype_list(args.dtype)
    out_path = Path(args.out) if args.out else C.default_out_path("host_device_split")
    writer = C.CsvWriter(out_path, [
        "N", "dtype", "precision",
        "e2e_us", "no_io_us", "host_io_us", "host_io_pct",
    ])

    with C.open_device() as device:
        for dtype in dtypes:
            precisions = C.parse_precision_list(args.precision, dtype)
            N_list = C.resolve_N_list(args.N, dtype)
            for N in N_list:
                for prec in precisions:
                    # 1) e2e: re-upload from torch every call.
                    torch_in = torch.randn(N, dtype=torch.float32)
                    ttnn_dtype, td = C.DTYPE_MAP[dtype]
                    feed = torch_in.to(td)

                    def e2e_call():
                        tt = ttnn.from_torch(
                            feed,
                            dtype=ttnn_dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device,
                        )
                        re, im = ttnn.experimental.fft(tt, precision=prec)
                        # force materialisation back to torch
                        ttnn.to_torch(re)
                        ttnn.to_torch(im)

                    try:
                        e2e_stats = C.time_call_us(
                            e2e_call, device,
                            warmup=args.warmup, iters=args.iters)
                    except Exception as e:
                        C.log(f"[skip e2e] N={N} {dtype}/{prec}: {e}")
                        continue

                    # 2) no_io: tensor already on device, no to_torch in loop.
                    tt_in, _ = C.make_input(
                        N, 1, dtype, device, seed=args.seed)

                    def hot_call():
                        C.call_fft(tt_in, precision=prec)

                    try:
                        hot_stats = C.time_call_us(
                            hot_call, device,
                            warmup=args.warmup, iters=args.iters)
                    except Exception as e:
                        C.log(f"[skip no_io] N={N} {dtype}/{prec}: {e}")
                        continue

                    e2e_us = e2e_stats["median_us"]
                    no_io_us = hot_stats["median_us"]
                    host_io_us = max(0.0, e2e_us - no_io_us)
                    host_io_pct = 100.0 * host_io_us / e2e_us if e2e_us > 0 else 0.0

                    writer.write({
                        "N": N, "dtype": dtype, "precision": prec,
                        "e2e_us":     f"{e2e_us:.3f}",
                        "no_io_us":   f"{no_io_us:.3f}",
                        "host_io_us": f"{host_io_us:.3f}",
                        "host_io_pct": f"{host_io_pct:.1f}",
                    })
                    C.log(
                        f"N={N:>8} {dtype}/{prec:>7}: "
                        f"e2e={e2e_us:.1f}us  no_io={no_io_us:.1f}us  "
                        f"host_io={host_io_us:.1f}us ({host_io_pct:.0f}%)")
                    C.cleanup()

    writer.close()
    C.log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
