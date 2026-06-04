#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bench_metal_trace.py — untraced vs traced (Metal Trace) call latency.

If the running ttnn build does not expose `ttnn.begin_trace_capture` /
`ttnn.end_trace_capture` / `ttnn.execute_trace`, the script falls back
to recording the untraced number only and writes 'unsupported' in the
traced columns. Either way the CSV row is produced so plot_results.py
does not break.

CSV columns:
    N, dtype, precision, untraced_us, traced_us, speedup
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
import ttnn


def _trace_supported() -> bool:
    return all(
        hasattr(ttnn, sym)
        for sym in ("begin_trace_capture", "end_trace_capture", "execute_trace")
    )


def main() -> int:
    p = C.base_argparser(__doc__ or "")
    args = p.parse_args()

    dtypes = C.parse_dtype_list(args.dtype)
    batch = 1
    out_path = Path(args.out) if args.out else C.default_out_path("metal_trace")
    writer = C.CsvWriter(out_path, [
        "N", "dtype", "precision", "untraced_us", "traced_us", "speedup",
    ])

    trace_ok = _trace_supported()
    if not trace_ok:
        C.log("WARN: this ttnn build does not expose trace API; "
              "filling traced columns with 'unsupported'.")

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

                    # Untraced baseline
                    def call():
                        C.call_fft(tt_in, precision=prec)

                    try:
                        base = C.time_call_us(
                            call, device,
                            warmup=args.warmup, iters=args.iters)
                    except Exception as e:
                        C.log(f"[skip untraced] N={N} dtype={dtype}"
                              f" prec={prec}: {e}")
                        continue

                    traced_med = None
                    if trace_ok:
                        try:
                            # Warm program cache before tracing.
                            for _ in range(2):
                                call()
                            C.synchronize(device)

                            tid = ttnn.begin_trace_capture(device, cq_id=0)
                            call()
                            ttnn.end_trace_capture(device, tid, cq_id=0)

                            def replay():
                                ttnn.execute_trace(device, tid, cq_id=0,
                                                   blocking=False)

                            stats = C.time_call_us(
                                replay, device, warmup=2, iters=args.iters)
                            traced_med = stats["median_us"]

                            if hasattr(ttnn, "release_trace"):
                                ttnn.release_trace(device, tid)
                        except Exception as e:
                            C.log(f"[trace failed] N={N} dtype={dtype}"
                                  f" prec={prec}: {e}")
                            traced_med = None

                    if traced_med is None:
                        traced_str = "unsupported"
                        speedup_str = "unsupported"
                    else:
                        traced_str = f"{traced_med:.3f}"
                        speedup_str = (
                            f"{base['median_us'] / traced_med:.2f}"
                            if traced_med > 0 else "0"
                        )

                    writer.write({
                        "N": N, "dtype": dtype, "precision": prec,
                        "untraced_us": f"{base['median_us']:.3f}",
                        "traced_us": traced_str,
                        "speedup": speedup_str,
                    })
                    C.log(
                        f"N={N:>8} {dtype}/{prec:>7}: "
                        f"untraced={base['median_us']:.1f}us  "
                        f"traced={traced_str}  "
                        f"speedup={speedup_str}")
                    C.cleanup()

    writer.close()
    C.log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
