#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""HPEC 2026: energy / J-per-sample harness for fft_universal.

Polls `tt-smi -s` (or `tt-smi --snapshot`) at a configurable rate while a
sustained FFT loop runs in a child process. Computes:
    * mean board power (W)              — direct from tt-smi
    * total energy over the run (J)     — integral of power dt
    * J per FFT call                    — energy / total iterations
    * J per sample                      — energy / (iterations * N)

Wraps the existing metal_example_fft_universal_benchmark as the "sustained
loop" — call it with a large --iters to ensure the device stays busy long
enough for tt-smi to converge (recommend >= 1000 iters and N >= 16384).

Usage:
  python tt_smi_energy_sampler.py \
      --binary build_Release/programming_examples/fft_universal/metal_example_fft_universal_benchmark \
      --N 16384 --iters 5000 \
      --interval 0.25 \
      --csv paper_results/energy_N16384.csv

Requires `tt-smi` in PATH. Tested with tt-smi 3.x snapshot output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import deque


def parse_tt_smi_snapshot(text: str) -> dict | None:
    """Parse the JSON portion of `tt-smi -s` (snapshot mode)."""
    # tt-smi -s prints a JSON blob. Defensive: try to find the first { ... }
    try:
        start = text.index('{')
        end   = text.rindex('}')
        obj   = json.loads(text[start:end + 1])
        return obj
    except Exception:
        return None


def extract_power_watts(snap: dict, device_idx: int = 0) -> float | None:
    """Return board power for `device_idx` from a tt-smi snapshot dict.

    tt-smi exposes telemetry as
        snap["device_info"][i]["telemetry"]["board_power"]   (Watts)
    or
        snap["device_info"][i]["telemetry"]["tdp"]           (older)

    We try both keys.
    """
    try:
        dev = snap["device_info"][device_idx]
    except (KeyError, IndexError, TypeError):
        return None
    tel = dev.get("telemetry", dev)
    for key in ("board_power", "power", "tdp", "Board Power", "TDP"):
        if key in tel:
            try:
                return float(re.sub(r"[^\d.]", "", str(tel[key])))
            except ValueError:
                continue
    return None


def sample_power_loop(stop_event, samples: list, interval: float, device_idx: int):
    """Background sampler. Appends (timestamp, watts) tuples."""
    tt_smi = shutil.which("tt-smi")
    if not tt_smi:
        sys.stderr.write("warn: tt-smi not in PATH — energy will be empty\n")
        return
    while not stop_event.is_set():
        t = time.monotonic()
        try:
            out = subprocess.run(
                [tt_smi, "-s"], capture_output=True, text=True, timeout=5
            ).stdout
        except subprocess.TimeoutExpired:
            continue
        snap = parse_tt_smi_snapshot(out)
        if snap is not None:
            w = extract_power_watts(snap, device_idx)
            if w is not None:
                samples.append((t, w))
        # Sleep until next interval (compensate for sampling latency)
        elapsed = time.monotonic() - t
        sleep_for = max(0.0, interval - elapsed)
        if stop_event.wait(sleep_for):
            return


def integrate_energy(samples: list[tuple[float, float]]) -> float:
    """Trapezoidal integration. Returns Joules."""
    if len(samples) < 2:
        return 0.0
    e = 0.0
    for (t1, w1), (t2, w2) in zip(samples[:-1], samples[1:]):
        e += 0.5 * (w1 + w2) * (t2 - t1)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True,
                    help="path to metal_example_fft_universal_benchmark")
    ap.add_argument("--N",        type=int, required=True)
    ap.add_argument("--iters",    type=int, default=2000)
    ap.add_argument("--interval", type=float, default=0.25,
                    help="seconds between tt-smi samples")
    ap.add_argument("--device",   type=int, default=0)
    ap.add_argument("--csv",      default="paper_results/energy.csv")
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        sys.exit(f"binary not found: {args.binary}")

    import threading
    stop_event = threading.Event()
    samples: list[tuple[float, float]] = []
    sampler = threading.Thread(
        target=sample_power_loop,
        args=(stop_event, samples, args.interval, args.device),
        daemon=True,
    )
    sampler.start()

    print(f"=== tt_smi_energy_sampler ===")
    print(f"  binary   : {args.binary}")
    print(f"  N        : {args.N}")
    print(f"  iters    : {args.iters}")
    print(f"  interval : {args.interval} s")
    print()

    t_run_start = time.monotonic()
    cmd = [args.binary, str(args.N), str(args.iters)]
    print("  running:", " ".join(shlex.quote(c) for c in cmd))
    rc = subprocess.run(cmd).returncode
    t_run_end = time.monotonic()
    wall = t_run_end - t_run_start

    stop_event.set()
    sampler.join(timeout=5)

    if rc != 0:
        sys.stderr.write(f"warn: binary exited with rc={rc}\n")

    energy_J = integrate_energy(samples)
    avg_W    = energy_J / wall if wall > 0 else 0.0
    j_call   = energy_J / args.iters if args.iters > 0 else 0.0
    j_sample = j_call / args.N        if args.N     > 0 else 0.0
    p_max    = max((w for _, w in samples), default=0.0)
    p_min    = min((w for _, w in samples), default=0.0)

    print()
    print(f"  wall time    : {wall:.2f} s")
    print(f"  samples      : {len(samples)} from tt-smi")
    print(f"  avg power    : {avg_W:.2f} W")
    print(f"  peak / min   : {p_max:.2f} / {p_min:.2f} W")
    print(f"  total energy : {energy_J:.2f} J")
    print(f"  J / fft      : {j_call*1000:.3f} mJ")
    print(f"  J / sample   : {j_sample*1e6:.3f} uJ")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "iters", "wall_s", "samples_taken",
                    "avg_power_W", "peak_W", "min_W",
                    "total_energy_J", "J_per_fft", "J_per_sample_uJ"])
        w.writerow([args.N, args.iters, f"{wall:.3f}", len(samples),
                    f"{avg_W:.3f}", f"{p_max:.3f}", f"{p_min:.3f}",
                    f"{energy_J:.3f}", f"{j_call:.6f}",
                    f"{j_sample*1e6:.6f}"])
    print(f"\n  wrote {args.csv}")


if __name__ == "__main__":
    main()
