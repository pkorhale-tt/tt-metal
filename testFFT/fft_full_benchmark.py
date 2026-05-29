# fft_full_benchmark.py
#
# Comprehensive benchmark for ttnn.fft / ttnn.ifft.
# Covers all 4 backends with timing + correctness vs torch.fft.fft.
#
#   Backends exercised (auto-dispatched by ttnn):
#     fp32 + pow2  + N <= 1M        →  fft_stockham
#     fp32 + pow2  + 1M < N <= 16M  →  fft_universal_xl
#     fp32 + non-pow2               →  fft_universal       (mixed-radix / Bluestein)
#     bf16 + any N                  →  fft_universal_bf16
#
# Run:    python3 fft_full_benchmark.py
# Notes:  Large N (>= 4M) takes minutes; comment out if your dev box is busy.

import time
import gc
import torch
import ttnn

# ───────────────────────────── Configuration ─────────────────────────────────
WARMUP_RUNS = 1  # ignore first call (program-cache miss + kernel compile)
TIMED_RUNS = 3  # take min of N runs


def tol_for(dtype, N):
    """Per-(dtype, N) tolerance.

    Calibrated from observed Wormhole errors. Matches the pytest tolerances
    in tests/ttnn/unit_tests/operations/experimental/test_fft.py.

    Two error regimes for fp32:
      * butterfly path (Stockham / UniversalXL / Bluestein primes) — pure
        log-N rounding, rel_err ~1e-7 floor;
      * packed-DFT path (composite non-pow2) — each matmul reduction has
        bf16 internal mantissa → ~1-3e-3 floor regardless of N.
    bf16 floor everywhere is ~3-5e-3 (true-bf16 FPU compute by design).
    """
    if dtype == ttnn.bfloat16:
        if N <= 32:
            return 1.5e-2
        if N <= 1024:
            return 2e-2
        return 5e-2

    # fp32 — split butterfly path vs packed-DFT path
    is_pow2 = (N & (N - 1)) == 0 and N > 0
    if is_pow2:
        if N <= 1024:
            return 5e-4
        if N <= 65_536:
            return 2e-3
        if N <= 1_048_576:
            return 5e-3
        return 2e-2
    else:
        # fft_universal — packed_dft path adds 1-3e-3 noise per matmul stage.
        # Bluestein primes still hit ~1e-7 because they use butterflies
        # internally; we set the tolerance for the worst case (composites).
        if N <= 32:
            return 2e-3
        if N <= 1024:
            return 5e-3
        return 1e-2


# Test cases: (label, dtype, N, expected_backend)
CASES = [
    # ── fp32 + pow2  →  fft_stockham (≤1M) ──────────────────────────────────
    ("Stockham (small pow2)", ttnn.float32, 8, "stockham"),
    ("Stockham (small pow2)", ttnn.float32, 64, "stockham"),
    ("Stockham (small pow2)", ttnn.float32, 1024, "stockham"),
    ("Stockham (mid pow2)", ttnn.float32, 4096, "stockham"),
    ("Stockham (mid pow2)", ttnn.float32, 65536, "stockham"),
    ("Stockham (large pow2)", ttnn.float32, 262144, "stockham"),
    ("Stockham (max pow2)", ttnn.float32, 1048576, "stockham"),
    # ── fp32 + pow2  →  fft_universal_xl (>1M, ≤16M) ────────────────────────
    ("UniversalXL (huge pow2)", ttnn.float32, 2097152, "universal_xl"),  # 2M
    ("UniversalXL (huge pow2)", ttnn.float32, 4194304, "universal_xl"),  # 4M
    # Comment back in if you have device + host RAM + patience:
    # ("UniversalXL (huge pow2)",    ttnn.float32,  8388608,  "universal_xl"), # 8M
    # ("UniversalXL (max pow2)",     ttnn.float32, 16777216,  "universal_xl"), # 16M
    # ── fp32 + non-pow2  →  fft_universal  (composite + prime/Bluestein) ────
    ("Universal (composite small)", ttnn.float32, 24, "universal"),
    ("Universal (composite mid)", ttnn.float32, 96, "universal"),
    ("Universal (composite 2^3·5^3)", ttnn.float32, 1000, "universal"),
    ("Universal (prime → Bluestein)", ttnn.float32, 97, "universal"),
    ("Universal (prime → Bluestein)", ttnn.float32, 1009, "universal"),
    # ── bf16 (any N)  →  fft_universal_bf16 ─────────────────────────────────
    ("Bf16 (small)", ttnn.bfloat16, 8, "bf16"),
    ("Bf16 (small pow2)", ttnn.bfloat16, 32, "bf16"),
    ("Bf16 (mid pow2)", ttnn.bfloat16, 256, "bf16"),
    ("Bf16 (large pow2)", ttnn.bfloat16, 1024, "bf16"),
    ("Bf16 (large pow2)", ttnn.bfloat16, 4096, "bf16"),
    ("Bf16 (composite non-pow2)", ttnn.bfloat16, 96, "bf16"),
    ("Bf16 (prime → Bluestein)", ttnn.bfloat16, 97, "bf16"),
]


# ─────────────────────────────── Helpers ─────────────────────────────────────
def time_call(fn, n_warm=WARMUP_RUNS, n_timed=TIMED_RUNS):
    """Run fn(), discarding warmup; return min wall time across timed runs."""
    for _ in range(n_warm):
        fn()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def measure_one(device, label, dtype, N, expected_backend):
    """Run a single (dtype, N) case; return a result dict."""
    torch.manual_seed(0)
    x_torch = torch.randn(N, dtype=torch.float32)

    # ── CPU reference + timing ────────────────────────────────────────────
    def cpu_fft():
        return torch.fft.fft(x_torch)

    t_cpu = time_call(cpu_fft)
    ref = cpu_fft()

    # ── Upload to device ─────────────────────────────────────────────────
    tt_x = ttnn.from_torch(
        x_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # ── ttnn.fft timing ──────────────────────────────────────────────────
    def device_fft():
        return ttnn.fft(tt_x)

    t_device = time_call(device_fft)
    re, im = device_fft()

    # ── Correctness ──────────────────────────────────────────────────────
    got = torch.complex(
        ttnn.to_torch(re).reshape(-1).to(torch.float32),
        ttnn.to_torch(im).reshape(-1).to(torch.float32),
    )
    rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
    tol = tol_for(dtype, N)

    # ── IFFT round-trip ──────────────────────────────────────────────────
    rec_re, _ = ttnn.ifft(re, im)
    rec = ttnn.to_torch(rec_re).reshape(-1).to(torch.float32)
    rt_rel = (torch.linalg.norm(rec - x_torch) / torch.linalg.norm(x_torch)).item()

    # ── Cleanup so memory doesn't blow up across cases ───────────────────
    del tt_x, re, im, rec_re, _, got, ref, rec
    gc.collect()

    return {
        "label": label,
        "backend": expected_backend,
        "dtype": "fp32" if dtype == ttnn.float32 else "bf16",
        "N": N,
        "rel_err": rel,
        "rt_rel_err": rt_rel,
        "tol": tol,
        "passed": rel < tol,
        "t_cpu_ms": t_cpu * 1000,
        "t_device_ms": t_device * 1000,
        "speedup_vs_cpu": t_cpu / t_device if t_device > 0 else 0.0,
    }


# ─────────────────────────────── Main ────────────────────────────────────────
def main():
    device = ttnn.open_device(device_id=0)
    results = []

    print(f"{'='*98}")
    print(f"  ttnn.fft full backend benchmark  ({WARMUP_RUNS} warmup + min of {TIMED_RUNS} timed)")
    print(f"{'='*98}")

    try:
        for label, dtype, N, backend in CASES:
            try:
                r = measure_one(device, label, dtype, N, backend)
                results.append(r)
                status = "PASS" if r["passed"] else "FAIL"
                print(
                    f"  [{status}] {r['dtype']:4s} N={r['N']:>9,d} | "
                    f"backend={r['backend']:13s} | "
                    f"rel_err={r['rel_err']:.2e} (tol {r['tol']:.0e}) | "
                    f"roundtrip={r['rt_rel_err']:.2e} | "
                    f"cpu={r['t_cpu_ms']:7.2f}ms  device={r['t_device_ms']:7.2f}ms  "
                    f"({r['speedup_vs_cpu']:5.2f}x vs cpu)"
                )
            except Exception as e:
                print(f"  [ERROR] {dtype} N={N} ({label}): {e}")
                results.append(
                    {
                        "label": label,
                        "dtype": dtype,
                        "N": N,
                        "passed": False,
                        "error": str(e),
                    }
                )
    finally:
        ttnn.close_device(device)

    # ─────────────────────────────── Summary ─────────────────────────────────
    print(f"\n{'='*98}\n  SUMMARY\n{'='*98}")
    n_total = len(results)
    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"  {n_pass}/{n_total} cases passed")

    print("\n  Per-backend pass rate:")
    for backend in ("stockham", "universal_xl", "universal", "bf16"):
        sub = [r for r in results if r.get("backend") == backend]
        if not sub:
            continue
        ok = sum(1 for r in sub if r.get("passed"))
        print(f"    {backend:13s}  {ok:>2d}/{len(sub):<2d}  cases")

    print("\n  Timings (median across N for each backend):")
    for backend in ("stockham", "universal_xl", "universal", "bf16"):
        sub = [r for r in results if r.get("backend") == backend and "t_device_ms" in r]
        if not sub:
            continue
        med = sorted(r["t_device_ms"] for r in sub)[len(sub) // 2]
        print(
            f"    {backend:13s}  median wall time = {med:7.2f} ms "
            f"(N range = {min(r['N'] for r in sub)}–{max(r['N'] for r in sub):,d})"
        )


if __name__ == "__main__":
    main()
