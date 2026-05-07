# fft_universal_xl — FFT for any pow2 N up to 2³⁰ (1 G)

Sibling of `fft_universal/` and `fft_universal_bf16/`. Combines a
**K-level recursive planner** with a **multi-stage dispatcher** that
calls existing fft_stockham kernels.

## What's in this folder

| File | Status |
|---|---|
| `fft_universal_xl_planner.hpp` | ✅ K-level factorisation planner (host-only, pure logic) |
| `fft_universal_xl_planner_test.cpp` | ✅ host-only unit test (no device required) |
| `fft_universal_xl_host.cpp` | ✅ **Option B** dispatcher: works for any pow2 N up to 1G **today** |
| `fft_universal_xl_test.cpp` | ✅ device round-trip test |
| `fft_universal_xl_demo.cpp` | ✅ minimal "how to call it" example |
| `CMakeLists.txt` | ✅ wired into the top-level build |
| `option_a_pass2_xl_design.md` | 📋 **Option A** kernel design + skeleton (~1 week to implement) |
| `option_c_kstep_kernel_design.md` | 📋 **Option C** K-step kernel design (~3-5 weeks to implement) |

## What you can run today (Option B)

```bash
cmake -S . -B build -DBUILD_PROGRAMMING_EXAMPLES=ON
ninja -C build \
    metal_example_fft_universal_xl_planner_test \
    metal_example_fft_universal_xl_test \
    metal_example_fft_universal_xl_demo

# 1. Planner sanity check (host-only, no device)
./build/programming_examples/fft_universal_xl/metal_example_fft_universal_xl_planner_test

# 2. Device round-trip across N=1024 ... 4M
./build/programming_examples/fft_universal_xl/metal_example_fft_universal_xl_test

# 3. Demo (defaults to N=2M, the smallest size that exercises the K=3 path)
./build/programming_examples/fft_universal_xl/metal_example_fft_universal_xl_demo 8388608
```

## How big-N is handled

For pow2 `N <= 1M` the dispatcher delegates straight to `fft_stockham`
(no XL-specific work, identical performance and accuracy).

For pow2 `N > 1M` (the new regime), the dispatcher applies a 5-step
Cooley-Tukey decomposition with the **smallest** factor as outer:

| Step | Where | What |
|---|---|---|
| 1 | device | F1 sequential `fft_stockham::fft` calls of length M = N / F1 |
| 2 | **host** ⚠ | outer twiddle multiply `Y[n1, k] *= exp(-2πi · n1·k / N)` (cached cos/sin table) |
| 3 | host | transpose Y (F1 × M) → Z (M × F1) — pure memcopy, no math |
| 4 | device | ONE batched `batch_fft(sub_N=F1, batch=M)` dispatch |
| 5 | host | reorder W (M × F1) → X (length N) — pure memcopy |

**Step 2 is the only host arithmetic.** That's what Option A removes.

## What "Option B" gives you vs what "Option A" would add

| Property | Option B (today) | Option A (after the kernel work) |
|---|---|---|
| Pow2 N up to 1G | ✅ | ✅ |
| No host arithmetic on data | ❌ (Step 2 is host) | ✅ |
| Correctness | ✅ identical | ✅ identical |
| Speed at N ≤ 1M | identical to fft_stockham | identical to fft_stockham |
| Speed at N = 8M | ~5-10s per FFT | ~1-2s per FFT (estimate) |
| Speed at N = 1G | tens of minutes | seconds (estimate) |

For typical workloads the Option-B speed at large N is **bottlenecked
by the F1 sequential outer FFT calls** (Step 1), NOT by the host
twiddle (Step 2). So Option A doesn't fully solve the speed problem at
huge N — that's what Option C is for.

## Recommended sequencing

1. ✅ **Done now** — Option B gives you a correct any-N path.
2. ⏭ **Next** — implement Option A from the design doc to remove Step 2's host arithmetic. ~1 week of focused kernel work.
3. ⏸ **Future** — Option C if and only if PCIe round-trips dominate after A. Most likely needed only for repeated huge-N calls.

## Planner output (already verified)

```
N = 1,048,576 (2²⁰)        → factors=[1024, 1024]      k=2
N = 2,097,152 (2²¹)        → factors=[1024, 1024, 2]   k=3
N = 8,388,608 (2²³)        → factors=[1024, 1024, 8]   k=3
N = 67,108,864 (2²⁶)       → factors=[1024, 1024, 64]  k=3
N = 1,073,741,824 (2³⁰)    → factors=[1024, 1024, 1024] k=3
```

For any pow2 N up to 2³⁰ the planner produces at most **3 factors**.
Use `metal_example_fft_universal_xl_planner_test` to verify.

## Honest limits of Option B (today)

* **No bf16 path yet.** The dispatcher only wraps `fft_stockham` (fp32).
  Adding bf16 means wrapping `fft_universal_bf16` analogously — about 50
  lines of code, blocked only on you saying you want it.
* **Sequential outer FFTs.** Step 1 calls `fft_stockham::fft` F1 times
  back-to-back. At N=8M with F1=8 that's only 8 calls (~5-10s total);
  at N=1G with F1=1024 that's 1024 calls (tens of minutes). Option C
  is the only fix for this.
* **Host twiddle at Step 2.** ~10ns per element. At N=1G that's ~10s
  of host CPU time on top of device time. Option A fixes this.
* **Accuracy at huge N.** Same compounding bf16-multiplier issue
  documented in `fft_universal/README.md`. Expect rel err ≈ 1e-5 at
  N=8M, growing to ~1e-3 at N=1G. fp32 baseline is no longer paper-grade
  past N=8M because of FPU bf16-multiplier rounding.

## See also

* Option A design + skeleton: `option_a_pass2_xl_design.md`
* Option C design: `option_c_kstep_kernel_design.md`
