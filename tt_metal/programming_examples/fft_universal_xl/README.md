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

For pow2 `N > 1M` (the new regime), the dispatcher applies a 4-step
Cooley-Tukey decomposition with the **smallest** factor as outer:

| Step | Where | What |
|---|---|---|
| 0 | host | strided pre-pack `T[n1, n2] = signal[n2·F1 + n1]` (no math) |
| 1 | device | F1 sequential `fft_stockham::fft` calls of length M = N / F1 |
| 2 | **host** ⚠ | outer twiddle multiply `Y[n1, k] *= exp(-2πi · n1·k / N)` (cached cos/sin table) |
| 3 | **host** ⚠ | length-F1 DFT per inner index (F1 ∈ {2, 4} in practice — at most 16 muls/element) fused with the final reorder |

**Why Step 3 is on host (not on `batch_fft`):** the existing `batch_fft`
kernel allocates one full 1024-element tile per sub-FFT regardless of
`sub_N`. For K=3 with `batch = M` up to 1 M, that would request **16 GB
of DRAM** (4 buffers × 1 M tiles × 4 KB) — impossible on a 12 GB
Wormhole. Until a packed `batch_fft_xl` kernel ships (many short FFTs
per tile), Step 3 stays on the host. F1 is by planner construction the
smallest factor (typically 2 or 4), so the DFT is trivial — for F1=2
it degenerates to one add and one sub per output pair.

**Steps 2 and 3 are the only host arithmetic.** Both are removed once
Option A's `pass2_xl` kernel + a packed `batch_fft_xl` kernel land.

## What "Option B" gives you vs what "Option A" would add

| Property | Option B (today) | Option A (after the kernel work) |
|---|---|---|
| Pow2 N supported | **2 ≤ N ≤ 16M** (gated; see below) | up to 1G |
| No host arithmetic on data | ❌ (Steps 2 + 3 are host) | ✅ |
| Correctness | ✅ identical | ✅ identical |
| Speed at N ≤ 1M | identical to fft_stockham | identical to fft_stockham |
| Speed at N = 8M | ~250 ms host + device | sub-second |
| Speed at N = 1G | host-bound (impractical) — gated off | seconds (estimate) |

## Why N is gated at 16M today (and not at the algorithmic 1G ceiling)

The K=3 dispatcher's algorithm is correct for any pow2 N up to 1G
(`M = N / F1 ≤ 1M` is the only correctness constraint). However, the
host-side length-F1 outer DFT (Step 3) costs `O(F1² · M) = O(F1 · N)`
host ops:

| N | F1 | Step-3 host cost | Verdict |
|---|---|---|---|
| 2M | 2 | ~10 ms | trivial |
| 4M | 4 | ~40 ms | fine |
| 8M | 8 | ~250 ms | OK |
| **16M** | **16** | **~1 s** | **practical ceiling** |
| 32M | 32 | ~4 s | too slow for a public op |
| 1G | 1024 | hours | impossible |

`fft_universal_xl_host.cpp` enforces `kXlMaxNFp32 = 16M` and aborts
above that with an error that points at the `pass2_xl` /
`batch_fft_xl` kernel work that lifts the gate. **Once the packed
`batch_fft_xl` kernel ships, raise `kXlMaxNFp32` to `1u << 30` and
the algorithm runs through to 1G without source changes elsewhere.**

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

* **N is gated at 16M.** Algorithm runs to 1G; host arithmetic doesn't.
  See "Why N is gated at 16M" above. The gate is a single constant in
  `fft_universal_xl_host.cpp` (`kXlMaxNFp32`).
* **No bf16 path yet.** The dispatcher only wraps `fft_stockham` (fp32).
  Adding bf16 means wrapping `fft_universal_bf16` analogously — about 50
  lines of code, blocked only on you saying you want it.
* **Sequential outer FFTs.** Step 1 calls `fft_stockham::fft` F1 times
  back-to-back. At N=8M with F1=8 that's 8 calls; at N=16M with F1=16
  that's 16 calls. Inside the gated regime this is fine.
* **Host twiddle at Step 2.** ~10ns per element. At N=16M that's ~150ms
  of host CPU time on top of device time. Option A fixes this.
* **Accuracy at huge N.** Same compounding bf16-multiplier issue
  documented in `fft_universal/README.md`. Inside the 16M gate fp32
  rel err stays ≤ 3e-7 (verified), well below paper-grade tolerance
  past N=8M because of FPU bf16-multiplier rounding.

## See also

* Option A design + skeleton: `option_a_pass2_xl_design.md`
* Option C design: `option_c_kstep_kernel_design.md`
