# fft_universal — FFT for ANY N (not just powers of two)

## Recommended operating range (measured)

| Target accuracy        | Max usable N            | Notes                                |
|------------------------|------------------------:|--------------------------------------|
| rel err ≤ 1e-6 (paper) |  **1,048,576 (2²⁰, 1M)** | Verified clean across 64K-1M sweep. 1M = 58 ms cached. |
| rel err ≤ 1e-3         |  same; degrades with N  | Path is fp32 throughout, no soft cliff. |
| Pow2 N ≥ 2,097,152 (2M)|     **NOT SUPPORTED**   | Stockham planner asserts `sub_N <= 1024`. |
| Composite N ≥ ~10M     |   numerically broken    | rel err ≈ 0.46 at N=10M (chained mixed-radix on bf16-multiplier FPU). |

For N > 1M, switch to `fft_universal_bf16/` — it handles up to ~2M
with rel err ≈ 2e-2 (normal bf16) and doesn't hit the planner cap.

---

`fft_universal::fft(md, signal)` computes the DFT of a complex signal of any
length N >= 2. Internally it reuses `fft_stockham/` for pow2 sub-FFTs of
length ≥ 64 and adds ONE new device kernel — the **packed direct-DFT
kernel** under `./kernel/` — for every sub-FFT of length ≤ 32.

## How it decides what to do

```
                ┌──────────────────────────────┐
                │   fft_universal::fft(x, N)   │
                └──────────────┬───────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ N == 1                → return x             │
        │ N is a power of two   → fft_stockham::fft    │
        │ N is composite        → Cooley–Tukey split   │
        │   (N = N1 · N2)         recurse on N1 and N2 │
        │ N is prime (>= 3)     → Bluestein (chirp-z)  │
        │                          pads to pow2 M,     │
        │                          one pow2 FFT + IFFT │
        │                                               │
        │ Every leaf of length ≤ 32 (pow2 or not, prime │
        │ or composite) short-circuits through the      │
        │ PACKED DIRECT-DFT kernel — 32 sub-FFTs per    │
        │ tile, complex 32×32 matmul, ONE dispatch.     │
        └─────────────────────────────────────────────┘
```

**Every non-trivial compute path executes on Wormhole.** The host only does
reshape / twiddle / transpose / dispatch logic.

* **Power-of-two (incl. small: 2, 4, 8, …)**: direct hand-off to the tuned
  device kernel — zero overhead.
* **Composite non-pow2**: split N = N1 · N2 (biggest pow2 factor × odd rest,
  or smallest prime × rest), recurse, twiddle-multiply, transpose, recurse.
  The engine operates on **batches** of sibling signals at every level — a
  split of an N1 × N2 problem generates `count · N1` siblings for pass-1 and
  `count · N2` siblings for pass-2, so **batching width GROWS with recursion
  depth**. Each sub-FFT pass collapses to exactly ONE `fft_stockham::batch_fft`
  dispatch across all 64 Tensix cores. Total device dispatches for a
  composite N is O(log N) — independent of how many sibling sub-FFTs live at
  any level.
* **Prime N ≥ 3**: Bluestein's algorithm turns the length-N DFT into a
  length-M cyclic convolution (M = next pow2ᵢ ≥ 2N-1), which is 1 forward FFT +
  1 inverse FFT on the device (both pow2 → `fft_stockham::batch_fft`). When
  Bluestein shows up as a sub-step of Cooley-Tukey with K sibling calls, all K
  pre-multiplies, K forward FFTs, K pointwise multiplies, K inverse FFTs, and
  K post-multiplies are fused: still exactly 2 device dispatches, regardless
  of K.

## Algorithm summary

| Stage            | Where it runs     | Why                                        |
|------------------|-------------------|--------------------------------------------|
| Dispatch         | host              | pure decision logic, negligible cost       |
| Cooley–Tukey     | host + device     | host reshape + twiddle, device sub-FFTs    |
| Bluestein        | host + device     | host chirps/multiplies, device does 2 FFTs |
| Every actual FFT | device (1–64 cores) | `fft_stockham::fft` (handles pow2 up to 1M) |

Plans are **cached** so second and later calls for the same N skip all
host-side prep:

* **`PackedDFTPlan`** (Opt #5) — keyed on `(N, count)`; holds the `N×N`
  twiddle tile (plus its negated-imag mirror for sign-free accumulation)
  and reusable host/device scratch buffers. Every sub-FFT leaf of length
  ≤ 32 goes through this plan in one dispatch.
* **`BluesteinPlan`** — keyed on `N`; holds the chirp `w[n]` and pre-computed
  `B_fft = FFT_M(b_ext)`. Only used for prime N ≥ 37 now (smaller primes
  go through the packed DFT).
* **`CooleyTukeyPlan`** — keyed on `(N1, N2)`; holds the `N1·N2` twiddle table
  `exp(-2πi · n1 · k2 / N)`. Removes all `cos/sin` from the per-iter hot path.
* **`fft_stockham`'s batch_fft / pass-2 / Stockham plans** — inherited via
  delegation (program builds + twiddle DRAM buffers cached per distinct
  `(sub_N, batch)`).

## API

```cpp
#include "fft_universal/fft_universal_host.cpp"

auto md = MeshDevice::create_unit_mesh(0);
std::vector<std::complex<float>> X = fft_universal::fft(md, signal);
```

## See also

* `fft/` — inner radix-2 kernel (L1-resident up to N=65,536).
* `fft_bf16/` — bf16 storage variant of the inner kernel.
* `fft_stockham/` — 4-pass Stockham orchestrator for power-of-two N > 65,536.
