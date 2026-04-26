# fft_universal — FFT for ANY N (not just powers of two)

`fft_universal::fft(md, signal)` computes the DFT of a complex signal of any
length N >= 2. Internally it reuses the existing power-of-two kernels from
`fft/` and `fft_stockham/` — no new device kernels are introduced.

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

Plans (chirp tables, B_fft for Bluestein, factorizations) are **cached**,
so second and later calls for the same N skip all host-side prep.

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
