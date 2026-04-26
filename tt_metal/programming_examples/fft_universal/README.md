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
  Every leaf of the recursion lands on a device dispatch. **When either side
  of the split is a power of two ≤ 1024**, that entire pass (up to thousands of
  sibling sub-FFTs) is fused into a single device dispatch via
  `fft_stockham::batch_fft`, which fans across all 64 Tensix cores — typically
  a 5–20× speedup over per-row dispatches on cases like `N = 1024·k`.
* **Prime N ≥ 3**: Bluestein's algorithm turns the length-N DFT into a
  length-M cyclic convolution (M = next pow2ᵢ ≥ 2N-1), which is 1 forward FFT +
  1 inverse FFT on the device (both pow2 → `fft_stockham`). Example: N=3 →
  two length-8 device FFTs; N=101 → two length-256 device FFTs.

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
