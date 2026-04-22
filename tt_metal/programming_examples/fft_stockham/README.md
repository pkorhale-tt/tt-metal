# FFT — Stockham (multi-pass) extension to N > 65,536

This folder is a **thin orchestrator** on top of `programming_examples/fft/`.
It implements the classic four-step Cooley-Tukey / Bailey / six-step Stockham
factorisation so we can run FFTs whose size exceeds what a single pass through
the inner radix-2 kernel can handle (currently `N <= 65,536`, set by the
`P = N / 1024` core-count rule and the 64-core Wormhole worker grid).

## What it adds over `fft/`

| | `fft/` (radix-2 single-pass)         | `fft_stockham/` (this folder)             |
|---|---|---|
| Max `N`                | 65,536                              | host-memory limited (tested to 1,048,576) |
| Inner algorithm        | Cooley-Tukey radix-2 DIT            | reuses the same kernel as the inner FFT   |
| DRAM round-trips       | 2 (in + out)                        | 2 *per pass* — total 4 for the full FFT   |
| Cross-core traffic     | log₂P pairwise butterflies in L1    | none in the orchestrator (all in inner)   |
| Twiddle / transpose    | n/a                                 | host-side fused multiply + transpose      |
| Plan cache             | yes (per inner `N`)                 | inherits the inner kernel's plan cache    |

## Algorithm in 4 steps

For `N = N1 * N2` with both halves ≤ 65,536:

1. Reshape input as a `(N1, N2)` row-major matrix.
2. **Pass 1**: FFT every row (length `N2`).  → `N1` sub-FFTs, each via the
   existing radix-2 kernel.
3. **Pass 2 (host)**: per-element twiddle multiply by `W_N^{i*j}` and
   transpose to `(N2, N1)` row-major.
4. **Pass 3**: FFT every row (length `N1`).  → `N2` sub-FFTs.
5. Reorder to natural 1D output: `X[k] = D[k % N2, k / N2]`.

## Public API

Drop-in replacement for `fft_example::fft`:

```cpp
#include "fft_stockham_host.cpp"

auto X = fft_stockham::fft(md, signal);   // any power-of-two N >= 2
```

For `N <= 65,536` the call short-circuits to the inner radix-2 kernel —
zero overhead vs calling `fft_example::fft` directly.

## Building and running

```bash
ninja -C build metal_example_fft_stockham_test
ninja -C build metal_example_fft_stockham_benchmark
ninja -C build metal_example_fft_stockham_demo

./build/programming_examples/fft_stockham/metal_example_fft_stockham_test
./build/programming_examples/fft_stockham/metal_example_fft_stockham_benchmark 1048576 50
./build/programming_examples/fft_stockham/metal_example_fft_stockham_demo
```

## MVP caveats and obvious follow-ups

This is a working MVP. Two clear optimisation directions:

1. **Device-side batch FFT** — today each sub-FFT is a separate enqueue, so
   the orchestrator pays N1+N2 worth of host-overhead per call. A single
   batch dispatch where 64 cores each run their own sub-FFT in parallel
   would amortise that to one enqueue per pass.
2. **Device-side twiddle + transpose kernel** — Pass 2 is currently a host
   loop. A BRISC-only data-movement kernel (read tile → multiply by per-
   element twiddle → write to transposed DRAM page) would move this onto
   the chip and remove the host bottleneck.

Together these would bring N=1M cached latency from the current ~hundreds
of ms down to well under 10 ms.

## Why we wrote this

The single-pass radix-2 kernel hits its hard wall at `N = 65,536` because
`P = N / 1024 = 64` already saturates the Wormhole worker grid. Stockham
is the standard way past that wall — it's what cuFFT / FFTW / Brown et al.
all use for "too big for one shared-memory pass" sizes — and importantly
it preserves L1-residency for the actual butterfly work: every sub-FFT in
both passes still runs entirely inside L1 with the same fast pairwise
cross-core butterflies. The DRAM traffic is bounded to 2 round-trips per
pass, not one per stage.
