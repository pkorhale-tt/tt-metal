# 4. Implementation

This section walks through the four device kernels and the host
planner that ties them together. Source line numbers refer to the
released branch at `programming_examples/fft_universal/`.

## 4.1 Power-of-two paths (Stockham, contributed)

For $N$ a power of two, we delegate to the existing
`fft_stockham::fft` kernel. That kernel itself has three regimes:

* **$N \le 1024$** — single-tile Stockham. One Tensix tile holds the
  entire signal; the compute kernel runs $\log_2 N$ in-tile passes
  and emits the result in one PCIe read. Up to 64 sub-FFTs run in
  parallel one-per-core if the caller invokes `batch_fft`.
* **$1024 < N \le 65\,536$** — six-step Stockham. The signal is laid
  out as a $\sqrt N \times \sqrt N$ matrix; the algorithm performs
  two row-FFT passes separated by twiddle multiply + transpose, all on
  the device, using up to 64 cores per pass.
* **$65\,536 < N \le 2^{20}$** — four-step Stockham. Similar but with
  rectangular factors; transpose is fused into the second pass to
  avoid an explicit shuffle.

We use these kernels unmodified, so the pow-2 leg of `fft_universal`
inherits Brown et al.'s [1] published throughput and accuracy directly.

## 4.2 Packed direct-DFT kernel (new)

For $N \le 32$ the recursion or Bluestein bottoms out at sub-FFTs of
length $\le 32$. The pre-existing `batch_fft` path would compute these
by zero-padding to length 32 (or whatever next pow-2 is at least $N$)
and running a single-tile Stockham, which leaves $1 - N/32$ of every
tile row empty. At $N = 7$ that is 78 % waste; at $N = 3$, 91 %.

The packed direct-DFT kernel removes this waste by exploiting the fact
that for $N \le 32$ the **entire $N \times N$ twiddle matrix fits in a
single tile**. We can therefore replace the recursive butterfly with a
single complex matrix–vector multiplication:

$$
X = W \cdot x, \qquad W_{kn} = e^{-2\pi i k n / N},
$$

executed as one Tensix FPU matmul. The trick to packing efficiency is
that we pack **32 sub-FFTs per tile**: row $r$ of the input tile holds
the $r$-th sibling sub-FFT (zero-padded in cols $[N, 32)$), so one
matmul retires 32 independent length-$N$ DFTs.

Tile efficiency becomes $N/32$ — between 6.25 % at $N = 2$ and 100 %
at $N = 32$ — versus $\le 3$ % for the pow-2 padding path, an
8–32× improvement on the small-$N$ leaves of every Bluestein and
Cooley–Tukey call.

### Core grid selection

Sub-FFT counts at recursion leaves are arbitrary positive integers,
not powers of two (e.g. $N = 3600 \to 60 \times 60 \to 6 \times 10$
produces `count = 360` length-10 leaves). We therefore use a more
permissive core-count rule than `batch_fft`'s "round up to next
pow-2":

```
raw_num_tiles  = ceil(count / 32)             # 32 sub-FFTs per tile
if raw_num_tiles <= 7:
    num_cores = raw_num_tiles                 # 1..7 directly
else:
    num_cores = ceil(raw_num_tiles / 8) * 8   # round to multiple of 8
    num_cores = min(num_cores, 64)
```

This allows `num_cores` to take any value in
$\{1,\dots,7,8,16,24,32,40,48,56,64\}$, matching the valid
`pick_batch_grid` rectangles. Any padded zero tiles produce zero
output, which the writer kernel discards.

### Numerical considerations

The Tensix FPU is bf16-multiplier with fp32 accumulation. For the
$32 \times 32$ packed-DFT matmul we leverage the HiFi4 mode, which
decomposes a single fp32 input into 4 bf16 partials and recombines in
fp32 accumulators — yielding effectively fp32-input, fp32-output
arithmetic at 1/4 the raw matmul throughput, but at numerical fidelity
indistinguishable from fp32 on FFT inputs (verified in §5.4).

## 4.3 Bluestein chirp-z (new orchestration over device kernels)

For prime $N \ge 37$ we use Bluestein's algorithm. Define
$M = 2^{\lceil \log_2(2N-1) \rceil}$ — the smallest power of two at
least $2N - 1$. Then the length-$N$ DFT is equivalent to a length-$M$
cyclic convolution of two chirped sequences, which our engine
implements as:

1. **Host pre-multiply** ($O(N)$): multiply input by the chirp
   $w[n] = e^{-\pi i n^2 / N}$.
2. **Device length-$M$ FFT** of the chirped, zero-padded input — one
   `fft_stockham::batch_fft` dispatch.
3. **Device pointwise complex multiply** with the pre-computed
   $B = \mathrm{FFT}_M(b)$ chirp — one batched complex-mul dispatch.
4. **Device length-$M$ IFFT** — one more `batch_fft` (conjugation
   trick).
5. **Host post-multiply** by $w[n]$ and crop back to length $N$.

The chirp $w[\cdot]$ and the constant spectrum $B$ are computed once
per $N$ and cached in the `BluesteinPlan`, so per-call host cost is
**three vector multiplies, all $O(N)$**, on top of two pow-2 device
FFTs and one device pointwise multiply.

**Batched fusion.** When Bluestein appears as a sub-step of
Cooley–Tukey (e.g. $N = 21 = 3 \cdot 7$ recurses to seven length-3
sub-FFTs and three length-7 sub-FFTs at pass-1, both prime), our
implementation fuses all sibling Bluestein calls into the same
$M$-batch. Regardless of how many primes appear at any level, each
Bluestein pass collapses to exactly two device dispatches.

## 4.4 Cooley–Tukey mixed-radix (new orchestration)

For composite non-pow-2 $N$ we use the classical mixed-radix split
$N = N_1 \cdot N_2$ where $N_1$ is chosen as the largest power-of-two
divisor of $N$ (or, failing that, the smallest prime factor). The
algorithm:

1. **Reshape** input as $N_1 \times N_2$, row-major (host, $O(N)$).
2. **Pass-1**: $N_1$ sub-FFTs of length $N_2$ (device, batched).
3. **Twiddle multiply** by $e^{-2\pi i n_1 k_2 / N}$ (device, one
   batched complex-mul over $N_1 \cdot N_2$ entries; the table is
   cached in `CooleyTukeyPlan`).
4. **Pass-2**: $N_2$ sub-FFTs of length $N_1$ (device, batched), after
   an in-place transpose (host, $O(N)$).
5. **Output reshape** as $N_1 N_2$, row-major (host, $O(N)$).

Pass-1 and Pass-2 each invoke `batched_siblings_fft` recursively, so
either or both may further split into Cooley–Tukey, route through
Bluestein, or short-circuit through the packed direct-DFT kernel
depending on $N_k$. Each pass — at any depth — collapses to **one**
device dispatch over all 64 cores. Total device dispatches for any
composite $N$ is therefore $O(\log N)$, independent of the recursion
fan-out, and the count-multiplicative batching of §3.5 keeps every
dispatch saturating the full grid.

## 4.5 Host glue: planner, plan cache, per-call work

The host does **only** the following on a cached call:

* **Plan lookup.** `unordered_map<N, plan>` lookup. ~ ns.
* **Reshape / transpose.** $O(N)$ memcpy with strided indices for the
  pass-1 / pass-2 transpose. Bandwidth-bound on the host, takes
  $\sim N / 5$ ns at our measurement machine's DRAM speed.
* **Chirp / twiddle multiplies.** $O(N)$ vector complex multiplies.
  We use a tight scalar loop; SIMD vectorisation would help but is
  not yet in place — see §7 (limitations).
* **PCIe write and read** of the input / output buffers.
  Bandwidth-bound (~16 GB/s on our n300).

We classify these as "host glue" and measure their wall-time fraction
explicitly (Fig. 3 in §5). For $N \ge 16\,384$ the host fraction
drops below 20 %, validating the claim that the planner–composer
architecture is asymptotically device-bound.

## 4.6 Always-on host/device timing

To support §5 we instrument every device dispatch with a
`profile::ScopeDevice` RAII timer (≈ 25 lines of code in
`fft_universal_host.cpp`). The timer accumulates into a thread-local
`profile::Budget` struct that the benchmark and sweep binaries reset
before each call and read after. Total wall time minus accumulated
device time gives the host fraction. The overhead of the timer itself
is two `chrono::steady_clock::now()` calls per dispatch (~ 40 ns) and
is included in the host bucket — it is below noise even at $N = 2$.

## 4.7 What is upstream-reproducible

Every component above lives in
`tt_metal/programming_examples/fft_universal/` in the public
`tenstorrent/tt-metal` repository. A single shell script,
`paper/run_paper_sweep.sh`, builds two binaries, runs the sweep, and
produces the input CSVs for the figures in §5. `paper/plot_universal.py`
renders the figures.
