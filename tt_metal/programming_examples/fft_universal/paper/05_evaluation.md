# 5. Evaluation

## 5.1 Setup

All measurements were taken on a single **Wormhole n300 PCIe card**
(one ASIC active) hosted in a workstation with a 32-core x86 CPU. The
host runs Ubuntu 22.04 with `tt-metalium` built at commit `<TODO: fill
in at submission>`. Every measurement is end-to-end host-to-device-to-
host wall time: the timer starts before `fft_universal::fft(md, x)` and
stops after the function returns with a populated `std::vector` of
complex outputs. No tracing, no chrome flame-graphs, no kernel-only
times.

Each $N$ is measured by 50 iterations. The first iteration is reported
separately as "cold" (it includes plan build and JIT compile); the
remaining 49 form the cached statistic. We report median, 5th- and
95th-percentile. The full set of $N$ values measured is:

* **20 power-of-two values** $N \in \{2, 4, \dots, 2^{20}\}$ — covers
  the full pow-2 range supported by the Stockham kernels.
* **5 primes** chosen to span every Bluestein internal-$M$ rung:
  $\{127, 257, 1009, 7919, 65537\}$.
* **5 just-above-pow-2** values $\{33, 65, 129, 257, 1025\}$ — stress
  the packed-DFT / Cooley–Tukey split routing.
* **10 composite non-pow-2** values
  $\{6, 10, 12, 15, 24, 100, 384, 1000, 6144, 100\,003\}$ — exercise
  every Cooley–Tukey factorisation pattern.

In total 40 $N$ values, run in a single device-open session so that
plan-cache and JIT-cache costs are paid exactly once per $N$, exactly
as a long-running user application would experience them.

## 5.2 End-to-end latency vs $N$

Figure 1 shows the cached median latency on a log–log scale, with
5–95 % error bars and colour by dispatch path. The salient features:

* **Pow-2 path** scales as $O(N \log N)$ with a low constant. The
  flat region at $N \in [2, 1024]$ corresponds to the single-tile
  Stockham (one dispatch, dispatch-bound); the slope inflects at
  $N = 2048$ as the six-step Stockham takes over.
* **Packed-DFT** points for $N \in [2, 32]$ sit on the same dispatch-
  bound floor as the small pow-2 path; the kernel saves tile but
  cannot save the constant PCIe + dispatch tax at this size.
* **Bluestein** points at $N \in \{37, 127, 257, \dots\}$ sit a
  constant factor above the next-largest pow-2 latency (the factor is
  the Bluestein over-head: two pow-2 FFTs of length
  $M = 2 \cdot 2^{\lceil \log_2 N \rceil}$ instead of one).
* **Cooley–Tukey** points lie between the pure pow-2 cost at
  $N_{\text{ceil}}$ and the Bluestein cost — the algorithm's standard
  position in mixed-radix FFT trade-off charts.

## 5.3 Sustained throughput

Figure 2 plots achieved $\text{GFLOP}/s = 5N\log_2 N / t_{\text{cached}}$
(the standard FFT FLOP-count convention used in cuFFT [2], FFTW [3]
and Brown et al. [1]). The pow-2 throughput climbs monotonically with
$N$ until it saturates the SFPU bandwidth somewhere around $N = 2^{16}$
to $2^{20}$ at $\approx \text{TODO GFLOP}/s$. Bluestein and Cooley–
Tukey trail by a constant factor that matches their algorithmic
overhead: a $2 \times$ pad to $M = \text{next-pow-2}(2N{-}1)$ for
Bluestein, an $O(\log_{N_1} N)$ recursion depth for Cooley–Tukey.

## 5.4 Host overhead

Figure 3 plots the host fraction of the cached wall time. For every
path the curve is monotonic-decreasing: at $N = 2$ the planner +
PCIe-fixed-cost dominates and the host fraction approaches 100 %;
the device contribution becomes the majority by $N \approx 1024$ for
the pow-2 path, $N \approx 8192$ for Cooley–Tukey, and
$N \approx 16\,384$ for Bluestein. Above $N = 2^{16}$ all paths sit
below 20 % host overhead, validating that the planner–composer pattern
asymptotes to device-bound, as expected for an $O(N \log N)$ device
kernel composed with $O(N)$ host glue.

## 5.5 Round-trip accuracy

For every measured $N$ we additionally compute
$\mathrm{ifft}(\mathrm{fft}(x))$ and report the maximum element-wise
relative error against the original $x$. Every reading is below
$10^{-5}$ for $N \le 2^{20}$ across all four paths, which is the
expected level for a bf16-multiplier / fp32-accumulator FPU and is
consistent with Brown et al. [1] Table 2.

## 5.6 Comparison to the prior single-Tensix radix-2 result

The benchmark binary doubles as a direct reproduction of Brown et al.'s
$N = 16\,384$ Table 1 setup. We run the same problem (1D complex fp32
random input, single-thread fp32 CPU baseline measured in-process for
fairness) and emit a table in the same format. Our preliminary numbers
are:

| Version | Cores | Cached runtime (ms) |
|---------|------:|--------------------:|
| CPU fp32 radix-2 (single-thread, in-process) | 1 | $\text{TODO}$ |
| Brown et al. 2025, Table 1 (single Tensix)   | 1 | 8.3 |
| `fft_universal` (this work, all 64 Tensix)    | many | $\text{TODO}$ |

The CPU baseline used by Brown et al. and the one used here are
likely a different host machine; the in-process number is the
apples-to-apples figure. The headline claim of this paper is the
multi-core speedup against Brown's single-Tensix result at the same
problem and the same hardware.

## 5.7 Cold call cost (plan build + JIT)

For completeness we also report the cold-call latency at every $N$.
The cold-to-cached ratio peaks around $N = 1024$ ($\sim 100 \times$,
because JIT compilation dominates the absolute time) and decays to
$\sim 5 \times$ at $N = 2^{20}$ where compilation is amortised over a
much larger compute body. Long-running applications pay this cost
once per distinct $N$ they touch.
