# 7. Conclusion and limitations

We presented `fft_universal`, an FFT implementation for Tenstorrent's
Wormhole accelerator that accepts arbitrary transform lengths $N \ge 2$
by routing each call through one of four cached device paths
(packed direct DFT, Stockham, Bluestein, Cooley–Tukey). We
contributed a new packed direct-DFT device kernel that improves
small-$N$ tile occupancy from $<3$ % to $6$–$100$ %, eliminating the
dominant inefficiency in Bluestein and mixed-radix recursion. We
measured end-to-end latency, sustained GFLOP/s, and host overhead
across six decades of $N$, and reproduced the prior single-Tensix
result of Brown et al. on the same hardware.

## 7.1 Limitations and disclosed honest issues

* **Host glue is single-thread scalar.** The chirp-/twiddle-multiplies
  and the pass-1 / pass-2 transposes are written as plain C++ loops
  and account for the majority of host wall time at $N \le 8192$.
  A SIMD or threaded host-side implementation would push the
  device-bound crossover lower; we leave this to future work.
* **Pow-2 ceiling is $2^{20}$**. The Stockham planner currently asserts
  `sub_N <= 1024`, capping pow-2 $N$ at $2^{20}$ in fp32. The
  bf16-storage variant in `fft_universal_bf16/` extends this to
  $\sim 2^{21}$ at lower numerical fidelity.
* **Composite $N \ge 10^7$** suffers numerical degradation
  (relative error $\sim 0.5$ at $N = 10^7$) because the chained
  mixed-radix recursion accumulates twiddle error on a bf16-multiplier
  FPU. fp32-multiplier hardware (Tenstorrent's Blackhole) would lift
  this limit.
* **Single ASIC only.** The n300 card has two Wormhole ASICs; we
  measure on one. Scaling Bluestein and Cooley–Tukey across the
  chip-to-chip link is a natural next step.

## 7.2 Future work

The packed direct-DFT idea generalises beyond $N \le 32$: any $N$
divisible by 32 can be expressed as a fused tile-batched matmul,
which would extend the new kernel's reach into the range currently
handled by single-tile Stockham. We are also exploring a
real-input variant that exploits Hermitian symmetry to halve PCIe
traffic, which is the dominant cost in our small-$N$ regime.
