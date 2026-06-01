# 1. Introduction

The Fast Fourier Transform (FFT) is the most studied
transform in scientific computing and remains the kernel that decides
whether a new accelerator is viable for signal-processing,
communications, image-reconstruction, and PDE workloads. Every
established accelerator — modern GPUs (cuFFT [2], rocFFT), CPUs
(FFTW [3], MKL), Cerebras, Habana, Graphcore — ships an FFT library
that handles **arbitrary** $N$, because real workloads do not, in
general, choose powers of two: radar pulse-Doppler is sized by PRF,
MRI by acquisition matrix, 5G OFDM by sub-carrier count.

Tenstorrent's Wormhole accelerator [4] is a 64-core (8 × 8 Tensix
grid) RISC-V-driven SIMD/SIMT machine with a bfloat16-multiplier /
fp32-accumulator FPU, a vector SFPU, and ~1.5 MB of L1 per core. It
is programmed directly through `tt-metalium`, an open C++ runtime that
exposes circular buffers, NoC primitives, and per-core kernels with
none of the abstraction overhead of CUDA's stream/event machinery.
The recent paper by Brown, Davies, and Le Clair [1] gave the first
published FFT result on Wormhole — a single-Tensix radix-2 Cooley–Tukey
implementation that achieves a 2.8× advantage over a single-thread
CPU at $N = 16384$ — but its scope is **power-of-two only**.

This paper closes that gap. We present `fft_universal`, an open
implementation built on `tt-metalium` that:

1. **Accepts any $N \ge 2$** in a single `fft(x)` entry point, by
   routing through four device kernels selected by a small host
   planner (§3).
2. **Adds a new packed direct-DFT device kernel** that computes 32
   sub-FFTs of length $\le 32$ per Tensix tile as a complex
   $32 \times 32$ matmul, replacing the wasteful pow-2 padding path
   for all small-$N$ leaves of the prime- and composite-$N$ algorithms
   (§4.2). Tile occupancy on those leaves jumps from $< 3\%$ to
   between 6 % and 100 %.
3. **Batches every recursive level**. A composite-$N$ split
   $N = N_1 \cdot N_2$ generates $\text{count} \cdot N_k$ sibling
   sub-FFTs at pass $k$; each pass collapses to **one**
   `batch_fft` dispatch across all 64 cores, so the device-dispatch
   count is $O(\log N)$ independent of recursion fan-out (§4.3).
4. **Discloses host overhead.** Every reported latency includes a
   host-percentage column measured by always-on `std::chrono` markers
   around every device dispatch. The planner–composer pattern matches
   how FFTW [3] and cuFFT [2] are commonly used (per-$N$ cached plan,
   per-call host glue for packing/twiddle look-up); we make this
   structurally explicit so readers can compare apples to apples.

Our evaluation (§5) sweeps $N$ across six decades on a single Wormhole
n300 card and reports:

* end-to-end median latency, with 5–95 % bands (Fig. 1);
* achieved sustained $\text{GFLOP}/s = 5N\log_2 N / t$, the Brown / cuFFT
  / FFTW convention (Fig. 2);
* host glue as a fraction of wall time, validating that the planner
  cost amortises to under 20 % above $N \approx 2^{14}$ (Fig. 3);
* a direct reproduction of Brown et al.'s $N = 16384$ Table 1 setup
  on the same hardware, in-process, with cached vs.\ cold timing
  separated (Tab. 2).

To our knowledge this is the first published FFT result on a
Tenstorrent accelerator that covers arbitrary $N$, the first that uses
all 64 Tensix cores per dispatch, and the first to publish a host /
device cost split for any FFT implementation on this hardware. The
implementation is upstream and reproducible: a single shell script
produces every figure in the paper.
