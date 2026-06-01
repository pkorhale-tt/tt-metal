# 2. Background: Wormhole and `tt-metalium`

## 2.1 The Wormhole accelerator

Wormhole is Tenstorrent's second-generation accelerator. The n300 PCIe
card carries two Wormhole ASICs joined by a high-bandwidth chip-to-chip
link; this paper uses a single ASIC (one device in `tt-metalium`
parlance). Each ASIC exposes an 8 × 8 grid of Tensix worker cores
(64 cores in total). Each Tensix is itself a small heterogeneous SoC:

* **Five RISC-V cores per Tensix.** Two are *data-movement* RISC-Vs
  ("BRISC0/1") used as the reader and writer kernels for NoC traffic
  in and out of the core. Three are *compute* RISC-Vs ("TRISC0/1/2")
  that drive the FPU and SFPU through micro-ops.
* **One FPU.** A 32 × 32 matrix engine with **bf16 multiplier and fp32
  accumulator** [4]. This is the unit our packed direct-DFT kernel
  targets: the small-$N$ DFT becomes a complex 32 × 32 matmul that
  retires in one tile.
* **One SFPU.** A 32-lane vector unit with native fp32 — used for
  twiddle multiplication in the multi-pass Stockham kernel.
* **~1.5 MB of L1 SRAM.** Code, kernel-args, and circular buffers
  live here. There is no shared cache across Tensix cores; data is
  exchanged through the NoC into another core's L1.

Cores communicate through the NoC and synchronise through **circular
buffers** (CBs): per-core L1 SRAM ring buffers indexed by 32 × 32 fp32
tiles (4 096 bytes). A `cb_push`/`cb_pop` pair is the only handshake
between the reader, compute, and writer kernels.

## 2.2 The `tt-metalium` runtime

Programs are expressed in `tt-metalium` as a host-side **program
descriptor** plus a set of per-core kernel source files. A program
descriptor specifies, for every core in the grid, which reader / writer
/ compute kernels run, what runtime arguments they receive, and which
CBs are bound. Once compiled, a program is **dispatchable** via
`EnqueueMeshWorkload` and the JIT-compiled binary is cached for the
process lifetime, so cached calls pay only PCIe write + dispatch +
PCIe read.

Dispatch is asynchronous: `EnqueueMeshWorkload` returns immediately.
Synchronisation with the host happens implicitly through blocking
reads (`ReadShard(..., blocking=true)`); explicit fences are also
available via `Finish()`. The user-visible cost of a single
"FFT of length $N$" call therefore decomposes as:

$$
t_{\text{wall}} \;=\; t_{\text{host plan}} + t_{\text{HtoD copy}} + t_{\text{dispatch}} + t_{\text{device compute}} + t_{\text{DtoH copy}} + t_{\text{host glue}},
$$

where on a **cached** call $t_{\text{host plan}}$ is amortised to zero
because the plan and the JIT'd binary live in process memory. §5.3
measures the (device) and (everything-else) components separately.

## 2.3 Why a power-of-two-only FFT is not enough

Brown et al. [1] reported the first FFT on Wormhole using a
single-Tensix radix-2 Cooley–Tukey decimation-in-time at $N = 16384$;
their Table 1 measures 8.3 ms on Wormhole versus 2.9 ms on a
single-thread fp32 CPU baseline (2.8× CPU advantage for end-to-end
single-call latency). Two important limitations of that result are:

1. **Power-of-two only.** Radix-2 cannot directly handle prime,
   $2k+1$, or any non-pow-2 $N$. Real workloads — radar, MRI, OFDM,
   astronomy — routinely require such lengths.
2. **Single-core.** The reported result uses one Tensix out of the
   available 64. Brown et al. note this and present a separate
   "Table 3" sketch for multi-core, but do not publish a numerical
   multi-core comparison.

Our work extends the Wormhole FFT story along both axes
simultaneously: any $N$, every core, every call.
