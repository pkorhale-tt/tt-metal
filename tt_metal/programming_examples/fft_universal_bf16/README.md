# fft_universal_bf16 — TRUE-bf16 FFT on Wormhole

Sibling of `fft_universal/` that keeps bf16 on the compute path end-to-end.
Every multiply that happens on the Tensix is a `bf16 × bf16 → fp32` FPU
matmul — no SFPU butterflies, no "bf16 storage with fp32 compute" detours.
The only fp32 rounding points are the host-side pack at input and the
pack-tile at output; everything between DRAM, CBs, and the FPU operand
path stays bf16.

## Why a separate binary?

`fft_bf16/` uses bf16 CBs but routes compute through the SFPU via
`UnpackToDestFp32 + fp32_dest_acc_en`. That means the butterfly math
actually runs in fp32 inside the SFPU — you only save DRAM bandwidth,
not compute. On Wormhole the SFPU has no native bf16 multiplier, so
**true bf16 compute is only reachable via FPU matmul**. That forces the
FFT itself to be expressed as matrix multiplies, which is what this
binary does.

## Phase status

| Phase | N coverage                    | Kernel                              | Status |
|-------|-------------------------------|-------------------------------------|--------|
| 1     | `N ∈ [2, 32]` (pow2, prime, composite) | TRUE-bf16 packed direct-DFT (this repo) | **done** |
| 2     | pow2 `N > 32`                 | radix-32 Stockham bf16 matmul       | pending |
| 2     | prime `N > 32`                | Bluestein → Phase 2 pow2 path       | pending |
| 2     | composite `N > 32`            | Cooley-Tukey → Phase 2 pow2 path    | pending |

Today, calling `fft_universal_bf16::fft(md, signal)` with `N > 32`
throws `std::runtime_error` with a clear "Phase 2 not yet implemented"
message. We intentionally refuse to fall back to the fp32 path — that
would silently break the precision contract of this binary.

## Layout

```
fft_universal_bf16/
├── CMakeLists.txt
├── README.md
├── sop.txt
├── fft_universal_bf16_host.cpp              # host library + dispatcher
├── fft_universal_bf16_test.cpp              # correctness vs double-precision DFT
├── fft_universal_bf16_benchmark.cpp         # cached vs cold timing
├── fft_universal_bf16_demo.cpp              # minimal pure-tone example
└── kernel/
    ├── packed_dft_bf16_common.h             # CB IDs, tile-size constants
    ├── packed_dft_bf16_reader.cpp           # BRISC0: stream A/B pairs
    ├── packed_dft_bf16_compute.cpp          # TRISC: 4 matmul_tiles per output
    └── packed_dft_bf16_writer.cpp           # BRISC1: flush output CBs to DRAM
```

## How the bf16-ness is achieved on device

1. Host prepares fp32 twiddles / input → tilizes → converts to bf16
   (2 B/element) → ships via `MeshBuffer`.
2. CBs declared `DataFormat::Float16_b` with 2048 B tile size (vs 4096 B
   for fp32). `CircularBufferConfig(...)` in the host file makes this
   explicit.
3. Compute kernel configured with `math_fidelity=HiFi4`,
   `fp32_dest_acc_en=true`, **no** `unpack_to_dest_mode` (that would
   break matmul — see comment in `packed_dft_bf16_compute.cpp`).
4. `mm_init(CB_A, CB_B, CB_OUT_R)` + `matmul_tiles(...)` — bf16 srcA ×
   bf16 srcB accumulates into fp32 DST.
5. `pack_tile(0, CB_OUT_*)` packs fp32 DST down to bf16 output CB —
   the only bf16 rounding on the compute path.

## Build

```bash
ninja -C build \
    metal_example_fft_universal_bf16_test \
    metal_example_fft_universal_bf16_benchmark \
    metal_example_fft_universal_bf16_demo
```

## Run

```bash
# Correctness sweep (all Phase 1 sizes + Phase 2 guard check)
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_test

# Cached-latency benchmark at N=32, 100 iters
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 32 100

# Pure-tone demo at bin 5, N=32
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 32 5
```

## Expected precision

bf16 has ~8 bits of mantissa → ULP ~4e-3 relative. For a length-N DFT
with `N ≤ 32` we accumulate N `bf16 × bf16 → fp32` products, so
rounding depth stays at O(log N) ≈ 5 bits. Random-input SNR lands
around **40-45 dB** (3-5e-3 relative error), and the test binary uses
1e-2 as its pass threshold.

If your application needs tighter precision, use `fft_universal/` (fp32,
~1e-6 relative).

## Roadmap to Phase 2

1. **radix-32 Stockham bf16 matmul kernel** for pow2 N > 32. A length-32²
   FFT becomes two 32-point DFT stages connected by a twiddle-scaled
   matmul (fold the per-pass twiddles into the W_32 matrix before the
   second multiply so the pointwise twiddle step stays on the FPU, not
   the SFPU). Recurse for N = 32³, 32⁴.
2. **bf16 pass-2 (four-step) kernel** for N > 1024 following the same
   matmul-only discipline.
3. **Bluestein / Cooley-Tukey delegation** — no new kernels needed
   beyond the pow2 engine.

Each step is testable in isolation. See `sop.txt` for the per-phase
build/test commands.
