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

| Phase | N coverage                              | Path                                                        | Status |
|-------|------------------------------------------|-------------------------------------------------------------|--------|
| 1     | `N ∈ [2, 32]` (pow2, prime, composite)   | TRUE-bf16 packed direct-DFT kernel (one FPU-matmul pass)    | **done** |
| 2a    | pow2 `N ∈ [64, 1024]`                    | Two-level Cooley-Tukey = 2 × Phase-1 kernel + host twiddle  | **done** |
| 2b    | pow2 `N > 1024`                          | Either recurse through 2a or device-side twiddle kernel     | pending |
| 2c    | prime `N > 32` (Bluestein)               | Pad to pow2 M, run 2a/2b pow2 path                          | pending |
| 2c    | composite non-pow2 `N`                   | Mixed-radix Cooley-Tukey → 2a/2b pow2 path                  | pending |

`fft_universal_bf16::fft(md, signal)` throws `std::runtime_error` with a
clear "not yet implemented" message for any N not covered above. We
intentionally refuse to fall back to the fp32 path — that would silently
break the precision contract of this binary.

### Phase 2a in one paragraph

For `N = N1 × N2` with `N1 = 32` and `N2 ∈ {2, 4, 8, 16, 32}`, the
inner-outer Cooley-Tukey split gives us two independent "32 sibling
length-M DFTs along rows of a 32×M matrix" passes, which is exactly
what the Phase-1 `packed_dft_bf16` kernel already computes. Between
the two passes we apply the pointwise complex twiddle `exp(-2πi · n1 · k2 / N)`
on the host in fp32 (pointwise = no reduction, so fp32-on-host is
*more* accurate than bf16-on-device and doesn't violate the
"true bf16 compute" contract — that contract is about the reduction
path, which is both passes and stays on-device in bf16). No new
kernels, no new LLK code — just ~100 lines of host orchestration.

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
# Correctness sweep (Phase 1 sizes + Phase 2a pow2 [64, 1024] + guard checks)
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_test

# Cached-latency benchmark at N=32 (Phase 1), 100 iters
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 32 100

# Cached-latency benchmark at N=1024 (Phase 2a, two-pass CT), 100 iters
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 1024 100

# Pure-tone demo at bin 5, N=32 (Phase 1)
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 32 5

# Pure-tone demo at bin 37, N=1024 (Phase 2a)
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 1024 37
```

## Expected precision

bf16 has ~8 bits of mantissa → ULP ~4e-3 relative.

* **Phase 1** (single pass, `N ≤ 32`): rounding depth ~5 bits.
  Random-input SNR ≈ **45-55 dB** (3-5e-3 relative). Test threshold: 1e-2.
* **Phase 2a** (two passes + one host twiddle round-trip,
  `N ∈ [64, 1024]`): roughly 2× the Phase-1 rounding depth on the
  reduction path, plus one extra bf16 round-trip for the host twiddle.
  Random-input SNR ≈ **38-45 dB** (5-8e-3 relative). Test threshold: 2e-2.

If your application needs tighter precision, use `fft_universal/`
(fp32, ~1e-6 relative).

## Roadmap beyond Phase 2a

1. **Phase 2b — pow2 `N > 1024`.** Two clean options, pick whichever
   benchmarks better:
   * *Recurse* through the Phase 2a kernel: e.g. `N = 32³ = 32768` splits
     as `32 × 1024`, where the outer DFT along `n1` is a length-32 DFT
     (one more Phase-1 call) and the inner 1024-point is a Phase-2a call.
   * *Device-side twiddle* kernel (mul_tiles + add_tiles/sub_tiles on
     bf16 CBs = true-bf16 FPU elementwise mul). Avoids the PCIe bounce
     between passes. Required anyway for large `N` where the host round
     trip dominates.
2. **Phase 2c — primes (Bluestein) and composite non-pow2 (mixed-radix).**
   Both ultimately call into the Phase 2a/2b pow2 engine. The chirp/
   mixed-radix scaffolding already exists in `fft_universal_host.cpp`;
   it just needs to dispatch to the bf16 pow2 path here.

Each step is testable in isolation. See `sop.txt` for the per-phase
build/test commands.
