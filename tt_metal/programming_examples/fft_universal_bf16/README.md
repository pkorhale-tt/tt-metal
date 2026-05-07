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

## Coverage

The `fft_universal_bf16::fft(md, signal)` dispatcher now handles **any
N ≥ 2**:

| N                                              | Path                                                          |
|------------------------------------------------|---------------------------------------------------------------|
| `N ∈ [2, 32]` (pow2, prime, composite)         | Phase 1: TRUE-bf16 packed direct-DFT (1 FPU matmul pass)      |
| pow2 `N > 32`                                  | Phase 2b: recursive 2-level CT with `N1 = 32`                 |
| composite non-pow2 `N > 32` with a divisor ≤ 32 | Phase 2b: mixed-radix CT with `N1 = largest divisor ≤ 32`    |
| prime `N > 32`                                 | Phase 2c: Bluestein's chirp-Z, `M = next_pow2(2N-1)` → 2b     |
| composite with no divisor ≤ 32 (e.g. `37²`)    | Phase 2c: Bluestein on `N` itself                             |

Every reduction path funnels through the same `packed_dft_bf16` kernel
on device (FPU bf16 matmul). The between-pass twiddle multiply runs on
the host in fp32 — pointwise, no reduction, so fp32 there is strictly
more accurate than bf16 would be.

### Why Pass-2 is on the host (and what would move it on-device)

We tried the obvious "Plan A" — dispatch the twiddle multiply through
`fft_stockham`'s on-device `pass2_compute` (SFPU fp32, 64 cores). It
**regressed performance** (4.06 ms → 7.31 ms at N=16384) because the
bf16 dispatcher already returns Pass-1's output to the host before
Pass-3 starts. Adding an on-device Pass-2 in the middle inserts an
**extra** WriteShard + ReadShard pair per call, not a removed one.

For Pass-2 to actually win on this path, the bf16 dispatcher needs to
keep intermediate buffers in DRAM across passes (no readback between
Pass-1 and Pass-3). That's a deeper refactor of the multi-pass plumbing
— see SOP "device residency for bf16 multi-pass". Until then, host
Pass-2 is the faster path.

## How the dispatch tree works

```
fft(N)
├── N ≤ 32         → packed_dft_bf16_batched(N, count=1)                   [Phase 1]
├── pow2(N)        → two_level_fft_bf16(N, N1=32)                          [Phase 2b pow2]
├── ÷≤32 exists    → two_level_fft_bf16(N, N1=largest_divisor_le_32(N))    [Phase 2b mixed]
└── otherwise      → bluestein_fft_bf16(N)                                 [Phase 2c]

two_level_fft_bf16(N, N1):
    pass-1: N1 sibling length-(N/N1) FFTs     (recurse via fft() if > 32)
    host-side fp32 twiddle + transpose
    pass-2: (N/N1) sibling length-N1 FFTs     (always ≤ 32 → Phase 1 kernel)

bluestein_fft_bf16(N):
    build chirp c, a = x·c (host fp32)
    M = next_pow2(2N-1), zero-pad, build b
    A = fft(a)                                [recurses → pow2 path]
    B = fft(b)                                [recurses → pow2 path]
    p = IFFT(A·B) via a third fft()
    X[k] = c[k]·p[k]  for k ∈ [0, N)
```

The reduction-critical math (every matmul) is on-device bf16. The
pointwise host steps (reshapes, twiddles, chirp multiplies) are fp32
because pointwise ops have zero reduction depth — fp32 there is
**strictly more accurate** than bf16 would be and costs ~µs of PCIe
bounce per call.

## Layout

```
fft_universal_bf16/
├── CMakeLists.txt
├── README.md
├── sop.txt
├── fft_universal_bf16_host.cpp              # host library + dispatcher (all phases)
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
# Full correctness sweep (Phase 1, pow2 up to 65536, primes, composites)
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_test

# Benchmark at any supported N, 100 iters
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 32    100   # Phase 1
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 1024  100   # pow2 depth-1
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 16384 100   # pow2 depth-3
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 3600  50    # mixed-radix
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_benchmark 1009  50    # Bluestein prime

# Pure-tone demo at bin k_in, length N
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 32    5
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 1024  37
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 37    10   # prime Bluestein
./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_demo 60    7    # mixed-radix
```

## Expected precision

bf16 has ~8 bits of mantissa → ULP ~4e-3 relative. Random complex input
with `|x| ≤ 1`:

| N                               | Path                 | Rel err   | SNR         |
|---------------------------------|----------------------|-----------|-------------|
| `N ≤ 32`                        | Phase 1              | 2-4e-3    | 48-56 dB    |
| pow2 `N ∈ [64, 1024]`           | Phase 2b depth-1     | 3-5e-3    | 46-50 dB    |
| pow2 `N ∈ [2048, 16384]`        | Phase 2b depth-2     | 5-10e-3   | 40-46 dB    |
| pow2 `N ∈ [32768, 65536]`       | Phase 2b depth-3     | 8-15e-3   | 36-42 dB    |
| composite `N` (mixed-radix)     | Phase 2b (varies)    | 3-8e-3    | 42-50 dB    |
| prime `N ≤ 251`                 | Bluestein, M ≤ 512   | 5-10e-3   | 40-46 dB    |
| prime `N ≤ 1009`                | Bluestein, M ≤ 2048  | 8-20e-3   | 34-42 dB    |
| hard composite (37²)            | Bluestein, M = 4096  | 10-25e-3  | 32-40 dB    |

The accuracy cost scales with **recursion depth** (each device round-trip
is ~2 bf16 roundings on the critical path). If your application needs
tighter precision, use `fft_universal/` (fp32, ~1e-6 relative).

## Performance notes (honest)

Phase 2b/2c is orchestrated correctness-first, not performance-first.
Each recursive level dispatches separately with no cross-sibling
batching. For large `N` you will see **many kernel dispatches per FFT**:

* `N = 2048`: ~65 dispatches (`pass-1` = 32 × Phase-2a, `pass-2` = 1).
* `N = 16384`: ~65 dispatches (N1=32, N2=512 ≤ 1024).
* `N = 65536`: ~2080 dispatches (N2=2048 needs another level).
* Bluestein `N = 1009`: 3 × ~65 = ~195 dispatches on `M = 2048`.

At ~0.14 ms per dispatch (warm) expect ~10 ms for `N = 2048`, ~10 ms for
`N = 16384`, ~300 ms for `N = 65536`. This is slow vs CPU for the
larger sizes — not because bf16 compute is slow, but because we are
paying dispatch overhead per sub-FFT. Future work to batch same-size
sub-FFTs into a single kernel would collapse those to 1-2 dispatches
per pass. See `sop.txt → FUTURE WORK`.

## Roadmap

1. **Batched pow2 kernel**: make `packed_dft_bf16` accept a `count`
   parameter that is *not* tied to `PACKED_ROWS_PER_TILE = 32`. Would
   let one dispatch cover all siblings of a pass, collapsing
   `pass-1 of N=2048` from 32 dispatches to 1.
2. **Device-side twiddle + transpose**: a small `bf16 × bf16 → bf16`
   elementwise kernel that avoids the host bounce between passes. The
   accuracy loss for going bf16 on the pointwise step is bounded
   because it has zero reduction depth.
3. **Multi-core dispatch**: shard `count` siblings across Tensix cores
   so the same dispatch gets wider parallelism.
