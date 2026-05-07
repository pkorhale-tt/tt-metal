# fft_universal_bf16 — TRUE-bf16 FFT on Wormhole

## Recommended operating range (measured)

| Target accuracy        | Max usable N            | Notes                                |
|------------------------|------------------------:|--------------------------------------|
| rel err ≤ 2.5e-2 (bf16)|         **2,097,152 (2M)** | Verified with 1-iter probe sweep. 2M = 438 ms cached. |
| rel err ≤ 5e-2         |          ≤ 4M (untested)   | Bf16 cliff is between 2M and 8M.     |
| **Anywhere above 4M**  |   **not recommended**      | At N = 8M rel err ≈ 0.19 (broken).   |

Use `fft_universal/` (fp32) when you need rel err ≤ 1e-5; it's clean
up to N = 1,048,576 (1M). For N > 1M, this bf16 path is **the only
working option** on the current build (fp32 Stockham planner asserts
out at sub-FFT lengths > 1024).

---

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

The right fix turned out to be the opposite direction: instead of
moving more work onto the device, **batch the recursive sub-FFTs into
fewer dispatches** so the host bounce per dispatch hurts less. See the
"Performance" section below for measurements.

For Pass-2 itself to actually win on-device, the bf16 dispatcher would
need to keep intermediate buffers in DRAM across passes (no readback
between Pass-1 and Pass-3). That's a deeper refactor of the multi-pass
plumbing — see SOP "device residency for bf16 multi-pass". Until then,
host Pass-2 is the faster path.

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

## Measured precision and runtime  (post-v3 batched path)

bf16 has ~8 bits of mantissa → ULP ~4e-3 relative. Numbers below are
measured on Wormhole, random complex input `|x| ≤ 1`, 100 cached iters
(50 for non-pow2). `rel err` is for `ifft(fft(x)) vs x` round-trip
(test suite `random` cases match the same band, see test output).

### Pow2 (Phase 2b)

| N         | FFT (ms) | IFFT (ms) | RT rel err | Path                       | Dispatches |
|-----------|---------:|----------:|-----------:|----------------------------|-----------:|
| 1024      |    0.084 |     0.083 |   9.55e-03 | depth-1 direct (N1=32)     |          1 |
| 2048      |    0.331 |     0.330 |   1.08e-02 | v3 batched two-level       |          2 |
| 4096      |    0.367 |     0.369 |   1.17e-02 | v3 batched two-level       |          2 |
| 8192      |    0.459 |     0.462 |   1.08e-02 | v3 batched two-level       |          2 |
| 16384     |    0.697 |     0.714 |   1.25e-02 | v3 batched two-level       |          2 |
| 32768     |    1.268 |     1.188 |   1.26e-02 | v3 batched (1 unique JIT)  |          3 |
| 65536     |   11.534 |    11.585 |   1.83e-02 | recursion + v3 inner       |        ~97 |
| 131,072   |   15.108 |       —   |   1.90e-02 | depth-3 recursion + v3     |          — |
| 262,144   |   21.421 |       —   |   1.47e-02 | depth-3 recursion + v3     |          — |
| 524,288   |   36.785 |       —   |   1.64e-02 | depth-3 recursion + v3     |          — |
| 1,048,576 |   68.444 |       —   |   2.09e-02 | depth-3 recursion + v3     |          — |
| 2,097,152 |  437.776 |       —   |   2.19e-02 | deep recursion             |          — |
| 8,388,608 | 1086.630 |  1146.401 |   1.86e-01 | **NUMERICALLY DEGRADED**   |          — |

`N <= 32768` runs all-batched (1-3 dispatches). `N = 65536` and above
fall outside the outer batched gate (`N2/32 > 32`) so they do
recursive sub-FFTs — each of which uses v3 internally, keeping cost
much lower than legacy per-sibling recursion would. **Up to N ≈ 2M
the round-trip rel err stays in the ~2e-2 band (normal bf16).** At
N = 8M cumulative bf16 roundings push rel err to ~0.19 — that size is
beyond the usable accuracy envelope of the current path.

### Mixed-radix (Phase 2b composite)

| N    | FFT (ms) | RT rel err | Path                        |
|------|---------:|-----------:|-----------------------------|
| 60   |    0.100 |   6.62e-03 | 30 × 2 (largest divisor)    |
| 100  |    0.098 |   7.95e-03 | 25 × 4                      |
| 360  |    0.100 |   8.09e-03 | 30 × 12                     |
| 3600 |    2.964 |   1.17e-02 | depth-2 mixed (30 × 120)    |

### Bluestein (Phase 2c)

| N    | M (FFT len) | FFT (ms) | RT rel err |
|------|------------:|---------:|-----------:|
| 37   |         128 |    0.292 |   8.56e-03 |
| 251  |         512 |    0.303 |   1.49e-02 |
| 509  |        1024 |    0.318 |   1.91e-02 |
| 1009 |        2048 |    1.019 |   1.90e-02 |
| 1369 |        4096 |    1.157 |   1.86e-02 |

Bluestein internally calls `fft(M)` three times, and all three benefit
from v3's batched path → these numbers are roughly `3 × pow2(M)` time.

### Round-trip precision band (test suite)

Worst observed `ifft(fft(x))` rel err across the full sweep is
**1.97e-02** (N=1369 Bluestein, depth-2 chained matmuls).
Worst forward FFT-vs-double rel err is **8.70e-03** (N=1009 Bluestein).
For 1e-6-class precision use `fft_universal/` (fp32 path).

## Performance summary

See [Measured precision and runtime](#measured-precision-and-runtime-post-v3-batched-path)
above for the full sweep. Headline: **N = 16384 in 0.7 ms**, within
24% of the fp32 path (0.56 ms).

**Optimisation timeline at N=16384:**

| Step | What changed                                        | ms    | Δ      |
|------|-----------------------------------------------------|-------|--------|
| v1   | per-sibling recursion + per-call host cos/sin       | 4.06  | —      |
| v2   | cached between-pass twiddle table                   | 3.52  | -13%   |
| v3   | `batched_two_level_fft_bf16` (collapse N1 sub-FFTs) | 0.70  | **-83%**|

The big v3 win came from **removing dispatches**, not redistributing
them. Going from 65 sequential dispatches to 3 batched ones eliminated
~3 ms of host-device round-trip overhead.

v3 also helps at sizes that fall outside its outer gate. At N=65536
(N2/32 > 32, so the outer call still recurses) the inner `fft(2048)`
sub-calls each pick up v3, giving ~11.5 ms instead of the ~50 ms a
fully-non-batched depth-3 path would cost.

## Roadmap

1. **bf16 batch_fft kernel** (analogue of fp32 `batch_fft`): would handle
   `N > 32` natively without recursing back through the dispatcher. Pushes
   `N >= 65536` into the batched fast path. Estimated 1-2 weeks of work.
2. **Device-side twiddle + transpose**: a small `bf16 × bf16 → bf16`
   elementwise kernel that avoids the host bounce between passes. Only
   profitable once #3 below removes the readback that motivated host
   twiddle in the first place.
3. **Device residency for multi-pass**: keep Pass-1 output in DRAM so
   Pass-2/3 can stream from there. Required before any on-device Pass-2
   plan can win (see "Plan A" failure above).
