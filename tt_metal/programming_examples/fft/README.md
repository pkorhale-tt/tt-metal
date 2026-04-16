# Multi-core radix-2 DIT FFT (fp32, Wormhole)

Tile-engine FFT sharded across `P = max(1, N/1024)` Tensix cores on a 1D row.
**FFT only** (no IFFT).

## Constraints

- `N` is a power of two.
- `2 <= N <= 8192` out of the box (P ∈ {1, 2, 4, 8}).
- 1D core row at `{0..P-1, 0}`. Extends trivially to a 2D grid (see bottom).

| N       | P  | Local stages | Cross-core stages |
|---------|----|--------------|-------------------|
| ≤ 1024  | 1  | `log2(N)`    | 0                 |
| 2048    | 2  | 10           | 1                 |
| 4096    | 4  | 10           | 2                 |
| 8192    | 8  | 10           | 3                 |

## Files

```
fft/
  CMakeLists.txt
  fft_host.cpp       # host setup: sharding, twiddles, semaphores, launch
  fft_test.cpp       # correctness + timing, single- and multi-core N
  kernel/
    fft_common.h     # CB indices, tile constants
    fft_reader.cpp   # BRISC0: DRAM I/O, state, local shuffle, cross-core exch.
    fft_compute.cpp  # TRISC : tile butterfly via SFPU (full fp32)
    fft_writer.cpp   # BRISC1: writes this core's state shard to DRAM
```

## Algorithm

Iterative **Cooley-Tukey DIT radix-2** in place over `log2(N)` stages.
Host bit-reverses the input; output comes back in natural order.

Each core owns a single 1024-element tile of the global state. Pair strides
split cleanly:

- `stride < 1024` → **local stage**: both elements of every pair live in the
  same tile → same code as the single-core kernel, BRISC scatters state into
  `CB_EVEN/CB_ODD`, compute runs the butterfly, BRISC gathers back.
- `stride >= 1024` → **cross-core stage**: at stage `s`, each core's partner
  is `my_core XOR (1 << (s-10))`.

### Cross-core butterfly (symmetric, one NoC round-trip per stage)

```
For each cross-core stage s:
   partner     = my_core XOR (1 << (s-10))
   is_c_even   = (my_core & (1 << (s-10))) == 0          // lower core of pair
   [async]  noc_write(state_{R,I}) -> partner's CB_RECV
            noc_semaphore_inc(partner's sem)
   [wait]   noc_semaphore_wait(my sem, cumulative_count)
   [compute] EVEN / ODD = (state, recv) if c_even else (recv, state)
             OUT0 = EVEN + W * ODD
             OUT1 = EVEN - W * ODD
   [keep]   state := OUT0 (c_even) or OUT1 (c_odd)
```

Both cores in a pair do identical math on the same `(E, O, W)` inputs. They
keep opposite halves of the butterfly output, so each core does exactly half
the radix-2 work, with full compute-engine utilisation.

### Why the semaphore is monotonic

Partners change every cross-core stage. A naive "inc + wait + reset" pattern
has a race: a faster core can increment its next-stage partner's semaphore
before that partner finishes resetting from the prior stage, losing the
increment. We instead never reset: each core waits for the cumulative count
`k+1` after the `k`-th cross-core stage. One inc per stage, one sem, no reset.

### Twiddle factors per stage

Stage `s`, slot `j` in a core's tile, twiddle `k` (so `W = exp(-2πi k / 2^(s+1))`):

- **Local (`s < 10` when P>1, or `s < log2N` when P=1):** `k = j & (2^s - 1)`.
  Identical for all cores at a given stage — the host writes the same tile
  into all `P` per-core twiddle pages.

- **Cross-core (`s >= 10`):**

  ```
  kshift    = s - 10
  c_low     = my_core & ~(1 << kshift)         // lower core of the pair
  c_in_grp  = c_low & ((1 << (kshift+1)) - 1)   // position in stage's group
  k         = c_in_grp * 1024 + j
  ```

  Both cores in a pair get the same `k_base` (determined by the pair's lower
  core), so they use the same twiddle and agree on `W·O`.

## Memory layout

- **Input / output DRAM**: `P` tiles per side (real, imag). Page `c` holds the
  bit-reversed chunk for core `c`.
- **Twiddles DRAM**: `log2(N) * P` tiles per side. Page `(s * P + c)` is core
  `c`'s twiddle tile for stage `s`. Local-stage pages are replicated across
  all P cores at the same stage; cross-core pages differ.
- **L1 per core**: `CB_STATE_{R,I}` (persistent state), `CB_RECV_{R,I}` (raw
  landing zone for the partner's tile), plus the usual pipelined EVEN/ODD/TW/
  OUT/TMP/TW_ODD CBs.

Per-core L1 footprint ≈ 128 KB (well under the 1.3 MB Tensix limit).

## Precision

Compute path is full fp32 on Wormhole:

- Butterfly math runs on the **SFPU** via `add_binary_tile`, `mul_binary_tile`,
  etc. The matrix engine (FPU `add_tiles` / `mul_tiles`) internally runs in
  bf16 on Wormhole even with `HiFi4` — we avoid it entirely.
- CB circuit is configured with `UnpackToDestMode::UnpackToDestFp32` so
  `copy_tile` feeds DEST in fp32 instead of down-converting to bf16.
- `fp32_dest_acc_en = true` keeps DEST at 32-bit and packs fp32 back to L1.

Observed relative error (random input, reference = O(N²) DFT):

| N    | abs err  | rel err |
|------|----------|---------|
| 64   | 1.4e-4   | 1.1e-5  |
| 256  | 1.3e-3   | 4.6e-5  |
| 1024 | 1.4e-2   | 2.0e-4  |

Abs error scales as `eps × sqrt(N) × sqrt(logN)` (random-walk stage
accumulation × signal magnitude growth). Relative error stays near Wormhole's
practical fp32 floor (~tf19 class, `~1e-5` per op).

## Build & run

From the tt-metal root:

```bash
cmake -B build -DBUILD_PROGRAMMING_EXAMPLES=ON
cmake --build build --target metal_example_fft_test -j
ARCH_NAME=wormhole_b0 ./build/programming_examples/fft/metal_example_fft_test
```

Add `TT_METAL_CLEAR_JIT_CACHE=1` in front of the run command to force a
kernel-binary rebuild (useful after editing kernel sources).

Expected output (abridged):

```
[PASS] N=16    FFT | abs=0.00e+00 rel=0.00e+00 | ... ms  (P=1)
[PASS] N=1024  FFT | abs=1.4e-02 rel=2.0e-04   | ... ms  (P=1)
[PASS] N=2048  FFT | abs=... rel=...           | ... ms  (P=2)
[PASS] N=4096  FFT | abs=... rel=...           | ... ms  (P=4)
[PASS] N=8192  FFT | abs=... rel=...           | ... ms  (P=8)
All tests PASSED.
```

## Going further

- **Larger N (P > 8).** Switch the logical core range from a 1D row to a 2D
  grid in `fft_host.cpp`. The kernels only use a core's *logical index* `c`
  (a single `uint32_t`), so the partner-bit XOR math is unchanged. You just
  need the host to:
  1. Enumerate cores in a stable order (row-major over the grid).
  2. Map logical index `c` to the right `CoreCoord{x,y}` and push NoC coords
     in the same order into the runtime-args lookup table.
- **Streamed / batched FFT.** If you have many FFTs to do, loop the reader's
  "load input → run stages → write output" for B batches. Amortises kernel
  launch and JIT compile across batches.
- **Reader/compute pipelining.** The pipelined CBs already have depth 2; the
  reader could kick off stage `s+1`'s scatter while compute is still on stage
  `s`'s butterfly. This halves per-stage latency in steady state.
