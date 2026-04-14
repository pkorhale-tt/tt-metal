# Multi-core 1D FFT for Tenstorrent Wormhole
## Based on Davies et al. 2025 + bounty issue #21412

---

## File structure

```
fft_kernel/
  kernels/
    fft_common.h      # CB indices, arg layouts, tile constants
    fft_reader.cpp    # BRISC-0: DRAM reads, bit-reversal, twiddle loads
    fft_compute.cpp   # TRISC: butterfly math via SFPU
    fft_writer.cpp    # BRISC-1: NOC sends, semaphore handshake, DRAM write
  host/
    fft_host.cpp      # Program setup, twiddle precompute, dispatch
    fft_test.cpp      # Correctness + perf tests
```

---

## Algorithm

**Cooley-Tukey DIT radix-2**, partitioned across cores.

```
Total stages = log2(N)
Local stages = log2(local_N)   = log2(N / num_cores)
NOC stages   = log2(num_cores)
```

**Stage progression:**
```
Stage 1..log2(local_N) :  butterfly pairs within one core's L1
                           → zero NOC traffic
Stage log2(local_N)+1..log2(N):  butterfly pairs span cores
                           → NOC unicast + semaphore per stage
```

---

## NOC handshake (critical path)

```
Writer (BRISC-1):                    Peer writer:
  noc_async_write(data → peer CB)      noc_async_write(data → my CB)
  noc_async_write_barrier()            noc_async_write_barrier()
  noc_semaphore_inc(peer_sem, 1)       noc_semaphore_inc(my_sem, 1)
  noc_semaphore_wait(my_sem, N-1) ←── wait for all N-1 peers
  noc_semaphore_set(my_sem, 0)         reset
  cb_push_back(CB_SYNC, 1)            unblock compute
```

**Rules that must not be violated:**
1. `write_barrier` before `semaphore_inc` — data must land before signal
2. `semaphore_set(0)` before `cb_push_back` — reset before compute races ahead
3. Use `noc_semaphore_inc` not volatile store — only safe cross-core signal

---

## Memory layout

### Input / Output (DRAM)
Interleaved complex: `[r0, i0, r1, i1, ..., r_{N-1}, i_{N-1}]`
fp32: 4 bytes/element, bf16: 2 bytes/element

### Twiddle factors (DRAM)
`twiddle[stage][k]` = `[cos(2πk/M), sin(2πk/M)]`
where `M = 2 * (1 << stage)` (butterfly group size)
Layout: stage-major, `(stage * N/2 + k) * 2` for real, `+1` for imag

### L1 per core
```
CB_LHS_R/I   : local_N elements real/imag (current stage input, LHS)
CB_RHS_R/I   : local_N elements real/imag (reordered butterfly partner)
CB_TWIDDLE   : local_N/2 twiddle factors
CB_OUT_R/I   : butterfly output
CB_SCRATCH   : incoming NOC data from partner core
CB_SYNC      : 1-element signal from writer to compute
CB_TMP/WR    : intermediate butterfly products
```

Total L1 usage (fp32, local_N=128):
```
  9 CBs × 128 × 4 = ~4.6KB  << 1.3MB limit  ✓
```

---

## IFFT

Identical kernel, two differences:
1. Twiddle factors conjugated: `sin` term negated in `precompute_twiddles`
2. After last stage: multiply by `1/N` via `scale_by_inv_N` in compute kernel

---

## bf16 vs fp32

| Mode | Tile size | Throughput | Max error |
|------|-----------|------------|-----------|
| fp32 | 4KB       | baseline   | ~1e-5     |
| bf16 | 2KB       | ~2x        | ~1e-2     |

bf16 halves L1 pressure and doubles effective bandwidth for twiddle reads.
Use fp32 when precision matters (signal processing), bf16 for ML pipelines.

---

## Supported sizes

Any `N = 2^k` where `N / num_cores = local_N` is also a power of 2.

| N     | num_cores | local_N | Local stages | NOC stages |
|-------|-----------|---------|--------------|------------|
| 64    | 1         | 64      | 6            | 0          |
| 256   | 4         | 64      | 6            | 2          |
| 1024  | 8         | 128     | 7            | 3          |
| 1024  | 32        | 32      | 5            | 5          |
| 4096  | 32        | 128     | 7            | 5          |

---

## Build

```bash
# From tt-metal root
./build_metal.sh

# Compile test
g++ -std=c++17 -O2 \
    -I tt_metal/api \
    host/fft_test.cpp \
    -L tt_metal/build/lib -ltt_metal \
    -o fft_test

./fft_test
```

---

## Known gotchas / future work

**Single-copy optimization (Davies paper §4.3)**
Currently doing 2 reorders per local stage. The single-copy optimization
(prepare next-stage order during write) halves data movement.
Requires linker script change to extend bss for second buffer.

**128-bit writes**
Lost when single-copy is applied (both src and dst are non-contiguous).
Keep 128-bit OR single-copy, not both — single-copy wins on net runtime.

**ThCon intrinsics for reordering**
`fft_reader.cpp` uses scalar BRISC copies for reorder (clear but slow).
Replace with `TT_LOADIND` / `TT_SETDMAREG` intrinsics from LLK library
for ~1ms reduction (Davies paper: 9.38ms → 7.56ms).

**Uneven core count**
Currently requires `num_cores` to be a power of 2 and to divide N evenly.
For arbitrary N: zero-pad input to next power of 2.
For arbitrary core count: requires uneven transposition (Davies §5 future work).

**2D FFT**
Build on top of this 1D primitive:
  1. FFT across rows (this kernel)
  2. Matrix transpose (use tt-nn's built-in multicore transpose)
  3. FFT across columns (this kernel again)
