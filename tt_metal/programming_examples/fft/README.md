# Single-core radix-2 DIT FFT (fp32, Wormhole)

Minimal, correct, tile-engine FFT. **FFT only** (no IFFT).

## Constraints

- `N` is a power of two, `2 <= N <= 1024` (fits in one 32x32 fp32 tile).
- Single Tensix core.

For larger `N` or multi-core, extend with NOC butterfly stages on top of this
kernel (notes at the bottom).

## Files

```
fft/
  CMakeLists.txt
  fft_host.cpp       # host program setup + twiddle precompute
  fft_test.cpp       # correctness test (compares against reference DFT)
  kernel/
    fft_common.h     # CB indices, tile constants
    fft_reader.cpp   # BRISC0: DRAM I/O, persistent state, per-stage shuffle
    fft_compute.cpp  # TRISC : tile-based radix-2 butterfly via SFPU
    fft_writer.cpp   # BRISC1: writes final state tile to DRAM
```

## Algorithm

Iterative **Cooley-Tukey DIT radix-2** in place over `log2(N)` stages.
Input is bit-reversed by the host, output comes out in natural order.

At stage `s` with `stride = 2^s`:

```
for each pair index p in [0, N/2):
    group = p >> s
    pos   = p & (stride - 1)
    lo    = group * 2*stride + pos
    hi    = lo + stride

    a      = state[lo]
    b      = state[hi]
    W      = twiddle[p]      // = exp(-j * 2*pi * pos / (2*stride))
    state[lo] = a + W*b
    state[hi] = a - W*b
```

The state is a single fp32 tile (1024 slots, first `N` used). The compute
engine operates on the whole tile each stage; the reader does the
stage-dependent gather/scatter on BRISC scalar code.

## Pipeline per stage

```
BRISC0  reader            TRISC  compute          BRISC1  writer
──────                    ───────                 ──────
load_twiddle -> CB_TW
gather state -> CB_EVEN/CB_ODD
                          wait EVEN/ODD/TW
                          cmul    odd,tw  -> Wodd
                          add     even,Wodd -> OUT0
                          sub     even,Wodd -> OUT1
                          push OUT0/OUT1
scatter OUT0/OUT1 -> state
...
push CB_SYNC                                      wait SYNC
                                                  write state -> DRAM
```

## Memory layout

- **Input / output DRAM**: two fp32 tiles (real, imag). Host packs the input
  into them bit-reversed so stage 0 already has contiguous pairs.
- **Twiddles DRAM**: `log2(N)` tiles per side (real, imag). Tile `s` holds the
  stage-`s` twiddle factor for each pair index `p` at slot `p`.
- **L1 / state**: two fp32 tiles (`CB_STATE_R`, `CB_STATE_I`), kept alive for
  the whole kernel.

Rough L1 footprint (fp32, 17 CBs × up to 2 tiles × 4 KB) ≈ 128 KB, well under
the 1.3 MB limit.

## Build & run

From the tt-metal root:

```bash
./build_metal.sh
./build/programming_examples/fft/metal_example_fft_test
```

Expected output:

```
[PASS] N=16    FFT | err=... | ...
[PASS] N=64    FFT | err=... | ...
[PASS] N=64    FFT | err=... | ... DC
[PASS] N=64    FFT | err=... | ... random
[PASS] N=256   FFT | err=... | ...
[PASS] N=1024  FFT | err=... | ...
All tests PASSED.
```

`err` is the max absolute complex error against the reference O(N^2) DFT;
threshold is `1e-3`.

## Extending to multi-core / larger N

Two independent axes:

1. **Larger `N` via multi-tile state on one core**: for `N > 1024` the state
   no longer fits in a single tile. Split state into `N/1024` tiles, run
   `log2(1024)` "within-tile" stages using this kernel, then add
   `log2(N) - 10` "cross-tile" stages that butterfly across tiles. The
   cross-tile stages are structurally identical to the NOC stages below;
   they just cross tile boundaries on the same core instead of core
   boundaries.

2. **Multi-core via NOC butterflies**: once `local_N = N / num_cores` is still
   a power of two and `<= 1024`, run this kernel per core for the first
   `log2(local_N)` stages. Then `log2(num_cores)` NOC stages where each core:
   - sends its state tile to its partner core (determined by the stage's
     stride bit in its core id),
   - waits for the partner's tile into `CB_SCRATCH_R/I`,
   - butterflies `state <-> scratch` and keeps its correct half.

   Add those NOC stages in `fft_writer.cpp` (which already owns NOC 1), with
   a semaphore for the partner handshake.

Both extensions reuse the reader/compute structure unchanged.
