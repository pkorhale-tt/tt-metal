# Multi-core radix-2 DIT FFT (bf16, Wormhole)

This is the **bfloat16** twin of `../fft/`. Same algorithm, same multi-core
sharding, same plan-cache, same PyTorch-style API — only the in-CB and
in-DRAM element format is `bfloat16` (2 B/element) instead of `fp32`
(4 B/element).

## What's different from `../fft/`?

| Aspect                          | `fft/` (fp32)              | `fft_bf16/` (bf16)             |
|---------------------------------|----------------------------|--------------------------------|
| In-CB / in-DRAM format          | `Float32`                  | `Float16_b`                    |
| Tile size                       | 4096 B                     | **2048 B**                     |
| DEST register precision         | fp32                       | fp32 (`UnpackToDestFp32`)      |
| Per-stage math                  | fp32 SFPU                  | fp32 SFPU                      |
| Inter-stage **storage** precision| fp32                      | **bf16** (truncated)           |
| Memory traffic                  | 1×                         | **0.5×**                       |
| Typical relative error          | `~2e-4` (N=1024)           | `~5e-2` (N=1024)               |
| API namespace                   | `fft_example`              | `fft_example_bf16`             |
| Python script tolerance         | `2e-3`                     | `2e-1`                         |

The compute kernel still operates in **fp32 inside DEST** (thanks to
`UnpackToDestFp32` + `fp32_dest_acc_en=true`); only the storage between
stages is rounded to bf16. So per-stage math is fp32 — the precision loss
is purely the round-trip through bf16-formatted CBs.

## Files

```
fft_bf16/
  CMakeLists.txt
  fft_host.cpp          # bf16-aware host: pack/unpack, twiddles, launch, plan cache
  fft_test.cpp          # correctness + timing (looser thresholds)
  fft_demo.cpp          # PyTorch-style examples
  fft_vs_torch.cpp      # dumper used by compare_with_torch.py
  fft_benchmark.cpp     # plan-cache speedup benchmark
  my_fft_app.cpp        # editable user playground
  compare_with_torch.py # accuracy comparison vs torch.fft.fft
  README.md             # (this file)
  command.txt           # full command reference
  kernel/
    fft_common.h        # CB indices, TILE_SIZE_BF16
    fft_reader.cpp      # BRISC0: DRAM I/O, local shuffle, cross-core exch.
    fft_compute.cpp     # TRISC : tile butterfly via SFPU (fp32 in DEST)
    fft_writer.cpp      # BRISC1: writes this core's state shard to DRAM
```

## Build & run

From the tt-metal repo root:

```bash
cmake -B build -DBUILD_PROGRAMMING_EXAMPLES=ON
cmake --build build --target metal_example_fft_bf16_test -j
ARCH_NAME=wormhole_b0 \
    ./build/programming_examples/fft_bf16/metal_example_fft_bf16_test
```

## Using the API

```cpp
#include "tt_metal/programming_examples/fft_bf16/fft_host.cpp"
using namespace fft_example_bf16;

auto md = MeshDevice::create_unit_mesh(0);
std::vector<float> signal = {10.f, 20.f, 30.f, 40.f};
auto spectrum = fft(md, signal);   // returns std::vector<std::complex<float>>
```

`signal.size()` must be a power of 2 in `[2, 65536]`. The first call for
each `N` builds + caches a plan; every call after that hits the cache.

## When to pick bf16 over fp32

- You're **memory-bandwidth bound** (the bf16 path moves half the bytes
  through L1/DRAM/NoC). The cross-core stages benefit the most.
- You can tolerate `~5e-2` relative error (typical bf16 FFT use case:
  feature extraction, log-magnitude spectra, ML preprocessing).
- For applications needing accurate inverse FFT round-trip, signal
  reconstruction, or numerical analysis — stick with the fp32 path.
