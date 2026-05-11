# `ttnn::experimental::fft` device kernels

All kernels reachable from `ttnn::experimental::fft` / `ttnn::experimental::ifft`,
staged in one place. Today (Phase 2) the program factory still calls the
original orchestrators under `tt_metal/programming_examples/fft*/`, which
themselves `CreateKernel(...)` against their in-tree paths. Phase 3 will
retarget those `CreateKernel` paths to the canonical copies here, at
which point the `programming_examples` copies become removable.

## Layout

```
device/kernels/
├── dataflow/                            (BRISC0 reader / BRISC1 writer)
│   ├── fft_reader.cpp                   ┐
│   ├── fft_writer.cpp                   │ inner radix-2 single-tile FFT
│   ├── fft_common.h                     ┘  (sub_N <= 1024)  — fp32
│   ├── batch_fft_reader.cpp             ┐
│   ├── batch_fft_writer.cpp             │ batched single-tile FFT, 64 cores,
│   ├── batch_fft_common.h               ┘  parallel sub-FFTs   — fp32
│   ├── pass2_reader.cpp                 ┐
│   ├── pass2_writer.cpp                 │ Stockham pass-2: per-element
│   ├── pass2_common.h                   ┘  twiddle multiply   — fp32
│   ├── packed_dft_reader.cpp            ┐
│   ├── packed_dft_writer.cpp            │ packed direct DFT for small /
│   ├── packed_dft_common.h              ┘  composite radices  — fp32
│   ├── packed_dft_bf16_reader.cpp       ┐
│   ├── packed_dft_bf16_writer.cpp       │ packed direct DFT, true-bf16 FPU
│   └── packed_dft_bf16_common.h         ┘  matmul reduction   — bf16
└── compute/                             (TRISC0/1/2 — FPU + SFPU)
    ├── fft_compute.cpp                  radix-2 butterfly via FPU matmul (fp32)
    ├── batch_fft_compute.cpp            same as above, batched per core   (fp32)
    ├── pass2_compute.cpp                complex multiply via SFPU         (fp32)
    ├── packed_dft_compute.cpp           packed direct DFT compute         (fp32)
    └── packed_dft_bf16_compute.cpp      packed direct DFT compute         (bf16)
```

19 files total: 12 dataflow (6 reader/writer pairs + 6 common headers),
7 compute (5 compute.cpp + 2 duplicated common.h — see below).

### Why two copies of `packed_dft{,_bf16}_common.h`

The tt-metal kernel build resolves bare `#include "X_common.h"` only
against the kernel's own directory. The `packed_dft` and
`packed_dft_bf16` kernel triples genuinely share state across both
compute and dataflow (inherited from the original flat `kernel/` layout
in `programming_examples/`), so each common.h is duplicated into both
`compute/` and `dataflow/`. Both copies carry a sync-warning header.
The other four common.h files (`fft`, `batch_fft`, `pass2`) are only
used by their reader/writer kernels, so they live in `dataflow/` only.

## Backend → kernel mapping

| `ttnn.experimental.fft` input | Backend       | Kernels used                                                               |
|--------------------------|--------------------|----------------------------------------------------------------------------|
| fp32, pow2, N ≤ 64K      | `fft_stockham`     | `fft_*`                                                                    |
| fp32, pow2, 64K < N ≤ 1M | `fft_stockham`     | `batch_fft_*` + `pass2_*`                                                  |
| fp32, pow2, 1M < N ≤ 16M | `fft_universal_xl` | (delegates to `fft_stockham`)                                              |
| fp32, non-pow2           | `fft_universal`    | `packed_dft_*` + `fft_stockham` kernels for pow2 sub-FFTs / Bluestein pad  |
| bf16, any N              | `fft_universal_bf16` | `packed_dft_bf16_*`                                                      |

## Provenance

Each file is a verbatim copy of the corresponding kernel under
`tt_metal/programming_examples/`:

| ttnn copy                                | Source of truth                                                            |
|------------------------------------------|----------------------------------------------------------------------------|
| `fft_*`                                  | `fft/kernel/fft_*.cpp`, `fft/kernel/fft_common.h`                          |
| `batch_fft_*`, `pass2_*`                 | `fft_stockham/kernel/{batch_fft,pass2}_*.cpp` + `*_common.h`               |
| `packed_dft_*`                           | `fft_universal/kernel/packed_dft_*.cpp` + `packed_dft_common.h`            |
| `packed_dft_bf16_*`                      | `fft_universal_bf16/kernel/packed_dft_bf16_*.cpp` + `packed_dft_bf16_common.h` |

Both copies must be kept in sync until Phase 3 retargets the orchestrator
`CreateKernel(...)` paths and the `programming_examples` copies are
deleted.

## Build / install

The parent `CMakeLists.txt` does
`file(GLOB_RECURSE kernels device/kernels/*)` and installs the whole tree
to `${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/fft/`,
so adding new files here requires no CMake changes — just rebuild.

## Phase 3 plan (single-Program rewrite)

The orchestrators in `programming_examples` each manage their own
`MeshWorkload` enqueues. Phase 3 will refactor them into free
`build_<backend>_program(md, N, in_buf, out_re_buf, out_im_buf)` builders
that emit one `Program` per call (no internal enqueues), `CreateKernel`
against the in-tree paths above, and let `FFTProgramFactory::create`
own the dispatch + caching.
