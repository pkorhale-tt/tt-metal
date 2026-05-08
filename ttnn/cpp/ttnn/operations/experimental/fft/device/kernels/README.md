# Stockham FFT kernels

Kernels for the `ttnn::experimental::fft` device op. Phase 2 (forthcoming)
will wire `fft_program_factory.cpp` to `CreateKernel(...)` these files; in
Phase 1 they are present-but-unreferenced so this PR can land without a
follow-up file-relocation PR.

## Layout

```
device/kernels/
├── dataflow/                    (BRISC0 reader / BRISC1 writer)
│   ├── fft_reader.cpp           ┐
│   ├── fft_writer.cpp           │ inner radix-2 single-tile FFT
│   ├── fft_common.h             ┘  (sub_N <= 1024)
│   ├── batch_fft_reader.cpp     ┐
│   ├── batch_fft_writer.cpp     │ batched single-tile FFT — 64 cores
│   ├── batch_fft_common.h       ┘  parallel sub-FFTs of length sub_N
│   ├── pass2_reader.cpp         ┐
│   ├── pass2_writer.cpp         │ Stockham pass-2: per-element twiddle
│   └── pass2_common.h           ┘  multiply, on-device (no transpose)
└── compute/                     (TRISC0/1/2 — FPU + SFPU)
    ├── fft_compute.cpp          radix-2 butterfly via FPU matmul
    ├── batch_fft_compute.cpp    same as above, batched per core
    └── pass2_compute.cpp        complex multiply via SFPU
```

## Provenance

Originally developed and validated end-to-end in:

* `tt_metal/programming_examples/fft/kernel/`           → `dataflow/fft_*` + `compute/fft_compute.cpp`
* `tt_metal/programming_examples/fft_stockham/kernel/`  → everything else

The programming-examples copies remain unchanged for now; once the ttnn
op is fully wired (Phase 2-C) the originals can be deleted in favour of
this canonical location.

## Pipeline (Phase 2 reference)

`ttnn::experimental::fft(x)` for length-N power-of-two will dispatch
the four-pass Stockham pipeline:

1. **batch_fft** (length-N₂, batch=N₁) — `batch_fft_*` kernels
2. **pass2** — on-device twiddle multiply `W_N^(i·j)`
3. **batch_fft** (length-N₁, batch=N₂) — `batch_fft_*` kernels
4. final reorder — currently host-side, lifts to a writer-kernel pass
   in Phase 2-C

For N ≤ 65,536 we collapse to the single-tile path (`fft_*`) directly.

See the orchestration logic in
`tt_metal/programming_examples/fft_stockham/fft_stockham_host.cpp`
(soon to be lifted into `device/stockham_host.hpp`).
