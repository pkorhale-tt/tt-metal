# `ttnn.experimental.fft` — Paper Kit

Self-contained measurement, analysis, and documentation kit for the
HPEC 2026 (or any subsequent venue) paper on
`ttnn.experimental.fft` / `ttnn.experimental.ifft`.

The kit lives **inside the op's own folder** so it travels with the
source: anyone who clones tt-metal at this commit can reproduce every
number in the paper by following `SOP.md`.

---

## Layout

```
paper/
├── README.md                # this file — overview + index
├── SOP.md                   # step-by-step procedure: how to run, what to read
├── ALGORITHMS.md            # per-backend math + dispatch rules
├── HOST_VS_DEVICE.md        # accounting: which files/lines run on host vs device
├── HOST_MATH.md             # math-explicit version: which formula, evaluated where
├── RECOMMENDED_EXTRAS.md    # additional measurements worth doing
└── scripts/
    ├── _common.py                  # shared helpers: timing, CSV, rel-err
    ├── bench_latency.py            # us/call vs N (fp32 + bf16, precise + fast, B in {1,8,64})
    ├── bench_throughput.py         # GFLOP/s = 5·B·N·log2(N) / latency
    ├── bench_accuracy.py           # rel-err vs torch.fft.fft fp64 reference
    ├── bench_program_cache.py      # first call vs cached call
    ├── bench_metal_trace.py        # untraced vs trace-replay
    ├── bench_brown_repro.py        # N=16384 hero replica (Brown 2025 Table 1)
    ├── bench_ifft_roundtrip.py     # ifft(fft(x)) ≈ x correctness sweep
    ├── bench_host_device_split.py  # measure host time around the call
    ├── run_all.sh                  # orchestrator: runs every bench, drops CSVs
    └── plot_results.py             # build PDFs/PNGs from the CSVs
```

The op itself (one folder up) is:

```
ttnn/cpp/ttnn/operations/experimental/fft/
├── fft.hpp / fft.cpp                public C++ API
├── fft_nanobind.{hpp,cpp}           nanobind binding (precision kwarg)
├── CMakeLists.txt                   kernel install glob
└── device/
    ├── fft_device_operation.{hpp,cpp}   device-op + program-hash
    ├── fft_device_operation_types.hpp   FFTPrecision / FFTBackend / FFTParams
    ├── fft_program_factory.{hpp,cpp}    dispatcher → backend orchestrator
    ├── stockham_host.hpp                fft_stockham orchestrator
    ├── universal_host.hpp               fft_universal (mixed-radix + Bluestein)
    ├── universal_bf16_host.hpp          fft_universal_bf16
    ├── universal_xl_host.hpp            fft_universal_xl (Option B host outer twiddle)
    ├── universal_xl_planner.hpp         XL plan factorization
    ├── fft_inner_host.hpp               shared inner-FFT helpers
    └── kernels/
        ├── compute/                     5 compute.cpp + 2 dup'd common.h
        └── dataflow/                    5 reader/writer pairs + 5 common.h
```

## Quick start

After building tt-metal with `TT_FFT_NATIVE=1` support and confirming
`python -c "import ttnn"` works:

```bash
cd $TT_METAL_HOME

# Run everything (≈ 20 min on Wormhole, default N list):
bash ttnn/cpp/ttnn/operations/experimental/fft/paper/scripts/run_all.sh

# Outputs land under $TT_METAL_HOME/paper_results/{csv,figs,logs}/
ls paper_results/
```

For step-by-step explanation of each script, read `SOP.md`.
For "what does each backend actually do" read `ALGORITHMS.md`.
For "where does work run, exactly" read `HOST_VS_DEVICE.md`.
For "which formula is evaluated where, with equations" read `HOST_MATH.md`.

## What this kit does NOT do

- Does not build tt-metal itself (assumes `_ttnn.so` is already linked).
- Does not run on multi-chip mesh (single-Wormhole assumption).
- Does not measure power directly (see `RECOMMENDED_EXTRAS.md` for the
  `tt-smi`-based sampler if you want energy figures).
- Does not auto-compare against external libraries like cuFFT or FFTW;
  those are separate side benches (see `RECOMMENDED_EXTRAS.md`).
