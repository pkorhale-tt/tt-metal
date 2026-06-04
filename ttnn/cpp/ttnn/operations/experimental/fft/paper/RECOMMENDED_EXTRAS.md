# Recommended extras for the paper

This file collects things that are not strictly required by the SOP
but would, in my opinion, **substantially strengthen the HPEC paper**.
They are listed in roughly descending order of expected impact.

The goal is to give a reviewer no reasonable reason to ask "but did
you measure …?".

---

## 1. Be explicit about the host round-trip cost

The single most likely reviewer challenge is:

> "You call this 'device-resident', but `fft_program_factory.cpp`
> reads the input tensor into a `std::vector<float>` on host every
> call. Why isn't the host I/O included in your numbers?"

`bench_host_device_split.py` already produces the data; **also include
a one-paragraph implementation-honesty note in §3 of the paper** that
says, in plain English:

> The op currently materialises the input/output tensor on the host
> before dispatch. We report both the user-visible wall time (which
> includes this round-trip) and the "tensor-already-on-device"
> steady-state time. The host round-trip is a known engineering gap
> being addressed by moving `read_real_as_fp32` and the conjugate
> trick into a device-side prologue/epilogue kernel.

The plot in `fig_host_device_split.pdf` makes this honest and
defensible.

---

## 2. Compare against at least one external library

The current kit measures `ttnn` against itself. For an HPEC paper
you want **at least one** of:

- **cuFFT** on an A100 / H100 (single-precision, same N list).
  Easy to script with `cupy.fft.fft`; the comparison fits on one plot.
- **FFTW** single-threaded and multithreaded on an x86 host.
  `pyfftw` is the lowest-friction Python wrapper.
- **MKL FFT** if a recent Intel host is available.

Even a single column showing "ttnn fp32 vs cuFFT fp32 at N = 16384,
B = 1" makes the contribution feel grounded. Brown 2025 already does
the cuFFT comparison; reproducing the same plot but with `ttnn` is
the cleanest "we beat / match / fall short by 1.8x" message.

These external benches deliberately live **outside** this kit, because
they need different environments. Suggested structure:

```
$TT_METAL_HOME/external_baselines/
├── cufft_baseline.py     # CUDA host
├── fftw_baseline.py      # x86 host
└── mkl_baseline.py       # x86 host
```

Then your `plot_results.py` can ingest those CSVs alongside the
`ttnn` ones for a head-to-head figure.

---

## 3. Energy / power measurement

`tt-smi` exposes per-die power telemetry. A tiny sampler that polls
`tt-smi -t json` every 10 ms while the bench runs gives you a
GFLOP/W column. The two-line addition to the paper

> "On Wormhole the median FFT at N = 16384 sustains XX GFLOP/s at
> YY W, i.e. ZZ GFLOP/W; cuFFT on an H100 SXM at the same N sustains
> XX GFLOP/W."

is a strong differentiator vs raw-throughput-only comparisons.

I deliberately did not script this — it's machine-specific and easy
to break. Recommended outline:

```python
# tt_smi_sampler.py (sketch)
import subprocess, json, time, threading
def sample(stop, out):
    while not stop.is_set():
        j = json.loads(subprocess.check_output(["tt-smi", "-t", "json"]))
        out.append((time.time(), j["device_info"][0]["power"]["instant"]))
        time.sleep(0.01)
```

---

## 4. Trace + program cache as a single "production" mode

`bench_metal_trace.py` and `bench_program_cache.py` measure the two
optimisations separately. The paper should also report **one
"production" number**: program cache on **plus** trace replay. That's
the realistic number a user gets after warmup, and it's typically
where the dispatch overhead disappears.

Add to `run_all.sh`:

```bash
run_bench bench_metal_trace \
    --dtype fp32 --precision precise \
    --N "32,1024,16384,65536" \
    --warmup 10 --iters 200 \
    --out "${CSV_DIR}/metal_trace_production.csv"
```

then add a "production" series to the throughput figure.

---

## 5. Strong-scaling sweep (batch)

Right now `--batch 1,8,64` is the default. For the paper's
sustained-throughput claim, add a **strong-scaling** curve for at
least one N (say 16384), showing GFLOP/s vs B for B in
{1, 2, 4, 8, 16, 32, 64, 128, 256}. This separates the "small-batch
dispatch dominated" regime from the "big-batch arithmetic dominated"
regime cleanly.

`bench_throughput.py` already does this — just expand `--batch`
in one call:

```bash
python bench_throughput.py --dtype fp32 --precision precise \
    --N 16384 --batch 1,2,4,8,16,32,64,128,256 --iters 50 \
    --out paper_results/csv/throughput_batch_sweep_N16384.csv
```

---

## 6. Numerical-precision "Brown comparison" plot

Brown 2025 reports both fp32 throughput and rel-err at N = 16384.
Reproduce **both** in the same figure: x-axis N, left y-axis GFLOP/s,
right y-axis rel-err, fp32 + bf16 lines.

`plot_results.py` does not generate this dual-axis plot yet because
it depends on which baseline you choose; the data are already in
`accuracy_*.csv` and `throughput_*.csv`. Easy to add once you decide
the layout — happy to bolt it on when you settle on the figure brief.

---

## 7. IFFT correctness budget

`bench_ifft_roundtrip.py` reports rel-err but the paper should also
state the **expected** error floor per dtype:

- fp32 / precise:   ≤ 5·10⁻⁵ for N ≤ 2²⁰  (matches Kahan-style bound on Stockham)
- fp32 / fast:      ≤ 5·10⁻³ for N ≤ 2²⁰
- bf16 / fast:      ≤ 5·10⁻² for N ≤ 2²⁰

Cite these as the design contract; the round-trip plot then becomes
"all measured points stay well under their bound", which is the
cleanest possible correctness story.

---

## 8. Ablation: SFPU vs FPU compute on the same N

The `precision={"precise","fast"}` knob is exactly an ablation
between the SFPU radix-2 path and the FPU bf16-mantissa matmul path,
on identical input. Add a side-by-side plot at one or two N values:

```bash
python bench_latency.py --dtype fp32 --precision both \
    --N 1024,4096,16384,65536 --batch 1 --iters 200 \
    --out paper_results/csv/ablation_precise_vs_fast.csv
```

then a bar chart in `plot_results.py`. This is exactly the kind of
"hardware-aware design decision" paragraph reviewers love.

---

## 9. Trace replay vs cuFFT plan reuse

cuFFT has the same "plan cache" notion (cufftPlan1d). For an even
fairer comparison against cuFFT, compare:

| ttnn step                    | cuFFT step             |
|------------------------------|------------------------|
| cold first call              | first cufftExecC2C    |
| warm (program cache)         | warm cufftExecC2C     |
| traced (Metal Trace)         | CUDA Graph replay      |

The three-row table makes the contribution especially crisp.

---

## 10. Versioning / reproducibility footnote

For the paper's "Reproducibility" appendix, capture in `run_all.sh`'s
log directory:

```bash
git rev-parse HEAD                        > $LOG_DIR/_commit.txt
git submodule status                      > $LOG_DIR/_submodules.txt
tt-smi -t json                            > $LOG_DIR/_tt_smi.json   # if available
python -c "import ttnn; print(ttnn.__version__)" > $LOG_DIR/_ttnn_version.txt
```

`run_all.sh` writes everything else; let's bolt these four lines on
when you do the final paper-pass.
