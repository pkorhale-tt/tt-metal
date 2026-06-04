# SOP — Extracting paper results from `ttnn.experimental.fft`

This is the operational checklist. Follow it top-to-bottom and you
will end up with the CSVs and PDFs the paper expects under
`$TT_METAL_HOME/paper_results/`.

For the math behind each backend see `ALGORITHMS.md`.
For where each computation runs see `HOST_VS_DEVICE.md`.

---

## 0. Prerequisites (one-time, per machine)

1. **Build tt-metal in Release.**

   ```bash
   cd $TT_METAL_HOME
   ./build_metal.sh -b Release
   # First time only — re-installing after a rebuild is unnecessary, the
   # rebuilt _ttnn.so is picked up automatically.
   pip install -e .
   ```

   Confirm:

   ```bash
   python -c "import ttnn; print(ttnn.experimental.fft)"
   ```

   should print a `<built-in function fft>` and *not* `ModuleNotFoundError`.

2. **Confirm the device is healthy.**

   ```bash
   tt-smi
   ```

   should list at least one Wormhole or Blackhole part with `BUSY = 0`.

3. **Confirm the unit tests pass first.**

   ```bash
   pytest tests/ttnn/unit_tests/operations/experimental/fft/test_fft.py -q
   ```

   No reason to measure performance numbers from a broken op.

---

## 1. What to run

A single orchestrator script runs everything in the right order:

```bash
bash ttnn/cpp/ttnn/operations/experimental/fft/paper/scripts/run_all.sh
```

By default this runs against the **production N list**:

| Backend                | N values used in the paper                                                |
|------------------------|---------------------------------------------------------------------------|
| `fft_stockham` (fp32)  | 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576           |
| `fft_universal` (fp32) | 5, 7, 11, 17, 25, 49, 96, 100, 360, 729, 1000, 4095, 8191, 65521, 524287   |
| `fft_universal_bf16`   | same as `fft_universal` + 32768, 131072 (so bf16 vs fp32 ratio is visible) |
| `fft_universal_xl`     | 2097152, 4194304, 8388608, 16777216                                        |

It produces:

```
paper_results/
├── csv/
│   ├── latency_fp32.csv
│   ├── latency_bf16.csv
│   ├── throughput_fp32.csv
│   ├── throughput_bf16.csv
│   ├── accuracy_fp32.csv
│   ├── accuracy_bf16.csv
│   ├── program_cache.csv
│   ├── metal_trace.csv
│   ├── brown_repro.csv
│   ├── ifft_roundtrip.csv
│   └── host_device_split.csv
├── figs/
│   ├── fig_latency.pdf
│   ├── fig_throughput.pdf
│   ├── fig_accuracy.pdf
│   ├── fig_program_cache.pdf
│   ├── fig_metal_trace.pdf
│   └── table_brown.tex     # \begin{tabular} fragment, paste straight into LaTeX
└── logs/
    └── <script>_<timestamp>.log    # full stdout of each bench
```

Each script is also runnable standalone (so you can iterate on one
plot without rerunning the whole sweep):

```bash
python ttnn/cpp/ttnn/operations/experimental/fft/paper/scripts/bench_latency.py \
    --dtype fp32 --precision precise --N 1024,4096,16384 --iters 100
```

---

## 2. What each script measures

### `bench_latency.py`
**What:** Median, p05, p95 wall-clock latency of `ttnn.experimental.fft` over
a sweep of N values, for every (dtype × precision × batch) combination.

**Knobs:**
- `--dtype {fp32,bf16,both}`
- `--precision {precise,fast,both}` (only fp32 has both; bf16 is forced to `fast`)
- `--N N1,N2,...` (default = paper N list)
- `--batch B` (default `1,8,64`)
- `--warmup 5` (calls to absorb plan / JIT cost)
- `--iters 50` (calls used for the percentile)

**CSV columns:**
`N, dtype, precision, batch, median_us, p05_us, p95_us, first_call_us`

**Paper use:** Fig. "latency vs N", and the "first-call vs warm-call"
table the program-cache section will reference.

### `bench_throughput.py`
**What:** GFLOP/s computed from `bench_latency.py` style timings, using
the conventional radix-2 cost model

    flops = 5 · B · N · log₂(N)
    gflops = flops / median_seconds / 1e9

**Knobs:** identical to `bench_latency.py`.

**CSV columns:** `N, dtype, precision, batch, median_us, gflops`

**Paper use:** Fig. "sustained GFLOP/s vs N" (one line per dtype × precision).

### `bench_accuracy.py`
**What:** Relative L2 error of `ttnn.experimental.fft` output vs
`torch.fft.fft(input.to(torch.complex128))`, then cast to the same
working precision so the floor is meaningful.

**CSV columns:** `N, dtype, precision, batch, rel_err`

**Paper use:** Numerical-precision table; precise-vs-fast trade-off plot.

### `bench_program_cache.py`
**What:** For each N, calls the op `--cold` times after clearing the
program cache, then `--warm` times once cached. Reports the
cold/warm ratio.

**CSV columns:** `N, dtype, precision, cold_median_us, warm_median_us, speedup`

**Paper use:** "Program-cache amortizes plan + JIT overhead by Nx" claim.

### `bench_metal_trace.py`
**What:** For each N, records the dispatch sequence into a `ttnn.Trace`
and replays it `--iters` times. Compares against the untraced call.

**CSV columns:** `N, dtype, untraced_us, traced_us, speedup`

**Paper use:** "Trace capture amortizes dispatch overhead by Nx"
claim, particularly important for small-N where dispatch dominates.

### `bench_brown_repro.py`
**What:** Replicates the hero-figure setup from Brown 2025 (length
N = 16384 complex FFT, fp32). Compares to numbers we record in the
script's docstring so you can drop the result straight into Table 1.

**CSV columns:** `N, dtype, precision, median_us, gflops, rel_err`
plus a `table_brown.tex` fragment.

### `bench_ifft_roundtrip.py`
**What:** For representative N values per backend, checks
`ifft(fft(x))` matches `x` within tolerance and reports the rel-err.

**CSV columns:** `N, dtype, precision, batch, fwd_then_inv_rel_err`

**Paper use:** Correctness assertion for the IFFT (conjugate-trick)
implementation.

### `bench_host_device_split.py`
**What:** Wraps `ttnn.experimental.fft` calls with `time.perf_counter_ns`
to measure wall time and, where available, contrasts with the
"already-on-device, no host I/O" cost by reading the same tensor on a
hot loop without re-uploading. Surfaces the universal host-copy cost
documented in `HOST_VS_DEVICE.md`.

**CSV columns:** `N, dtype, e2e_us, no_io_us, host_io_us, host_io_pct`

**Paper use:** Honest disclosure of the host round-trip cost. This is
the data we should cite when we say "ttnn.experimental.fft is
device-resident" — it'll allow the reviewer to see how much of the
wall-clock time is in fact device work today.

### `plot_results.py`
**What:** Reads every CSV under `paper_results/csv/` and produces the
PDFs under `paper_results/figs/`. No measurement is done here; this is
a pure post-processing step so you can iterate on plot styling without
re-running the device.

---

## 3. Reproducing a single figure

If you only want one figure (say the throughput plot):

```bash
python paper/scripts/bench_throughput.py --dtype both --precision both \
    --N 32,256,1024,4096,16384,65536,262144,1048576 \
    --iters 100 --batch 1 \
    --out paper_results/csv/throughput_fp32.csv
python paper/scripts/plot_results.py --only throughput
```

(`--only` filters by the filename stem under `figs/`.)

---

## 4. Bringing the data into the LaTeX source

The `figs/` directory is the contract: every paper figure should
`\includegraphics{paper_results/figs/fig_latency.pdf}`. The brown
replication is special — it emits a complete `\begin{tabular} ...
\end{tabular}` fragment in `figs/table_brown.tex` so you can do

```latex
\input{paper_results/figs/table_brown.tex}
```

and it Just Works.

---

## 5. Reproducing on a clean machine

The kit is committed under the op folder, so:

```bash
git clone …/tt-metal
cd tt-metal
git checkout pkorhale/experimental-fft        # or whichever commit your paper cites
./build_metal.sh -b Release
pip install -e .
bash ttnn/cpp/ttnn/operations/experimental/fft/paper/scripts/run_all.sh
```

is the entire reproduction recipe. Add a citation to your paper's
"Reproducibility" appendix linking the commit hash and that command
sequence.
