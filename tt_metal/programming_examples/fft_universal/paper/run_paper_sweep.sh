#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# HPEC 2026 paper full benchmark driver. Produces every CSV/figure the
# paper references in §5. Run from the tt-metal checkout root on a
# Wormhole + libfftw3-dev machine.
#
#   bash tt_metal/programming_examples/fft_universal/paper/run_paper_sweep.sh
#
# Outputs (under $OUT_DIR, default $TT_METAL_HOME/paper_results):
#   universal_sweep.csv           — main run, packed-DFT ON,  B=1
#   universal_sweep_noPDFT.csv    — ablation: packed-DFT OFF, B=1
#   universal_sweep_B64.csv       — batched throughput,       B=64
#   fftw_baseline.csv             — host CPU FFTW3, single-thread
#   brown_repro_N16384.txt        — Brown 2025 Table-1 replica
#   brown_repro_N1048576.txt      — large-N replica
#   energy_N16384.csv             — energy / J-per-sample (if tt-smi avail)
#   build.log                     — build log
#
# Env knobs:
#   TT_METAL_HOME   — required
#   BUILD_DIR       — default $TT_METAL_HOME/build_Release
#   OUT_DIR         — default $TT_METAL_HOME/paper_results
#   ITERS           — default 50 (sweep iters/N)
#   ITERS_BENCH     — default 200 (Brown replica iters)
#   ITERS_FFTW      — default 200
#   SKIP_FFTW=1     — skip fftw_baseline section
#   SKIP_ENERGY=1   — skip tt-smi energy section

set -euo pipefail

: "${TT_METAL_HOME:?Set TT_METAL_HOME to your tt-metal checkout}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_HOME/build_Release}"
OUT_DIR="${OUT_DIR:-$TT_METAL_HOME/paper_results}"
ITERS="${ITERS:-50}"
ITERS_BENCH="${ITERS_BENCH:-200}"
ITERS_FFTW="${ITERS_FFTW:-200}"

mkdir -p "$OUT_DIR/figs"

echo "============================================================"
echo " HPEC 2026 fft_universal full sweep"
echo "  tt-metal      : $TT_METAL_HOME"
echo "  build dir     : $BUILD_DIR"
echo "  out dir       : $OUT_DIR"
echo "  iters/N       : $ITERS"
echo "  bench iters   : $ITERS_BENCH"
echo "============================================================"
echo

# ─── Step 1: build all binaries we'll need ───────────────────────────
echo ">>> Step 1: build binaries"
TARGETS=(
    metal_example_fft_universal_sweep
    metal_example_fft_universal_benchmark
)
# fftw_baseline is optional — only present if libfftw3-dev was found at
# configure time. Quietly ignore if the target doesn't exist.
if [[ "${SKIP_FFTW:-0}" != "1" ]]; then
    TARGETS+=(fftw_baseline)
fi

ninja -C "$BUILD_DIR" "${TARGETS[@]}" 2>&1 | tee "$OUT_DIR/build.log" || {
    echo "warn: some targets failed to build, continuing with what's available"
}

SWEEP_BIN="$BUILD_DIR/programming_examples/fft_universal/metal_example_fft_universal_sweep"
BENCH_BIN="$BUILD_DIR/programming_examples/fft_universal/metal_example_fft_universal_benchmark"
FFTW_BIN="$BUILD_DIR/programming_examples/fftw_baseline/fftw_baseline"

if [[ ! -x "$SWEEP_BIN" ]]; then
    echo "FAIL: sweep binary not found at $SWEEP_BIN"
    echo "      did the build succeed?  did you set -DBUILD_PROGRAMMING_EXAMPLES=ON ?"
    exit 1
fi

# ─── Step 2: main sweep (Fig 1, 2, 3, 4) ─────────────────────────────
echo
echo ">>> Step 2: main sweep, B=1, packed-DFT ON"
"$SWEEP_BIN" \
    --csv "$OUT_DIR/universal_sweep.csv" \
    --iters "$ITERS" \
    --round-trip \
    2>&1 | tee "$OUT_DIR/universal_sweep.log"

# ─── Step 3: ablation, packed-DFT OFF (Fig 7) ────────────────────────
echo
echo ">>> Step 3: ablation, packed-DFT DISABLED, B=1"
"$SWEEP_BIN" \
    --csv "$OUT_DIR/universal_sweep_noPDFT.csv" \
    --iters "$ITERS" \
    --disable-packed-dft \
    --N-list "2,3,5,7,8,10,13,15,16,24,32,64,128,256,1024" \
    2>&1 | tee "$OUT_DIR/universal_sweep_noPDFT.log"

# ─── Step 4: batched throughput (Fig 8) ──────────────────────────────
echo
echo ">>> Step 4: batched throughput, B=64"
"$SWEEP_BIN" \
    --csv "$OUT_DIR/universal_sweep_B64.csv" \
    --iters 10 \
    --batch 64 \
    --N-list "64,256,1024,4096,16384,65536,262144,1048576,7919,100003" \
    2>&1 | tee "$OUT_DIR/universal_sweep_B64.log"

# ─── Step 5: Brown 2025 replica ──────────────────────────────────────
echo
echo ">>> Step 5: Brown 2025 Table-1 replica at N=16384"
"$BENCH_BIN" 16384 "$ITERS_BENCH" 2>&1 | tee "$OUT_DIR/brown_repro_N16384.txt"

echo
echo ">>> Step 5b: large-N replica at N=1048576"
"$BENCH_BIN" 1048576 50 2>&1 | tee "$OUT_DIR/brown_repro_N1048576.txt"

# ─── Step 6: FFTW baseline (Fig 5, 6) ────────────────────────────────
if [[ "${SKIP_FFTW:-0}" != "1" ]] && [[ -x "$FFTW_BIN" ]]; then
    echo
    echo ">>> Step 6: host CPU FFTW3 baseline"
    "$FFTW_BIN" \
        --csv "$OUT_DIR/fftw_baseline.csv" \
        --iters "$ITERS_FFTW" \
        2>&1 | tee "$OUT_DIR/fftw_baseline.log"
else
    echo
    echo ">>> Step 6: FFTW baseline skipped (no libfftw3-dev or SKIP_FFTW=1)"
fi

# ─── Step 7: energy ──────────────────────────────────────────────────
if [[ "${SKIP_ENERGY:-0}" != "1" ]] && command -v tt-smi >/dev/null 2>&1; then
    echo
    echo ">>> Step 7: energy (tt-smi sampling, N=16384, 5000 iters)"
    python "$TT_METAL_HOME/tt_metal/programming_examples/fft_universal/paper/tt_smi_energy_sampler.py" \
        --binary "$BENCH_BIN" \
        --N 16384 --iters 5000 \
        --csv "$OUT_DIR/energy_N16384.csv" \
        2>&1 | tee "$OUT_DIR/energy_N16384.log" || \
        echo "warn: energy sampler failed, continuing"
else
    echo
    echo ">>> Step 7: energy skipped (no tt-smi or SKIP_ENERGY=1)"
fi

# ─── Step 8: figures ─────────────────────────────────────────────────
echo
echo ">>> Step 8: generate paper figures"
PLOTDIR="$TT_METAL_HOME/tt_metal/programming_examples/fft_universal/paper"

python "$PLOTDIR/plot_universal.py" \
    --csv "$OUT_DIR/universal_sweep.csv" \
    --out "$OUT_DIR/figs/" || echo "warn: plot_universal failed"

python "$PLOTDIR/combine_results.py" \
    --wh       "$OUT_DIR/universal_sweep.csv" \
    --fftw     "$OUT_DIR/fftw_baseline.csv" \
    --no-pdft  "$OUT_DIR/universal_sweep_noPDFT.csv" \
    --batched  "$OUT_DIR/universal_sweep_B64.csv" \
    --out      "$OUT_DIR/figs/" || echo "warn: combine_results failed"

echo
echo "============================================================"
echo " Done. All outputs in: $OUT_DIR"
echo "  figures:        $OUT_DIR/figs/*.pdf"
echo "  CSVs:           $OUT_DIR/*.csv"
echo "  benchmark logs: $OUT_DIR/*.log, $OUT_DIR/*.txt"
echo "============================================================"
