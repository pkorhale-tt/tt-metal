#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# HPEC 2026 paper benchmark driver.
# Run from inside a built tt-metal tree on a Wormhole machine:
#
#   cd $TT_METAL_HOME
#   bash tt_metal/programming_examples/fft_universal/paper/run_paper_sweep.sh
#
# Produces:
#   paper_results/universal_sweep.csv        — main sweep (Figs 1, 2, 4)
#   paper_results/brown_repro_N16384.txt     — Brown Table-1 replica (Fig 3)
#   paper_results/build.log                  — build log
#
# Then:
#   python tt_metal/programming_examples/fft_universal/paper/plot_universal.py \
#       --csv paper_results/universal_sweep.csv \
#       --out paper_results/figs/

set -euo pipefail

# Use the standard tt-metal build dir layout.
: "${TT_METAL_HOME:?Set TT_METAL_HOME to your tt-metal checkout}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_HOME/build}"
OUT_DIR="${OUT_DIR:-$TT_METAL_HOME/paper_results}"
ITERS="${ITERS:-50}"

mkdir -p "$OUT_DIR/figs"

echo ">>> Step 1: build sweep + benchmark binaries"
ninja -C "$BUILD_DIR" \
    metal_example_fft_universal_sweep \
    metal_example_fft_universal_benchmark \
    2>&1 | tee "$OUT_DIR/build.log"

SWEEP_BIN="$BUILD_DIR/programming_examples/fft_universal/metal_example_fft_universal_sweep"
BENCH_BIN="$BUILD_DIR/programming_examples/fft_universal/metal_example_fft_universal_benchmark"

if [[ ! -x "$SWEEP_BIN" ]]; then
    echo "FAIL: sweep binary not found at $SWEEP_BIN"; exit 1
fi

echo
echo ">>> Step 2: full N sweep (this is the main paper data)"
"$SWEEP_BIN" \
    --csv "$OUT_DIR/universal_sweep.csv" \
    --iters "$ITERS" \
    --round-trip \
    2>&1 | tee "$OUT_DIR/universal_sweep.log"

echo
echo ">>> Step 3: Brown et al. 2025 N=16384 Table-1 replica"
"$BENCH_BIN" 16384 200 \
    2>&1 | tee "$OUT_DIR/brown_repro_N16384.txt"

echo
echo ">>> Step 4: bonus — N=1048576 for the 'big-N' Table-3 replica"
"$BENCH_BIN" 1048576 50 \
    2>&1 | tee "$OUT_DIR/brown_repro_N1048576.txt"

echo
echo ">>> Done. Inputs ready for plot_universal.py:"
echo "    $OUT_DIR/universal_sweep.csv"
echo "    $OUT_DIR/brown_repro_N16384.txt"
echo "    $OUT_DIR/brown_repro_N1048576.txt"
