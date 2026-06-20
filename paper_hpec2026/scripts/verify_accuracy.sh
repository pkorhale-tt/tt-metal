#!/usr/bin/env bash
# =============================================================================
# Accuracy verification — run before putting any numbers in the paper
# Checks all algorithm tiers against numpy float64 reference
# Run from repo root:
#
#   bash paper_hpec2026/scripts/verify_accuracy.sh
# =============================================================================

set -euo pipefail

SCRIPT="tests/ttnn/unit_tests/operations/experimental/fft/fft_energy_compare.py"
OUT_DIR="paper_hpec2026/results/accuracy"
mkdir -p "$OUT_DIR"

check() {
    local N=$1
    local DTYPE=$2
    local BATCH=${3:-4}
    echo -n "  check N=$N dtype=$DTYPE batch=$BATCH ... "
    python "$SCRIPT" \
        --backend check \
        --n "$N" \
        --dtype "$DTYPE" \
        --batch "$BATCH" \
        --json-out "$OUT_DIR/acc_N${N}_${DTYPE}.json" \
    | grep -E "max rel|PASS|FAIL"
}

echo "================================================================"
echo " Accuracy check — WH vs numpy float64 reference"
echo "================================================================"

echo ""
echo "--- Stockham (N<=1024) ---"
check 64    float32 8
check 512   float32 8
check 1024  float32 8
check 1024  bfloat16 8

echo ""
echo "--- Two-pass ---"
check 8192    float32 4
check 65536   float32 4
check 1048576 float32 2
check 1048576 bfloat16 2

echo ""
echo "--- Three-pass ---"
check 2097152  float32 1
check 16777216 float32 1
check 16777216 bfloat16 1

echo ""
echo "--- Bluestein (non-power-of-2) ---"
check 97    float32 4
check 1000  float32 4
check 9999  float32 2
check 64512 float32 2

echo ""
echo "================================================================"
echo " All accuracy checks done. Results in: $OUT_DIR/"
echo "================================================================"
