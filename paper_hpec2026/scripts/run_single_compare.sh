#!/usr/bin/env bash
# =============================================================================
# Quick single-point compare: WH vs torch CPU, with energy
# Run from repo root:
#
#   bash paper_hpec2026/scripts/run_single_compare.sh 1048576 float32
#   bash paper_hpec2026/scripts/run_single_compare.sh 16777216 bfloat16
# =============================================================================

set -euo pipefail

N="${1:-1048576}"
DTYPE="${2:-float32}"
WH_POWER="${3:-42}"
CPU_POWER="${4:-353}"

SCRIPT="tests/ttnn/unit_tests/operations/experimental/fft/fft_energy_compare.py"
OUT_DIR="paper_hpec2026/results"
mkdir -p "$OUT_DIR"

echo ">>> compare N=$N dtype=$DTYPE wh=${WH_POWER}W cpu=${CPU_POWER}W"

python "$SCRIPT" \
    --backend compare \
    --n "$N" \
    --dtype "$DTYPE" \
    --warmup 10 \
    --iters  50 \
    --wh-power  "$WH_POWER"  \
    --cpu-power "$CPU_POWER" \
    --cpu-backend torch \
    --json-out "$OUT_DIR/compare_N${N}_${DTYPE}.json"

echo ""
echo "Result saved to: $OUT_DIR/compare_N${N}_${DTYPE}.json"
