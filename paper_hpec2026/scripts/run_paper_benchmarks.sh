#!/usr/bin/env bash
# =============================================================================
# HPEC 2026 — Paper benchmark runner
# Run this script on the Wormhole machine from the repo root:
#
#   cd /proj_sw/user_dev/pkorhale/fft_metal_clean
#   source python_env/bin/activate
#   bash paper_hpec2026/scripts/run_paper_benchmarks.sh
#
# Output: paper_hpec2026/results/  (one JSON per (N, dtype))
# =============================================================================

set -euo pipefail

SCRIPT="tests/ttnn/unit_tests/operations/experimental/fft/benchmark_fft.py"
OUT_DIR="paper_hpec2026/results"
mkdir -p "$OUT_DIR"

# Measured power — update these if your machine differs
WH_POWER=42
CPU_POWER=353

WARMUP=10
RUNS=50

echo "================================================================"
echo " HPEC 2026 FFT Benchmark"
echo " WH power : ${WH_POWER} W"
echo " CPU power: ${CPU_POWER} W"
echo " warmup=${WARMUP}  runs=${RUNS}"
echo "================================================================"

# ------------------------------------------------------------------
# TABLE I — device throughput across all algorithm tiers
# ------------------------------------------------------------------
echo ""
echo ">>> TABLE I: all tiers, fp32 + bf16"
python "$SCRIPT" \
    --dtype both \
    --warmup "$WARMUP" \
    --runs   "$RUNS"   \
    --wh-power  "$WH_POWER"  \
    --cpu-power "$CPU_POWER" \
    --csv "$OUT_DIR/table1_all_tiers.csv"

# ------------------------------------------------------------------
# TABLE II — energy comparison at key N values
# (fp32: crossover point, peak throughput point)
# (bf16: peak energy advantage point)
# ------------------------------------------------------------------
echo ""
echo ">>> TABLE II fp32 — N=131072, 1048576"
python "$SCRIPT" \
    --two-pass \
    --dtype fp32 \
    --warmup "$WARMUP" \
    --runs   "$RUNS"   \
    --wh-power  "$WH_POWER"  \
    --cpu-power "$CPU_POWER" \
    --csv "$OUT_DIR/table2_twpass_fp32.csv"

echo ""
echo ">>> TABLE II bf16 — N=16777216 (three-pass, best energy ratio)"
python "$SCRIPT" \
    --three-pass \
    --dtype bf16 \
    --warmup "$WARMUP" \
    --runs   "$RUNS"   \
    --wh-power  "$WH_POWER"  \
    --cpu-power "$CPU_POWER" \
    --csv "$OUT_DIR/table2_threepass_bf16.csv"

# ------------------------------------------------------------------
# Bluestein — for coverage table
# ------------------------------------------------------------------
echo ""
echo ">>> Bluestein fp32 (coverage)"
python "$SCRIPT" \
    --bluestein \
    --dtype fp32 \
    --warmup "$WARMUP" \
    --runs   "$RUNS"   \
    --wh-power  "$WH_POWER"  \
    --cpu-power "$CPU_POWER" \
    --csv "$OUT_DIR/bluestein_fp32.csv"

echo ""
echo "================================================================"
echo " All done. Results saved to: $OUT_DIR/"
echo " Files:"
ls -lh "$OUT_DIR/"
echo "================================================================"
