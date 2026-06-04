#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# run_all.sh — orchestrate every paper benchmark and produce all CSVs + figures.
#
# Outputs land under $TT_FFT_PAPER_RESULTS_DIR (defaults to
# $TT_METAL_HOME/paper_results).
#
# Usage:
#     bash ttnn/cpp/ttnn/operations/experimental/fft/paper/scripts/run_all.sh
#     SMOKE=1 bash …/run_all.sh        # short N list for sanity (≈ 2 min)
#     SKIP_PLOTS=1 bash …/run_all.sh   # benches only, no matplotlib step

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${TT_FFT_PAPER_RESULTS_DIR:-${TT_METAL_HOME:-$PWD}/paper_results}"
LOG_DIR="${RESULTS_DIR}/logs"
CSV_DIR="${RESULTS_DIR}/csv"
FIG_DIR="${RESULTS_DIR}/figs"

mkdir -p "${LOG_DIR}" "${CSV_DIR}" "${FIG_DIR}"

export TT_FFT_PAPER_RESULTS_DIR="${RESULTS_DIR}"

ts() { date +"%Y%m%d_%H%M%S"; }

run_bench() {
    local stem="$1"; shift
    local log="${LOG_DIR}/${stem}_$(ts).log"
    echo
    echo "──────────────────────────────────────────────────────────────────"
    echo "▶ ${stem}"
    echo "  args:  $*"
    echo "  log :  ${log}"
    echo "──────────────────────────────────────────────────────────────────"
    if python "${SCRIPT_DIR}/${stem}.py" "$@" 2>&1 | tee "${log}"; then
        echo "✔ ${stem} ok"
    else
        echo "✘ ${stem} failed — see ${log}"
    fi
}

# ────────────────────────── N lists ─────────────────────────────────────
if [[ "${SMOKE:-0}" == "1" ]]; then
    N_FP32="32,1024,4096,16384"
    N_BF16="32,1024,4096,16384"
    N_XL="2097152"
    ITERS=20
    WARMUP=3
else
    N_FP32="32,64,128,256,512,1024,4096,16384,65536,262144,1048576"
    N_BF16="5,7,11,17,32,128,1024,4096,16384,32768,131072,524287"
    N_XL="2097152,4194304,8388608,16777216"
    ITERS=50
    WARMUP=5
fi

# ────────────────────────── benches ─────────────────────────────────────

# Latency — split per dtype so the CSV is paper-ready.
run_bench bench_latency \
    --dtype fp32 --precision both --N "${N_FP32}" \
    --batch 1,8,64 --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${CSV_DIR}/latency_fp32.csv"

run_bench bench_latency \
    --dtype bf16 --precision fast --N "${N_BF16}" \
    --batch 1,8,64 --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${CSV_DIR}/latency_bf16.csv"

# XL is fp32-only and slow → smaller batch list, fewer iters.
run_bench bench_latency \
    --dtype fp32 --precision precise --N "${N_XL}" \
    --batch 1 --warmup 2 --iters 10 \
    --out "${CSV_DIR}/latency_fp32_xl.csv"

# Throughput
run_bench bench_throughput \
    --dtype fp32 --precision both --N "${N_FP32}" \
    --batch 1,8 --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${CSV_DIR}/throughput_fp32.csv"

run_bench bench_throughput \
    --dtype bf16 --precision fast --N "${N_BF16}" \
    --batch 1,8 --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${CSV_DIR}/throughput_bf16.csv"

# Accuracy (1 batch is enough for the rel-err point)
run_bench bench_accuracy \
    --dtype fp32 --precision both --N "${N_FP32}" \
    --batch 1 --warmup 0 --iters 1 \
    --out "${CSV_DIR}/accuracy_fp32.csv"

run_bench bench_accuracy \
    --dtype bf16 --precision fast --N "${N_BF16}" \
    --batch 1 --warmup 0 --iters 1 \
    --out "${CSV_DIR}/accuracy_bf16.csv"

# Program cache
run_bench bench_program_cache \
    --dtype both --precision both \
    --N "32,1024,16384,262144" \
    --warmup 0 --iters 20 \
    --out "${CSV_DIR}/program_cache.csv"

# Metal trace
run_bench bench_metal_trace \
    --dtype both --precision both \
    --N "32,1024,16384,65536" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${CSV_DIR}/metal_trace.csv"

# Brown 2025 hero replication
run_bench bench_brown_repro \
    --dtype both --precision both \
    --warmup 10 --iters 200 \
    --out "${CSV_DIR}/brown_repro.csv"

# IFFT round-trip
run_bench bench_ifft_roundtrip \
    --dtype both --precision both \
    --N "32,128,1024,4096,16384,65536" \
    --batch 1 --warmup 0 --iters 1 \
    --out "${CSV_DIR}/ifft_roundtrip.csv"

# Host vs device split
run_bench bench_host_device_split \
    --dtype both --precision both \
    --N "32,1024,16384,65536,262144,1048576" \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${CSV_DIR}/host_device_split.csv"

# ────────────────────────── plots ───────────────────────────────────────
if [[ "${SKIP_PLOTS:-0}" != "1" ]]; then
    echo
    echo "▶ plot_results.py"
    python "${SCRIPT_DIR}/plot_results.py"
fi

echo
echo "All results: ${RESULTS_DIR}"
ls -1 "${CSV_DIR}" 2>/dev/null | sed 's/^/  csv: /'
ls -1 "${FIG_DIR}" 2>/dev/null | sed 's/^/  fig: /'
