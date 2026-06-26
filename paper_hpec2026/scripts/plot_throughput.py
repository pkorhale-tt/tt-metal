"""
plot_throughput.py — Export benchmark data as CSV for HPEC 2026 paper.

Usage:
    python plot_throughput.py

Output:
    throughput.csv in the same directory as this script.

Data is from the benchmark run on yyzc-wh-03 (Wormhole n150, 50-run median).
Edit the DATA list below if you re-run benchmarks.
"""

import csv
import os

# Each entry: (N, tier, dtype, gflops_per_s)
DATA = [
    # Stockham fp32 (B=64)
    (32,         "Stockham",    "fp32", 0.30),
    (64,         "Stockham",    "fp32", 0.64),
    (128,        "Stockham",    "fp32", 1.32),
    (256,        "Stockham",    "fp32", 2.58),
    (512,        "Stockham",    "fp32", 4.24),
    (1024,       "Stockham",    "fp32", 6.22),

    # Stockham bf16 (B=64)
    (32,         "Stockham",    "bf16", 0.21),
    (64,         "Stockham",    "bf16", 0.54),
    (128,        "Stockham",    "bf16", 1.08),
    (256,        "Stockham",    "bf16", 2.29),
    (512,        "Stockham",    "bf16", 3.70),
    (1024,       "Stockham",    "bf16", 5.46),

    # Two-pass fp32 (B=1)
    (2048,       "Two-pass",    "fp32", 0.24),
    (4096,       "Two-pass",    "fp32", 0.44),
    (8192,       "Two-pass",    "fp32", 0.74),
    (65536,      "Two-pass",    "fp32", 2.21),
    (131072,     "Two-pass",    "fp32", 2.60),
    (1048576,    "Two-pass",    "fp32", 3.61),

    # Two-pass bf16 (B=1)
    (2048,       "Two-pass",    "bf16", 0.23),
    (4096,       "Two-pass",    "bf16", 0.43),
    (8192,       "Two-pass",    "bf16", 0.72),
    (65536,      "Two-pass",    "bf16", 2.09),
    (131072,     "Two-pass",    "bf16", 2.46),
    (1048576,    "Two-pass",    "bf16", 3.44),

    # Three-pass fp32 (B=1)
    (2097152,    "Three-pass",  "fp32", 1.29),
    (8388608,    "Three-pass",  "fp32", 1.76),
    (16777216,   "Three-pass",  "fp32", 1.99),

    # Three-pass bf16 (B=1)
    (2097152,    "Three-pass",  "bf16", 1.24),
    (8388608,    "Three-pass",  "bf16", 1.67),
    (16777216,   "Three-pass",  "bf16", 1.89),

    # Bluestein fp32 (B=1)
    (97,         "Bluestein",   "fp32", 0.01),
    (127,        "Bluestein",   "fp32", 0.01),
    (257,        "Bluestein",   "fp32", 0.01),
    (509,        "Bluestein",   "fp32", 0.02),
    (1000,       "Bluestein",   "fp32", 0.03),
    (3000,       "Bluestein",   "fp32", 0.07),
    (9999,       "Bluestein",   "fp32", 0.14),
    (64512,      "Bluestein",   "fp32", 0.44),
    (525312,     "Bluestein",   "fp32", 0.14),
    (786432,     "Bluestein",   "fp32", 0.21),

    # Bluestein bf16 (B=1)
    (97,         "Bluestein",   "bf16", 0.00),
    (127,        "Bluestein",   "bf16", 0.00),
    (257,        "Bluestein",   "bf16", 0.01),
    (509,        "Bluestein",   "bf16", 0.02),
    (1000,       "Bluestein",   "bf16", 0.03),
    (3000,       "Bluestein",   "bf16", 0.07),
    (9999,       "Bluestein",   "bf16", 0.13),
    (64512,      "Bluestein",   "bf16", 0.25),
    (525312,     "Bluestein",   "bf16", 0.13),
    (786432,     "Bluestein",   "bf16", 0.20),
]


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "throughput.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "tier", "dtype", "gflops_per_s"])
        for row in DATA:
            writer.writerow(row)

    print(f"Saved: {csv_path}")
    print(f"Rows:  {len(DATA)}")


if __name__ == "__main__":
    main()
