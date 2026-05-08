// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// packed_dft_bf16_common.h — shared layout for the TRUE-bf16 packed direct-DFT
// kernel trio.
//
// Identical dispatch shape to the fp32 packed_dft kernels (see
// ../fft_universal/kernel/packed_dft_common.h) but the circular buffers and
// DRAM tiles carry bfloat16 (2 B / element) and the FPU matmul runs with
// bf16 srcA / srcB into an fp32 DST accumulator (fp32_dest_acc_en=true),
// then packs fp32 → bf16 back into the output CB. That's the only "true bf16
// compute" path on Wormhole — the SFPU has no native bf16 math, so a
// matmul-based DFT is what buys us genuine bf16 throughput.
//
// Per-tile math stays the same:
//   out[i, k] = Σ_n in[i, n] * T[n, k]      where T[n, k] = exp(-2πi k n / N)
// Complex 32x32 matmul is 4 real 32x32 matmuls that all accumulate into DST:
//   out_R = in_R · T_R  +  in_I · (-T_I)
//   out_I = in_R · T_I  +  in_I · T_R
// Host pre-negates T_I once into T_I_neg so every device matmul is adding.
//
// Tile layout (bf16 storage):
//   Tile size          = 32 × 32 × 2 bytes = 2048 B
//   Packed rows / tile = 32 sub-FFTs       (one sub-FFT per tile row)
//   Cols 0..N-1        = sub-FFT slots     (valid data)
//   Cols N..31         = zero padding      (matmul propagates zeros)
//
// Precision notes:
//   FPU matmul does bf16 × bf16 multiplies with fp32 accumulation inside
//   the 32-element reduction. N ≤ 32 so we accumulate at most 32 bf16 × bf16
//   products per output element, keeping the rounding depth at ~log2(32)=5
//   bits worst case — well inside bf16's ~8 bits of mantissa. Expected
//   output SNR on random inputs: 40-45 dB (vs fp32 reference).

#pragma once

constexpr uint32_t CB_A       = 0;  // streaming A input  (in_R or in_I, per round)
constexpr uint32_t CB_B       = 1;  // streaming B input  (T_R / T_I / T_I_neg, per round)
constexpr uint32_t CB_OUT_R   = 2;  // output tile, real part
constexpr uint32_t CB_OUT_I   = 3;  // output tile, imag part

constexpr uint32_t PACKED_DFT_BF16_NUM_CBS = 4;

constexpr uint32_t PACKED_ROWS_PER_TILE = 32;   // sub-FFTs per tile
constexpr uint32_t TILE_HW              = 32;
constexpr uint32_t TILE_ELEMS           = TILE_HW * TILE_HW;   // 1024
constexpr uint32_t TILE_SIZE_BF16       = TILE_ELEMS * 2;       // 2048 bytes
