// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────
//  DUPLICATE — keep byte-for-byte in sync with
//    ../dataflow/packed_dft_common.h
//  Two copies exist because the tt-metal kernel build resolves "common.h"
//  includes only against the kernel's own directory, so a single shared
//  header in dataflow/ is invisible to compute/packed_dft_compute.cpp and
//  vice versa. Edit BOTH files for any change.
// ─────────────────────────────────────────────────────────────────────────
//
// packed_dft_common.h — Shared layout for the PACKED DIRECT-DFT kernel trio.
//
// Purpose
// -------
// The existing fft_stockham::batch_fft kernel stores one sub-FFT per tile
// (sub_N valid elements + 1024-sub_N padding zeros).  For small sub_N (3, 5,
// 7, 8, 16, ...) that wastes ~99% of every DRAM/PCIe byte we ship.  This
// kernel fixes that for N <= 32 by packing exactly 32 sub-FFTs per tile —
// each sub-FFT is one row (32 slots) of a 32x32 tile, using the first N
// slots and zero-padding the remaining (32 - N) slots.
//
// The direct DFT is then a complex 32x32 matmul per packed tile:
//   out[i, k] = Σ_n in[i, n] * T[n, k]      where T[n, k] = exp(-2πi k n / N)
// with `i` = sub-FFT index within the packed tile, and n, k running over
// the sub-FFT's own axis.  Positions [N, 32) of T are zero, so the padded
// input/twiddle slots naturally contribute 0 and the same padding pattern
// propagates cleanly to the output.
//
// Complex 32x32 matmul = 4 real 32x32 matmuls:
//   out_R = in_R · T_R  +  in_I · (-T_I)     ← both accumulate into DST(0)
//   out_I = in_R · T_I  +  in_I · T_R        ← both accumulate into DST(0)
// The host pre-negates T_I once and ships T_I_neg as a third twiddle tile
// so every matmul is an *adding* matmul — no SFPU subtract needed.
//
// Tile layout
// -----------
//   Input tile t  ==  packed rows [t * 32 .. t * 32 + 31] (zero-padded at
//   the end if count % 32 != 0). Real and imag parts live in separate DRAM
//   buffers so we can use plain fp32 tiles throughout.
//
//   Twiddle tiles (one per DFT size N, single tile each):
//     tw_r_buf     : T_R[n, k] = cos(-2π k n / N),   zeros outside [0, N)²
//     tw_i_buf     : T_I[n, k] = sin(-2π k n / N),   zeros outside [0, N)²
//     tw_i_neg_buf : -T_I[n, k]
//
// Dispatch model
// --------------
//   Each core owns `tiles_per_core` consecutive tiles of the packed batch
//   and consumes / produces them in order. No cross-core stages exist —
//   sub-FFT math is purely intra-tile, so the core is fully parallel over
//   tiles. num_cores = min(num_tiles, 64), same formula the batch_fft and
//   pass2 plans use.
//
// CB plan
// -------
//   Reader pushes 4 (A, B) tile pairs per output tile into CB_A / CB_B:
//     pair 1: (in_R, T_R)      for out_R += in_R · T_R
//     pair 2: (in_I, T_I_neg)  for out_R += in_I · (-T_I)
//     pair 3: (in_R, T_I)      for out_I += in_R · T_I
//     pair 4: (in_I, T_R)      for out_I += in_I · T_R
//   Compute performs 4 matmul_tiles and 2 pack_tile (CB_OUT_R, CB_OUT_I).
//   Writer drains (CB_OUT_R, CB_OUT_I) and writes to DRAM.
//
//   CB depths: CB_A / CB_B are 4 tiles deep so the reader can queue all
//   four pairs upfront and let the compute pipeline drain them in order.

#pragma once

constexpr uint32_t CB_A       = 0;  // streaming A input  (in_R or in_I, per round)
constexpr uint32_t CB_B       = 1;  // streaming B input  (T_R / T_I / T_I_neg, per round)
constexpr uint32_t CB_OUT_R   = 2;  // output tile, real part
constexpr uint32_t CB_OUT_I   = 3;  // output tile, imag part

constexpr uint32_t PACKED_DFT_NUM_CBS = 4;

constexpr uint32_t PACKED_ROWS_PER_TILE = 32;   // i.e. sub-FFTs per tile
constexpr uint32_t TILE_HW              = 32;
constexpr uint32_t TILE_ELEMS           = TILE_HW * TILE_HW;   // 1024
constexpr uint32_t TILE_SIZE_FP32       = TILE_ELEMS * 4;       // 4096 bytes
