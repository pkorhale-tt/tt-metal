// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
//  fft_common.h — shared layout for the single-core FFT programming example
//
//  Design (FFT-only, no IFFT):
//    * Entire signal up to N=1024 lives in a single fp32 tile per channel
//      (real, imag).  The "state" is held in CB_STATE_{R,I}, which the reader
//      owns across all log2(N) butterfly stages.
//    * Every CB is sized at a full 32x32 fp32 tile (TILE_SIZE_FP32), so that
//      compute-engine tile ops (add_tiles / mul_tiles / sub_tiles) operate on
//      well-defined, consistent-sized data.
//    * For each stage s:
//        reader: gather `even[i] = state[pair_lo(i, s)]`
//                       `odd[i]  = state[pair_hi(i, s)]`
//                load stage-s twiddle tile from DRAM,
//                push EVEN/ODD/TW  -> compute
//        compute: out0 = even + W*odd,  out1 = even - W*odd
//        reader: scatter out0,out1 back into state at pair_lo/pair_hi.
//    * writer: after the reader signals CB_SYNC, writes state to DRAM.
// ============================================================================

// ── Circular Buffer indices ────────────────────────────────────────────────
constexpr uint32_t CB_EVEN_R    = 0;   // lo element of each butterfly pair
constexpr uint32_t CB_EVEN_I    = 1;
constexpr uint32_t CB_ODD_R     = 2;   // hi element of each butterfly pair
constexpr uint32_t CB_ODD_I     = 3;
constexpr uint32_t CB_TW_R      = 4;   // stage twiddle factors (per pair)
constexpr uint32_t CB_TW_I      = 5;
constexpr uint32_t CB_OUT0_R    = 6;   // even + W*odd
constexpr uint32_t CB_OUT0_I    = 7;
constexpr uint32_t CB_OUT1_R    = 8;   // even - W*odd
constexpr uint32_t CB_OUT1_I    = 9;
constexpr uint32_t CB_TMP_R     = 10;  // cmul intermediates
constexpr uint32_t CB_TMP_I     = 11;
constexpr uint32_t CB_TW_ODD_R  = 12;  // W * odd
constexpr uint32_t CB_TW_ODD_I  = 13;
constexpr uint32_t CB_STATE_R   = 14;  // persistent state (reader-owned)
constexpr uint32_t CB_STATE_I   = 15;
constexpr uint32_t CB_SYNC      = 16;  // reader -> writer signal

constexpr uint32_t NUM_CBS = 17;

// ── Tile geometry ──────────────────────────────────────────────────────────
constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;       // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;          // 4096 bytes
