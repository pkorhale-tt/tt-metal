// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
//  fft_common.h — shared layout for the (multi-core) FFT bf16 example.
//
//  Same multi-core algorithm as the fp32 version (see ../fft/), only the
//  in-CB element format is bfloat16 (2 bytes/element). Compute is still
//  performed in fp32 in the DEST register via UnpackToDestFp32, so the
//  butterfly math itself is fp32; only the storage between stages is bf16.
//  This trades ~3 decimal digits of precision for half the memory traffic.
// ============================================================================

// ── Circular Buffer indices ────────────────────────────────────────────────
constexpr uint32_t CB_EVEN_R    = 0;
constexpr uint32_t CB_EVEN_I    = 1;
constexpr uint32_t CB_ODD_R     = 2;
constexpr uint32_t CB_ODD_I     = 3;
constexpr uint32_t CB_TW_R      = 4;
constexpr uint32_t CB_TW_I      = 5;
constexpr uint32_t CB_OUT0_R    = 6;
constexpr uint32_t CB_OUT0_I    = 7;
constexpr uint32_t CB_OUT1_R    = 8;
constexpr uint32_t CB_OUT1_I    = 9;
constexpr uint32_t CB_TMP_R     = 10;
constexpr uint32_t CB_TMP_I     = 11;
constexpr uint32_t CB_TW_ODD_R  = 12;
constexpr uint32_t CB_TW_ODD_I  = 13;
constexpr uint32_t CB_STATE_R   = 14;
constexpr uint32_t CB_STATE_I   = 15;
constexpr uint32_t CB_SYNC      = 16;
constexpr uint32_t CB_RECV_R    = 17;
constexpr uint32_t CB_RECV_I    = 18;

constexpr uint32_t NUM_CBS = 19;

// ── Tile geometry (bfloat16) ───────────────────────────────────────────────
constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;       // 1024
constexpr uint32_t TILE_SIZE_BF16 = TILE_ELEMS * 2;          // 2048 bytes
