// fft_compute_f32.cpp — Radix-2 DIT butterfly compute kernel (VERIFIED v2)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  STATUS: No logic changes.  Comments updated to match fixed pipeline.
// ══════════════════════════════════════════════════════════════════════
//
//  CB ownership (matches fixed reader + writer):
//
//  CB  0  cb_stage0_even_r  ← reader writes (stage 0 only)
//  CB  1  cb_stage0_even_i  ← reader writes (stage 0 only)
//  CB  2  cb_stage0_odd_r   ← reader writes (stage 0 only)
//  CB  3  cb_stage0_odd_i   ← reader writes (stage 0 only)
//  CB  4  cb_tw_r            ← reader writes (all stages)
//  CB  5  cb_tw_i            ← reader writes (all stages)
//  CB  6  cb_next_even_r    ← writer writes (stage 1+), compute reads
//  CB  7  cb_next_even_i    ← writer writes (stage 1+), compute reads
//  CB  8  cb_next_odd_r     ← writer writes (stage 1+), compute reads
//  CB  9  cb_next_odd_i     ← writer writes (stage 1+), compute reads
//  CB 16  cb_out0_r          → compute writes, writer reads
//  CB 17  cb_out0_i          → compute writes, writer reads
//  CB 18  cb_out1_r          → compute writes, writer reads
//  CB 19  cb_out1_i          → compute writes, writer reads
//  CB 20  cb_tmp0            internal scratch (depth=1)
//  CB 21  cb_tmp1            internal scratch (depth=1)
//  CB 22  cb_tmp2            t_r scratch       (depth=1)
//  CB 23  cb_tmp3            t_i scratch       (depth=1)
//
//  Per-butterfly compute sequence (5 FPU sessions):
//
//  A1: tw_r*odd_r → tmp0,  tw_i*odd_i → tmp1
//   B: tmp0 - tmp1 → tmp2   (t_r = real part of twiddle×odd)
//  A2: tw_r*odd_i → tmp0,  tw_i*odd_r → tmp1
//   C: tmp0 + tmp1 → tmp3   (t_i = imag part of twiddle×odd)
//   D: even+t → out0,  even-t → out1
//
//  Each session has exactly one tile_regs_acquire … tile_regs_release
//  with no CB operations inside it.  This is required because CB ops
//  and tile_regs ops share internal state on Tensix.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENTS
// ══════════════════════════════════════════════════════════════════════
//
//  [0] num_stages      (log2_row)
//  [1] tiles_per_stage (tiles_per_row)
//  [2] rows_per_core
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(2);

    // ── CB indices ────────────────────────────────────────────────────
    constexpr uint32_t cb_stage0_even_r = 0;
    constexpr uint32_t cb_stage0_even_i = 1;
    constexpr uint32_t cb_stage0_odd_r  = 2;
    constexpr uint32_t cb_stage0_odd_i  = 3;

    constexpr uint32_t cb_next_even_r   = 6;
    constexpr uint32_t cb_next_even_i   = 7;
    constexpr uint32_t cb_next_odd_r    = 8;
    constexpr uint32_t cb_next_odd_i    = 9;

    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;

    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    // Scratch CBs (depth=1 each)
    constexpr uint32_t cb_tmp0   = 20;   // tw_r*odd_r, then tw_r*odd_i
    constexpr uint32_t cb_tmp1   = 21;   // tw_i*odd_i, then tw_i*odd_r
    constexpr uint32_t cb_tmp2   = 22;   // t_r  = tw_r*odd_r - tw_i*odd_i
    constexpr uint32_t cb_tmp3   = 23;   // t_i  = tw_r*odd_i + tw_i*odd_r

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) {
        return;
    }

    // Sticky FPU config — valid for the lifetime of this kernel.
    // Called once here so the FPU is not reconfigured on every tile.
    binary_op_init_common(cb_stage0_even_r, cb_stage0_odd_r, cb_tmp0);

    for (uint32_t row = 0; row < rows_per_core; row++) {
        for (uint32_t stage = 0; stage < num_stages; stage++) {

            // Stage 0 reads from the reader-filled CBs (0-3).
            // Stage 1+ reads from the writer-shuffled CBs (6-9).
            const uint32_t cb_even_r = (stage == 0) ? cb_stage0_even_r : cb_next_even_r;
            const uint32_t cb_even_i = (stage == 0) ? cb_stage0_even_i : cb_next_even_i;
            const uint32_t cb_odd_r  = (stage == 0) ? cb_stage0_odd_r  : cb_next_odd_r;
            const uint32_t cb_odd_i  = (stage == 0) ? cb_stage0_odd_i  : cb_next_odd_i;

            for (uint32_t t = 0; t < tiles_per_stage; t++) {

                // ── Wait for all inputs up front ──────────────────────
                cb_wait_front(cb_tw_r,   1);
                cb_wait_front(cb_tw_i,   1);
                cb_wait_front(cb_odd_r,  1);
                cb_wait_front(cb_odd_i,  1);
                cb_wait_front(cb_even_r, 1);
                cb_wait_front(cb_even_i, 1);

                // =====================================================
                // SESSION A1: tw_r*odd_r → tmp0,  tw_i*odd_i → tmp1
                // =====================================================
                cb_reserve_back(cb_tmp0, 1);
                tile_regs_acquire();
                mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);
                mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
                cb_push_back(cb_tmp0, 1);

                cb_reserve_back(cb_tmp1, 1);
                tile_regs_acquire();
                mul_tiles_init(cb_tw_i, cb_odd_i, cb_tmp1);
                mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp1);
                tile_regs_release();
                cb_push_back(cb_tmp1, 1);

                // =====================================================
                // SESSION B: t_r = tmp0 - tmp1 → tmp2
                // =====================================================
                cb_wait_front(cb_tmp0, 1);
                cb_wait_front(cb_tmp1, 1);

                cb_reserve_back(cb_tmp2, 1);
                tile_regs_acquire();
                sub_tiles_init(cb_tmp0, cb_tmp1, cb_tmp2);
                sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp2);
                tile_regs_release();
                cb_push_back(cb_tmp2, 1);

                cb_pop_front(cb_tmp0, 1);
                cb_pop_front(cb_tmp1, 1);

                // =====================================================
                // SESSION A2: tw_r*odd_i → tmp0,  tw_i*odd_r → tmp1
                // =====================================================
                cb_reserve_back(cb_tmp0, 1);
                tile_regs_acquire();
                mul_tiles_init(cb_tw_r, cb_odd_i, cb_tmp0);
                mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
                cb_push_back(cb_tmp0, 1);

                cb_reserve_back(cb_tmp1, 1);
                tile_regs_acquire();
                mul_tiles_init(cb_tw_i, cb_odd_r, cb_tmp1);
                mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp1);
                tile_regs_release();
                cb_push_back(cb_tmp1, 1);

                // Twiddle and odd inputs fully consumed — release them.
                cb_pop_front(cb_tw_r, 1);
                cb_pop_front(cb_tw_i, 1);
                cb_pop_front(cb_odd_r, 1);
                cb_pop_front(cb_odd_i, 1);

                // =====================================================
                // SESSION C: t_i = tmp0 + tmp1 → tmp3
                // =====================================================
                cb_wait_front(cb_tmp0, 1);
                cb_wait_front(cb_tmp1, 1);

                cb_reserve_back(cb_tmp3, 1);
                tile_regs_acquire();
                add_tiles_init(cb_tmp0, cb_tmp1, cb_tmp3);
                add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp3);
                tile_regs_release();
                cb_push_back(cb_tmp3, 1);

                cb_pop_front(cb_tmp0, 1);
                cb_pop_front(cb_tmp1, 1);

                // =====================================================
                // SESSION D: butterfly outputs
                //   out0 = even + t   (upper butterfly arm)
                //   out1 = even - t   (lower butterfly arm)
                // =====================================================
                cb_wait_front(cb_tmp2, 1);
                cb_wait_front(cb_tmp3, 1);

                // out0_r = even_r + t_r
                cb_reserve_back(cb_out0_r, 1);
                tile_regs_acquire();
                add_tiles_init(cb_even_r, cb_tmp2, cb_out0_r);
                add_tiles(cb_even_r, cb_tmp2, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out0_r);
                tile_regs_release();
                cb_push_back(cb_out0_r, 1);

                // out0_i = even_i + t_i
                cb_reserve_back(cb_out0_i, 1);
                tile_regs_acquire();
                add_tiles_init(cb_even_i, cb_tmp3, cb_out0_i);
                add_tiles(cb_even_i, cb_tmp3, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out0_i);
                tile_regs_release();
                cb_push_back(cb_out0_i, 1);

                // out1_r = even_r - t_r
                cb_reserve_back(cb_out1_r, 1);
                tile_regs_acquire();
                sub_tiles_init(cb_even_r, cb_tmp2, cb_out1_r);
                sub_tiles(cb_even_r, cb_tmp2, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out1_r);
                tile_regs_release();
                cb_push_back(cb_out1_r, 1);

                // out1_i = even_i - t_i
                cb_reserve_back(cb_out1_i, 1);
                tile_regs_acquire();
                sub_tiles_init(cb_even_i, cb_tmp3, cb_out1_i);
                sub_tiles(cb_even_i, cb_tmp3, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out1_i);
                tile_regs_release();
                cb_push_back(cb_out1_i, 1);

                // Even inputs and scratch t_r/t_i consumed.
                cb_pop_front(cb_even_r, 1);
                cb_pop_front(cb_even_i, 1);
                cb_pop_front(cb_tmp2, 1);
                cb_pop_front(cb_tmp3, 1);
            }
        }
    }
}