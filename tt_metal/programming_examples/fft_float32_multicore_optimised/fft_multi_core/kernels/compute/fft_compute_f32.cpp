// fft_compute_f32.cpp  — MULTICORE butterfly kernel (FIXED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ROOT CAUSE OF CORRUPTION (previous version):
//   mul_tiles() advances the LLK read pointer of each source CB internally.
//   When two separate tile_regs sessions both called mul_tiles on the same
//   source CBs (tw_r, tw_i, odd_r, odd_i), the second session read the NEXT
//   tile in those CBs — a different row's data — producing NaN/Inf.
//
// FIX — 4-phase butterfly, each CB read exactly once:
//
//   Phase 1 (session A): Compute all 4 partial products from tw/odd CBs.
//                        Pack all 4 to scratch CBs. Pop tw/odd immediately.
//   Phase 2 (session B): Subtract partials → t_r. Pack to cb_tmp0.
//   Phase 3 (session C): Add partials → t_i. Pack to cb_tmp1.
//   Phase 4 (session D): Compute even±t, pack all 4 outputs. Pop even+scratch.
//
//   tw_r, tw_i, odd_r, odd_i: read in Phase 1 only, popped immediately after.
//   even_r, even_i: read in Phase 4 only, popped immediately after.
//   No CB is read in more than one session.
//
// OPTIMIZATIONS:
//   - mul/add/sub_tiles_init() called once per stage (outside tile loop)
//   - All input CBs waited on together at start of tile loop
//   - All 4 output CBs packed before tile_regs_release() in Phase 4
//
// CB map:
//   0  cb_even_r    stage input even real
//   1  cb_even_i    stage input even imag
//   2  cb_odd_r     stage input odd  real
//   3  cb_odd_i     stage input odd  imag
//   4  cb_tw_r      expanded twiddle real
//   5  cb_tw_i      expanded twiddle imag
//  16  cb_out0_r    butterfly sum real
//  17  cb_out0_i    butterfly sum imag
//  18  cb_out1_r    butterfly diff real
//  19  cb_out1_i    butterfly diff imag
//  20  cb_tmp0      scratch (tw_r*odd_r and tw_r*odd_i staging, then t_r)
//  21  cb_tmp1      scratch (tw_i*odd_i and tw_i*odd_r staging, then t_i)

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

// CB indices for the two extra scratch CBs used as t_i partials staging
// We reuse cb_tmp0/cb_tmp1 with a push/pop protocol — they are FIFO CBs.
// Depth must be >= 2 tiles (configured in host). After Phase 1 they hold:
//   cb_tmp0: [tw_r*odd_r, tw_r*odd_i]  (2 tiles)
//   cb_tmp1: [tw_i*odd_i, tw_i*odd_r]  (2 tiles)
// After Phase 2 (t_r sub), cb_tmp0 front = t_r (1 tile).
// After Phase 3 (t_i add), cb_tmp1 front = t_i (1 tile).

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_tmp0   = 20;
    constexpr uint32_t cb_tmp1   = 21;

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        // OPTIMIZATION: init once per stage, sticky until next stage.
        mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);

        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // OPTIMIZATION: coalesced wait — all inputs before any math.
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // ── PHASE 1: 4 partial products from tw and odd CBs ───────────
            // Each of tw_r, tw_i, odd_r, odd_i is read ONCE in this session.
            // Results packed to cb_tmp0 (x2) and cb_tmp1 (x2) before release.
            // tw/odd CBs popped immediately after — never read again.

            cb_reserve_back(cb_tmp0, 2);
            cb_reserve_back(cb_tmp1, 2);

            tile_regs_acquire();

            mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);   // slot 0 = tw_r * odd_r

            mul_tiles_init(cb_tw_i, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);   // slot 1 = tw_i * odd_i

            mul_tiles_init(cb_tw_r, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 2);   // slot 2 = tw_r * odd_i

            mul_tiles_init(cb_tw_i, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 3);   // slot 3 = tw_i * odd_r

            tile_regs_commit();
            tile_regs_wait();

            // Pack: tmp0[0] = tw_r*odd_r, tmp0[1] = tw_r*odd_i
            pack_tile(0, cb_tmp0);
            pack_tile(2, cb_tmp0);
            // Pack: tmp1[0] = tw_i*odd_i, tmp1[1] = tw_i*odd_r
            pack_tile(1, cb_tmp1);
            pack_tile(3, cb_tmp1);

            tile_regs_release();

            cb_push_back(cb_tmp0, 2);
            cb_push_back(cb_tmp1, 2);

            // Pop all tw/odd CBs — fully consumed, never re-read.
            cb_pop_front(cb_tw_r,  1);
            cb_pop_front(cb_tw_i,  1);
            cb_pop_front(cb_odd_r, 1);
            cb_pop_front(cb_odd_i, 1);

            // ── PHASE 2: t_r = (tw_r*odd_r) - (tw_i*odd_i) ──────────────
            // tmp0 front = tw_r*odd_r, tmp1 front = tw_i*odd_i

            cb_wait_front(cb_tmp0, 2);
            cb_wait_front(cb_tmp1, 2);

            cb_reserve_back(cb_tmp0, 1); // will hold t_r

            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1, cb_tmp0);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);    // slot 0 = t_r
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0);                    // pack t_r
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);

            // Pop the two partial products just consumed (indices 0 of each)
            cb_pop_front(cb_tmp0, 1); // tw_r*odd_r
            cb_pop_front(cb_tmp1, 1); // tw_i*odd_i

            // ── PHASE 3: t_i = (tw_r*odd_i) + (tw_i*odd_r) ──────────────
            // tmp0 front = tw_r*odd_i, tmp1 front = tw_i*odd_r

            cb_reserve_back(cb_tmp1, 1); // will hold t_i

            tile_regs_acquire();
            add_tiles_init(cb_tmp0, cb_tmp1, cb_tmp1);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);    // slot 0 = t_i
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1);                    // pack t_i
            tile_regs_release();

            cb_push_back(cb_tmp1, 1);

            // Pop the two partial products just consumed
            cb_pop_front(cb_tmp0, 1); // tw_r*odd_i
            cb_pop_front(cb_tmp1, 1); // tw_i*odd_r

            // ── PHASE 4: output even ± t ─────────────────────────────────
            // cb_tmp0 front = t_r, cb_tmp1 front = t_i
            // even_r, even_i still live (not yet popped)
            // All 4 outputs computed and packed in one session.

            cb_wait_front(cb_tmp0, 1); // t_r
            cb_wait_front(cb_tmp1, 1); // t_i

            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);

            tile_regs_acquire();

            add_tiles_init(cb_even_r, cb_tmp0, cb_out0_r);
            add_tiles(cb_even_r, cb_tmp0, 0, 0, 0);  // slot 0 = even_r + t_r

            add_tiles_init(cb_even_i, cb_tmp1, cb_out0_i);
            add_tiles(cb_even_i, cb_tmp1, 0, 0, 1);  // slot 1 = even_i + t_i

            sub_tiles_init(cb_even_r, cb_tmp0, cb_out1_r);
            sub_tiles(cb_even_r, cb_tmp0, 0, 0, 2);  // slot 2 = even_r - t_r

            sub_tiles_init(cb_even_i, cb_tmp1, cb_out1_i);
            sub_tiles(cb_even_i, cb_tmp1, 0, 0, 3);  // slot 3 = even_i - t_i

            tile_regs_commit();
            tile_regs_wait();

            // Pack all 4 outputs while register file is live
            pack_tile(0, cb_out0_r);
            pack_tile(1, cb_out0_i);
            pack_tile(2, cb_out1_r);
            pack_tile(3, cb_out1_i);

            tile_regs_release();

            cb_push_back(cb_out0_r, 1);
            cb_push_back(cb_out0_i, 1);
            cb_push_back(cb_out1_r, 1);
            cb_push_back(cb_out1_i, 1);

            cb_pop_front(cb_even_r, 1);
            cb_pop_front(cb_even_i, 1);
            cb_pop_front(cb_tmp0,   1); // t_r
            cb_pop_front(cb_tmp1,   1); // t_i
        }
    }
}