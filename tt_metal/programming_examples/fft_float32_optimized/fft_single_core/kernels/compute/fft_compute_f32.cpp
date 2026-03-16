// fft_compute_f32.cpp  — OPTIMISED MULTI-STAGE FFT
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//   1. direction evaluated ONCE — inner loop no longer branches on it.
//   2. IFFT per-tile twiddle negation removed — host pre-negates tw_i.
//   3. cb_neg_tw_i (CB 24) removed entirely.
//   4. Updated API: binary_op_init_common / mul_tiles_init / add_tiles_init /
//      sub_tiles_init all now take a third ocb argument (output CB).

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

void kernel_main() {
    const uint32_t direction       = get_arg_val<uint32_t>(0);
    const uint32_t num_stages      = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(2);

    // direction no longer used inside kernel — twiddle sign baked on host.
    (void)direction;

    // CB indices
    constexpr uint32_t cb_in_even_r   = 0;
    constexpr uint32_t cb_in_even_i   = 1;
    constexpr uint32_t cb_in_odd_r    = 2;
    constexpr uint32_t cb_in_odd_i    = 3;
    constexpr uint32_t cb_tw_r        = 4;
    constexpr uint32_t cb_tw_i        = 5;
    constexpr uint32_t cb_pong_odd_r  = 6;
    constexpr uint32_t cb_pong_odd_i  = 7;
    constexpr uint32_t cb_ping_even_r = 10;
    constexpr uint32_t cb_ping_even_i = 11;
    constexpr uint32_t cb_ping_odd_r  = 12;
    constexpr uint32_t cb_ping_odd_i  = 13;
    constexpr uint32_t cb_pong_even_r = 14;
    constexpr uint32_t cb_pong_even_i = 15;
    constexpr uint32_t cb_out0_r      = 16;
    constexpr uint32_t cb_out0_i      = 17;
    constexpr uint32_t cb_out1_r      = 18;
    constexpr uint32_t cb_out1_i      = 19;
    constexpr uint32_t cb_tmp0        = 20;
    constexpr uint32_t cb_tmp1        = 21;
    constexpr uint32_t cb_tw_odd_r    = 22;
    constexpr uint32_t cb_tw_odd_i    = 23;
    // CB 24 removed — IFFT negation now done on host.

    // One-time init: icb0, icb1, ocb  (updated 3-arg API)
    binary_op_init_common(cb_in_even_r, cb_in_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        // ── Select source CBs ─────────────────────────────────────
        uint32_t src_even_r, src_even_i;
        uint32_t src_odd_r,  src_odd_i;

        if (stage == 0) {
            src_even_r = cb_in_even_r;
            src_even_i = cb_in_even_i;
            src_odd_r  = cb_in_odd_r;
            src_odd_i  = cb_in_odd_i;
        } else if ((stage & 1) == 1) {
            src_even_r = cb_ping_even_r;
            src_even_i = cb_ping_even_i;
            src_odd_r  = cb_ping_odd_r;
            src_odd_i  = cb_ping_odd_i;
        } else {
            src_even_r = cb_pong_even_r;
            src_even_i = cb_pong_even_i;
            src_odd_r  = cb_pong_odd_r;
            src_odd_i  = cb_pong_odd_i;
        }

        // ── Select destination CBs ────────────────────────────────
        uint32_t dst0_r, dst0_i;
        uint32_t dst1_r, dst1_i;

        if (stage == num_stages - 1) {
            dst0_r = cb_out0_r;  dst0_i = cb_out0_i;
            dst1_r = cb_out1_r;  dst1_i = cb_out1_i;
        } else if ((stage & 1) == 0) {
            dst0_r = cb_ping_even_r;  dst0_i = cb_ping_even_i;
            dst1_r = cb_ping_odd_r;   dst1_i = cb_ping_odd_i;
        } else {
            dst0_r = cb_pong_even_r;  dst0_i = cb_pong_even_i;
            dst1_r = cb_pong_odd_r;   dst1_i = cb_pong_odd_i;
        }

        // ── Process all tiles in this stage ───────────────────────
        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            cb_wait_front(cb_tw_r,    1);
            cb_wait_front(cb_tw_i,    1);
            cb_wait_front(src_odd_r,  1);
            cb_wait_front(src_odd_i,  1);
            cb_wait_front(src_even_r, 1);
            cb_wait_front(src_even_i, 1);

            // ════════════════════════════════════════════════════════
            // Step 1: t_r = tw_r * odd_r  -  tw_i * odd_i
            // ════════════════════════════════════════════════════════

            // tmp0 = tw_r * odd_r
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, src_odd_r, cb_tmp0);
            mul_tiles(cb_tw_r, src_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = tw_i * odd_i
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, src_odd_i, cb_tmp1);
            mul_tiles(cb_tw_i, src_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            // tw_odd_r = tmp0 - tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_r);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tw_odd_r);
            tile_regs_release();
            cb_push_back(cb_tw_odd_r, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // ════════════════════════════════════════════════════════
            // Step 2: t_i = tw_r * odd_i  +  tw_i * odd_r
            // ════════════════════════════════════════════════════════

            // tmp0 = tw_r * odd_i
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, src_odd_i, cb_tmp0);
            mul_tiles(cb_tw_r, src_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = tw_i * odd_r
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, src_odd_r, cb_tmp1);
            mul_tiles(cb_tw_i, src_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            // tw_odd_i = tmp0 + tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_i);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tw_odd_i);
            tile_regs_release();
            cb_push_back(cb_tw_odd_i, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // ════════════════════════════════════════════════════════
            // Step 3: out0 = even + t
            // ════════════════════════════════════════════════════════
            cb_wait_front(cb_tw_odd_r, 1);
            cb_wait_front(cb_tw_odd_i, 1);

            // out0_r = even_r + tw_odd_r
            cb_reserve_back(dst0_r, 1);
            tile_regs_acquire();
            add_tiles_init(src_even_r, cb_tw_odd_r, dst0_r);
            add_tiles(src_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst0_r);
            tile_regs_release();
            cb_push_back(dst0_r, 1);

            // out0_i = even_i + tw_odd_i
            cb_reserve_back(dst0_i, 1);
            tile_regs_acquire();
            add_tiles_init(src_even_i, cb_tw_odd_i, dst0_i);
            add_tiles(src_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst0_i);
            tile_regs_release();
            cb_push_back(dst0_i, 1);

            // ════════════════════════════════════════════════════════
            // Step 4: out1 = even - t
            // ════════════════════════════════════════════════════════

            // out1_r = even_r - tw_odd_r
            cb_reserve_back(dst1_r, 1);
            tile_regs_acquire();
            sub_tiles_init(src_even_r, cb_tw_odd_r, dst1_r);
            sub_tiles(src_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst1_r);
            tile_regs_release();
            cb_push_back(dst1_r, 1);

            // out1_i = even_i - tw_odd_i
            cb_reserve_back(dst1_i, 1);
            tile_regs_acquire();
            sub_tiles_init(src_even_i, cb_tw_odd_i, dst1_i);
            sub_tiles(src_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst1_i);
            tile_regs_release();
            cb_push_back(dst1_i, 1);

            // ════════════════════════════════════════════════════════
            // Step 5: Pop all consumed inputs
            // ════════════════════════════════════════════════════════
            cb_pop_front(cb_tw_r,     1);
            cb_pop_front(cb_tw_i,     1);
            cb_pop_front(src_odd_r,   1);
            cb_pop_front(src_odd_i,   1);
            cb_pop_front(src_even_r,  1);
            cb_pop_front(src_even_i,  1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}