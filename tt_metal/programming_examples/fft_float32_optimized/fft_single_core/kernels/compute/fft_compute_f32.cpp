// fft_compute_f32.cpp  — OPTIMISED MULTI-STAGE FFT
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//   1. direction is evaluated ONCE above all loops into is_ifft (bool).
//      Original code re-evaluated "if (direction == 1)" every tile×stage —
//      needless branch on a value that never changes.
//   2. IFFT per-tile twiddle negation branch removed entirely.
//      The host now pre-negates tw_i when direction==1 before upload, so
//      cb_tw_i already holds the correct sign.  Deleted: cb_reserve_back,
//      tile_regs_acquire/commit/wait/release, copy_tile, negative_tile,
//      pack_tile, cb_push_back, cb_wait_front for cb_neg_tw_i — that is
//      9 extra API calls per tile per stage saved on the IFFT path.
//   3. cb_neg_tw_i (CB 24) no longer referenced here.  Its CreateCircularBuffer
//      call should also be removed on the host (frees 4 KB of L1 SRAM).
//   4. tw_i_to_use variable removed — all uses replaced with cb_tw_i directly.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t direction       = get_arg_val<uint32_t>(0);
    const uint32_t num_stages      = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(2);

    // Evaluate direction ONCE — never changes during kernel execution.
    // Keeping it for potential future use (e.g. post-IFFT 1/N scaling flag),
    // but the inner loop no longer branches on it.
    // NOTE: actual twiddle sign flip for IFFT is now done on the host.
    (void)direction;  // suppress unused-variable warning if no other use

    // CB indices
    constexpr uint32_t cb_in_even_r   = 0;
    constexpr uint32_t cb_in_even_i   = 1;
    constexpr uint32_t cb_in_odd_r    = 2;
    constexpr uint32_t cb_in_odd_i    = 3;
    constexpr uint32_t cb_tw_r        = 4;
    constexpr uint32_t cb_tw_i        = 5;   // sign already correct for FFT or IFFT
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

    // Scratch CBs for intermediate computation
    constexpr uint32_t cb_tmp0     = 20;  // tw_r * odd_r
    constexpr uint32_t cb_tmp1     = 21;  // tw_i * odd_i
    constexpr uint32_t cb_tw_odd_r = 22;  // t_r = tw_r*odd_r - tw_i*odd_i
    constexpr uint32_t cb_tw_odd_i = 23;  // t_i = tw_r*odd_i + tw_i*odd_r
    // CB 24 (cb_neg_tw_i) removed — negation now done on host.

    // Initialize binary compute engine once.
    binary_op_init_common(cb_in_even_r, cb_in_odd_r);

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
            dst0_r = cb_out0_r;
            dst0_i = cb_out0_i;
            dst1_r = cb_out1_r;
            dst1_i = cb_out1_i;
        } else if ((stage & 1) == 0) {
            dst0_r = cb_ping_even_r;
            dst0_i = cb_ping_even_i;
            dst1_r = cb_ping_odd_r;
            dst1_i = cb_ping_odd_i;
        } else {
            dst0_r = cb_pong_even_r;
            dst0_i = cb_pong_even_i;
            dst1_r = cb_pong_odd_r;
            dst1_i = cb_pong_odd_i;
        }

        // ── Process all tiles in this stage ───────────────────────
        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // ════════════════════════════════════════════════════════
            // Wait for all inputs
            // ════════════════════════════════════════════════════════
            cb_wait_front(cb_tw_r,    1);
            cb_wait_front(cb_tw_i,    1);
            cb_wait_front(src_odd_r,  1);
            cb_wait_front(src_odd_i,  1);
            cb_wait_front(src_even_r, 1);
            cb_wait_front(src_even_i, 1);

            // ════════════════════════════════════════════════════════
            // Step 1: Compute t_r = tw_r * odd_r - tw_i * odd_i
            //
            // cb_tw_i already holds the correct sign:
            //   forward FFT → host stored  -sin(angle)
            //   inverse FFT → host stored  +sin(angle)  (conjugate)
            // No per-tile negation branch needed here.
            // ════════════════════════════════════════════════════════

            // tmp0 = tw_r * odd_r
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, src_odd_r);
            mul_tiles(cb_tw_r, src_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = tw_i * odd_i
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, src_odd_i);
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
            sub_tiles_init(cb_tmp0, cb_tmp1);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tw_odd_r);
            tile_regs_release();
            cb_push_back(cb_tw_odd_r, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // ════════════════════════════════════════════════════════
            // Step 2: Compute t_i = tw_r * odd_i + tw_i * odd_r
            // ════════════════════════════════════════════════════════

            // tmp0 = tw_r * odd_i
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, src_odd_i);
            mul_tiles(cb_tw_r, src_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = tw_i * odd_r
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, src_odd_r);
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
            add_tiles_init(cb_tmp0, cb_tmp1);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tw_odd_i);
            tile_regs_release();
            cb_push_back(cb_tw_odd_i, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // ════════════════════════════════════════════════════════
            // Step 3: Compute out0 = even + t
            // ════════════════════════════════════════════════════════
            cb_wait_front(cb_tw_odd_r, 1);
            cb_wait_front(cb_tw_odd_i, 1);

            // out0_r = even_r + tw_odd_r
            cb_reserve_back(dst0_r, 1);
            tile_regs_acquire();
            add_tiles_init(src_even_r, cb_tw_odd_r);
            add_tiles(src_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst0_r);
            tile_regs_release();
            cb_push_back(dst0_r, 1);

            // out0_i = even_i + tw_odd_i
            cb_reserve_back(dst0_i, 1);
            tile_regs_acquire();
            add_tiles_init(src_even_i, cb_tw_odd_i);
            add_tiles(src_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst0_i);
            tile_regs_release();
            cb_push_back(dst0_i, 1);

            // ════════════════════════════════════════════════════════
            // Step 4: Compute out1 = even - t
            // ════════════════════════════════════════════════════════

            // out1_r = even_r - tw_odd_r
            cb_reserve_back(dst1_r, 1);
            tile_regs_acquire();
            sub_tiles_init(src_even_r, cb_tw_odd_r);
            sub_tiles(src_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst1_r);
            tile_regs_release();
            cb_push_back(dst1_r, 1);

            // out1_i = even_i - tw_odd_i
            cb_reserve_back(dst1_i, 1);
            tile_regs_acquire();
            sub_tiles_init(src_even_i, cb_tw_odd_i);
            sub_tiles(src_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst1_i);
            tile_regs_release();
            cb_push_back(dst1_i, 1);

            // ════════════════════════════════════════════════════════
            // Step 5: Pop all consumed inputs
            // ════════════════════════════════════════════════════════
            cb_pop_front(cb_tw_r,    1);
            cb_pop_front(cb_tw_i,    1);
            cb_pop_front(src_odd_r,  1);
            cb_pop_front(src_odd_i,  1);
            cb_pop_front(src_even_r, 1);
            cb_pop_front(src_even_i, 1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}