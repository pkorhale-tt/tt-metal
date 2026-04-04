// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

// ---------------------------------------------------------------------------
// BUG FIX: All *_tiles_init calls were previously inside tile_regs_acquire()
// blocks, which deadlocks the kernel.  *_tiles_init programs the UNPACK
// subunit; tile_regs_acquire() locks the dst register for the MATH core.
// If UNPACK is programmed while MATH holds the lock, the two sub-threads
// deadlock waiting on each other.
//
// Rule: *_tiles_init must ALWAYS appear BEFORE tile_regs_acquire().
// ---------------------------------------------------------------------------

void kernel_main() {
    const uint32_t num_stages    = get_arg_val<uint32_t>(0);
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

    constexpr uint32_t cb_tmp0     = 20;
    constexpr uint32_t cb_tmp1     = 21;
    constexpr uint32_t cb_tw_odd_r = 22;
    constexpr uint32_t cb_tw_odd_i = 23;

    // One-time initialisation of shared binary-op state (unpacker CB
    // pointers, packer CB pointer, etc.).  Individual *_tiles_init calls
    // below re-program only the operation-specific state before each use.
    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // -----------------------------------------------------------------
            // t_r = tw_r * odd_r - tw_i * odd_i
            // -----------------------------------------------------------------

            // tmp0 = tw_r * odd_r
            mul_tiles_init(cb_tw_r, cb_odd_r);          // ← OUTSIDE acquire
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = tw_i * odd_i
            mul_tiles_init(cb_tw_i, cb_odd_i);          // ← OUTSIDE acquire
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            // tw_odd_r = tmp0 - tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            sub_tiles_init(cb_tmp0, cb_tmp1);           // ← OUTSIDE acquire
            cb_reserve_back(cb_tw_odd_r, 1);
            tile_regs_acquire();
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tw_odd_r);
            tile_regs_release();
            cb_push_back(cb_tw_odd_r, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // -----------------------------------------------------------------
            // t_i = tw_r * odd_i + tw_i * odd_r
            // -----------------------------------------------------------------

            // tmp0 = tw_r * odd_i
            mul_tiles_init(cb_tw_r, cb_odd_i);          // ← OUTSIDE acquire
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // tmp1 = tw_i * odd_r
            mul_tiles_init(cb_tw_i, cb_odd_r);          // ← OUTSIDE acquire
            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            // tw_odd_i = tmp0 + tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            add_tiles_init(cb_tmp0, cb_tmp1);           // ← OUTSIDE acquire
            cb_reserve_back(cb_tw_odd_i, 1);
            tile_regs_acquire();
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tw_odd_i);
            tile_regs_release();
            cb_push_back(cb_tw_odd_i, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // tw_odd_r and tw_odd_i are now ready; wait before using them
            // as srcB in the next four operations.
            cb_wait_front(cb_tw_odd_r, 1);
            cb_wait_front(cb_tw_odd_i, 1);

            // -----------------------------------------------------------------
            // out0 = even + t
            // -----------------------------------------------------------------

            // out0_r = even_r + tw_odd_r
            add_tiles_init(cb_even_r, cb_tw_odd_r);     // ← OUTSIDE acquire
            cb_reserve_back(cb_out0_r, 1);
            tile_regs_acquire();
            add_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0_r);
            tile_regs_release();
            cb_push_back(cb_out0_r, 1);

            // out0_i = even_i + tw_odd_i
            add_tiles_init(cb_even_i, cb_tw_odd_i);     // ← OUTSIDE acquire
            cb_reserve_back(cb_out0_i, 1);
            tile_regs_acquire();
            add_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0_i);
            tile_regs_release();
            cb_push_back(cb_out0_i, 1);

            // -----------------------------------------------------------------
            // out1 = even - t
            // -----------------------------------------------------------------

            // out1_r = even_r - tw_odd_r
            sub_tiles_init(cb_even_r, cb_tw_odd_r);     // ← OUTSIDE acquire
            cb_reserve_back(cb_out1_r, 1);
            tile_regs_acquire();
            sub_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out1_r);
            tile_regs_release();
            cb_push_back(cb_out1_r, 1);

            // out1_i = even_i - tw_odd_i
            sub_tiles_init(cb_even_i, cb_tw_odd_i);     // ← OUTSIDE acquire
            cb_reserve_back(cb_out1_i, 1);
            tile_regs_acquire();
            sub_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out1_i);
            tile_regs_release();
            cb_push_back(cb_out1_i, 1);

            // Release all input tiles consumed this butterfly iteration
            cb_pop_front(cb_tw_r,    1);
            cb_pop_front(cb_tw_i,    1);
            cb_pop_front(cb_odd_r,   1);
            cb_pop_front(cb_odd_i,   1);
            cb_pop_front(cb_even_r,  1);
            cb_pop_front(cb_even_i,  1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}