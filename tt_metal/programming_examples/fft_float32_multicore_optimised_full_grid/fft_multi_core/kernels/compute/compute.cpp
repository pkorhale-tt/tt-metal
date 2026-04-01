// fft_compute_f32.cpp - FULLY FIXED
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// KEY FIX: All {mul,sub,add}_tiles_init() calls are now OUTSIDE
// tile_regs_acquire()/tile_regs_release() blocks.
// The API contract is:
//   _init()              ← configure UNPACK + math engine
//   tile_regs_acquire()  ← lock dest regs
//   _tiles(...)          ← execute
//   tile_regs_commit()   ← signal math done
//   tile_regs_wait()     ← wait for pack ack
//   pack_tile(...)       ← pack to CB
//   tile_regs_release()  ← unlock dest regs

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

// Helper macro to keep the acquire/compute/pack/release block
// uniform and avoid accidental mis-ordering.
//
//  MATH_OP(init_call, exec_call, dst_cb)
//
// Expands to:
//   init_call;
//   cb_reserve_back(dst_cb, 1);
//   tile_regs_acquire();
//   exec_call;
//   tile_regs_commit();
//   tile_regs_wait();
//   pack_tile(0, dst_cb);
//   tile_regs_release();
//   cb_push_back(dst_cb, 1);
//
// The caller is responsible for cb_wait_front on src CBs beforehand
// and cb_pop_front on src CBs afterwards.

#define MATH_OP(init_call, exec_call, dst_cb)   \
    do {                                         \
        init_call;                               \
        cb_reserve_back(dst_cb, 1);              \
        tile_regs_acquire();                     \
        exec_call;                               \
        tile_regs_commit();                      \
        tile_regs_wait();                        \
        pack_tile(0, dst_cb);                    \
        tile_regs_release();                     \
        cb_push_back(dst_cb, 1);                 \
    } while(0)

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_even_r   = 0;
    constexpr uint32_t cb_even_i   = 1;
    constexpr uint32_t cb_odd_r    = 2;
    constexpr uint32_t cb_odd_i    = 3;
    constexpr uint32_t cb_tw_r     = 4;
    constexpr uint32_t cb_tw_i     = 5;
    constexpr uint32_t cb_out0_r   = 16;
    constexpr uint32_t cb_out0_i   = 17;
    constexpr uint32_t cb_out1_r   = 18;
    constexpr uint32_t cb_out1_i   = 19;
    constexpr uint32_t cb_tmp0     = 20;
    constexpr uint32_t cb_tmp1     = 21;
    constexpr uint32_t cb_tw_odd_r = 22;
    constexpr uint32_t cb_tw_odd_i = 23;

    // binary_op_init_common configures the unpack/pack pipeline once.
    // Use the first source pair that will be consumed (tw_r, odd_r).
    binary_op_init_common(cb_tw_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // ── Wait for all inputs ──────────────────────────────────
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // ── t_r = tw_r*odd_r ─────────────────────────────────────
            MATH_OP(
                mul_tiles_init(cb_tw_r, cb_odd_r),
                mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0),
                cb_tmp0
            );

            // ── t_r -= tw_i*odd_i (compute tw_i*odd_i first) ─────────
            MATH_OP(
                mul_tiles_init(cb_tw_i, cb_odd_i),
                mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0),
                cb_tmp1
            );

            // tw_odd_r = tmp0 - tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            MATH_OP(
                sub_tiles_init(cb_tmp0, cb_tmp1),
                sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0),
                cb_tw_odd_r
            );
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // ── t_i = tw_r*odd_i ─────────────────────────────────────
            MATH_OP(
                mul_tiles_init(cb_tw_r, cb_odd_i),
                mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0),
                cb_tmp0
            );

            // ── t_i += tw_i*odd_r (compute tw_i*odd_r first) ─────────
            MATH_OP(
                mul_tiles_init(cb_tw_i, cb_odd_r),
                mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0),
                cb_tmp1
            );

            // tw_odd_i = tmp0 + tmp1
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            MATH_OP(
                add_tiles_init(cb_tmp0, cb_tmp1),
                add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0),
                cb_tw_odd_i
            );
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // Wait until both twiddle-rotated odd terms are ready
            cb_wait_front(cb_tw_odd_r, 1);
            cb_wait_front(cb_tw_odd_i, 1);

            // ── out0 = even + t ──────────────────────────────────────
            MATH_OP(
                add_tiles_init(cb_even_r, cb_tw_odd_r),
                add_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0),
                cb_out0_r
            );
            MATH_OP(
                add_tiles_init(cb_even_i, cb_tw_odd_i),
                add_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0),
                cb_out0_i
            );

            // ── out1 = even − t ──────────────────────────────────────
            MATH_OP(
                sub_tiles_init(cb_even_r, cb_tw_odd_r),
                sub_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0),
                cb_out1_r
            );
            MATH_OP(
                sub_tiles_init(cb_even_i, cb_tw_odd_i),
                sub_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0),
                cb_out1_i
            );

            // ── Pop consumed inputs ───────────────────────────────────
            cb_pop_front(cb_tw_r,     1);
            cb_pop_front(cb_tw_i,     1);
            cb_pop_front(cb_odd_r,    1);
            cb_pop_front(cb_odd_i,    1);
            cb_pop_front(cb_even_r,   1);
            cb_pop_front(cb_even_i,   1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}