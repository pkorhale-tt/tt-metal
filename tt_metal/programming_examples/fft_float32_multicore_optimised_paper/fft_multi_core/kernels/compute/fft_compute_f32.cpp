// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

void kernel_main() {
    const uint32_t num_stages = get_arg_val(0);
    const uint32_t tiles_per_stage = get_arg_val(1);

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

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {
            cb_wait_front(cb_tw_r, 1);
            cb_wait_front(cb_tw_i, 1);
            cb_wait_front(cb_odd_r, 1);
            cb_wait_front(cb_odd_i, 1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // t_r = tw_r*odd_r - tw_i*odd_i
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

            // t_i = tw_r*odd_i + tw_i*odd_r
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

            cb_wait_front(cb_tw_odd_r, 1);
            cb_wait_front(cb_tw_odd_i, 1);

            // out0 = even + t
            cb_reserve_back(cb_out0_r, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_r, cb_tw_odd_r, cb_out0_r);
            add_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0_r);
            tile_regs_release();
            cb_push_back(cb_out0_r, 1);

            cb_reserve_back(cb_out0_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_i, cb_tw_odd_i, cb_out0_i);
            add_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0_i);
            tile_regs_release();
            cb_push_back(cb_out0_i, 1);

            // out1 = even - t
            cb_reserve_back(cb_out1_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_r, cb_tw_odd_r, cb_out1_r);
            sub_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out1_r);
            tile_regs_release();
            cb_push_back(cb_out1_r, 1);

            cb_reserve_back(cb_out1_i, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_i, cb_tw_odd_i, cb_out1_i);
            sub_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out1_i);
            tile_regs_release();
            cb_push_back(cb_out1_i, 1);

            cb_pop_front(cb_tw_r, 1);
            cb_pop_front(cb_tw_i, 1);
            cb_pop_front(cb_odd_r, 1);
            cb_pop_front(cb_odd_i, 1);
            cb_pop_front(cb_even_r, 1);
            cb_pop_front(cb_even_i, 1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}