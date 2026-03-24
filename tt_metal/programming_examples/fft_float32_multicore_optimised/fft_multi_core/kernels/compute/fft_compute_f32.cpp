// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(2);

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
    constexpr uint32_t cb_tmp2   = 22;
    constexpr uint32_t cb_tmp3   = 23;

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) {
        return;
    }

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t row = 0; row < rows_per_core; row++) {
        for (uint32_t stage = 0; stage < num_stages; stage++) {
            for (uint32_t t = 0; t < tiles_per_stage; t++) {
                cb_wait_front(cb_even_r, 1);
                cb_wait_front(cb_even_i, 1);
                cb_wait_front(cb_odd_r,  1);
                cb_wait_front(cb_odd_i,  1);
                cb_wait_front(cb_tw_r,   1);
                cb_wait_front(cb_tw_i,   1);

                // tmp0 = tw_r * odd_r
                cb_reserve_back(cb_tmp0, 1);
                tile_regs_acquire();
                copy_tile(cb_tw_r, 0, 0);
                copy_tile(cb_odd_r, 0, 1);
                mul_tiles_init(cb_tw_r, cb_odd_r);
                mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
                cb_push_back(cb_tmp0, 1);

                // tmp1 = tw_i * odd_i
                cb_reserve_back(cb_tmp1, 1);
                tile_regs_acquire();
                copy_tile(cb_tw_i, 0, 0);
                copy_tile(cb_odd_i, 0, 1);
                mul_tiles_init(cb_tw_i, cb_odd_i);
                mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp1);
                tile_regs_release();
                cb_push_back(cb_tmp1, 1);

                // tmp2 = tw_r * odd_i
                cb_reserve_back(cb_tmp2, 1);
                tile_regs_acquire();
                copy_tile(cb_tw_r, 0, 0);
                copy_tile(cb_odd_i, 0, 1);
                mul_tiles_init(cb_tw_r, cb_odd_i);
                mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp2);
                tile_regs_release();
                cb_push_back(cb_tmp2, 1);

                // tmp3 = tw_i * odd_r
                cb_reserve_back(cb_tmp3, 1);
                tile_regs_acquire();
                copy_tile(cb_tw_i, 0, 0);
                copy_tile(cb_odd_r, 0, 1);
                mul_tiles_init(cb_tw_i, cb_odd_r);
                mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp3);
                tile_regs_release();
                cb_push_back(cb_tmp3, 1);

                cb_wait_front(cb_tmp0, 1);
                cb_wait_front(cb_tmp1, 1);
                cb_wait_front(cb_tmp2, 1);
                cb_wait_front(cb_tmp3, 1);

                // out0_r = even_r + (tmp0 - tmp1)
                cb_reserve_back(cb_out0_r, 1);
                tile_regs_acquire();
                copy_tile(cb_tmp0, 0, 0);
                copy_tile(cb_tmp1, 0, 1);
                sub_tiles_init(cb_tmp0, cb_tmp1);
                sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
                cb_pop_front(cb_tmp0, 1);
                cb_wait_front(cb_tmp0, 1);

                tile_regs_acquire();
                copy_tile(cb_even_r, 0, 0);
                copy_tile(cb_tmp0, 0, 1);
                add_tiles_init(cb_even_r, cb_tmp0);
                add_tiles(cb_even_r, cb_tmp0, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out0_r);
                tile_regs_release();
                cb_push_back(cb_out0_r, 1);

                // out1_r = even_r - (tmp0 - tmp1)
                cb_reserve_back(cb_out1_r, 1);
                tile_regs_acquire();
                copy_tile(cb_even_r, 0, 0);
                copy_tile(cb_tmp0, 0, 1);
                sub_tiles_init(cb_even_r, cb_tmp0);
                sub_tiles(cb_even_r, cb_tmp0, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out1_r);
                tile_regs_release();
                cb_push_back(cb_out1_r, 1);

                // out0_i = even_i + (tmp2 + tmp3)
                cb_reserve_back(cb_out0_i, 1);
                tile_regs_acquire();
                copy_tile(cb_tmp2, 0, 0);
                copy_tile(cb_tmp3, 0, 1);
                add_tiles_init(cb_tmp2, cb_tmp3);
                add_tiles(cb_tmp2, cb_tmp3, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp2);
                tile_regs_release();
                cb_pop_front(cb_tmp2, 1);
                cb_wait_front(cb_tmp2, 1);

                tile_regs_acquire();
                copy_tile(cb_even_i, 0, 0);
                copy_tile(cb_tmp2, 0, 1);
                add_tiles_init(cb_even_i, cb_tmp2);
                add_tiles(cb_even_i, cb_tmp2, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out0_i);
                tile_regs_release();
                cb_push_back(cb_out0_i, 1);

                // out1_i = even_i - (tmp2 + tmp3)
                cb_reserve_back(cb_out1_i, 1);
                tile_regs_acquire();
                copy_tile(cb_even_i, 0, 0);
                copy_tile(cb_tmp2, 0, 1);
                sub_tiles_init(cb_even_i, cb_tmp2);
                sub_tiles(cb_even_i, cb_tmp2, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out1_i);
                tile_regs_release();
                cb_push_back(cb_out1_i, 1);

                cb_pop_front(cb_even_r, 1);
                cb_pop_front(cb_even_i, 1);
                cb_pop_front(cb_odd_r,  1);
                cb_pop_front(cb_odd_i,  1);
                cb_pop_front(cb_tw_r,   1);
                cb_pop_front(cb_tw_i,   1);

                cb_pop_front(cb_tmp0, 1);
                cb_pop_front(cb_tmp1, 1);
                cb_pop_front(cb_tmp2, 1);
                cb_pop_front(cb_tmp3, 1);
            }
        }
    }
}
}  // namespace NAMESPACE