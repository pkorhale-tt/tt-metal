// fft_compute_f32.cpp — BATCHED TILE OPERATIONS
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0


#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
namespace NAMESPACE {
void MAIN {
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
    
    binary_op_init_common(cb_even_r, cb_odd_r);
    
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);
            
            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);
            
            tile_regs_acquire();
            
            mul_tiles_init();
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 2);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 3);
            
            sub_tiles_init();
            sub_tiles(cb_tw_r, cb_tw_i, 0, 1, 4);
            
            add_tiles_init();
            add_tiles(cb_tw_r, cb_tw_i, 2, 3, 5);
            
            copy_tile_init();
            copy_tile(cb_even_r, 0, 6);
            copy_tile(cb_even_i, 0, 7);
            
            add_tiles_init();
            add_tiles(cb_even_r, cb_tw_r, 6, 4, 8);
            add_tiles(cb_even_i, cb_tw_i, 7, 5, 9);
            
            sub_tiles_init();
            sub_tiles(cb_even_r, cb_tw_r, 6, 4, 10);
            sub_tiles(cb_even_i, cb_tw_i, 7, 5, 11);
            
            tile_regs_commit();
            tile_regs_wait();
            
            pack_tile(8,  cb_out0_r);
            pack_tile(9,  cb_out0_i);
            pack_tile(10, cb_out1_r);
            pack_tile(11, cb_out1_i);
            
            tile_regs_release();
            
            cb_push_back(cb_out0_r, 1);
            cb_push_back(cb_out0_i, 1);
            cb_push_back(cb_out1_r, 1);
            cb_push_back(cb_out1_i, 1);
            
            cb_pop_front(cb_tw_r,   1);
            cb_pop_front(cb_tw_i,   1);
            cb_pop_front(cb_odd_r,  1);
            cb_pop_front(cb_odd_i,  1);
            cb_pop_front(cb_even_r, 1);
            cb_pop_front(cb_even_i, 1);
        }
    }
}
}
