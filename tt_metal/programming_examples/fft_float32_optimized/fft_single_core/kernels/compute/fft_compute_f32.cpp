// fft_compute_f32_opt.cpp — BATCHED TILE OPERATIONS
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

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
    
    // Initialize FPU once (outside loops)
    binary_op_init_common(cb_even_r, cb_odd_r);
    
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {
            // Wait for all inputs
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);
            
            // Reserve all output slots upfront
            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);
            
            // ═══════════════════════════════════════════════════════
            // BATCHED COMPUTATION: All operations before commit
            // ═══════════════════════════════════════════════════════
            tile_regs_acquire();
            
            // ─── Compute t = tw * odd (complex multiply) ──────────
            // t_r = tw_r*odd_r - tw_i*odd_i
            // t_i = tw_r*odd_i + tw_i*odd_r
            
            mul_tiles_init();
            
            // DST[0] = tw_r * odd_r
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
            
            // DST[1] = tw_i * odd_i
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);
            
            // DST[2] = tw_r * odd_i
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 2);
            
            // DST[3] = tw_i * odd_r
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 3);
            
            // DST[4] = t_r = DST[0] - DST[1]
            sub_tiles_init();
            sub_tiles(cb_tw_r, cb_tw_i, 0, 1, 4);  // Use DST regs directly
            
            // DST[5] = t_i = DST[2] + DST[3]
            add_tiles_init();
            add_tiles(cb_tw_r, cb_tw_i, 2, 3, 5);
            
            // ─── Load even values into DST[6-7] ───────────────────
            copy_tile_init();
            copy_tile(cb_even_r, 0, 6);
            copy_tile(cb_even_i, 0, 7);
            
            // ─── Compute out0 = even + t ──────────────────────────
            add_tiles_init();
            add_tiles(cb_even_r, cb_tw_r, 6, 4, 8);   // DST[8] = out0_r
            add_tiles(cb_even_i, cb_tw_i, 7, 5, 9);   // DST[9] = out0_i
            
            // ─── Compute out1 = even - t ──────────────────────────
            sub_tiles_init();
            sub_tiles(cb_even_r, cb_tw_r, 6, 4, 10);  // DST[10] = out1_r
            sub_tiles(cb_even_i, cb_tw_i, 7, 5, 11);  // DST[11] = out1_i
            
            // ═══════════════════════════════════════════════════════
            // SINGLE COMMIT: All 4 outputs ready
            // ═══════════════════════════════════════════════════════
            tile_regs_commit();
            tile_regs_wait();
            
            // Pack all outputs
            pack_tile(8,  cb_out0_r);
            pack_tile(9,  cb_out0_i);
            pack_tile(10, cb_out1_r);
            pack_tile(11, cb_out1_i);
            
            tile_regs_release();
            
            // Push outputs and pop inputs
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