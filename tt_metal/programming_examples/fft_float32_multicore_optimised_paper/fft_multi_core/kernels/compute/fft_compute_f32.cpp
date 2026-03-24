// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PRODUCTION FFT COMPUTE KERNEL
// Architecture: Decoupled data movement, optimized for Tensix SFPU
// Based on Tenstorrent Wormhole FFT paper design

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

void MAIN {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    
    // CB indices
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
    constexpr uint32_t cb_tmp_r  = 20;  // Temp for twiddle * odd (real)
    constexpr uint32_t cb_tmp_i  = 21;  // Temp for twiddle * odd (imag)
    
    // ═══════════════════════════════════════════════════════════
    // OPTIMIZATION: Initialize SFPU once (not per iteration)
    // Paper: "twiddle factors are calculated by the compute engine 
    //         on initialisation and stored in SRAM"
    // ═══════════════════════════════════════════════════════════
    binary_op_init_common(cb_even_r, cb_odd_r, cb_out0_r);
    
    mul_tiles_init();   // Initialize multiply operation once
    add_tiles_init();   // Initialize add operation once
    sub_tiles_init();   // Initialize subtract operation once
    
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t tile_idx = 0; tile_idx < tiles_per_stage; tile_idx++) {
            
            // ═══════════════════════════════════════════════════
            // Wait for data movement cores to provide inputs
            // Architecture: Decoupled - compute waits on CBs
            // ═══════════════════════════════════════════════════
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            
            // ═══════════════════════════════════════════════════
            // Reserve output space BEFORE acquiring tile registers
            // ═══════════════════════════════════════════════════
            cb_reserve_back(cb_tmp_r, 1);
            cb_reserve_back(cb_tmp_i, 1);
            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);
            
            // ═══════════════════════════════════════════════════
            // COMPUTE SESSION: Complex butterfly
            // Paper Figure 3: "compute engine" handles operations
            // 
            // Complex multiply: (tw_r + j*tw_i) * (odd_r + j*odd_i)
            //   = (tw_r*odd_r - tw_i*odd_i) + j(tw_r*odd_i + tw_i*odd_r)
            // ═══════════════════════════════════════════════════
            
            acquire_dst(tt::DstMode::Half);  // Acquire compute resources
            
            // Step 1: Compute twiddle * odd
            // tmp_r = tw_r * odd_r - tw_i * odd_i
            // tmp_i = tw_r * odd_i + tw_i * odd_r
            
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);  // reg[0] = tw_r * odd_r
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);  // reg[1] = tw_i * odd_i
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 2);  // reg[2] = tw_r * odd_i
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 3);  // reg[3] = tw_i * odd_r
            
            sub_tiles(0, 1, 0, 0, 4);  // reg[4] = tmp_r = reg[0] - reg[1]
            add_tiles(2, 3, 0, 0, 5);  // reg[5] = tmp_i = reg[2] + reg[3]
            
            pack_tile(4, cb_tmp_r);  // Write tmp_r to CB
            pack_tile(5, cb_tmp_i);  // Write tmp_i to CB
            
            cb_push_back(cb_tmp_r, 1);
            cb_push_back(cb_tmp_i, 1);
            
            // Step 2: Butterfly - now wait for tmp to be ready
            cb_wait_front(cb_tmp_r, 1);
            cb_wait_front(cb_tmp_i, 1);
            
            // out0 = even + tmp (upper butterfly output)
            // out1 = even - tmp (lower butterfly output)
            
            add_tiles(cb_even_r, cb_tmp_r, 0, 0, 6);  // reg[6] = out0_r
            add_tiles(cb_even_i, cb_tmp_i, 0, 0, 7);  // reg[7] = out0_i
            sub_tiles(cb_even_r, cb_tmp_r, 0, 0, 8);  // reg[8] = out1_r
            sub_tiles(cb_even_i, cb_tmp_i, 0, 0, 9);  // reg[9] = out1_i
            
            pack_tile(6, cb_out0_r);
            pack_tile(7, cb_out0_i);
            pack_tile(8, cb_out1_r);
            pack_tile(9, cb_out1_i);
            
            release_dst(tt::DstMode::Half);  // Release compute resources
            
            // ═══════════════════════════════════════════════════
            // Push outputs and pop consumed inputs
            // ═══════════════════════════════════════════════════
            cb_push_back(cb_out0_r, 1);
            cb_push_back(cb_out0_i, 1);
            cb_push_back(cb_out1_r, 1);
            cb_push_back(cb_out1_i, 1);
            
            cb_pop_front(cb_even_r, 1);
            cb_pop_front(cb_even_i, 1);
            cb_pop_front(cb_odd_r, 1);
            cb_pop_front(cb_odd_i, 1);
            cb_pop_front(cb_tw_r, 1);
            cb_pop_front(cb_tw_i, 1);
            cb_pop_front(cb_tmp_r, 1);
            cb_pop_front(cb_tmp_i, 1);
        }
    }
}