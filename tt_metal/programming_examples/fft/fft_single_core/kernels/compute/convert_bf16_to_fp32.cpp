// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Kernel: Converts bfloat16 input tiles to float32 for computation

#include <stdint.h>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

// ═══════════════════════════════════════════════════════════════════════════
// INPUT CBs (bfloat16 from reader)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_data0_r_bf16 = tt::CBIndex::c_0;      // LHS Real (bf16)
constexpr auto cb_data0_i_bf16 = tt::CBIndex::c_1;      // LHS Imag (bf16)
constexpr auto cb_data1_r_bf16 = tt::CBIndex::c_2;      // RHS Real (bf16)
constexpr auto cb_data1_i_bf16 = tt::CBIndex::c_3;      // RHS Imag (bf16)
constexpr auto cb_twiddle_r_bf16 = tt::CBIndex::c_4;    // Twiddle Real (bf16)
constexpr auto cb_twiddle_i_bf16 = tt::CBIndex::c_5;    // Twiddle Imag (bf16)

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT CBs (float32 for computation)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_data0_r_fp32 = tt::CBIndex::c_6;      // LHS Real (fp32)
constexpr auto cb_data0_i_fp32 = tt::CBIndex::c_7;      // LHS Imag (fp32)
constexpr auto cb_data1_r_fp32 = tt::CBIndex::c_8;      // RHS Real (fp32)
constexpr auto cb_data1_i_fp32 = tt::CBIndex::c_9;      // RHS Imag (fp32)
constexpr auto cb_twiddle_r_fp32 = tt::CBIndex::c_10;   // Twiddle Real (fp32)
constexpr auto cb_twiddle_i_fp32 = tt::CBIndex::c_11;   // Twiddle Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Convert single tile from bf16 to fp32
// ═══════════════════════════════════════════════════════════════════════════
inline void convert_tile_bf16_to_fp32(uint32_t cb_in_bf16, uint32_t cb_out_fp32) {
    // Wait for input tile (bf16)
    cb_wait_front(cb_in_bf16, 1);
    
    // Reserve output space (fp32)
    cb_reserve_back(cb_out_fp32, 1);
    
    // Acquire DST registers
    tile_regs_acquire();
    
    // Copy bf16 tile to DST register
    // DST registers are fp32, so automatic conversion happens!
    copy_tile_to_dst_init_short(cb_in_bf16);
    copy_tile(cb_in_bf16, 0, 0);  // cb_in[0] → DST[0] (now fp32)
    
    tile_regs_commit();
    tile_regs_wait();
    
    // Pack DST[0] to fp32 output CB
    pack_tile(0, cb_out_fp32);
    
    tile_regs_release();
    
    // Update circular buffers
    cb_push_back(cb_out_fp32, 1);
    cb_pop_front(cb_in_bf16, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN KERNEL
// ═══════════════════════════════════════════════════════════════════════════
namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        
        // Convert all 6 input tiles from bf16 → fp32
        convert_tile_bf16_to_fp32(cb_data0_r_bf16, cb_data0_r_fp32);
        convert_tile_bf16_to_fp32(cb_data0_i_bf16, cb_data0_i_fp32);
        convert_tile_bf16_to_fp32(cb_data1_r_bf16, cb_data1_r_fp32);
        convert_tile_bf16_to_fp32(cb_data1_i_bf16, cb_data1_i_fp32);
        convert_tile_bf16_to_fp32(cb_twiddle_r_bf16, cb_twiddle_r_fp32);
        convert_tile_bf16_to_fp32(cb_twiddle_i_bf16, cb_twiddle_i_fp32);
    }
}
}