// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Kernel: Converts float32 output tiles to bfloat16 for writing to DRAM

#include <stdint.h>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

// ═══════════════════════════════════════════════════════════════════════════
// INPUT CBs (float32 from FFT compute kernel)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_out0_r_fp32 = tt::CBIndex::c_12;      // Output LHS Real (fp32)
constexpr auto cb_out0_i_fp32 = tt::CBIndex::c_13;      // Output LHS Imag (fp32)
constexpr auto cb_out1_r_fp32 = tt::CBIndex::c_14;      // Output RHS Real (fp32)
constexpr auto cb_out1_i_fp32 = tt::CBIndex::c_15;      // Output RHS Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT CBs (bfloat16 for writer)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_out0_r_bf16 = tt::CBIndex::c_16;      // Output LHS Real (bf16)
constexpr auto cb_out0_i_bf16 = tt::CBIndex::c_17;      // Output LHS Imag (bf16)
constexpr auto cb_out1_r_bf16 = tt::CBIndex::c_18;      // Output RHS Real (bf16)
constexpr auto cb_out1_i_bf16 = tt::CBIndex::c_19;      // Output RHS Imag (bf16)

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Convert single tile from fp32 to bf16
// ═══════════════════════════════════════════════════════════════════════════
inline void convert_tile_fp32_to_bf16(uint32_t cb_in_fp32, uint32_t cb_out_bf16) {
    // Wait for input tile (fp32)
    cb_wait_front(cb_in_fp32, 1);
    
    // Reserve output space (bf16)
    cb_reserve_back(cb_out_bf16, 1);
    
    // Acquire DST registers
    tile_regs_acquire();
    
    // Copy fp32 tile to DST register
    copy_tile_to_dst_init_short(cb_in_fp32);
    copy_tile(cb_in_fp32, 0, 0);  // cb_in[0] → DST[0]
    
    tile_regs_commit();
    tile_regs_wait();
    
    // Pack DST[0] to bf16 output CB
    // This is where fp32 → bf16 truncation happens (only once!)
    pack_tile(0, cb_out_bf16);
    
    tile_regs_release();
    
    // Update circular buffers
    cb_push_back(cb_out_bf16, 1);
    cb_pop_front(cb_in_fp32, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN KERNEL
// ═══════════════════════════════════════════════════════════════════════════
namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        
        // Convert all 4 output tiles from fp32 → bf16
        convert_tile_fp32_to_bf16(cb_out0_r_fp32, cb_out0_r_bf16);
        convert_tile_fp32_to_bf16(cb_out0_i_fp32, cb_out0_i_bf16);
        convert_tile_fp32_to_bf16(cb_out1_r_fp32, cb_out1_r_bf16);
        convert_tile_fp32_to_bf16(cb_out1_i_fp32, cb_out1_i_bf16);
    }
}
}