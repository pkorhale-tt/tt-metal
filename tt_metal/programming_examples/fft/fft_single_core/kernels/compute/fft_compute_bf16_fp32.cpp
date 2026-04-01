// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FFT Compute Kernel: bf16 input → fp32 compute → bf16 output

#include <stdint.h>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/common.h"

// Operations
#define ADD 0
#define SUB 1
#define MUL 2

// ═══════════════════════════════════════════════════════════════════════════
// INPUT CBs (bfloat16)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_in0_r = tt::CBIndex::c_0;       // LHS Real (bf16)
constexpr auto cb_in0_i = tt::CBIndex::c_1;       // LHS Imag (bf16)
constexpr auto cb_in1_r = tt::CBIndex::c_2;       // RHS Real (bf16)
constexpr auto cb_in1_i = tt::CBIndex::c_3;       // RHS Imag (bf16)
constexpr auto cb_tw_r = tt::CBIndex::c_4;        // Twiddle Real (bf16)
constexpr auto cb_tw_i = tt::CBIndex::c_5;        // Twiddle Imag (bf16)

// ═══════════════════════════════════════════════════════════════════════════
// INTERMEDIATE CBs (float32) - for fp32 computation
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_fp32_in0_r = tt::CBIndex::c_6;  // LHS Real (fp32)
constexpr auto cb_fp32_in0_i = tt::CBIndex::c_7;  // LHS Imag (fp32)
constexpr auto cb_fp32_in1_r = tt::CBIndex::c_8;  // RHS Real (fp32)
constexpr auto cb_fp32_in1_i = tt::CBIndex::c_9;  // RHS Imag (fp32)
constexpr auto cb_fp32_tw_r = tt::CBIndex::c_10;  // Twiddle Real (fp32)
constexpr auto cb_fp32_tw_i = tt::CBIndex::c_11;  // Twiddle Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT INTERMEDIATE CBs (float32)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_fp32_out0_r = tt::CBIndex::c_12; // Out LHS Real (fp32)
constexpr auto cb_fp32_out0_i = tt::CBIndex::c_13; // Out LHS Imag (fp32)
constexpr auto cb_fp32_out1_r = tt::CBIndex::c_14; // Out RHS Real (fp32)
constexpr auto cb_fp32_out1_i = tt::CBIndex::c_15; // Out RHS Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT CBs (bfloat16)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_out0_r = tt::CBIndex::c_16;     // Out LHS Real (bf16)
constexpr auto cb_out0_i = tt::CBIndex::c_17;     // Out LHS Imag (bf16)
constexpr auto cb_out1_r = tt::CBIndex::c_18;     // Out RHS Real (bf16)
constexpr auto cb_out1_i = tt::CBIndex::c_19;     // Out RHS Imag (bf16)

// ═══════════════════════════════════════════════════════════════════════════
// TEMP CBs (float32)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_tmp0 = tt::CBIndex::c_24;       // Temp 0 (fp32)
constexpr auto cb_tmp1 = tt::CBIndex::c_25;       // Temp 1 (fp32)
constexpr auto cb_f0 = tt::CBIndex::c_26;         // f0 (fp32)
constexpr auto cb_f1 = tt::CBIndex::c_27;         // f1 (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Reblock bf16 tile to fp32 tile (copy with format conversion)
// ═══════════════════════════════════════════════════════════════════════════
inline void reblock_bf16_to_fp32(uint32_t cb_bf16, uint32_t cb_fp32) {
    // Wait for bf16 input
    cb_wait_front(cb_bf16, 1);
    
    // Reserve fp32 output
    cb_reserve_back(cb_fp32, 1);
    
    // Copy tile: bf16 → DST (automatically fp32) → fp32 CB
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_bf16);
    copy_tile(cb_bf16, 0, 0);  // tile 0 from cb → DST[0]
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_fp32);     // DST[0] → cb_fp32
    tile_regs_release();
    
    cb_push_back(cb_fp32, 1);
    cb_pop_front(cb_bf16, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Reblock fp32 tile to bf16 tile (copy with format conversion)
// ═══════════════════════════════════════════════════════════════════════════
inline void reblock_fp32_to_bf16(uint32_t cb_fp32, uint32_t cb_bf16) {
    // Wait for fp32 input
    cb_wait_front(cb_fp32, 1);
    
    // Reserve bf16 output
    cb_reserve_back(cb_bf16, 1);
    
    // Copy tile: fp32 → DST → bf16 CB (truncation happens here)
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_fp32);
    copy_tile(cb_fp32, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_bf16);
    tile_regs_release();
    
    cb_push_back(cb_bf16, 1);
    cb_pop_front(cb_fp32, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Math op on fp32 CBs (pops both inputs)
// ═══════════════════════════════════════════════════════════════════════════
template <int OP>
inline void math_op_fp32(uint32_t in0, uint32_t in1, uint32_t out) {
    cb_wait_front(in0, 1);
    cb_wait_front(in1, 1);
    cb_reserve_back(out, 1);
    
    binary_op_init_common(in0, in1, out);
    
    if constexpr (OP == ADD) {
        add_tiles_init(in0, in1);
    } else if constexpr (OP == SUB) {
        sub_tiles_init(in0, in1);
    } else if constexpr (OP == MUL) {
        mul_tiles_init(in0, in1);
    }
    
    tile_regs_acquire();
    
    if constexpr (OP == ADD) {
        add_tiles(in0, in1, 0, 0, 0);
    } else if constexpr (OP == SUB) {
        sub_tiles(in0, in1, 0, 0, 0);
    } else if constexpr (OP == MUL) {
        mul_tiles(in0, in1, 0, 0, 0);
    }
    
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    
    cb_push_back(out, 1);
    cb_pop_front(in0, 1);
    cb_pop_front(in1, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Math op on fp32 CBs (keeps inputs - no pop)
// ═══════════════════════════════════════════════════════════════════════════
template <int OP>
inline void math_op_fp32_keep(uint32_t in0, uint32_t in1, uint32_t out) {
    cb_wait_front(in0, 1);
    cb_wait_front(in1, 1);
    cb_reserve_back(out, 1);
    
    binary_op_init_common(in0, in1, out);
    
    if constexpr (OP == ADD) {
        add_tiles_init(in0, in1);
    } else if constexpr (OP == SUB) {
        sub_tiles_init(in0, in1);
    } else if constexpr (OP == MUL) {
        mul_tiles_init(in0, in1);
    }
    
    tile_regs_acquire();
    
    if constexpr (OP == ADD) {
        add_tiles(in0, in1, 0, 0, 0);
    } else if constexpr (OP == SUB) {
        sub_tiles(in0, in1, 0, 0, 0);
    } else if constexpr (OP == MUL) {
        mul_tiles(in0, in1, 0, 0, 0);
    }
    
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    
    cb_push_back(out, 1);
    // Don't pop inputs!
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN KERNEL
// ═══════════════════════════════════════════════════════════════════════════
namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    
    // Initialize
    copy_tile_to_dst_init_short(cb_in0_r);
    
    for (uint32_t t = 0; t < num_tiles; t++) {
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: Convert all bf16 inputs → fp32
        // ═══════════════════════════════════════════════════════════════════
        reblock_bf16_to_fp32(cb_in0_r, cb_fp32_in0_r);
        reblock_bf16_to_fp32(cb_in0_i, cb_fp32_in0_i);
        reblock_bf16_to_fp32(cb_in1_r, cb_fp32_in1_r);
        reblock_bf16_to_fp32(cb_in1_i, cb_fp32_in1_i);
        reblock_bf16_to_fp32(cb_tw_r, cb_fp32_tw_r);
        reblock_bf16_to_fp32(cb_tw_i, cb_fp32_tw_i);
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: FFT Butterfly in fp32
        // ═══════════════════════════════════════════════════════════════════
        
        // f0 = in1_r * tw_r - in1_i * tw_i
        math_op_fp32_keep<MUL>(cb_fp32_in1_r, cb_fp32_tw_r, cb_tmp0);
        math_op_fp32_keep<MUL>(cb_fp32_in1_i, cb_fp32_tw_i, cb_tmp1);
        math_op_fp32<SUB>(cb_tmp0, cb_tmp1, cb_f0);
        
        // f1 = in1_r * tw_i + in1_i * tw_r
        math_op_fp32_keep<MUL>(cb_fp32_in1_r, cb_fp32_tw_i, cb_tmp0);
        math_op_fp32_keep<MUL>(cb_fp32_in1_i, cb_fp32_tw_r, cb_tmp1);
        math_op_fp32<ADD>(cb_tmp0, cb_tmp1, cb_f1);
        
        // Done with in1 and tw
        cb_pop_front(cb_fp32_in1_r, 1);
        cb_pop_front(cb_fp32_in1_i, 1);
        cb_pop_front(cb_fp32_tw_r, 1);
        cb_pop_front(cb_fp32_tw_i, 1);
        
        // Butterfly outputs
        // out0_r = in0_r + f0
        math_op_fp32_keep<ADD>(cb_fp32_in0_r, cb_f0, cb_fp32_out0_r);
        // out1_r = in0_r - f0
        math_op_fp32<SUB>(cb_fp32_in0_r, cb_f0, cb_fp32_out1_r);
        
        // out0_i = in0_i + f1
        math_op_fp32_keep<ADD>(cb_fp32_in0_i, cb_f1, cb_fp32_out0_i);
        // out1_i = in0_i - f1
        math_op_fp32<SUB>(cb_fp32_in0_i, cb_f1, cb_fp32_out1_i);
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: Convert all fp32 outputs → bf16
        // ═══════════════════════════════════════════════════════════════════
        reblock_fp32_to_bf16(cb_fp32_out0_r, cb_out0_r);
        reblock_fp32_to_bf16(cb_fp32_out0_i, cb_out0_i);
        reblock_fp32_to_bf16(cb_fp32_out1_r, cb_out1_r);
        reblock_fp32_to_bf16(cb_fp32_out1_i, cb_out1_i);
    }
}
}
