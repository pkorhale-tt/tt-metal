// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combined FFT Compute Kernel: fft_compute_bf16_fp32.cpp
// 
// This kernel performs:
// 1. Convert bf16 inputs → fp32
// 2. FFT Butterfly computation in fp32 (full precision)
// 3. Convert fp32 outputs → bf16

#include <stdint.h>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/common.h"

// ═══════════════════════════════════════════════════════════════════════════
// OPERATION DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════
#define ADD 0
#define SUB 1
#define MUL 2

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
// INTERMEDIATE CBs (float32 for computation)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_data0_r_fp32 = tt::CBIndex::c_6;      // LHS Real (fp32)
constexpr auto cb_data0_i_fp32 = tt::CBIndex::c_7;      // LHS Imag (fp32)
constexpr auto cb_data1_r_fp32 = tt::CBIndex::c_8;      // RHS Real (fp32)
constexpr auto cb_data1_i_fp32 = tt::CBIndex::c_9;      // RHS Imag (fp32)
constexpr auto cb_twiddle_r_fp32 = tt::CBIndex::c_10;   // Twiddle Real (fp32)
constexpr auto cb_twiddle_i_fp32 = tt::CBIndex::c_11;   // Twiddle Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// FFT OUTPUT CBs (float32)
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
// TEMPORARY CBs (float32)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_tmp0 = tt::CBIndex::c_24;             // Temp 0 (fp32)
constexpr auto cb_tmp1 = tt::CBIndex::c_25;             // Temp 1 (fp32)
constexpr auto cb_f0 = tt::CBIndex::c_26;               // Butterfly Real (fp32)
constexpr auto cb_f1 = tt::CBIndex::c_27;               // Butterfly Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Convert bf16 tile to fp32 tile
// ═══════════════════════════════════════════════════════════════════════════
inline void convert_bf16_to_fp32(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Convert fp32 tile to bf16 tile
// ═══════════════════════════════════════════════════════════════════════════
inline void convert_fp32_to_bf16(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Math operation (consumes both inputs)
// ═══════════════════════════════════════════════════════════════════════════
template <int OP>
inline void math_op(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    cb_wait_front(in0_cb, 1);
    cb_wait_front(in1_cb, 1);
    cb_reserve_back(out_cb, 1);
    
    binary_op_init_common(in0_cb, in1_cb, out_cb);
    
    if constexpr (OP == ADD) {
        add_tiles_init(in0_cb, in1_cb);
    } else if constexpr (OP == SUB) {
        sub_tiles_init(in0_cb, in1_cb);
    } else if constexpr (OP == MUL) {
        mul_tiles_init(in0_cb, in1_cb);
    }
    
    tile_regs_acquire();
    
    if constexpr (OP == ADD) {
        add_tiles(in0_cb, in1_cb, 0, 0, 0);
    } else if constexpr (OP == SUB) {
        sub_tiles(in0_cb, in1_cb, 0, 0, 0);
    } else if constexpr (OP == MUL) {
        mul_tiles(in0_cb, in1_cb, 0, 0, 0);
    }
    
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    
    cb_push_back(out_cb, 1);
    cb_pop_front(in0_cb, 1);
    cb_pop_front(in1_cb, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Math operation (keeps inputs - doesn't pop)
// ═══════════════════════════════════════════════════════════════════════════
template <int OP>
inline void math_op_keep(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    cb_wait_front(in0_cb, 1);
    cb_wait_front(in1_cb, 1);
    cb_reserve_back(out_cb, 1);
    
    binary_op_init_common(in0_cb, in1_cb, out_cb);
    
    if constexpr (OP == ADD) {
        add_tiles_init(in0_cb, in1_cb);
    } else if constexpr (OP == SUB) {
        sub_tiles_init(in0_cb, in1_cb);
    } else if constexpr (OP == MUL) {
        mul_tiles_init(in0_cb, in1_cb);
    }
    
    tile_regs_acquire();
    
    if constexpr (OP == ADD) {
        add_tiles(in0_cb, in1_cb, 0, 0, 0);
    } else if constexpr (OP == SUB) {
        sub_tiles(in0_cb, in1_cb, 0, 0, 0);
    } else if constexpr (OP == MUL) {
        mul_tiles(in0_cb, in1_cb, 0, 0, 0);
    }
    
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    
    cb_push_back(out_cb, 1);
    // Don't pop inputs - keep them for reuse!
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN KERNEL
// ═══════════════════════════════════════════════════════════════════════════
namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    
    // Initialize copy_tile
    copy_tile_to_dst_init_short(cb_data0_r_bf16);
    
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: Convert ALL inputs from bf16 → fp32
        // ═══════════════════════════════════════════════════════════════════
        convert_bf16_to_fp32(cb_data0_r_bf16, cb_data0_r_fp32);
        convert_bf16_to_fp32(cb_data0_i_bf16, cb_data0_i_fp32);
        convert_bf16_to_fp32(cb_data1_r_bf16, cb_data1_r_fp32);
        convert_bf16_to_fp32(cb_data1_i_bf16, cb_data1_i_fp32);
        convert_bf16_to_fp32(cb_twiddle_r_bf16, cb_twiddle_r_fp32);
        convert_bf16_to_fp32(cb_twiddle_i_bf16, cb_twiddle_i_fp32);
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: FFT BUTTERFLY COMPUTATION (all in fp32)
        // ═══════════════════════════════════════════════════════════════════
        
        // --- f0 = data1_r * twiddle_r - data1_i * twiddle_i ---
        math_op_keep<MUL>(cb_data1_r_fp32, cb_twiddle_r_fp32, cb_tmp0);
        math_op_keep<MUL>(cb_data1_i_fp32, cb_twiddle_i_fp32, cb_tmp1);
        math_op<SUB>(cb_tmp0, cb_tmp1, cb_f0);
        
        // --- f1 = data1_r * twiddle_i + data1_i * twiddle_r ---
        math_op_keep<MUL>(cb_data1_r_fp32, cb_twiddle_i_fp32, cb_tmp0);
        math_op_keep<MUL>(cb_data1_i_fp32, cb_twiddle_r_fp32, cb_tmp1);
        math_op<ADD>(cb_tmp0, cb_tmp1, cb_f1);
        
        // Pop data1 and twiddle (no longer needed)
        cb_pop_front(cb_data1_r_fp32, 1);
        cb_pop_front(cb_data1_i_fp32, 1);
        cb_pop_front(cb_twiddle_r_fp32, 1);
        cb_pop_front(cb_twiddle_i_fp32, 1);
        
                // --- Butterfly: out0 = data0 + f, out1 = data0 - f ---
        cb_wait_front(cb_f0, 1);
        cb_wait_front(cb_f1, 1);
        
        // --- out0_r = data0_r + f0 ---
        math_op_keep<ADD>(cb_data0_r_fp32, cb_f0, cb_out0_r_fp32);
        
        // --- out1_r = data0_r - f0 ---
        math_op<SUB>(cb_data0_r_fp32, cb_f0, cb_out1_r_fp32);
        // data0_r and f0 are now popped
        
        // --- out0_i = data0_i + f1 ---
        math_op_keep<ADD>(cb_data0_i_fp32, cb_f1, cb_out0_i_fp32);
        
        // --- out1_i = data0_i - f1 ---
        math_op<SUB>(cb_data0_i_fp32, cb_f1, cb_out1_i_fp32);
        // data0_i and f1 are now popped
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: Convert ALL outputs from fp32 → bf16
        // ═══════════════════════════════════════════════════════════════════
        convert_fp32_to_bf16(cb_out0_r_fp32, cb_out0_r_bf16);
        convert_fp32_to_bf16(cb_out0_i_fp32, cb_out0_i_bf16);
        convert_fp32_to_bf16(cb_out1_r_fp32, cb_out1_r_bf16);
        convert_fp32_to_bf16(cb_out1_i_fp32, cb_out1_i_bf16);
    }
}
}
        