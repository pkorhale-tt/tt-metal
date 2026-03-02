// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Kernel: FFT Butterfly computation in float32

#include <stdint.h>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

// ═══════════════════════════════════════════════════════════════════════════
// OPERATION DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════
#define ADD 0
#define SUB 1
#define MUL 2

// ═══════════════════════════════════════════════════════════════════════════
// INPUT CBs (float32 from conversion kernel)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_data0_r = tt::CBIndex::c_6;       // LHS Real (fp32)
constexpr auto cb_data0_i = tt::CBIndex::c_7;       // LHS Imag (fp32)
constexpr auto cb_data1_r = tt::CBIndex::c_8;       // RHS Real (fp32)
constexpr auto cb_data1_i = tt::CBIndex::c_9;       // RHS Imag (fp32)
constexpr auto cb_twiddle_r = tt::CBIndex::c_10;    // Twiddle Real (fp32)
constexpr auto cb_twiddle_i = tt::CBIndex::c_11;    // Twiddle Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT CBs (float32 for output conversion)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_out0_r = tt::CBIndex::c_12;       // Output LHS Real (fp32)
constexpr auto cb_out0_i = tt::CBIndex::c_13;       // Output LHS Imag (fp32)
constexpr auto cb_out1_r = tt::CBIndex::c_14;       // Output RHS Real (fp32)
constexpr auto cb_out1_i = tt::CBIndex::c_15;       // Output RHS Imag (fp32)

// ═══════════════════════════════════════════════════════════════════════════
// TEMPORARY CBs (float32)
// ═══════════════════════════════════════════════════════════════════════════
constexpr auto cb_tmp0 = tt::CBIndex::c_24;         // Temp 0
constexpr auto cb_tmp1 = tt::CBIndex::c_25;         // Temp 1
constexpr auto cb_f0 = tt::CBIndex::c_26;           // Butterfly Real
constexpr auto cb_f1 = tt::CBIndex::c_27;           // Butterfly Imag

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
// HELPER: Math operation (keeps inputs, doesn't pop)
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
    
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        
        // ═══════════════════════════════════════════════════════════════════
        // COMPLEX MULTIPLICATION: f = twiddle × data1
        // f0 = data1_r * twiddle_r - data1_i * twiddle_i
        // f1 = data1_r * twiddle_i + data1_i * twiddle_r
        // ═══════════════════════════════════════════════════════════════════
        
        // --- f0 = data1_r * twiddle_r - data1_i * twiddle_i ---
        
        // tmp0 = data1_r * twiddle_r (keep inputs)
        math_op_keep<MUL>(cb_data1_r, cb_twiddle_r, cb_tmp0);
        
        // tmp1 = data1_i * twiddle_i (keep inputs)
        math_op_keep<MUL>(cb_data1_i, cb_twiddle_i, cb_tmp1);
        
        // f0 = tmp0 - tmp1
        math_op<SUB>(cb_tmp0, cb_tmp1, cb_f0);
        
        // --- f1 = data1_r * twiddle_i + data1_i * twiddle_r ---
        
        // tmp0 = data1_r * twiddle_i (keep inputs)
        math_op_keep<MUL>(cb_data1_r, cb_twiddle_i, cb_tmp0);
        
        // tmp1 = data1_i * twiddle_r (keep inputs)
        math_op_keep<MUL>(cb_data1_i, cb_twiddle_r, cb_tmp1);
        
        // f1 = tmp0 + tmp1
        math_op<ADD>(cb_tmp0, cb_tmp1, cb_f1);
        
        // Pop data1 and twiddle (no longer needed)
        cb_pop_front(cb_data1_r, 1);
        cb_pop_front(cb_data1_i, 1);
        cb_pop_front(cb_twiddle_r, 1);
        cb_pop_front(cb_twiddle_i, 1);
        
        // ═══════════════════════════════════════════════════════════════════
        // BUTTERFLY: out0 = data0 + f, out1 = data0 - f
        // ═══════════════════════════════════════════════════════════════════
        
        // Wait for f0 and f1
        cb_wait_front(cb_f0, 1);
        cb_wait_front(cb_f1, 1);
        
        // --- out0_r = data0_r + f0 ---
        math_op_keep<ADD>(cb_data0_r, cb_f0, cb_out0_r);
        
        // --- out1_r = data0_r - f0 ---
        math_op<SUB>(cb_data0_r, cb_f0, cb_out1_r);
        // data0_r and f0 are now popped
        
        // --- out0_i = data0_i + f1 ---
        math_op_keep<ADD>(cb_data0_i, cb_f1, cb_out0_i);
        
        // --- out1_i = data0_i - f1 ---
        math_op<SUB>(cb_data0_i, cb_f1, cb_out1_i);
        // data0_i and f1 are now popped
    }
}
}