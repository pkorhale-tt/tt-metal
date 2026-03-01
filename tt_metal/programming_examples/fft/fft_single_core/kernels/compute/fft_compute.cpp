// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/common.h"

// Define operations
#define ADD 0
#define SUB 1
#define MUL 2

// Define circular buffer indices
constexpr auto cb_data0_r = tt::CBIndex::c_0; // LHS Real
constexpr auto cb_data0_i = tt::CBIndex::c_1; // LHS Imag
constexpr auto cb_data1_r = tt::CBIndex::c_2; // RHS Real
constexpr auto cb_data1_i = tt::CBIndex::c_3; // RHS Imag
constexpr auto cb_twiddle_r = tt::CBIndex::c_4; // Twiddle Real
constexpr auto cb_twiddle_i = tt::CBIndex::c_5; // Twiddle Imag
constexpr auto cb_out0_r = tt::CBIndex::c_16; // Output LHS Real
constexpr auto cb_out0_i = tt::CBIndex::c_17; // Output LHS Imag
constexpr auto cb_out1_r = tt::CBIndex::c_18; // Output RHS Real
constexpr auto cb_out1_i = tt::CBIndex::c_19; // Output RHS Imag

// Temp buffers
constexpr auto cb_tmp0 = tt::CBIndex::c_24;
constexpr auto cb_tmp1 = tt::CBIndex::c_25;
constexpr auto cb_f0   = tt::CBIndex::c_26; // Temp Butterfly Real (f0)
constexpr auto cb_f1   = tt::CBIndex::c_27; // Temp Butterfly Imag (f1)


template <int OP>
inline void math_op(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb)
{
    // Need to initialize unpacker/math for each operation
    binary_op_init_common(in0_cb, in1_cb, out_cb);
    if constexpr (OP == ADD) {
        add_tiles_init(in0_cb, in1_cb);
    } else if constexpr (OP == SUB) {
        sub_tiles_init(in0_cb, in1_cb);
    } else if constexpr (OP == MUL) {
        mul_tiles_init(in0_cb, in1_cb);
    }

    tile_regs_acquire();
    
    // Perform standard FPU operation on 1st tile of in0 and in1, storing to DST reg 0
    if constexpr (OP == ADD) {
        add_tiles(in0_cb, in1_cb, 0, 0, 0); // cb0[0] + cb1[0] -> dst[0]
    } else if constexpr (OP == SUB) {
        sub_tiles(in0_cb, in1_cb, 0, 0, 0); // cb0[0] - cb1[0] -> dst[0]
    } else if constexpr (OP == MUL) {
        mul_tiles(in0_cb, in1_cb, 0, 0, 0); // cb0[0] * cb1[0] -> dst[0]
    }
    
    tile_regs_commit();
    tile_regs_wait();
    
    cb_reserve_back(out_cb, 1);
    pack_tile(0, out_cb); // pack the result from dst[0]
    
    tile_regs_release();
    cb_push_back(out_cb, 1);
}

namespace NAMESPACE {

void MAIN {
    uint32_t numSteps = get_arg_val<uint32_t>(0);
    
    // Note: We initialize copy_tile first (it uses unpacker internally)
    copy_tile_to_dst_init_short(cb_data1_r); 
    
    for (uint32_t step = 0; step < numSteps; step++)
    {
        // Wait for RHS data and twiddle factors
        cb_wait_front(cb_data1_r, 1);
        cb_wait_front(cb_data1_i, 1);
        cb_wait_front(cb_twiddle_r, 1);
        cb_wait_front(cb_twiddle_i, 1);
        
        // ---- f0 = r1*wr - i1*wi ----
        math_op<MUL>(cb_data1_r, cb_twiddle_r, cb_tmp0);
        cb_wait_front(cb_tmp0, 1);

        math_op<MUL>(cb_data1_i, cb_twiddle_i, cb_tmp1);
        cb_wait_front(cb_tmp1, 1);

        math_op<SUB>(cb_tmp0, cb_tmp1, cb_f0);
        
        // pop temp cb
        cb_pop_front(cb_tmp0, 1);
        cb_pop_front(cb_tmp1, 1);
        
        // ---- f1 = r1*wi + i1*wr ----
        math_op<MUL>(cb_data1_r, cb_twiddle_i, cb_tmp0);
        cb_wait_front(cb_tmp0, 1);

        math_op<MUL>(cb_data1_i, cb_twiddle_r, cb_tmp1);
        cb_wait_front(cb_tmp1, 1);

        math_op<ADD>(cb_tmp0, cb_tmp1, cb_f1);

        // pop temp cb
        cb_pop_front(cb_tmp0, 1);
        cb_pop_front(cb_tmp1, 1);
        
        // Pop twiddle factors (we won't need them anymore for this tile)
        cb_pop_front(cb_twiddle_r, 1);
        cb_pop_front(cb_twiddle_i, 1);
        
        // Wait for LHS data and the computed f factors
        cb_wait_front(cb_data0_r, 1);
        cb_wait_front(cb_data0_i, 1);
        cb_wait_front(cb_f0, 1);
        cb_wait_front(cb_f1, 1);
        
        // ---- Butterfly ----
        // out1 (RHS output) = data0 - f
        math_op<SUB>(cb_data0_r, cb_f0, cb_out1_r);
        math_op<SUB>(cb_data0_i, cb_f1, cb_out1_i);
        
        // out0 (LHS output) = data0 + f
        math_op<ADD>(cb_data0_r, cb_f0, cb_out0_r);
        math_op<ADD>(cb_data0_i, cb_f1, cb_out0_i);
        
        // Free inputs and temp f
        cb_pop_front(cb_data0_r, 1);
        cb_pop_front(cb_data0_i, 1);
        cb_pop_front(cb_data1_r, 1);
        cb_pop_front(cb_data1_i, 1);
        cb_pop_front(cb_f0, 1);
        cb_pop_front(cb_f1, 1);
    }
}
}
