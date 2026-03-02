// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/common.h"

#define ADD 0
#define SUB 1
#define MUL 2

constexpr auto cb_data0_r = tt::CBIndex::c_0;
constexpr auto cb_data0_i = tt::CBIndex::c_1;
constexpr auto cb_data1_r = tt::CBIndex::c_2;
constexpr auto cb_data1_i = tt::CBIndex::c_3;
constexpr auto cb_tw_r = tt::CBIndex::c_4;
constexpr auto cb_tw_i = tt::CBIndex::c_5;

constexpr auto cb_out0_r = tt::CBIndex::c_16;
constexpr auto cb_out0_i = tt::CBIndex::c_17;
constexpr auto cb_out1_r = tt::CBIndex::c_18;
constexpr auto cb_out1_i = tt::CBIndex::c_19;

constexpr auto cb_tmp0 = tt::CBIndex::c_24;
constexpr auto cb_tmp1 = tt::CBIndex::c_25;
constexpr auto cb_f0 = tt::CBIndex::c_26;
constexpr auto cb_f1 = tt::CBIndex::c_27;

template <int OP>
inline void math_op(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
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
    
    cb_reserve_back(out_cb, 1);
    pack_tile(0, out_cb);
    
    tile_regs_release();
    cb_push_back(out_cb, 1);
}

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    
    copy_tile_to_dst_init_short(cb_data1_r);
    
    for (uint32_t t = 0; t < num_tiles; t++) {
        // Wait for inputs
        cb_wait_front(cb_data0_r, 1);
        cb_wait_front(cb_data0_i, 1);
        cb_wait_front(cb_data1_r, 1);
        cb_wait_front(cb_data1_i, 1);
        cb_wait_front(cb_tw_r, 1);
        cb_wait_front(cb_tw_i, 1);
        
        // f0 = data1_r * tw_r - data1_i * tw_i
        math_op<MUL>(cb_data1_r, cb_tw_r, cb_tmp0);
        cb_wait_front(cb_tmp0, 1);
        math_op<MUL>(cb_data1_i, cb_tw_i, cb_tmp1);
        cb_wait_front(cb_tmp1, 1);
        math_op<SUB>(cb_tmp0, cb_tmp1, cb_f0);
        cb_pop_front(cb_tmp0, 1);
        cb_pop_front(cb_tmp1, 1);
        
        // f1 = data1_r * tw_i + data1_i * tw_r
        math_op<MUL>(cb_data1_r, cb_tw_i, cb_tmp0);
        cb_wait_front(cb_tmp0, 1);
        math_op<MUL>(cb_data1_i, cb_tw_r, cb_tmp1);
        cb_wait_front(cb_tmp1, 1);
        math_op<ADD>(cb_tmp0, cb_tmp1, cb_f1);
        cb_pop_front(cb_tmp0, 1);
        cb_pop_front(cb_tmp1, 1);
        
        // Pop twiddles
        cb_pop_front(cb_tw_r, 1);
        cb_pop_front(cb_tw_i, 1);
        
        // Wait for f
        cb_wait_front(cb_f0, 1);
        cb_wait_front(cb_f1, 1);
        
        // Butterfly: out1 = data0 - f
        math_op<SUB>(cb_data0_r, cb_f0, cb_out1_r);
        math_op<SUB>(cb_data0_i, cb_f1, cb_out1_i);
        
        // Butterfly: out0 = data0 + f
        math_op<ADD>(cb_data0_r, cb_f0, cb_out0_r);
        math_op<ADD>(cb_data0_i, cb_f1, cb_out0_i);
        
        // Free inputs
        cb_pop_front(cb_data0_r, 1);
        cb_pop_front(cb_data0_i, 1);
        cb_pop_front(cb_data1_r, 1);
        cb_pop_front(cb_data1_i, 1);
        cb_pop_front(cb_f0, 1);
        cb_pop_front(cb_f1, 1);
    }
}
}