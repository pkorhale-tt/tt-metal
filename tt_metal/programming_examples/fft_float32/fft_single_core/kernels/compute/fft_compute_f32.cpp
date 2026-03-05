// fft_compute_f32.cpp
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

constexpr uint32_t ADD = 0;
constexpr uint32_t SUB = 1;
constexpr uint32_t MUL = 2;
constexpr uint32_t NEG = 3;

//-------------------------
// SFPU Unary Operation
//-------------------------
template <uint32_t OPERATION>
FORCE_INLINE void unary_sfpu_op(uint32_t cb_in, uint32_t cb_tgt, bool wait_in = true, bool pop_in = true) {
    if (wait_in) cb_wait_front(cb_in, 1);
    
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    
    if constexpr (OPERATION == NEG) {
        negative_tile_init();
        negative_tile(0);
    }
    
    tile_regs_commit();
    
    if (pop_in) cb_pop_front(cb_in, 1);
    
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

//-------------------------
// Standard FPU Binary Operation
//-------------------------
template <uint32_t OPERATION>
FORCE_INLINE void binary_fpu_op(
    uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
    bool wait_in1 = true, bool wait_in2 = true,
    bool pop_in1 = true, bool pop_in2 = true
) {
    if (wait_in1) cb_wait_front(cb_in_1, 1);
    if (wait_in2) cb_wait_front(cb_in_2, 1);
    
    tile_regs_acquire();
    
    if constexpr (OPERATION == ADD) {
        add_tiles_init(cb_in_1, cb_in_2);
        add_tiles(cb_in_1, cb_in_2, 0, 0, 0);
    } else if constexpr (OPERATION == SUB) {
        sub_tiles_init(cb_in_1, cb_in_2);
        sub_tiles(cb_in_1, cb_in_2, 0, 0, 0);
    } else if constexpr (OPERATION == MUL) {
        mul_tiles_init(cb_in_1, cb_in_2);
        mul_tiles(cb_in_1, cb_in_2, 0, 0, 0);
    }
    
    tile_regs_commit();
    
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);
    
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

//-------------------------
// Complex Multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
//-------------------------
FORCE_INLINE void complex_multiply(
    uint32_t cb_a_r, uint32_t cb_a_i,
    uint32_t cb_b_r, uint32_t cb_b_i,
    uint32_t cb_out_r, uint32_t cb_out_i,
    uint32_t cb_tmp0, uint32_t cb_tmp1,
    bool wait_a = true, bool wait_b = true,
    bool pop_a = true, bool pop_b = true
) {
    if (wait_a) {
        cb_wait_front(cb_a_r, 1);
        cb_wait_front(cb_a_i, 1);
    }
    if (wait_b) {
        cb_wait_front(cb_b_r, 1);
        cb_wait_front(cb_b_i, 1);
    }
    
    // Real part: ac - bd
    binary_fpu_op<MUL>(cb_a_r, cb_b_r, cb_tmp0, false, false, false, false);
    binary_fpu_op<MUL>(cb_a_i, cb_b_i, cb_tmp1, false, false, false, false);
    binary_fpu_op<SUB>(cb_tmp0, cb_tmp1, cb_out_r, true, true, true, true);
    
    // Imaginary part: ad + bc
    binary_fpu_op<MUL>(cb_a_r, cb_b_i, cb_tmp0, false, false, false, false);
    binary_fpu_op<MUL>(cb_a_i, cb_b_r, cb_tmp1, false, false, false, false);
    binary_fpu_op<ADD>(cb_tmp0, cb_tmp1, cb_out_i, true, true, true, true);
    
    if (pop_a) {
        cb_pop_front(cb_a_r, 1);
        cb_pop_front(cb_a_i, 1);
    }
    if (pop_b) {
        cb_pop_front(cb_b_r, 1);
        cb_pop_front(cb_b_i, 1);
    }
}

//-------------------------
// FFT Butterfly Operation
//-------------------------
FORCE_INLINE void fft_butterfly(
    uint32_t cb_even_r, uint32_t cb_even_i,
    uint32_t cb_odd_r, uint32_t cb_odd_i,
    uint32_t cb_tw_r, uint32_t cb_tw_i,
    uint32_t cb_out0_r, uint32_t cb_out0_i,
    uint32_t cb_out1_r, uint32_t cb_out1_i,
    uint32_t cb_tmp0, uint32_t cb_tmp1,
    uint32_t cb_tw_odd_r, uint32_t cb_tw_odd_i
) {
    // Compute W * O[k]
    complex_multiply(
        cb_odd_r, cb_odd_i,
        cb_tw_r, cb_tw_i,
        cb_tw_odd_r, cb_tw_odd_i,
        cb_tmp0, cb_tmp1,
        true, true, true, true
    );
    
    cb_wait_front(cb_even_r, 1);
    cb_wait_front(cb_even_i, 1);
    cb_wait_front(cb_tw_odd_r, 1);
    cb_wait_front(cb_tw_odd_i, 1);
    
    // X[k] = E[k] + W * O[k]
    binary_fpu_op<ADD>(cb_even_r, cb_tw_odd_r, cb_out0_r, false, false, false, false);
    binary_fpu_op<ADD>(cb_even_i, cb_tw_odd_i, cb_out0_i, false, false, false, false);
    
    // X[k + N/2] = E[k] - W * O[k]
    binary_fpu_op<SUB>(cb_even_r, cb_tw_odd_r, cb_out1_r, false, false, true, true);
    binary_fpu_op<SUB>(cb_even_i, cb_tw_odd_i, cb_out1_i, false, false, true, true);
    
    cb_pop_front(cb_even_r, 1);
    cb_pop_front(cb_even_i, 1);
}

void kernel_main() {
    uint32_t direction = get_arg_val<uint32_t>(0);
    uint32_t num_butterflies = get_arg_val<uint32_t>(1);
    
    // Input CBs
    constexpr auto cb_even_r = tt::CBIndex::c_0;
    constexpr auto cb_even_i = tt::CBIndex::c_1;
    constexpr auto cb_odd_r = tt::CBIndex::c_2;
    constexpr auto cb_odd_i = tt::CBIndex::c_3;
    constexpr auto cb_tw_r = tt::CBIndex::c_4;
    constexpr auto cb_tw_i = tt::CBIndex::c_5;
    
    // Output CBs
    constexpr auto cb_out0_r = tt::CBIndex::c_16;
    constexpr auto cb_out0_i = tt::CBIndex::c_17;
    constexpr auto cb_out1_r = tt::CBIndex::c_18;
    constexpr auto cb_out1_i = tt::CBIndex::c_19;
    
    // Intermediate CBs
    constexpr auto cb_tmp0 = tt::CBIndex::c_20;
    constexpr auto cb_tmp1 = tt::CBIndex::c_21;
    constexpr auto cb_tw_odd_r = tt::CBIndex::c_22;
    constexpr auto cb_tw_odd_i = tt::CBIndex::c_23;
    constexpr auto cb_neg_tmp = tt::CBIndex::c_24;
    
    // Initialize
    unary_op_init_common(cb_odd_r, cb_out0_r);
    copy_tile_to_dst_init_short(cb_odd_r);
    
    for (uint32_t bf = 0; bf < num_butterflies; bf++) {
        if (direction == 1) {
            // Inverse FFT: negate twiddle imaginary
            cb_wait_front(cb_tw_i, 1);
            unary_sfpu_op<NEG>(cb_tw_i, cb_neg_tmp, false, true);
            
            fft_butterfly(
                cb_even_r, cb_even_i,
                cb_odd_r, cb_odd_i,
                cb_tw_r, cb_neg_tmp,
                cb_out0_r, cb_out0_i,
                cb_out1_r, cb_out1_i,
                cb_tmp0, cb_tmp1,
                cb_tw_odd_r, cb_tw_odd_i
            );
            cb_pop_front(cb_neg_tmp, 1);
        } else {
            // Forward FFT
            fft_butterfly(
                cb_even_r, cb_even_i,
                cb_odd_r, cb_odd_i,
                cb_tw_r, cb_tw_i,
                cb_out0_r, cb_out0_i,
                cb_out1_r, cb_out1_i,
                cb_tmp0, cb_tmp1,
                cb_tw_odd_r, cb_tw_odd_i
            );
        }
    }
}

}  // namespace NAMESPACE