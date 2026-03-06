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

constexpr uint32_t ADD = 0;
constexpr uint32_t SUB = 1;
constexpr uint32_t MUL = 2;
constexpr uint32_t NEG = 3;
#define USE_SFPU 0

//-------------------------
// SFPU Unary Operation
//-------------------------
template <uint32_t OPERATION, bool CB_OP_IN = false>
FORCE_INLINE void unary_sfpu_op(uint32_t cb_in, uint32_t cb_tgt) {
    if constexpr (CB_OP_IN) cb_wait_front(cb_in, 1);
    
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    
    if constexpr (OPERATION == NEG) {
        negative_tile_init();
        negative_tile(0);
    }
    
    tile_regs_commit();
    if constexpr (CB_OP_IN) cb_pop_front(cb_in, 1);
    
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

//-------------------------
// Binary SFPU Operation
//-------------------------
template <uint32_t OPERATION, bool CB_OP_IN1 = false, bool CB_OP_IN2 = false>
FORCE_INLINE void maths_sfpu_op(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt) {
    if constexpr (CB_OP_IN1) cb_wait_front(cb_in_1, 1);
    if constexpr (CB_OP_IN2) cb_wait_front(cb_in_2, 1);
    
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);
    
    if constexpr (OPERATION == ADD) {
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else if constexpr (OPERATION == SUB) {
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
    } else if constexpr (OPERATION == MUL) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    }
    
    tile_regs_commit();
    if constexpr (CB_OP_IN1) cb_pop_front(cb_in_1, 1);
    if constexpr (CB_OP_IN2) cb_pop_front(cb_in_2, 1);
    
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

//-------------------------
// Standard FPU Binary Operation
//-------------------------
template <uint32_t OPERATION, bool CB_OP_IN1 = false, bool CB_OP_IN2 = false>
FORCE_INLINE void maths_mm_op(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt) {
    if constexpr (CB_OP_IN1) cb_wait_front(cb_in_1, 1);
    if constexpr (CB_OP_IN2) cb_wait_front(cb_in_2, 1);
    
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
    
    if constexpr (CB_OP_IN1) cb_pop_front(cb_in_1, 1);
    if constexpr (CB_OP_IN2) cb_pop_front(cb_in_2, 1);
    
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
    uint32_t cb_tmp0, uint32_t cb_tmp1
) {
    // Real part: ac - bd
#ifdef USE_SFPU
    maths_sfpu_op<MUL, false, false>(cb_a_r, cb_b_r, cb_tmp0);
    maths_sfpu_op<MUL, false, false>(cb_a_i, cb_b_i, cb_tmp1);
    maths_sfpu_op<SUB, true, true>(cb_tmp0, cb_tmp1, cb_out_r);
#else
    maths_mm_op<MUL, false, false>(cb_a_r, cb_b_r, cb_tmp0);
    maths_mm_op<MUL, false, false>(cb_a_i, cb_b_i, cb_tmp1);
    maths_mm_op<SUB, true, true>(cb_tmp0, cb_tmp1, cb_out_r);
#endif

    // Imaginary part: ad + bc
#ifdef USE_SFPU
    maths_sfpu_op<MUL, false, false>(cb_a_r, cb_b_i, cb_tmp0);
    maths_sfpu_op<MUL, false, false>(cb_a_i, cb_b_r, cb_tmp1);
    maths_sfpu_op<ADD, true, true>(cb_tmp0, cb_tmp1, cb_out_i);
#else
    maths_mm_op<MUL, false, false>(cb_a_r, cb_b_i, cb_tmp0);
    maths_mm_op<MUL, false, false>(cb_a_i, cb_b_r, cb_tmp1);
    maths_mm_op<ADD, true, true>(cb_tmp0, cb_tmp1, cb_out_i);
#endif
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
    cb_wait_front(cb_odd_r, 1);
    cb_wait_front(cb_odd_i, 1);
    cb_wait_front(cb_tw_r, 1);
    cb_wait_front(cb_tw_i, 1);
    
    // Compute W * O[k]
    complex_multiply(
        cb_odd_r, cb_odd_i,
        cb_tw_r, cb_tw_i,
        cb_tw_odd_r, cb_tw_odd_i,
        cb_tmp0, cb_tmp1
    );
    
    cb_pop_front(cb_odd_r, 1);
    cb_pop_front(cb_odd_i, 1);
    
    cb_wait_front(cb_even_r, 1);
    cb_wait_front(cb_even_i, 1);
    cb_wait_front(cb_tw_odd_r, 1);
    cb_wait_front(cb_tw_odd_i, 1);
    
    // X[k] = E[k] + W * O[k]
#ifdef USE_SFPU
    maths_sfpu_op<ADD, false, false>(cb_even_r, cb_tw_odd_r, cb_out0_r);
    maths_sfpu_op<ADD, false, false>(cb_even_i, cb_tw_odd_i, cb_out0_i);
#else
    maths_mm_op<ADD, false, false>(cb_even_r, cb_tw_odd_r, cb_out0_r);
    maths_mm_op<ADD, false, false>(cb_even_i, cb_tw_odd_i, cb_out0_i);
#endif
    
    // X[k + N/2] = E[k] - W * O[k]
#ifdef USE_SFPU
    maths_sfpu_op<SUB, false, false>(cb_even_r, cb_tw_odd_r, cb_out1_r);
    maths_sfpu_op<SUB, false, false>(cb_even_i, cb_tw_odd_i, cb_out1_i);
#else
    maths_mm_op<SUB, false, false>(cb_even_r, cb_tw_odd_r, cb_out1_r);
    maths_mm_op<SUB, false, false>(cb_even_i, cb_tw_odd_i, cb_out1_i);
#endif
    
    cb_pop_front(cb_even_r, 1);
    cb_pop_front(cb_even_i, 1);
    cb_pop_front(cb_tw_odd_r, 1);
    cb_pop_front(cb_tw_odd_i, 1);
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
            unary_sfpu_op<NEG, true>(cb_tw_i, cb_neg_tmp);
            
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
        
        // Pop twiddles after processing
        cb_pop_front(cb_tw_r, 1);
        if (direction == 1) {
            // Only need to pop if not inverse because we pushed to neg_tmp
        } else {
            cb_pop_front(cb_tw_i, 1);
        }
    }
}