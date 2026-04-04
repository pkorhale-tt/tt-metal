// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0
//
// fft_compute_f32.cpp  –  fixed for tt-metal LLK-direct API

#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_pack_api.h"

// ---------------------------------------------------------------------------
// maths_sfpu helpers using the LLK API directly
// Each helper:
//   1. Calls the op-specific init OUTSIDE acquire (deadlock fix, paper Listing 1.3)
//   2. Reserves output CB page
//   3. acquire → unpack both inputs into dst[0], dst[1] → op → commit
//   4. Optionally pops input CBs
//   5. wait → pack dst[0] to output CB → release → push
// ---------------------------------------------------------------------------

inline void maths_sfpu_mul(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    llk_math_eltwise_binary_init<ELEWISE_BINARY_MUL, NONE, MATH_FIDELITY>();
    llk_unpack_AB_init<BroadcastType::NONE>(cb_in_1, cb_in_2);

    cb_reserve_back(cb_tgt, 1);

    llk_math_wait_for_dest_available();
    llk_unpack_AB(cb_in_1, cb_in_2, 0, 0);
    llk_math_eltwise_binary<ELEWISE_BINARY_MUL, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(0);
    llk_math_dest_section_done();

    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);

    llk_packer_wait_for_math_done();
    llk_pack<false, false>(0, cb_tgt, 0);
    llk_pack_dest_section_done();
    cb_push_back(cb_tgt, 1);
}

inline void maths_sfpu_sub(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    llk_math_eltwise_binary_init<ELEWISE_BINARY_SUB, NONE, MATH_FIDELITY>();
    llk_unpack_AB_init<BroadcastType::NONE>(cb_in_1, cb_in_2);

    cb_reserve_back(cb_tgt, 1);

    llk_math_wait_for_dest_available();
    llk_unpack_AB(cb_in_1, cb_in_2, 0, 0);
    llk_math_eltwise_binary<ELEWISE_BINARY_SUB, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(0);
    llk_math_dest_section_done();

    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);

    llk_packer_wait_for_math_done();
    llk_pack<false, false>(0, cb_tgt, 0);
    llk_pack_dest_section_done();
    cb_push_back(cb_tgt, 1);
}

inline void maths_sfpu_add(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    llk_math_eltwise_binary_init<ELEWISE_BINARY_ADD, NONE, MATH_FIDELITY>();
    llk_unpack_AB_init<BroadcastType::NONE>(cb_in_1, cb_in_2);

    cb_reserve_back(cb_tgt, 1);

    llk_math_wait_for_dest_available();
    llk_unpack_AB(cb_in_1, cb_in_2, 0, 0);
    llk_math_eltwise_binary<ELEWISE_BINARY_ADD, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(0);
    llk_math_dest_section_done();

    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);

    llk_packer_wait_for_math_done();
    llk_pack<false, false>(0, cb_tgt, 0);
    llk_pack_dest_section_done();
    cb_push_back(cb_tgt, 1);
}

void kernel_main() {
    const uint32_t num_steps  = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_data0_r   = 0;
    constexpr uint32_t cb_data0_i   = 1;
    constexpr uint32_t cb_data1_r   = 2;
    constexpr uint32_t cb_data1_i   = 3;
    constexpr uint32_t cb_twiddle_r = 4;
    constexpr uint32_t cb_twiddle_i = 5;
    constexpr uint32_t cb_out0_r    = 16;
    constexpr uint32_t cb_out0_i    = 17;
    constexpr uint32_t cb_out1_r    = 18;
    constexpr uint32_t cb_out1_i    = 19;
    constexpr uint32_t cb_int0      = 20;
    constexpr uint32_t cb_int1      = 21;
    constexpr uint32_t cb_f0        = 22;
    constexpr uint32_t cb_f1        = 23;

    for (uint32_t step = 0; step < num_steps; ++step) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {

            cb_wait_front(cb_data1_r, 1);
            cb_wait_front(cb_data1_i, 1);

            // f0 = data1_r*tw_r - data1_i*tw_i
            maths_sfpu_mul(cb_data1_r, cb_twiddle_r, cb_int0);
            maths_sfpu_mul(cb_data1_i, cb_twiddle_i, cb_int1);
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            maths_sfpu_sub(cb_int0, cb_int1, cb_f0, true, true);

            // f1 = data1_r*tw_i + data1_i*tw_r
            maths_sfpu_mul(cb_data1_r, cb_twiddle_i, cb_int0);
            maths_sfpu_mul(cb_data1_i, cb_twiddle_r, cb_int1);
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            maths_sfpu_add(cb_int0, cb_int1, cb_f1, true, true);

            cb_wait_front(cb_data0_r, 1);
            cb_wait_front(cb_data0_i, 1);

            // out1 = data0 - f
            cb_wait_front(cb_f0, 1);
            maths_sfpu_sub(cb_data0_r, cb_f0, cb_out1_r);
            cb_wait_front(cb_f1, 1);
            maths_sfpu_sub(cb_data0_i, cb_f1, cb_out1_i);

            // out0 = data0 + f
            cb_wait_front(cb_f0, 1);
            maths_sfpu_add(cb_data0_r, cb_f0, cb_out0_r);
            cb_wait_front(cb_f1, 1);
            maths_sfpu_add(cb_data0_i, cb_f1, cb_out0_i);

            cb_pop_front(cb_data0_r, 1);
            cb_pop_front(cb_data0_i, 1);
            cb_pop_front(cb_data1_r, 1);
            cb_pop_front(cb_data1_i, 1);
            cb_pop_front(cb_f0,      1);
            cb_pop_front(cb_f1,      1);
        }
    }
}