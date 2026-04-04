// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"

inline void maths_sfpu_mul(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

inline void maths_sfpu_sub(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

inline void maths_sfpu_add(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

void kernel_main() {
    const uint32_t num_steps  = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_data0_r   = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i   = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r   = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i   = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;
    constexpr uint32_t cb_out0_r    = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i    = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r    = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i    = tt::CBIndex::c_19;
    constexpr uint32_t cb_int0      = tt::CBIndex::c_20;
    constexpr uint32_t cb_int1      = tt::CBIndex::c_21;
    constexpr uint32_t cb_f0        = tt::CBIndex::c_22;
    constexpr uint32_t cb_f1        = tt::CBIndex::c_23;

    // Global init: prime the unpacker for the first CB it will see
    copy_tile_to_dst_init_short(cb_data1_r);

    for (uint32_t step = 0; step < num_steps; ++step) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_data1_r, 1);
            cb_wait_front(cb_data1_i, 1);
            cb_wait_front(cb_twiddle_r, 1);
            cb_wait_front(cb_twiddle_i, 1);

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

            // out1 = data0 - f  (odd output)
            cb_wait_front(cb_f0, 1);
            maths_sfpu_sub(cb_data0_r, cb_f0, cb_out1_r);
            cb_wait_front(cb_f1, 1);
            maths_sfpu_sub(cb_data0_i, cb_f1, cb_out1_i);

            // out0 = data0 + f  (even output)
            cb_wait_front(cb_f0, 1);
            maths_sfpu_add(cb_data0_r, cb_f0, cb_out0_r);
            cb_wait_front(cb_f1, 1);
            maths_sfpu_add(cb_data0_i, cb_f1, cb_out0_i);

            cb_pop_front(cb_data0_r, 1);
            cb_pop_front(cb_data0_i, 1);
            cb_pop_front(cb_data1_r, 1);
            cb_pop_front(cb_data1_i, 1);
            cb_pop_front(cb_twiddle_r, 1);
            cb_pop_front(cb_twiddle_i, 1);
            cb_pop_front(cb_f0, 1);
            cb_pop_front(cb_f1, 1);
        }
    }
}