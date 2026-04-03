// SPDX-FileCopyrightText: © 2026 OpenAI
// SPDX-License-Identifier: Apache-2.0
//
// Paper-style FFT compute kernel for Wormhole / TT-Metalium.
// Safe version: explicit intermediate CBs, no direct register-to-register SFPU chaining.
// Matches the paper's basic design: reader reorders per stage, compute performs complex butterflies,
// writer scatters back to row-major order for the next stage.

#include <cstdint>
#include <compute_kernel_api.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/tile_move_copy.h>
#include <compute_kernel_api/eltwise_binary.h>

namespace {

enum class BinaryOp : uint32_t {
    Mul = 0,
    Add = 1,
    Sub = 2,
};

inline void binary_front_to_cb(const uint32_t cb_a, const uint32_t cb_b, const uint32_t cb_out, const BinaryOp op) {
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile(cb_a, 0, 0);
    copy_tile(cb_b, 0, 1);

    switch (op) {
        case BinaryOp::Mul:
            mul_binary_tile_init();
            mul_binary_tile(0, 1);
            break;
        case BinaryOp::Add:
            add_binary_tile_init();
            add_binary_tile(0, 1);
            break;
        case BinaryOp::Sub:
            sub_binary_tile_init();
            sub_binary_tile(0, 1);
            break;
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out, 0);
    tile_regs_release();

    cb_push_back(cb_out, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t num_stages        = get_arg_val<uint32_t>(0);
    const uint32_t rows_this_core    = get_arg_val<uint32_t>(1);
    const uint32_t pair_tiles_per_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    constexpr uint32_t cb_mul0   = tt::CBIndex::c_20;
    constexpr uint32_t cb_mul1   = tt::CBIndex::c_21;
    constexpr uint32_t cb_tmp_r  = tt::CBIndex::c_22;
    constexpr uint32_t cb_tmp_i  = tt::CBIndex::c_23;

    // Standard binary-op init pattern for compute kernels.
    binary_op_init_common(cb_even_r, cb_odd_r, cb_out0_r);
    add_tiles_init(cb_even_r, cb_odd_r, false);
    sub_tiles_init(cb_even_r, cb_odd_r, false);
    mul_tiles_init(cb_even_r, cb_odd_r, false);

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        for (uint32_t row = 0; row < rows_this_core; ++row) {
            for (uint32_t t = 0; t < pair_tiles_per_row; ++t) {
                // Inputs for this butterfly tile are kept alive until all outputs are formed.
                cb_wait_front(cb_even_r, 1);
                cb_wait_front(cb_even_i, 1);
                cb_wait_front(cb_odd_r, 1);
                cb_wait_front(cb_odd_i, 1);
                cb_wait_front(cb_tw_r, 1);
                cb_wait_front(cb_tw_i, 1);

                // tmp_r = odd_r * tw_r - odd_i * tw_i
                binary_front_to_cb(cb_odd_r, cb_tw_r, cb_mul0, BinaryOp::Mul);
                binary_front_to_cb(cb_odd_i, cb_tw_i, cb_mul1, BinaryOp::Mul);
                binary_front_to_cb(cb_mul0, cb_mul1, cb_tmp_r, BinaryOp::Sub);
                cb_pop_front(cb_mul0, 1);
                cb_pop_front(cb_mul1, 1);

                // tmp_i = odd_r * tw_i + odd_i * tw_r
                binary_front_to_cb(cb_odd_r, cb_tw_i, cb_mul0, BinaryOp::Mul);
                binary_front_to_cb(cb_odd_i, cb_tw_r, cb_mul1, BinaryOp::Mul);
                binary_front_to_cb(cb_mul0, cb_mul1, cb_tmp_i, BinaryOp::Add);
                cb_pop_front(cb_mul0, 1);
                cb_pop_front(cb_mul1, 1);

                // out0 = even + tmp ; out1 = even - tmp
                binary_front_to_cb(cb_even_r, cb_tmp_r, cb_out0_r, BinaryOp::Add);
                binary_front_to_cb(cb_even_i, cb_tmp_i, cb_out0_i, BinaryOp::Add);
                binary_front_to_cb(cb_even_r, cb_tmp_r, cb_out1_r, BinaryOp::Sub);
                binary_front_to_cb(cb_even_i, cb_tmp_i, cb_out1_i, BinaryOp::Sub);

                // All consumers for current tile are done.
                cb_pop_front(cb_even_r, 1);
                cb_pop_front(cb_even_i, 1);
                cb_pop_front(cb_odd_r, 1);
                cb_pop_front(cb_odd_i, 1);
                cb_pop_front(cb_tw_r, 1);
                cb_pop_front(cb_tw_i, 1);
                cb_pop_front(cb_tmp_r, 1);
                cb_pop_front(cb_tmp_i, 1);
            }
        }
    }
}
