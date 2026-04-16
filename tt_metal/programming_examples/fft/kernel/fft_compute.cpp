// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_compute.cpp — TRISC / compute
//
// For each of the LOG2N stages, wait for EVEN/ODD/TW tiles from the reader,
// do a radix-2 DIT butterfly on the whole tile (split-complex):
//
//     W*odd  :  (odd_r + i*odd_i) * (tw_r + i*tw_i)
//     out0   =  even + W*odd
//     out1   =  even - W*odd
//
// then push OUT0/OUT1 back to the reader for the scatter step.
//
// All tile ops operate on full 32x32 fp32 tiles (TILE_SIZE_FP32).  Unused
// slots in a short (N<1024) tile are zero-padded by the host and stay zero
// through every stage, so they don't affect the valid region.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

constexpr auto CB_EVEN_R    = tt::CBIndex::c_0;
constexpr auto CB_EVEN_I    = tt::CBIndex::c_1;
constexpr auto CB_ODD_R     = tt::CBIndex::c_2;
constexpr auto CB_ODD_I     = tt::CBIndex::c_3;
constexpr auto CB_TW_R      = tt::CBIndex::c_4;
constexpr auto CB_TW_I      = tt::CBIndex::c_5;
constexpr auto CB_OUT0_R    = tt::CBIndex::c_6;
constexpr auto CB_OUT0_I    = tt::CBIndex::c_7;
constexpr auto CB_OUT1_R    = tt::CBIndex::c_8;
constexpr auto CB_OUT1_I    = tt::CBIndex::c_9;
constexpr auto CB_TMP_R     = tt::CBIndex::c_10;
constexpr auto CB_TMP_I     = tt::CBIndex::c_11;
constexpr auto CB_TW_ODD_R  = tt::CBIndex::c_12;
constexpr auto CB_TW_ODD_I  = tt::CBIndex::c_13;

constexpr uint32_t LOG2N = get_compile_time_arg_val(0);

enum : uint32_t { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2 };

// Run a binary tile op (add/sub/mul) on CBs a,b -> CB out. Inputs are expected
// to already be waited on; out is pushed back (not popped). Inputs are NOT
// popped here -- caller controls lifetime.
template <uint32_t OP>
FORCE_INLINE void binop_push(uint32_t a, uint32_t b, uint32_t out) {
    if      constexpr (OP == OP_ADD) { add_tiles_init(a, b); }
    else if constexpr (OP == OP_SUB) { sub_tiles_init(a, b); }
    else if constexpr (OP == OP_MUL) { mul_tiles_init(a, b); }

    tile_regs_acquire();
    if      constexpr (OP == OP_ADD) { add_tiles(a, b, 0, 0, 0); }
    else if constexpr (OP == OP_SUB) { sub_tiles(a, b, 0, 0, 0); }
    else if constexpr (OP == OP_MUL) { mul_tiles(a, b, 0, 0, 0); }
    tile_regs_commit();

    cb_reserve_back(out, 1);
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}

// Complex multiply: (ar+ i*ai) * (br + i*bi) -> (outr + i*outi)
//   outr = ar*br - ai*bi
//   outi = ar*bi + ai*br
// Assumes ar/ai/br/bi already waited-on. Does not pop them.
FORCE_INLINE void cmul(
    uint32_t ar, uint32_t ai, uint32_t br, uint32_t bi,
    uint32_t outr, uint32_t outi)
{
    // outr = ar*br - ai*bi
    binop_push<OP_MUL>(ar, br, CB_TMP_R);
    binop_push<OP_MUL>(ai, bi, CB_TMP_I);
    cb_wait_front(CB_TMP_R, 1);
    cb_wait_front(CB_TMP_I, 1);
    binop_push<OP_SUB>(CB_TMP_R, CB_TMP_I, outr);
    cb_pop_front(CB_TMP_R, 1);
    cb_pop_front(CB_TMP_I, 1);

    // outi = ar*bi + ai*br
    binop_push<OP_MUL>(ar, bi, CB_TMP_R);
    binop_push<OP_MUL>(ai, br, CB_TMP_I);
    cb_wait_front(CB_TMP_R, 1);
    cb_wait_front(CB_TMP_I, 1);
    binop_push<OP_ADD>(CB_TMP_R, CB_TMP_I, outi);
    cb_pop_front(CB_TMP_R, 1);
    cb_pop_front(CB_TMP_I, 1);
}

void kernel_main() {
    // binary_op_init_common requires a representative (a, b, out) triple.
    binary_op_init_common(CB_EVEN_R, CB_ODD_R, CB_OUT0_R);

    for (uint32_t s = 0; s < LOG2N; ++s) {
        cb_wait_front(CB_EVEN_R, 1);
        cb_wait_front(CB_EVEN_I, 1);
        cb_wait_front(CB_ODD_R,  1);
        cb_wait_front(CB_ODD_I,  1);
        cb_wait_front(CB_TW_R,   1);
        cb_wait_front(CB_TW_I,   1);

        // W * odd  ->  CB_TW_ODD
        cmul(CB_ODD_R, CB_ODD_I, CB_TW_R, CB_TW_I, CB_TW_ODD_R, CB_TW_ODD_I);

        // After cmul we no longer need odd or twiddle.
        cb_pop_front(CB_ODD_R, 1);
        cb_pop_front(CB_ODD_I, 1);
        cb_pop_front(CB_TW_R,  1);
        cb_pop_front(CB_TW_I,  1);

        cb_wait_front(CB_TW_ODD_R, 1);
        cb_wait_front(CB_TW_ODD_I, 1);

        // out0 = even + W*odd     out1 = even - W*odd
        binop_push<OP_ADD>(CB_EVEN_R, CB_TW_ODD_R, CB_OUT0_R);
        binop_push<OP_ADD>(CB_EVEN_I, CB_TW_ODD_I, CB_OUT0_I);
        binop_push<OP_SUB>(CB_EVEN_R, CB_TW_ODD_R, CB_OUT1_R);
        binop_push<OP_SUB>(CB_EVEN_I, CB_TW_ODD_I, CB_OUT1_I);

        cb_pop_front(CB_EVEN_R,   1);
        cb_pop_front(CB_EVEN_I,   1);
        cb_pop_front(CB_TW_ODD_R, 1);
        cb_pop_front(CB_TW_ODD_I, 1);
    }
}
