// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// complex_mul_compute.cpp — TRISC compute for the elementwise complex
// multiply used by the Bluestein chirp pre/post multiplies.
//
// For each tile pair (A, B) waited on the input CBs, this kernel computes
//     OUT_R = A_R * B_R - A_I * B_I
//     OUT_I = A_R * B_I + A_I * B_R
// using the SFPU binary-op path (full IEEE fp32, "precise" mode).
//
// Structurally identical to pass2_compute.cpp; the CB layout matches
// `complex_mul_common.h`. We keep this as a separate kernel rather than
// reusing pass2 because the pass2 host plan is hard-coded to Stockham's
// (N1, N2) geometry, which is the wrong shape for arbitrary-length
// Bluestein chirp tables.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"

constexpr auto CB_A_R   = tt::CBIndex::c_0;
constexpr auto CB_A_I   = tt::CBIndex::c_1;
constexpr auto CB_B_R   = tt::CBIndex::c_2;
constexpr auto CB_B_I   = tt::CBIndex::c_3;
constexpr auto CB_OUT_R = tt::CBIndex::c_4;
constexpr auto CB_OUT_I = tt::CBIndex::c_5;
constexpr auto CB_TMP_R = tt::CBIndex::c_6;
constexpr auto CB_TMP_I = tt::CBIndex::c_7;

enum : uint32_t { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2 };

template <uint32_t OP>
FORCE_INLINE void sfpu_binop_push(uint32_t a, uint32_t b, uint32_t out) {
    tile_regs_acquire();

    copy_tile_to_dst_init_short(a);
    copy_tile(a, 0, 0);
    copy_tile_to_dst_init_short_with_dt(a, b);
    copy_tile(b, 0, 1);

    if      constexpr (OP == OP_ADD) { add_binary_tile_init(); add_binary_tile(0, 1, 0); }
    else if constexpr (OP == OP_SUB) { sub_binary_tile_init(); sub_binary_tile(0, 1, 0); }
    else if constexpr (OP == OP_MUL) { mul_binary_tile_init(); mul_binary_tile(0, 1, 0); }

    tile_regs_commit();

    cb_reserve_back(out, 1);
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    unary_op_init_common(CB_A_R, CB_OUT_R);
    copy_tile_to_dst_init_short(CB_A_R);

    for (uint32_t k = 0; k < num_tiles; ++k) {
        cb_wait_front(CB_A_R, 1);
        cb_wait_front(CB_A_I, 1);
        cb_wait_front(CB_B_R, 1);
        cb_wait_front(CB_B_I, 1);

        // OUT_R = A_R * B_R - A_I * B_I
        sfpu_binop_push<OP_MUL>(CB_A_R, CB_B_R, CB_TMP_R);
        sfpu_binop_push<OP_MUL>(CB_A_I, CB_B_I, CB_TMP_I);
        cb_wait_front(CB_TMP_R, 1);
        cb_wait_front(CB_TMP_I, 1);
        sfpu_binop_push<OP_SUB>(CB_TMP_R, CB_TMP_I, CB_OUT_R);
        cb_pop_front(CB_TMP_R, 1);
        cb_pop_front(CB_TMP_I, 1);

        // OUT_I = A_R * B_I + A_I * B_R
        sfpu_binop_push<OP_MUL>(CB_A_R, CB_B_I, CB_TMP_R);
        sfpu_binop_push<OP_MUL>(CB_A_I, CB_B_R, CB_TMP_I);
        cb_wait_front(CB_TMP_R, 1);
        cb_wait_front(CB_TMP_I, 1);
        sfpu_binop_push<OP_ADD>(CB_TMP_R, CB_TMP_I, CB_OUT_I);
        cb_pop_front(CB_TMP_R, 1);
        cb_pop_front(CB_TMP_I, 1);

        cb_pop_front(CB_A_R, 1);
        cb_pop_front(CB_A_I, 1);
        cb_pop_front(CB_B_R, 1);
        cb_pop_front(CB_B_I, 1);
    }
}
