// fft_compute_f32.cpp  — MINIMAL DRY RUN
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  MINIMAL DRY RUN — no FPU, no copy_tile.
//
//  Per stage per tile:
//    1. wait + pop twiddle r/i        (consume, discard)
//    2. wait + pop odd r/i            (consume, discard)
//    3. wait + pop even r/i           (consume, discard)
//    4. push zeros to dst0 r/i        (reserve_back + push_back)
//    5. push zeros to dst1 r/i        (reserve_back + push_back)
//
//  "Push zeros" = reserve a CB slot, do NOT write anything to it
//  (L1 memory is uninitialised but the CB slot exists), push_back.
//  The writer will write whatever garbage is in L1 to DRAM —
//  that is fine, we only care that the kernel COMPLETES.
//
//  If this hangs → CB routing is still wrong.
//  If this completes → CB routing is correct, FPU init was the bug.
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    const uint32_t direction       = get_arg_val<uint32_t>(0);
    const uint32_t num_stages      = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(2);

    constexpr auto cb_in_even_r   = tt::CBIndex::c_0;
    constexpr auto cb_in_even_i   = tt::CBIndex::c_1;
    constexpr auto cb_in_odd_r    = tt::CBIndex::c_2;
    constexpr auto cb_in_odd_i    = tt::CBIndex::c_3;
    constexpr auto cb_tw_r        = tt::CBIndex::c_4;
    constexpr auto cb_tw_i        = tt::CBIndex::c_5;

    constexpr auto cb_ping_even_r = tt::CBIndex::c_10;
    constexpr auto cb_ping_even_i = tt::CBIndex::c_11;
    constexpr auto cb_ping_odd_r  = tt::CBIndex::c_12;
    constexpr auto cb_ping_odd_i  = tt::CBIndex::c_13;

    constexpr auto cb_pong_even_r = tt::CBIndex::c_14;
    constexpr auto cb_pong_even_i = tt::CBIndex::c_15;
    constexpr auto cb_pong_odd_r  = tt::CBIndex::c_6;
    constexpr auto cb_pong_odd_i  = tt::CBIndex::c_7;

    constexpr auto cb_out0_r      = tt::CBIndex::c_16;
    constexpr auto cb_out0_i      = tt::CBIndex::c_17;
    constexpr auto cb_out1_r      = tt::CBIndex::c_18;
    constexpr auto cb_out1_i      = tt::CBIndex::c_19;

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        uint32_t src_even_r, src_even_i;
        uint32_t src_odd_r,  src_odd_i;

        if (stage == 0) {
            src_even_r = cb_in_even_r;   src_even_i = cb_in_even_i;
            src_odd_r  = cb_in_odd_r;    src_odd_i  = cb_in_odd_i;
        } else if ((stage & 1) == 1) {
            src_even_r = cb_ping_even_r; src_even_i = cb_ping_even_i;
            src_odd_r  = cb_ping_odd_r;  src_odd_i  = cb_ping_odd_i;
        } else {
            src_even_r = cb_pong_even_r; src_even_i = cb_pong_even_i;
            src_odd_r  = cb_pong_odd_r;  src_odd_i  = cb_pong_odd_i;
        }

        uint32_t dst0_r, dst0_i;
        uint32_t dst1_r, dst1_i;

        if (stage == num_stages - 1) {
            dst0_r = cb_out0_r;      dst0_i = cb_out0_i;
            dst1_r = cb_out1_r;      dst1_i = cb_out1_i;
        } else if ((stage & 1) == 0) {
            dst0_r = cb_ping_even_r; dst0_i = cb_ping_even_i;
            dst1_r = cb_ping_odd_r;  dst1_i = cb_ping_odd_i;
        } else {
            dst0_r = cb_pong_even_r; dst0_i = cb_pong_even_i;
            dst1_r = cb_pong_odd_r;  dst1_i = cb_pong_odd_i;
        }

        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // Consume all inputs
            cb_wait_front(cb_tw_r,    1); cb_pop_front(cb_tw_r,    1);
            cb_wait_front(cb_tw_i,    1); cb_pop_front(cb_tw_i,    1);
            cb_wait_front(src_odd_r,  1); cb_pop_front(src_odd_r,  1);
            cb_wait_front(src_odd_i,  1); cb_pop_front(src_odd_i,  1);
            cb_wait_front(src_even_r, 1); cb_pop_front(src_even_r, 1);
            cb_wait_front(src_even_i, 1); cb_pop_front(src_even_i, 1);

            // Push uninitialised slots to outputs (no FPU needed)
            cb_reserve_back(dst0_r, 1); cb_push_back(dst0_r, 1);
            cb_reserve_back(dst0_i, 1); cb_push_back(dst0_i, 1);
            cb_reserve_back(dst1_r, 1); cb_push_back(dst1_r, 1);
            cb_reserve_back(dst1_i, 1); cb_push_back(dst1_i, 1);
        }
    }
}