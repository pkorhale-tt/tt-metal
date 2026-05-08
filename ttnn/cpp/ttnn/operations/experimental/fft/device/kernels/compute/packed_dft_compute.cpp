// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// packed_dft_compute.cpp — TRISC compute for the PACKED DIRECT-DFT kernel.
//
// For each input tile `t` the reader pushes 4 (A, B) tile pairs into
// (CB_A, CB_B). We run 4 matmul_tiles calls whose accumulating semantics
// (DST += A · B) give us both complex-output tiles directly, no SFPU subtract
// required:
//
//   tile_regs_acquire()                     // DST(0) = 0
//     matmul(A=in_R, B=T_R)                 // DST(0) += in_R · T_R
//     matmul(A=in_I, B=T_I_neg)             // DST(0) += in_I · (-T_I)
//   pack_tile(0, CB_OUT_R)                  // out_R = in_R·T_R − in_I·T_I
//   tile_regs_release()
//
//   tile_regs_acquire()
//     matmul(A=in_R, B=T_I)                 // DST(0) += in_R · T_I
//     matmul(A=in_I, B=T_R)                 // DST(0) += in_I · T_R
//   pack_tile(0, CB_OUT_I)                  // out_I = in_R·T_I + in_I·T_R
//   tile_regs_release()
//
// pack_tile(dst_idx, cb_id) is legal for any CB of the same data format as
// the one mm_init was initialised with; we exploit that to pack the two
// halves of the complex result into CB_OUT_R and CB_OUT_I respectively.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "packed_dft_common.h"

void kernel_main() {
    const uint32_t tiles_per_core = get_compile_time_arg_val(0);

    // One full config at the start; the FPU stays in matmul mode the whole
    // run, and we only swap which tiles live in CB_A / CB_B (the reader's
    // job). Using CB_OUT_R as the nominal out CB here — pack_tile below
    // overrides the target CB per call, which is valid as long as all three
    // CBs share one data format.
    mm_init(CB_A, CB_B, CB_OUT_R);

    for (uint32_t k = 0; k < tiles_per_core; ++k) {
        // ── out_R = in_R · T_R  +  in_I · T_I_neg ────────────────────────
        tile_regs_acquire();

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_OUT_R, 1);
        pack_tile(0, CB_OUT_R);
        cb_push_back(CB_OUT_R, 1);
        tile_regs_release();

        // ── out_I = in_R · T_I  +  in_I · T_R ────────────────────────────
        tile_regs_acquire();

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);
        matmul_tiles(CB_A, CB_B, 0, 0, 0);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_OUT_I, 1);
        pack_tile(0, CB_OUT_I);
        cb_push_back(CB_OUT_I, 1);
        tile_regs_release();
    }
}
