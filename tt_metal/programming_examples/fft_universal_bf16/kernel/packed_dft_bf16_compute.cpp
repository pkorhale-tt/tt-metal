// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// packed_dft_bf16_compute.cpp — TRISC compute for the TRUE-bf16 packed
// direct-DFT kernel.
//
// The compute sequence is identical in shape to the fp32 variant
// (../fft_universal/kernel/packed_dft_compute.cpp) — four accumulating
// matmul_tiles per output tile, two pack_tile into (CB_OUT_R, CB_OUT_I).
// What differs is the *data format on the FPU operand path*:
//
//   * CB_A / CB_B are Float16_b (bf16, set host-side). The unpacker feeds
//     srcA / srcB in bf16 — native FPU format for bf16 × bf16 multiplies.
//   * fp32_dest_acc_en=true → DST is fp32, so the running
//     reduction-sum across the two accumulating matmul calls keeps fp32
//     precision before we pack back to bf16.
//   * No unpack_to_dest_mode override (that would bypass srcA/srcB and
//     break matmul — same trap the fp32 version documents).
//
// pack_tile(dst_idx, cb_id) honours each CB's own data format, so DST
// (fp32) → CB_OUT_R / CB_OUT_I (bf16) conversion happens automatically
// during the pack. That single conversion is the *only* bf16 rounding
// per output tile — internal matmul accumulation stays fp32.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "packed_dft_bf16_common.h"

void kernel_main() {
    const uint32_t tiles_per_core = get_compile_time_arg_val(0);

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
