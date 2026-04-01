// fft_compute_f32.cpp  — MULTICORE: per-core butterfly kernel
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Each core handles a contiguous slice of N/2 butterfly pairs.
// core_id ∈ [0, num_cores), each owns (half_N / num_cores) pairs per stage.
//
// CB map (identical to single-core version — each core has its own CBs):
//   0  cb_even_r    stage input even real
//   1  cb_even_i    stage input even imag
//   2  cb_odd_r     stage input odd  real
//   3  cb_odd_i     stage input odd  imag
//   4  cb_tw_r      expanded twiddle real  (reader fills per stage)
//   5  cb_tw_i      expanded twiddle imag
//  16  cb_out0_r    butterfly sum real
//  17  cb_out0_i    butterfly sum imag
//  18  cb_out1_r    butterfly diff real
//  19  cb_out1_i    butterfly diff imag
//  20  cb_tmp0      scratch
//  21  cb_tmp1      scratch
//  22  cb_tw_odd_r  W*odd real
//  23  cb_tw_odd_i  W*odd imag
//
// No changes from original — compute kernel was correct.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

void kernel_main() {
    // arg 0: total FFT stages (log2N)
    // arg 1: tiles this core handles per stage
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_even_r   = 0;
    constexpr uint32_t cb_even_i   = 1;
    constexpr uint32_t cb_odd_r    = 2;
    constexpr uint32_t cb_odd_i    = 3;
    constexpr uint32_t cb_tw_r     = 4;
    constexpr uint32_t cb_tw_i     = 5;
    constexpr uint32_t cb_out0_r   = 16;
    constexpr uint32_t cb_out0_i   = 17;
    constexpr uint32_t cb_out1_r   = 18;
    constexpr uint32_t cb_out1_i   = 19;
    constexpr uint32_t cb_tmp0     = 20;
    constexpr uint32_t cb_tmp1     = 21;
    constexpr uint32_t cb_tw_odd_r = 22;
    constexpr uint32_t cb_tw_odd_i = 23;

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // ── t_r = tw_r*odd_r − tw_i*odd_i ───────────────────────
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0); tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, cb_odd_i, cb_tmp1);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1); tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            cb_wait_front(cb_tmp0, 1); cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_r);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tw_odd_r); tile_regs_release();
            cb_push_back(cb_tw_odd_r, 1);
            cb_pop_front(cb_tmp0, 1); cb_pop_front(cb_tmp1, 1);

            // ── t_i = tw_r*odd_i + tw_i*odd_r ───────────────────────
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0); tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, cb_odd_r, cb_tmp1);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1); tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            cb_wait_front(cb_tmp0, 1); cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_i);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tw_odd_i); tile_regs_release();
            cb_push_back(cb_tw_odd_i, 1);
            cb_pop_front(cb_tmp0, 1); cb_pop_front(cb_tmp1, 1);

            cb_wait_front(cb_tw_odd_r, 1); cb_wait_front(cb_tw_odd_i, 1);

            // ── out0 = even + t ──────────────────────────────────────
            cb_reserve_back(cb_out0_r, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_r, cb_tw_odd_r, cb_out0_r);
            add_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out0_r); tile_regs_release();
            cb_push_back(cb_out0_r, 1);

            cb_reserve_back(cb_out0_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_i, cb_tw_odd_i, cb_out0_i);
            add_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out0_i); tile_regs_release();
            cb_push_back(cb_out0_i, 1);

            // ── out1 = even − t ──────────────────────────────────────
            cb_reserve_back(cb_out1_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_r, cb_tw_odd_r, cb_out1_r);
            sub_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out1_r); tile_regs_release();
            cb_push_back(cb_out1_r, 1);

            cb_reserve_back(cb_out1_i, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_i, cb_tw_odd_i, cb_out1_i);
            sub_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out1_i); tile_regs_release();
            cb_push_back(cb_out1_i, 1);

            // ── Pop consumed inputs ──────────────────────────────────
            cb_pop_front(cb_tw_r,     1);
            cb_pop_front(cb_tw_i,     1);
            cb_pop_front(cb_odd_r,    1);
            cb_pop_front(cb_odd_i,    1);
            cb_pop_front(cb_even_r,   1);
            cb_pop_front(cb_even_i,   1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}