// fft_compute_f32.cpp  — CORRECTED MULTI-STAGE FFT
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Architecture: pre-staged inputs.
// The host pre-computes the correct even/odd split for EVERY stage and
// uploads them all to DRAM before launch.  The reader feeds each stage's
// even/odd tiles through CBs 0-3, the compute kernel performs the butterfly,
// and the writer drains the final stage's output from CBs 16-19.
// No intermediate ping-pong CBs are needed — the host already knows the
// correct grouping for every stage.
//
// CB map:
//   0  cb_even_r   current stage even real  (reader writes, compute reads)
//   1  cb_even_i   current stage even imag
//   2  cb_odd_r    current stage odd  real
//   3  cb_odd_i    current stage odd  imag
//   4  cb_tw_r     twiddle real  (reader writes, compute reads)
//   5  cb_tw_i     twiddle imag
//  16  cb_out_r    final output real  lower half  (compute writes, writer reads)
//  17  cb_out_i    final output imag  lower half
//  18  cb_out_r2   final output real  upper half
//  19  cb_out_i2   final output imag  upper half
//  20  cb_tmp0     scratch
//  21  cb_tmp1     scratch
//  22  cb_tw_odd_r scratch t_r = tw_r*odd_r - tw_i*odd_i
//  23  cb_tw_odd_i scratch t_i = tw_r*odd_i + tw_i*odd_r

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_even_r    = 0;
    constexpr uint32_t cb_even_i    = 1;
    constexpr uint32_t cb_odd_r     = 2;
    constexpr uint32_t cb_odd_i     = 3;
    constexpr uint32_t cb_tw_r      = 4;
    constexpr uint32_t cb_tw_i      = 5;
    constexpr uint32_t cb_out_r     = 16;
    constexpr uint32_t cb_out_i     = 17;
    constexpr uint32_t cb_out_r2    = 18;
    constexpr uint32_t cb_out_i2    = 19;
    constexpr uint32_t cb_tmp0      = 20;
    constexpr uint32_t cb_tmp1      = 21;
    constexpr uint32_t cb_tw_odd_r  = 22;
    constexpr uint32_t cb_tw_odd_i  = 23;

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        bool is_last = (stage == num_stages - 1);

        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // ── t_r = tw_r * odd_r  -  tw_i * odd_i ──────────────

            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, cb_odd_i, cb_tmp1);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_r);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tw_odd_r);
            tile_regs_release();
            cb_push_back(cb_tw_odd_r, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            // ── t_i = tw_r * odd_i  +  tw_i * odd_r ──────────────

            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, cb_odd_r, cb_tmp1);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_i);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tw_odd_i);
            tile_regs_release();
            cb_push_back(cb_tw_odd_i, 1);
            cb_pop_front(cb_tmp0, 1);
            cb_pop_front(cb_tmp1, 1);

            cb_wait_front(cb_tw_odd_r, 1);
            cb_wait_front(cb_tw_odd_i, 1);

            // ── out0_r = even_r + t_r ─────────────────────────────
            // For intermediate stages, out0 feeds the next stage's even.
            // For last stage, out0 goes to output CBs.
            // In both cases we write to cb_out_r/cb_out_i and
            // the reader/writer handle routing — compute always writes to
            // the same CBs, keeping the kernel stateless across stages.

            cb_reserve_back(cb_out_r, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_r, cb_tw_odd_r, cb_out_r);
            add_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out_r);
            tile_regs_release();
            cb_push_back(cb_out_r, 1);

            // ── out0_i = even_i + t_i ─────────────────────────────
            cb_reserve_back(cb_out_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_i, cb_tw_odd_i, cb_out_i);
            add_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out_i);
            tile_regs_release();
            cb_push_back(cb_out_i, 1);

            // ── out1_r = even_r - t_r ─────────────────────────────
            cb_reserve_back(cb_out_r2, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_r, cb_tw_odd_r, cb_out_r2);
            sub_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out_r2);
            tile_regs_release();
            cb_push_back(cb_out_r2, 1);

            // ── out1_i = even_i - t_i ─────────────────────────────
            cb_reserve_back(cb_out_i2, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_i, cb_tw_odd_i, cb_out_i2);
            sub_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out_i2);
            tile_regs_release();
            cb_push_back(cb_out_i2, 1);

            // ── Pop all consumed inputs ────────────────────────────
            cb_pop_front(cb_tw_r,    1);
            cb_pop_front(cb_tw_i,    1);
            cb_pop_front(cb_odd_r,   1);
            cb_pop_front(cb_odd_i,   1);
            cb_pop_front(cb_even_r,  1);
            cb_pop_front(cb_even_i,  1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);
        }
    }
}