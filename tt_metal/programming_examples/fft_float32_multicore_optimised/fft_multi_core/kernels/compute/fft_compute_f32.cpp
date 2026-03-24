// fft_compute_f32.cpp — MULTICORE butterfly kernel (FINAL, deadlock-free)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  DEADLOCK FIX (vs previous version)
// ══════════════════════════════════════════════════════════════════════
//
//  Root cause: tile_regs_acquire() was called before cb_wait_front() for
//  input CBs, or cb_reserve_back was called before inputs were confirmed.
//  This caused circular blocking between the compute and reader/writer kernels.
//
//  INVARIANT enforced throughout:
//    cb_wait_front(all inputs) THEN tile_regs_acquire()
//    Never block inside a tile_regs session.
//
// ══════════════════════════════════════════════════════════════════════
//  BUTTERFLY — 4 sessions, each CB pair read exactly once
// ══════════════════════════════════════════════════════════════════════
//
//  Session A: 4× mul → pack 4 partials to tmp0[0,1] and tmp1[0,1]
//             Consumes: cb_tw_r, cb_tw_i, cb_odd_r, cb_odd_i
//             Produces: tmp0 (depth 2), tmp1 (depth 2)
//
//  Session B: sub(tmp0[0], tmp1[0]) → t_r packed to tmp0 (1 tile)
//             Consumes: first tile of tmp0 and tmp1
//
//  Session C: add(tmp0[1→0], tmp1[1→0]) → t_i packed to tmp1 (1 tile)
//             After session B pop, tmp0[1] is now at front (index 0).
//             Same for tmp1[1]. So we use tile index 0.
//             Consumes: second (now front) tile of tmp0 and tmp1
//             Produces: tmp1 with t_i (1 tile)
//             (tmp0 already holds t_r from session B)
//
//  Session D: 4× add/sub (even±t) → pack 4 outputs
//             Consumes: cb_even_r, cb_even_i, tmp0 (t_r), tmp1 (t_i)
//             Produces: cb_out0_r/i, cb_out1_r/i
//
//  CB depth requirements:
//    cb_tmp0: depth ≥ 2 (holds 2 tiles after session A)
//    cb_tmp1: depth ≥ 2
//    All others: depth = tiles_per_stage (1 for N=1024)
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_tmp0   = 20;   // depth ≥ 2: holds [tw_r*odd_r, tw_r*odd_i]
    constexpr uint32_t cb_tmp1   = 21;   // depth ≥ 2: holds [tw_i*odd_i, tw_i*odd_r]

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        // Optimization: init once per stage (sticky FPU config).
        mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);

        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // ── SESSION A: complex multiply W * odd ───────────────────────
            //
            // All cb_wait_front calls BEFORE tile_regs_acquire — no deadlock.
            // even_r/i are waited here too so reader fully drains before we
            // occupy any shared compute resources.

            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            cb_reserve_back(cb_tmp0, 2);   // space for tw_r*odd_r and tw_r*odd_i
            cb_reserve_back(cb_tmp1, 2);   // space for tw_i*odd_i and tw_i*odd_r

            tile_regs_acquire();

            mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);   // slot 0 = tw_r * odd_r

            mul_tiles_init(cb_tw_i, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);   // slot 1 = tw_i * odd_i

            mul_tiles_init(cb_tw_r, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 2);   // slot 2 = tw_r * odd_i

            mul_tiles_init(cb_tw_i, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 3);   // slot 3 = tw_i * odd_r

            tile_regs_commit();
            tile_regs_wait();

            // Pack all 4 partial products before release.
            // tmp0 gets [tw_r*odd_r (slot 0), tw_r*odd_i (slot 2)]
            // tmp1 gets [tw_i*odd_i (slot 1), tw_i*odd_r (slot 3)]
            pack_tile(0, cb_tmp0);
            pack_tile(2, cb_tmp0);
            pack_tile(1, cb_tmp1);
            pack_tile(3, cb_tmp1);

            tile_regs_release();

            cb_push_back(cb_tmp0, 2);
            cb_push_back(cb_tmp1, 2);

            // Pop all tw/odd inputs — consumed, never read again.
            cb_pop_front(cb_tw_r,  1);
            cb_pop_front(cb_tw_i,  1);
            cb_pop_front(cb_odd_r, 1);
            cb_pop_front(cb_odd_i, 1);

            // ── SESSION B: t_r = tmp0[0] - tmp1[0] ───────────────────────
            //
            // tmp0 front = tw_r*odd_r, tmp1 front = tw_i*odd_i.
            // We subtract them to get t_r, pack the result back to tmp0.
            // Then pop the consumed tiles and push the result.

            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);

            // Pack result into a fresh tmp0 slot.
            // We pop old front AFTER packing (pack reads registers, not CB).
            cb_reserve_back(cb_tmp0, 1);

            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1, cb_tmp0);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);   // slot 0 = t_r
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp0);
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);

            // Pop exactly the one tile we just consumed from each CB.
            cb_pop_front(cb_tmp0, 1);   // tw_r*odd_r consumed
            cb_pop_front(cb_tmp1, 1);   // tw_i*odd_i consumed
            // tmp0 now has: [t_r]
            // tmp1 now has: [tw_i*odd_r]  (the second tile from session A)

            // ── SESSION C: t_i = tmp0_remaining[0] + tmp1_remaining[0] ───
            //
            // After session B pops, the remaining tiles in each CB slide to front:
            //   tmp0 front = tw_r*odd_i  (was index 1 in session A)
            //   tmp1 front = tw_i*odd_r  (was index 1 in session A)
            // Plus tmp0 also has t_r (just pushed), which is behind tw_r*odd_i.
            //
            // Wait: actually after session B's reserve+push+pop sequence:
            //   tmp0: [tw_r*odd_i, t_r]  — tw_r*odd_i is at front, t_r behind it
            //   Wait, no. CB is FIFO. Session A pushed [tw_r*odd_r, tw_r*odd_i].
            //   Session B popped tw_r*odd_r (front), then pushed t_r.
            //   So tmp0 is now: [tw_r*odd_i, t_r]  (tw_r*odd_i at front)
            //   tmp1: session A pushed [tw_i*odd_i, tw_i*odd_r].
            //   Session B popped tw_i*odd_i (front).
            //   So tmp1 is now: [tw_i*odd_r]
            //
            // For t_i = tw_r*odd_i + tw_i*odd_r:
            //   source A = tmp0 front = tw_r*odd_i ✓
            //   source B = tmp1 front = tw_i*odd_r ✓
            // After this session we pop both, leaving tmp0 = [t_r], tmp1 = [] then push t_i.

            cb_wait_front(cb_tmp0, 1);   // tw_r*odd_i
            cb_wait_front(cb_tmp1, 1);   // tw_i*odd_r

            cb_reserve_back(cb_tmp1, 1);

            tile_regs_acquire();
            add_tiles_init(cb_tmp0, cb_tmp1, cb_tmp1);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);   // slot 0 = t_i
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();

            cb_push_back(cb_tmp1, 1);

            cb_pop_front(cb_tmp0, 1);   // tw_r*odd_i consumed
            cb_pop_front(cb_tmp1, 1);   // tw_i*odd_r consumed
            // tmp0 now has: [t_r]
            // tmp1 now has: [t_i]

            // ── SESSION D: out0 = even+t,  out1 = even-t ─────────────────
            //
            // even_r/i were waited in session A and not yet popped.
            // tmp0 front = t_r, tmp1 front = t_i.

            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            // even_r, even_i already at front from session A wait.

            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);

            tile_regs_acquire();

            add_tiles_init(cb_even_r, cb_tmp0, cb_out0_r);
            add_tiles(cb_even_r, cb_tmp0, 0, 0, 0);   // slot 0 = even_r + t_r

            add_tiles_init(cb_even_i, cb_tmp1, cb_out0_i);
            add_tiles(cb_even_i, cb_tmp1, 0, 0, 1);   // slot 1 = even_i + t_i

            sub_tiles_init(cb_even_r, cb_tmp0, cb_out1_r);
            sub_tiles(cb_even_r, cb_tmp0, 0, 0, 2);   // slot 2 = even_r - t_r

            sub_tiles_init(cb_even_i, cb_tmp1, cb_out1_i);
            sub_tiles(cb_even_i, cb_tmp1, 0, 0, 3);   // slot 3 = even_i - t_i

            tile_regs_commit();
            tile_regs_wait();

            // Pack all 4 outputs while register file is live.
            pack_tile(0, cb_out0_r);
            pack_tile(1, cb_out0_i);
            pack_tile(2, cb_out1_r);
            pack_tile(3, cb_out1_i);

            tile_regs_release();

            cb_push_back(cb_out0_r, 1);
            cb_push_back(cb_out0_i, 1);
            cb_push_back(cb_out1_r, 1);
            cb_push_back(cb_out1_i, 1);

            cb_pop_front(cb_even_r, 1);
            cb_pop_front(cb_even_i, 1);
            cb_pop_front(cb_tmp0,   1);   // t_r
            cb_pop_front(cb_tmp1,   1);   // t_i
        }
    }
}