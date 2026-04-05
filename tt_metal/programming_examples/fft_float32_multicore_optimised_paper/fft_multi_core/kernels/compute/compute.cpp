// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

// ── Compute kernel fix ────────────────────────────────────────────────────────
//
// ROOT CAUSE OF DEADLOCK:
//   The original maths_sfpu_* helpers called mul_binary_tile_init(),
//   sub_binary_tile_init(), add_binary_tile_init() INSIDE the
//   tile_regs_acquire/commit block.  In TT-Metal 0.68 these init functions
//   reconfigure TRISC1 (MATH) hardware.  Calling them inside the DST lock
//   window causes TRISC1 to spin on a hardware-ready signal while TRISC2
//   (PACK) is already blocked in tile_regs_wait() — a livelock that never
//   resolves.
//
//   Additionally, copy_tile_to_dst_init_short_with_dt was called between
//   copies of same-format (Float32) CBs.  This is unnecessary and can
//   misconfigure the UNPACK engine.
//
// FIX:
//   Use the high-level mul_tiles / sub_tiles / add_tiles API, which:
//     1. Takes *_tiles_init OUTSIDE the tile_regs block (TRISC0 / TRISC1
//        configuration happens before DST lock is acquired).
//     2. Handles UNPACK internally — no explicit copy_tile needed.
//     3. Follows the standard TT-Metal 0.68 eltwise-binary pattern.
//
//   The entire radix-2 butterfly is computed in ONE tile_regs cycle using
//   distinct DST slots (0-7 for intermediates, 8-11 for outputs), eliminating
//   all intermediate CBs (cb_int0, cb_int1, cb_f0, cb_f1).
//
//   DST slot plan (each slot = one 32×32 tile):
//     Slot 0: data1_r * tw_r          (partial f0)
//     Slot 1: data1_i * tw_i          (partial f0)
//     Slot 2: f0 = slot0 - slot1
//     Slot 3: data1_r * tw_i          (partial f1)
//     Slot 4: data1_i * tw_r          (partial f1)
//     Slot 5: f1 = slot3 + slot4
//     Slot 6: data0_r (copy via sub_tiles against zero, or via add)
//     Slot 7: data0_i
//     Slot 8:  out0_r = data0_r + f0
//     Slot 9:  out0_i = data0_i + f1
//     Slot 10: out1_r = data0_r - f0
//     Slot 11: out1_i = data0_i - f1
//
//   Simpler implementation: use separate tile_regs cycles (one per binary op)
//   with inits OUTSIDE the cycle.  This avoids the multi-init-in-one-block
//   problem and keeps the code structure close to the paper.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

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

    // Intermediate CBs — still needed because we use separate tile_regs
    // cycles for each binary op; DST is freed between cycles.
    constexpr uint32_t cb_int0      = tt::CBIndex::c_20;
    constexpr uint32_t cb_int1      = tt::CBIndex::c_21;
    constexpr uint32_t cb_f0        = tt::CBIndex::c_22;
    constexpr uint32_t cb_f1        = tt::CBIndex::c_23;

    for (uint32_t step = 0; step < num_steps; ++step) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {

            // ── Wait for reader-supplied inputs ──────────────────────────────
            cb_wait_front(cb_data1_r,   1);
            cb_wait_front(cb_data1_i,   1);
            cb_wait_front(cb_twiddle_r, 1);
            cb_wait_front(cb_twiddle_i, 1);

            // ── f0 = data1_r * tw_r  −  data1_i * tw_i ──────────────────────

            // int0 = data1_r * tw_r
            mul_tiles_init(cb_data1_r, cb_twiddle_r);   // OUTSIDE tile_regs
            cb_reserve_back(cb_int0, 1);
            tile_regs_acquire();
            mul_tiles(cb_data1_r, cb_twiddle_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_int0);
            tile_regs_release();
            cb_push_back(cb_int0, 1);

            // int1 = data1_i * tw_i
            mul_tiles_init(cb_data1_i, cb_twiddle_i);   // OUTSIDE tile_regs
            cb_reserve_back(cb_int1, 1);
            tile_regs_acquire();
            mul_tiles(cb_data1_i, cb_twiddle_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_int1);
            tile_regs_release();
            cb_push_back(cb_int1, 1);

            // f0 = int0 - int1
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            sub_tiles_init(cb_int0, cb_int1);            // OUTSIDE tile_regs
            cb_reserve_back(cb_f0, 1);
            tile_regs_acquire();
            sub_tiles(cb_int0, cb_int1, 0, 0, 0);
            tile_regs_commit();
            cb_pop_front(cb_int0, 1);
            cb_pop_front(cb_int1, 1);
            tile_regs_wait();
            pack_tile(0, cb_f0);
            tile_regs_release();
            cb_push_back(cb_f0, 1);

            // ── f1 = data1_r * tw_i  +  data1_i * tw_r ──────────────────────

            // int0 = data1_r * tw_i
            mul_tiles_init(cb_data1_r, cb_twiddle_i);
            cb_reserve_back(cb_int0, 1);
            tile_regs_acquire();
            mul_tiles(cb_data1_r, cb_twiddle_i, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_int0);
            tile_regs_release();
            cb_push_back(cb_int0, 1);

            // int1 = data1_i * tw_r
            mul_tiles_init(cb_data1_i, cb_twiddle_r);
            cb_reserve_back(cb_int1, 1);
            tile_regs_acquire();
            mul_tiles(cb_data1_i, cb_twiddle_r, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_int1);
            tile_regs_release();
            cb_push_back(cb_int1, 1);

            // f1 = int0 + int1
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            add_tiles_init(cb_int0, cb_int1);
            cb_reserve_back(cb_f1, 1);
            tile_regs_acquire();
            add_tiles(cb_int0, cb_int1, 0, 0, 0);
            tile_regs_commit();
            cb_pop_front(cb_int0, 1);
            cb_pop_front(cb_int1, 1);
            tile_regs_wait();
            pack_tile(0, cb_f1);
            tile_regs_release();
            cb_push_back(cb_f1, 1);

            // ── Apply butterfly: out0 = data0 + f, out1 = data0 - f ─────────
            cb_wait_front(cb_data0_r, 1);
            cb_wait_front(cb_data0_i, 1);
            cb_wait_front(cb_f0,      1);
            cb_wait_front(cb_f1,      1);

            // out0_r = data0_r + f0
            add_tiles_init(cb_data0_r, cb_f0);
            cb_reserve_back(cb_out0_r, 1);
            tile_regs_acquire();
            add_tiles(cb_data0_r, cb_f0, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0_r);
            tile_regs_release();
            cb_push_back(cb_out0_r, 1);

            // out0_i = data0_i + f1
            add_tiles_init(cb_data0_i, cb_f1);
            cb_reserve_back(cb_out0_i, 1);
            tile_regs_acquire();
            add_tiles(cb_data0_i, cb_f1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0_i);
            tile_regs_release();
            cb_push_back(cb_out0_i, 1);

            // out1_r = data0_r - f0
            sub_tiles_init(cb_data0_r, cb_f0);
            cb_reserve_back(cb_out1_r, 1);
            tile_regs_acquire();
            sub_tiles(cb_data0_r, cb_f0, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out1_r);
            tile_regs_release();
            cb_push_back(cb_out1_r, 1);

            // out1_i = data0_i - f1
            sub_tiles_init(cb_data0_i, cb_f1);
            cb_reserve_back(cb_out1_i, 1);
            tile_regs_acquire();
            sub_tiles(cb_data0_i, cb_f1, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out1_i);
            tile_regs_release();
            cb_push_back(cb_out1_i, 1);

            // ── Pop all inputs ───────────────────────────────────────────────
            cb_pop_front(cb_data0_r,   1);
            cb_pop_front(cb_data0_i,   1);
            cb_pop_front(cb_data1_r,   1);
            cb_pop_front(cb_data1_i,   1);
            cb_pop_front(cb_twiddle_r, 1);
            cb_pop_front(cb_twiddle_i, 1);
            cb_pop_front(cb_f0,        1);
            cb_pop_front(cb_f1,        1);
        }
    }
}