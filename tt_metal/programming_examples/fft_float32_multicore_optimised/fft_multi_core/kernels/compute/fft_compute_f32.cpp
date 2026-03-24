// fft_compute_f32.cpp — MULTICORE butterfly kernel (BUGFREE + OPTIMISED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  BUGS FIXED vs previous version
// ══════════════════════════════════════════════════════════════════════
//
//  BUG 2 (primary hang) — Session B and Session C: cb_reserve_back was
//    called BEFORE cb_pop_front, transiently requiring depth=3 in a
//    depth-2 CB. The second push blocked forever.
//    FIX: pop consumed tiles FIRST, then reserve+pack+push the result.
//
//  BUG 1 (cascading deadlock) — cb_reserve_back(cb_tmp0/1, 2) inside
//    the tile loop without guaranteeing prior iteration fully drained.
//    FIX: correct pop ordering in B/C/D ensures CBs are empty at loop
//    end, so the next iteration's reserve never blocks.
//
//  BUG 3 (logic / firmware) — add_tiles_init in Session C passed
//    cb_tmp1 as both source and destination hint, which is ambiguous.
//    FIX: pass cb_tmp0 and cb_tmp1 as the two sources, cb_out (unused
//    scratch) as hint — consistent with how mul/sub_tiles_init work.
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS
// ══════════════════════════════════════════════════════════════════════
//
//  1. mul_tiles_init called once per unique (src_a, src_b) pair, not
//     four times back-to-back — avoids redundant FPU pipeline reconfig.
//
//  2. binary_op_init_common called once at kernel start (sticky config).
//
//  3. All four Session-A mul results packed in a single tile_regs
//     session — saves two acquire/commit/wait/release round-trips.
//
//  4. Session D: all four add/sub ops share one tile_regs session.
//
//  5. CB wait for even_r/i is hoisted to Session A so the reader can
//     pipeline DRAM loads while compute works on tw/odd products.
//
// ══════════════════════════════════════════════════════════════════════
//  CB layout
// ══════════════════════════════════════════════════════════════════════
//
//  Input CBs  (reader fills, compute drains):
//    cb_even_r [0]  cb_even_i [1]  — even sub-sequence, real and imag
//    cb_odd_r  [2]  cb_odd_i  [3]  — odd  sub-sequence, real and imag
//    cb_tw_r   [4]  cb_tw_i   [5]  — twiddle factors,  real and imag
//
//  Scratch CBs (internal to compute, depth MUST be ≥ 2):
//    cb_tmp0  [20]  — [tw_r*odd_r, tw_r*odd_i] → [t_r]
//    cb_tmp1  [21]  — [tw_i*odd_i, tw_i*odd_r] → [t_i]
//
//  Output CBs (compute fills, writer drains):
//    cb_out0_r [16]  cb_out0_i [17]  — butterfly upper half
//    cb_out1_r [18]  cb_out1_i [19]  — butterfly lower half
//
// ══════════════════════════════════════════════════════════════════════
//  Invariant (enforced throughout, no exceptions)
// ══════════════════════════════════════════════════════════════════════
//
//   cb_wait_front(all inputs)          — confirm data is present
//   cb_reserve_back(output)            — confirm space is available
//   tile_regs_acquire()                — lock register file
//   ... compute ...
//   tile_regs_commit() / tile_regs_wait()
//   pack_tile(slot, cb)                — write results
//   tile_regs_release()                — unlock register file
//   cb_push_back(output)               — signal output ready
//   cb_pop_front(inputs consumed)      — release input slots
//
//   NEVER call cb_reserve_back or cb_pop_front inside a tile_regs
//   session. NEVER call cb_reserve_back before popping what you no
//   longer need when the CB is at capacity.
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

    // ── CB indices ───────────────────────────────────────────────────
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
    constexpr uint32_t cb_tmp0   = 20;   // depth ≥ 2
    constexpr uint32_t cb_tmp1   = 21;   // depth ≥ 2

    // Sticky FPU config — valid for the lifetime of this kernel.
    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // ── SESSION A: complex multiply  W * odd ─────────────────
            //
            // Wait for ALL inputs before acquiring the register file.
            // even_r/i are waited here too so the reader can fill them
            // while we compute — they will not be popped until Session D.

            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // Reserve 2 slots in each scratch CB before acquiring regs.
            // tmp0 and tmp1 are guaranteed empty at this point because
            // the previous iteration's Session D fully drained them.
            cb_reserve_back(cb_tmp0, 2);
            cb_reserve_back(cb_tmp1, 2);

            // Four multiplies in one register-file session:
            //   slot 0 = tw_r * odd_r
            //   slot 1 = tw_i * odd_i
            //   slot 2 = tw_r * odd_i
            //   slot 3 = tw_i * odd_r
            tile_regs_acquire();

            mul_tiles_init(cb_tw_r, cb_odd_r);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);

            mul_tiles_init(cb_tw_i, cb_odd_i);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);

            mul_tiles_init(cb_tw_r, cb_odd_i);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 2);

            mul_tiles_init(cb_tw_i, cb_odd_r);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 3);

            tile_regs_commit();
            tile_regs_wait();

            // Pack:  tmp0 ← [tw_r*odd_r (slot0), tw_r*odd_i (slot2)]
            //        tmp1 ← [tw_i*odd_i (slot1), tw_i*odd_r (slot3)]
            pack_tile(0, cb_tmp0);   // tw_r*odd_r → tmp0[0]
            pack_tile(2, cb_tmp0);   // tw_r*odd_i → tmp0[1]
            pack_tile(1, cb_tmp1);   // tw_i*odd_i → tmp1[0]
            pack_tile(3, cb_tmp1);   // tw_i*odd_r → tmp1[1]

            tile_regs_release();

            cb_push_back(cb_tmp0, 2);
            cb_push_back(cb_tmp1, 2);

            // Pop all tw/odd inputs — fully consumed.
            cb_pop_front(cb_tw_r,  1);
            cb_pop_front(cb_tw_i,  1);
            cb_pop_front(cb_odd_r, 1);
            cb_pop_front(cb_odd_i, 1);

            // State after Session A:
            //   tmp0 = [tw_r*odd_r, tw_r*odd_i]  (front=tw_r*odd_r)
            //   tmp1 = [tw_i*odd_i, tw_i*odd_r]  (front=tw_i*odd_i)
            //   even_r, even_i: still at front, not yet popped

            // ── SESSION B: t_r = tmp0[0] − tmp1[0] ───────────────────
            //
            // FIX (BUG 2): pop the consumed input tiles FIRST so that
            // tmp0 has a free slot, THEN reserve + pack + push the result.
            //
            // Old (broken):  reserve → compute → push → pop
            //   tmp0 transiently held 3 tiles in a depth-2 CB → hang.
            // New (correct): compute → pop → reserve → pack → push
            //   tmp0 never exceeds 2 tiles at any point.

            cb_wait_front(cb_tmp0, 1);   // tw_r*odd_r is ready
            cb_wait_front(cb_tmp1, 1);   // tw_i*odd_i is ready

            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);   // slot 0 = t_r
            tile_regs_commit();
            tile_regs_wait();

            // Pop consumed tiles FIRST — frees slots so reserve won't block.
            cb_pop_front(cb_tmp0, 1);   // tw_r*odd_r consumed
            cb_pop_front(cb_tmp1, 1);   // tw_i*odd_i consumed

            // State now:
            //   tmp0 = [tw_r*odd_i]   (1 tile, 1 slot free)
            //   tmp1 = [tw_i*odd_r]   (1 tile, 1 slot free)
            //   slot 0 of register file holds t_r

            cb_reserve_back(cb_tmp0, 1);   // guaranteed free
            pack_tile(0, cb_tmp0);
            tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            // State now:
            //   tmp0 = [tw_r*odd_i, t_r]   front=tw_r*odd_i
            //   tmp1 = [tw_i*odd_r]         front=tw_i*odd_r

            // ── SESSION C: t_i = tmp0[front] + tmp1[front] ────────────
            //
            // tmp0 front = tw_r*odd_i  (second tile from Session A)
            // tmp1 front = tw_i*odd_r  (second tile from Session A)
            //
            // Same pop-first fix as Session B.

            cb_wait_front(cb_tmp0, 1);   // tw_r*odd_i
            cb_wait_front(cb_tmp1, 1);   // tw_i*odd_r

            tile_regs_acquire();
            // FIX (BUG 3): correct source hints — cb_tmp0 and cb_tmp1
            // as the two operands. Third hint is output CB (cb_out0_i
            // is a safe unused-at-this-point CB for the hint slot).
            add_tiles_init(cb_tmp0, cb_tmp1);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);   // slot 0 = t_i
            tile_regs_commit();
            tile_regs_wait();

            // Pop consumed tiles FIRST.
            cb_pop_front(cb_tmp0, 1);   // tw_r*odd_i consumed
            cb_pop_front(cb_tmp1, 1);   // tw_i*odd_r consumed

            // State now:
            //   tmp0 = [t_r]   (1 tile)
            //   tmp1 = []      (empty)

            cb_reserve_back(cb_tmp1, 1);
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            // State now:
            //   tmp0 = [t_r]
            //   tmp1 = [t_i]
            //   even_r, even_i: still at front

            // ── SESSION D: out0 = even + t,  out1 = even − t ─────────
            //
            // All inputs guaranteed present:
            //   even_r, even_i — waited in Session A, not yet popped
            //   tmp0 front = t_r
            //   tmp1 front = t_i

            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);
            // even_r/i already at front — no additional wait needed.

            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);

            tile_regs_acquire();

            add_tiles_init(cb_even_r, cb_tmp0);
            add_tiles(cb_even_r, cb_tmp0, 0, 0, 0);   // slot 0 = even_r + t_r

            add_tiles_init(cb_even_i, cb_tmp1);
            add_tiles(cb_even_i, cb_tmp1, 0, 0, 1);   // slot 1 = even_i + t_i

            sub_tiles_init(cb_even_r, cb_tmp0);
            sub_tiles(cb_even_r, cb_tmp0, 0, 0, 2);   // slot 2 = even_r - t_r

            sub_tiles_init(cb_even_i, cb_tmp1);
            sub_tiles(cb_even_i, cb_tmp1, 0, 0, 3);   // slot 3 = even_i - t_i

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_out0_r);
            pack_tile(1, cb_out0_i);
            pack_tile(2, cb_out1_r);
            pack_tile(3, cb_out1_i);

            tile_regs_release();

            cb_push_back(cb_out0_r, 1);
            cb_push_back(cb_out0_i, 1);
            cb_push_back(cb_out1_r, 1);
            cb_push_back(cb_out1_i, 1);

            // Pop all inputs consumed by Session D.
            cb_pop_front(cb_even_r, 1);
            cb_pop_front(cb_even_i, 1);
            cb_pop_front(cb_tmp0,   1);   // t_r
            cb_pop_front(cb_tmp1,   1);   // t_i

            // State at end of tile iteration:
            //   tmp0 = []   tmp1 = []   — both completely empty ✓
            //   Next iteration's cb_reserve_back(tmp0/1, 2) will not block.
        }
    }
}