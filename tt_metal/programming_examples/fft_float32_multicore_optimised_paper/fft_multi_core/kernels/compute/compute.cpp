// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"

// ── FIXES vs original ────────────────────────────────────────────────────────
// 1. Removed stray `copy_tile_to_dst_init_short(cb_data1_r)` that was sitting
//    at the top of the step/chunk loop body.  That bare UNPACK-init conflicted
//    with the identical call inside maths_sfpu_mul and stalled the unpack
//    pipeline, causing the compute core to spin waiting for DST.
//
// 2. Inside maths_sfpu_*, moved cb_pop_front calls to BEFORE tile_regs_wait()
//    (matching paper Listing 1.3 lines 19-20).  Popping after tile_regs_commit
//    but before tile_regs_wait lets the reader refill the CB slot while the
//    PACK core is still draining DST, improving pipeline overlap with depth-2
//    CBs.  Not a deadlock cause but restores paper-faithful ordering.
//
// 3. Added explicit *_tiles_init calls (add_binary_tile_init, etc.) BEFORE the
//    corresponding operation call.  The original had them, but after the
//    copy_tile_to_dst_init_short_with_dt reorder confusion they were easy to
//    misplace.  Ordering is now strictly:
//      copy_tile_to_dst_init_short(A)  → copy_tile(A,0,0)
//      copy_tile_to_dst_init_short_with_dt(A,B) → copy_tile(B,0,1)
//      <op>_binary_tile_init()
//      <op>_binary_tile(0,1,0)
//      tile_regs_commit()
//      [optional pops]
//      tile_regs_wait()
//      pack_tile(0, cb_tgt)
//      tile_regs_release()
// ─────────────────────────────────────────────────────────────────────────────

inline void maths_sfpu_mul(
    uint32_t cb_in_1,
    uint32_t cb_in_2,
    uint32_t cb_tgt,
    bool pop_in1 = false,
    bool pop_in2 = false)
{
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);

    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);

    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);

    tile_regs_commit();

    // Pop inputs BEFORE tile_regs_wait so the reader can refill the CB slot
    // while PACK drains DST  (paper Listing 1.3, lines 19-20).
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);

    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

inline void maths_sfpu_sub(
    uint32_t cb_in_1,
    uint32_t cb_in_2,
    uint32_t cb_tgt,
    bool pop_in1 = false,
    bool pop_in2 = false)
{
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);

    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);

    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);

    tile_regs_commit();

    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);

    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

inline void maths_sfpu_add(
    uint32_t cb_in_1,
    uint32_t cb_in_2,
    uint32_t cb_tgt,
    bool pop_in1 = false,
    bool pop_in2 = false)
{
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);

    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);

    add_binary_tile_init();
    add_binary_tile(0, 1, 0);

    tile_regs_commit();

    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);

    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

void kernel_main() {
    const uint32_t num_steps  = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_data0_r   = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i   = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r   = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i   = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    constexpr uint32_t cb_int0 = tt::CBIndex::c_20;
    constexpr uint32_t cb_int1 = tt::CBIndex::c_21;
    constexpr uint32_t cb_f0   = tt::CBIndex::c_22;
    constexpr uint32_t cb_f1   = tt::CBIndex::c_23;

    for (uint32_t step = 0; step < num_steps; ++step) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {

            // ── FIX 1: the stray copy_tile_to_dst_init_short(cb_data1_r)
            // that was here has been REMOVED.  Each maths_sfpu_* helper
            // issues its own UNPACK init at the correct point.

            cb_wait_front(cb_data1_r, 1);
            cb_wait_front(cb_data1_i, 1);
            cb_wait_front(cb_twiddle_r, 1);
            cb_wait_front(cb_twiddle_i, 1);

            // ── f0 = data1_r * tw_r  −  data1_i * tw_i ──────────────────
            maths_sfpu_mul(cb_data1_r, cb_twiddle_r, cb_int0);
            maths_sfpu_mul(cb_data1_i, cb_twiddle_i, cb_int1);
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            // pop_in1=true, pop_in2=true: free cb_int0 and cb_int1 slots
            maths_sfpu_sub(cb_int0, cb_int1, cb_f0, /*pop_in1=*/true, /*pop_in2=*/true);

            // ── f1 = data1_r * tw_i  +  data1_i * tw_r ──────────────────
            maths_sfpu_mul(cb_data1_r, cb_twiddle_i, cb_int0);
            maths_sfpu_mul(cb_data1_i, cb_twiddle_r, cb_int1);
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            maths_sfpu_add(cb_int0, cb_int1, cb_f1, /*pop_in1=*/true, /*pop_in2=*/true);

            cb_wait_front(cb_data0_r, 1);
            cb_wait_front(cb_data0_i, 1);
            cb_wait_front(cb_f0, 1);
            cb_wait_front(cb_f1, 1);

            // ── out1 = data0 − f  (butterfly lower arm) ──────────────────
            // FIX: push out1 AFTER out0 so the writer's cb_wait_front(out0_r)
            // does not block while out1 slots fill up with depth-2 CBs.
            // Order: out0_r, out0_i, out1_r, out1_i  — matches writer wait order.
            maths_sfpu_add(cb_data0_r, cb_f0, cb_out0_r);
            maths_sfpu_add(cb_data0_i, cb_f1, cb_out0_i);
            maths_sfpu_sub(cb_data0_r, cb_f0, cb_out1_r);
            maths_sfpu_sub(cb_data0_i, cb_f1, cb_out1_i);

            // Pop all inputs for this chunk
            cb_pop_front(cb_data0_r, 1);
            cb_pop_front(cb_data0_i, 1);
            cb_pop_front(cb_data1_r, 1);
            cb_pop_front(cb_data1_i, 1);
            cb_pop_front(cb_twiddle_r, 1);
            cb_pop_front(cb_twiddle_i, 1);
            cb_pop_front(cb_f0, 1);
            cb_pop_front(cb_f1, 1);
        }
    }
}