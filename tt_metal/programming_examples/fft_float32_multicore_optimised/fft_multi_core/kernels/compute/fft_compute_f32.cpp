// fft_compute_f32.cpp — MULTICORE butterfly kernel (FIXED v2)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  ROOT CAUSE OF HANG (fixed here)
// ══════════════════════════════════════════════════════════════════════
//
//  The previous version called cb_pop_front and cb_reserve_back between
//  tile_regs_wait() and tile_regs_release() in Sessions B and C.
//  This violates the hardware invariant:
//
//    NEVER call cb_reserve_back or cb_pop_front inside a tile_regs
//    session (i.e. between tile_regs_acquire and tile_regs_release).
//
//  The firmware CB state machine and register-file arbitration interact
//  such that issuing a CB operation while the register file is locked
//  causes an unresolvable stall — the kernel hangs indefinitely.
//
//  Additionally, the previous Session A pushed 2 tiles to each of
//  tmp0 and tmp1 (filling both depth-2 CBs), then Session B tried to
//  reserve 1 more slot in tmp0 *before* acquiring registers — impossible
//  because tmp0 was already full.  The two problems reinforced each other.
//
// ══════════════════════════════════════════════════════════════════════
//  FIX STRATEGY
// ══════════════════════════════════════════════════════════════════════
//
//  Split the original 4-multiply Session A into two independent sessions
//  (A1 and A2), each producing one pair of partial products.  Introduce
//  two new scratch CBs (tmp2, tmp3, depth=1) to hold t_r and t_i so that
//  the intermediate and final partial products never share a CB queue.
//
//  Every tile_regs session now follows the strict invariant:
//
//    1. cb_reserve_back(output, n)        — outside session, before acquire
//    2. tile_regs_acquire()
//    3. <compute>
//    4. tile_regs_commit() / tile_regs_wait()
//    5. pack_tile(slot, cb)               — inside session, after wait
//    6. tile_regs_release()               — inside session, last op
//    7. cb_push_back(output, n)           — outside session
//    8. cb_pop_front(inputs consumed, n)  — outside session
//
//  This is verified by dry-run below.
//
// ══════════════════════════════════════════════════════════════════════
//  DRY-RUN (per tile iteration, all depths shown)
// ══════════════════════════════════════════════════════════════════════
//
//  Initial state: tmp0=[] tmp1=[] tmp2=[] tmp3=[]
//
//  SESSION A1  (tw_r*odd_r → tmp0,  tw_i*odd_i → tmp1)
//    reserve(tmp0,1) → depth=1, 0 tiles present → OK
//    reserve(tmp1,1) → depth=1, 0 tiles present → OK
//    acquire → mul → commit/wait → pack(0,tmp0) pack(1,tmp1) → release
//    push(tmp0,1)  push(tmp1,1)
//    State: tmp0=[tw_r*odd_r]  tmp1=[tw_i*odd_i]
//
//  SESSION B   (t_r = tmp0[0] − tmp1[0] → tmp2)
//    wait(tmp0,1) wait(tmp1,1)
//    reserve(tmp2,1) → depth=1, 0 tiles present → OK
//    acquire → sub → commit/wait → pack(0,tmp2) → release
//    pop(tmp0,1)  pop(tmp1,1)
//    push(tmp2,1)
//    State: tmp0=[]  tmp1=[]  tmp2=[t_r]
//
//  SESSION A2  (tw_r*odd_i → tmp0,  tw_i*odd_r → tmp1)
//    reserve(tmp0,1) → 0 tiles present → OK
//    reserve(tmp1,1) → 0 tiles present → OK
//    acquire → mul → commit/wait → pack(0,tmp0) pack(1,tmp1) → release
//    pop(tw_r,1) pop(tw_i,1) pop(odd_r,1) pop(odd_i,1)   ← all tw/odd done
//    push(tmp0,1)  push(tmp1,1)
//    State: tmp0=[tw_r*odd_i]  tmp1=[tw_i*odd_r]  tmp2=[t_r]
//
//  SESSION C   (t_i = tmp0[0] + tmp1[0] → tmp3)
//    wait(tmp0,1) wait(tmp1,1)
//    reserve(tmp3,1) → depth=1, 0 tiles present → OK
//    acquire → add → commit/wait → pack(0,tmp3) → release
//    pop(tmp0,1)  pop(tmp1,1)
//    push(tmp3,1)
//    State: tmp0=[]  tmp1=[]  tmp2=[t_r]  tmp3=[t_i]
//
//  SESSION D   (butterfly outputs)
//    wait(tmp2,1) wait(tmp3,1)   even_r/i waited in A1, still present
//    reserve(out0_r,1) reserve(out0_i,1) reserve(out1_r,1) reserve(out1_i,1)
//    acquire
//      add(even_r,tmp2→reg0)  add(even_i,tmp3→reg1)
//      sub(even_r,tmp2→reg2)  sub(even_i,tmp3→reg3)
//    commit/wait
//      pack(0,out0_r) pack(1,out0_i) pack(2,out1_r) pack(3,out1_i)
//    release
//    push(out0_r,1) push(out0_i,1) push(out1_r,1) push(out1_i,1)
//    pop(even_r,1) pop(even_i,1) pop(tmp2,1) pop(tmp3,1)
//    State: ALL scratch CBs empty ✓  Next iteration reserve will not block ✓
//
// ══════════════════════════════════════════════════════════════════════
//  CB layout
// ══════════════════════════════════════════════════════════════════════
//
//  Input CBs  (reader fills, compute drains):
//    cb_even_r [0]  cb_even_i [1]
//    cb_odd_r  [2]  cb_odd_i  [3]
//    cb_tw_r   [4]  cb_tw_i   [5]
//
//  Scratch CBs (internal, depth=1 each):
//    cb_tmp0 [20]  tw_r*odd_r  then  tw_r*odd_i
//    cb_tmp1 [21]  tw_i*odd_i  then  tw_i*odd_r
//    cb_tmp2 [22]  t_r  (= tw_r*odd_r − tw_i*odd_i)
//    cb_tmp3 [23]  t_i  (= tw_r*odd_i + tw_i*odd_r)
//
//  Output CBs (compute fills, writer drains):
//    cb_out0_r [16]  cb_out0_i [17]
//    cb_out1_r [18]  cb_out1_i [19]
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

    // ── CB indices ────────────────────────────────────────────────────
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
    constexpr uint32_t cb_tmp0   = 20;   // depth=1  tw_r*odd_r / tw_r*odd_i
    constexpr uint32_t cb_tmp1   = 21;   // depth=1  tw_i*odd_i / tw_i*odd_r
    constexpr uint32_t cb_tmp2   = 22;   // depth=1  t_r result
    constexpr uint32_t cb_tmp3   = 23;   // depth=1  t_i result

    // Sticky FPU config — valid for the lifetime of this kernel.
    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // ── Wait for all inputs up front ──────────────────────────
            // even_r/i are waited here and held until Session D.
            // tw/odd are waited here; popped after Session A2.
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // ═══════════════════════════════════════════════════════════
            //  SESSION A1 — tw_r*odd_r → tmp0,  tw_i*odd_i → tmp1
            // ═══════════════════════════════════════════════════════════
            // Both scratch CBs are empty at this point (guaranteed by
            // Session D of the previous iteration draining them fully).
            cb_reserve_back(cb_tmp0, 1);   // 0 tiles present, depth=1 → OK
            cb_reserve_back(cb_tmp1, 1);   // 0 tiles present, depth=1 → OK

            tile_regs_acquire();

            mul_tiles_init(cb_tw_r, cb_odd_r);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);   // reg[0] = tw_r*odd_r

            mul_tiles_init(cb_tw_i, cb_odd_i);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 1);   // reg[1] = tw_i*odd_i

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp0);   // tw_r*odd_r → tmp0
            pack_tile(1, cb_tmp1);   // tw_i*odd_i → tmp1

            tile_regs_release();

            // CB pushes and pops strictly outside the session.
            cb_push_back(cb_tmp0, 1);
            cb_push_back(cb_tmp1, 1);
            // tw_r, tw_i, odd_r, odd_i NOT popped yet — still needed for A2.

            // State: tmp0=[tw_r*odd_r]  tmp1=[tw_i*odd_i]

            // ═══════════════════════════════════════════════════════════
            //  SESSION B — t_r = tmp0[0] − tmp1[0] → tmp2
            // ═══════════════════════════════════════════════════════════
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);

            // Reserve output BEFORE acquire — tmp2 is empty (depth=1) → OK.
            cb_reserve_back(cb_tmp2, 1);

            tile_regs_acquire();

            sub_tiles_init(cb_tmp0, cb_tmp1);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);   // reg[0] = t_r

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp2);   // t_r → tmp2

            tile_regs_release();

            // CB pops and pushes outside session.
            cb_pop_front(cb_tmp0, 1);   // tw_r*odd_r consumed
            cb_pop_front(cb_tmp1, 1);   // tw_i*odd_i consumed
            cb_push_back(cb_tmp2, 1);   // t_r ready

            // State: tmp0=[]  tmp1=[]  tmp2=[t_r]

            // ═══════════════════════════════════════════════════════════
            //  SESSION A2 — tw_r*odd_i → tmp0,  tw_i*odd_r → tmp1
            // ═══════════════════════════════════════════════════════════
            // tmp0 and tmp1 are empty now — reserve is safe.
            cb_reserve_back(cb_tmp0, 1);
            cb_reserve_back(cb_tmp1, 1);

            tile_regs_acquire();

            mul_tiles_init(cb_tw_r, cb_odd_i);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);   // reg[0] = tw_r*odd_i

            mul_tiles_init(cb_tw_i, cb_odd_r);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 1);   // reg[1] = tw_i*odd_r

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp0);   // tw_r*odd_i → tmp0
            pack_tile(1, cb_tmp1);   // tw_i*odd_r → tmp1

            tile_regs_release();

            // All tw and odd inputs are now fully consumed — pop them.
            cb_pop_front(cb_tw_r,  1);
            cb_pop_front(cb_tw_i,  1);
            cb_pop_front(cb_odd_r, 1);
            cb_pop_front(cb_odd_i, 1);

            cb_push_back(cb_tmp0, 1);
            cb_push_back(cb_tmp1, 1);

            // State: tmp0=[tw_r*odd_i]  tmp1=[tw_i*odd_r]  tmp2=[t_r]

            // ═══════════════════════════════════════════════════════════
            //  SESSION C — t_i = tmp0[0] + tmp1[0] → tmp3
            // ═══════════════════════════════════════════════════════════
            cb_wait_front(cb_tmp0, 1);
            cb_wait_front(cb_tmp1, 1);

            // Reserve output BEFORE acquire — tmp3 is empty (depth=1) → OK.
            cb_reserve_back(cb_tmp3, 1);

            tile_regs_acquire();

            add_tiles_init(cb_tmp0, cb_tmp1);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);   // reg[0] = t_i

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_tmp3);   // t_i → tmp3

            tile_regs_release();

            cb_pop_front(cb_tmp0, 1);   // tw_r*odd_i consumed
            cb_pop_front(cb_tmp1, 1);   // tw_i*odd_r consumed
            cb_push_back(cb_tmp3, 1);   // t_i ready

            // State: tmp0=[]  tmp1=[]  tmp2=[t_r]  tmp3=[t_i]
            // even_r, even_i: still at front (waited in A1, not yet popped)

            // ═══════════════════════════════════════════════════════════
            //  SESSION D — butterfly outputs
            //    out0_r = even_r + t_r    (upper half, real)
            //    out0_i = even_i + t_i    (upper half, imag)
            //    out1_r = even_r − t_r    (lower half, real)
            //    out1_i = even_i − t_i    (lower half, imag)
            // ═══════════════════════════════════════════════════════════
            cb_wait_front(cb_tmp2, 1);
            cb_wait_front(cb_tmp3, 1);
            // even_r/i already at front — no additional wait needed.

            // Reserve all four output slots before acquiring the register file.
            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);

            tile_regs_acquire();

            add_tiles_init(cb_even_r, cb_tmp2);
            add_tiles(cb_even_r, cb_tmp2, 0, 0, 0);   // reg[0] = even_r + t_r

            add_tiles_init(cb_even_i, cb_tmp3);
            add_tiles(cb_even_i, cb_tmp3, 0, 0, 1);   // reg[1] = even_i + t_i

            sub_tiles_init(cb_even_r, cb_tmp2);
            sub_tiles(cb_even_r, cb_tmp2, 0, 0, 2);   // reg[2] = even_r − t_r

            sub_tiles_init(cb_even_i, cb_tmp3);
            sub_tiles(cb_even_i, cb_tmp3, 0, 0, 3);   // reg[3] = even_i − t_i

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
            cb_pop_front(cb_tmp2,   1);   // t_r
            cb_pop_front(cb_tmp3,   1);   // t_i

            // ── End of tile iteration ─────────────────────────────────
            // All scratch CBs are guaranteed empty:
            //   tmp0=[]  tmp1=[]  tmp2=[]  tmp3=[]
            // Next iteration's Session A1 reserve will succeed immediately.
        }
    }
}