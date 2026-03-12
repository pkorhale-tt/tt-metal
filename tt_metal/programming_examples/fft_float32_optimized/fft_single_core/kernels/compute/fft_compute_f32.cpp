// fft_compute_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  KEY DESIGN DECISION — single global init per op type
// ═══════════════════════════════════════════════════════════════════
//
//  On TT-Metal Tensix, *_tiles_init() programs the unpacker ONCE
//  for a given CB pair. The unpacker is NOT per-tile — it is a
//  hardware configuration that persists across tile operations.
//
//  The correct pattern is:
//    mul_tiles_init(cb_a, cb_b)    ← configure unpacker for THIS pair
//    mul_tiles(cb_a, cb_b, ...)    ← execute with THIS pair
//    mul_tiles_init(cb_c, cb_d)    ← reconfigure for NEXT pair
//    mul_tiles(cb_c, cb_d, ...)    ← execute with NEXT pair
//
//  WRONG (was causing hang):
//    mul_tiles_init(cb_a, cb_b)
//    mul_tiles(cb_a, cb_b, ...)    ← ok
//    mul_tiles(cb_c, cb_d, ...)    ← HANG: unpacker still set for (a,b)
//
//  The fix: call *_tiles_init() immediately before EVERY *_tiles()
//  call, unconditionally. The hardware cost is a few cycles per call
//  — acceptable because each tile is 1024 floats.
//
//  binary_op_init_common() is called once at startup to set up the
//  packer infrastructure. It does NOT configure the unpacker CB pair.
//
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

// ── CB index map ────────────────────────────────────────────────────
//
//  INPUT CBs
//    c_0  in_even_r   stage-0 even real   depth=2
//    c_1  in_even_i   stage-0 even imag   depth=2
//    c_2  in_odd_r    stage-0 odd  real   depth=2
//    c_3  in_odd_i    stage-0 odd  imag   depth=2
//    c_4  tw_r        twiddle cos          depth=2
//    c_5  tw_i        twiddle sin          depth=2
//
//  INTER-STAGE PING
//    c_10 ping_even_r   depth=tiles_per_stage
//    c_11 ping_even_i   depth=tiles_per_stage
//    c_12 ping_odd_r    depth=tiles_per_stage
//    c_13 ping_odd_i    depth=tiles_per_stage
//
//  INTER-STAGE PONG
//    c_14 pong_even_r   depth=tiles_per_stage
//    c_15 pong_even_i   depth=tiles_per_stage
//    c_6  pong_odd_r    depth=tiles_per_stage
//    c_7  pong_odd_i    depth=tiles_per_stage
//
//  OUTPUT CBs
//    c_16 out0_r   X[k]     real   depth=2
//    c_17 out0_i   X[k]     imag   depth=2
//    c_18 out1_r   X[k+N/2] real   depth=2
//    c_19 out1_i   X[k+N/2] imag   depth=2
//
//  SCRATCH
//    c_20 tmp0        depth=1
//    c_21 tmp1        depth=1
//    c_22 tw_odd_r    depth=1
//    c_23 tw_odd_i    depth=1
//    c_24 neg_tw_i    depth=1


// ═══════════════════════════════════════════════════════════════════
//  PRIMITIVES — always init before execute
//
//  Each function calls its own *_tiles_init() immediately before
//  *_tiles(). This is the only safe pattern when CB args vary.
// ═══════════════════════════════════════════════════════════════════

FORCE_INLINE void tile_add(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out)
{
    add_tiles_init(cb_a, cb_b);
    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

FORCE_INLINE void tile_sub(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out)
{
    sub_tiles_init(cb_a, cb_b);
    tile_regs_acquire();
    sub_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

FORCE_INLINE void tile_mul(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out)
{
    mul_tiles_init(cb_a, cb_b);
    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

FORCE_INLINE void tile_neg(uint32_t cb_in, uint32_t cb_out)
{
    copy_tile_to_dst_init_short(cb_in);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    negative_tile_init();
    negative_tile(0);
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  COMPLEX MULTIPLY: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
//
//  BORROWS:  cb_a_r, cb_a_i, cb_b_r, cb_b_i  (caller waited, will pop)
//  PRODUCES: cb_out_r, cb_out_i               (caller must wait+pop)
//  OWNS tmp: cb_tmp0, cb_tmp1                 (popped internally)
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void complex_mul(
    uint32_t cb_a_r,   uint32_t cb_a_i,
    uint32_t cb_b_r,   uint32_t cb_b_i,
    uint32_t cb_out_r, uint32_t cb_out_i,
    uint32_t cb_tmp0,  uint32_t cb_tmp1)
{
    // Real: ac - bd
    tile_mul(cb_a_r, cb_b_r, cb_tmp0);   // tmp0 = ac
    tile_mul(cb_a_i, cb_b_i, cb_tmp1);   // tmp1 = bd
    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    tile_sub(cb_tmp0, cb_tmp1, cb_out_r); // out_r = ac - bd
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);

    // Imag: ad + bc
    tile_mul(cb_a_r, cb_b_i, cb_tmp0);   // tmp0 = ad
    tile_mul(cb_a_i, cb_b_r, cb_tmp1);   // tmp1 = bc
    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    tile_add(cb_tmp0, cb_tmp1, cb_out_i); // out_i = ad + bc
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  FFT BUTTERFLY
//  X[k]     = E[k] + W·O[k]
//  X[k+N/2] = E[k] - W·O[k]
//
//  CRITICAL: cb_even and cb_odd MUST be different CBs.
//
//  OWNS (waits + pops): cb_even_r/i, cb_odd_r/i, cb_tw_odd_r/i
//  BORROWS (caller waited, caller pops): cb_tw_r, cb_tw_i
//  PRODUCES: cb_out0_r/i, cb_out1_r/i
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void butterfly(
    uint32_t cb_even_r,   uint32_t cb_even_i,
    uint32_t cb_odd_r,    uint32_t cb_odd_i,
    uint32_t cb_tw_r,     uint32_t cb_tw_i,
    uint32_t cb_out0_r,   uint32_t cb_out0_i,
    uint32_t cb_out1_r,   uint32_t cb_out1_i,
    uint32_t cb_tmp0,     uint32_t cb_tmp1,
    uint32_t cb_tw_odd_r, uint32_t cb_tw_odd_i)
{
    // Step 1: W·O[k]
    cb_wait_front(cb_odd_r, 1);
    cb_wait_front(cb_odd_i, 1);

    complex_mul(
        cb_odd_r,    cb_odd_i,
        cb_tw_r,     cb_tw_i,
        cb_tw_odd_r, cb_tw_odd_i,
        cb_tmp0,     cb_tmp1
    );

    cb_pop_front(cb_odd_r, 1);
    cb_pop_front(cb_odd_i, 1);
    // tw_r, tw_i NOT popped — caller owns them

    // Step 2: ADD / SUB with even
    // Safe: cb_even is a DIFFERENT CB from cb_odd
    cb_wait_front(cb_even_r,   1);
    cb_wait_front(cb_even_i,   1);
    cb_wait_front(cb_tw_odd_r, 1);
    cb_wait_front(cb_tw_odd_i, 1);

    tile_add(cb_even_r, cb_tw_odd_r, cb_out0_r); // X[k]     real
    tile_add(cb_even_i, cb_tw_odd_i, cb_out0_i); // X[k]     imag
    tile_sub(cb_even_r, cb_tw_odd_r, cb_out1_r); // X[k+N/2] real
    tile_sub(cb_even_i, cb_tw_odd_i, cb_out1_i); // X[k+N/2] imag

    cb_pop_front(cb_even_r,   1);
    cb_pop_front(cb_even_i,   1);
    cb_pop_front(cb_tw_odd_r, 1);
    cb_pop_front(cb_tw_odd_i, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  PROCESS ONE STAGE
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void process_stage(
    uint32_t cb_even_r,   uint32_t cb_even_i,
    uint32_t cb_odd_r,    uint32_t cb_odd_i,
    uint32_t cb_tw_r,     uint32_t cb_tw_i,
    uint32_t cb_out0_r,   uint32_t cb_out0_i,
    uint32_t cb_out1_r,   uint32_t cb_out1_i,
    uint32_t cb_tmp0,     uint32_t cb_tmp1,
    uint32_t cb_tw_odd_r, uint32_t cb_tw_odd_i,
    uint32_t cb_neg_tw_i,
    uint32_t num_tiles,
    bool     is_ifft)
{
    for (uint32_t t = 0; t < num_tiles; t++) {

        if (is_ifft) {
            // Conjugate twiddle: negate sin component
            cb_wait_front(cb_tw_i, 1);
            tile_neg(cb_tw_i, cb_neg_tw_i);
            cb_pop_front(cb_tw_i, 1);

            cb_wait_front(cb_tw_r,     1);
            cb_wait_front(cb_neg_tw_i, 1);

            butterfly(
                cb_even_r,    cb_even_i,
                cb_odd_r,     cb_odd_i,
                cb_tw_r,      cb_neg_tw_i,
                cb_out0_r,    cb_out0_i,
                cb_out1_r,    cb_out1_i,
                cb_tmp0,      cb_tmp1,
                cb_tw_odd_r,  cb_tw_odd_i
            );

            cb_pop_front(cb_tw_r,     1);
            cb_pop_front(cb_neg_tw_i, 1);

        } else {
            cb_wait_front(cb_tw_r, 1);
            cb_wait_front(cb_tw_i, 1);

            butterfly(
                cb_even_r,    cb_even_i,
                cb_odd_r,     cb_odd_i,
                cb_tw_r,      cb_tw_i,
                cb_out0_r,    cb_out0_i,
                cb_out1_r,    cb_out1_i,
                cb_tmp0,      cb_tmp1,
                cb_tw_odd_r,  cb_tw_odd_i
            );

            cb_pop_front(cb_tw_r, 1);
            cb_pop_front(cb_tw_i, 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  KERNEL MAIN
//
//  Runtime args:
//    0: direction        0=forward, 1=inverse
//    1: num_stages       log2(N)
//    2: tiles_per_stage  (N/2 + TILE_SIZE-1) / TILE_SIZE
// ═══════════════════════════════════════════════════════════════════
void kernel_main() {
    const uint32_t direction       = get_arg_val<uint32_t>(0);
    const uint32_t num_stages      = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(2);

    // ── CB declarations ────────────────────────────────────────────
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

    constexpr auto cb_tmp0        = tt::CBIndex::c_20;
    constexpr auto cb_tmp1        = tt::CBIndex::c_21;
    constexpr auto cb_tw_odd_r    = tt::CBIndex::c_22;
    constexpr auto cb_tw_odd_i    = tt::CBIndex::c_23;
    constexpr auto cb_neg_tw_i    = tt::CBIndex::c_24;

    // ── Global init — packer infrastructure only ───────────────────
    // binary_op_init_common sets up the packer/dest infrastructure.
    // It does NOT fix the unpacker to any CB pair.
    // Each tile_add/tile_sub/tile_mul calls its own *_tiles_init()
    // immediately before executing, which is the only correct pattern
    // when source CBs vary across calls.
    binary_op_init_common(cb_in_even_r, cb_in_odd_r, cb_out0_r);

    const bool is_ifft = (direction == 1);

    // ══════════════════════════════════════════════════════════════
    //  STAGE LOOP
    //
    //  stage  | src_even        src_odd       | dst X[k]      dst X[k+N/2]
    //  -------+---------------------------------+---------------------------
    //  0      | c_0/c_1         c_2/c_3       | ping_even     ping_odd
    //  1(odd) | ping_even       ping_odd      | pong_even     pong_odd
    //  2(even)| pong_even       pong_odd      | ping_even     ping_odd
    //  ...
    //  last   | (ping or pong)  (ping or pong)| c_16..c_19
    //
    //  num_stages==1: stage 0 is last → dst = DRAM output directly.
    // ══════════════════════════════════════════════════════════════

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        // ── Source selection ───────────────────────────────────────
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

        // ── Destination selection ──────────────────────────────────
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

        process_stage(
            src_even_r,  src_even_i,
            src_odd_r,   src_odd_i,
            cb_tw_r,     cb_tw_i,
            dst0_r,      dst0_i,
            dst1_r,      dst1_i,
            cb_tmp0,     cb_tmp1,
            cb_tw_odd_r, cb_tw_odd_i,
            cb_neg_tw_i,
            tiles_per_stage,
            is_ifft
        );
    }
}