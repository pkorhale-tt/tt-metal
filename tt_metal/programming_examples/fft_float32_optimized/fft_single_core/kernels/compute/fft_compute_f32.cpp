// fft_compute_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  ARCHITECTURE OVERVIEW
// ═══════════════════════════════════════════════════════════════════
//
//  KEY CHANGE vs original: ALL log2(N) stages run on device.
//  Host launches ONE kernel. No PCIe round-trips between stages.
//
//  Data flow per stage:
//    Stage 0:  DRAM input  ──► [butterfly] ──► L1 ping-even + ping-odd
//    Stage 1:  ping-even/odd──► [butterfly] ──► L1 pong-even + pong-odd
//    Stage 2:  pong-even/odd──► [butterfly] ──► L1 ping-even + ping-odd
//    ...
//    Stage N-1:ping/pong   ──► [butterfly] ──► DRAM output
//
//  Ping-pong L1 inter-stage buffers (SEPARATE even/odd CBs):
//    ping_even_r / ping_even_i  (c_10 / c_11)
//    ping_odd_r  / ping_odd_i   (c_12 / c_13)
//    pong_even_r / pong_even_i  (c_14 / c_15)
//    pong_odd_r  / pong_odd_i   (c_6  / c_7 )
//
//  WHY SEPARATE even/odd CBs:
//    butterfly() needs TWO tiles simultaneously — one even, one odd.
//    If even and odd share a CB, butterfly consumes the first tile
//    (for odd), then deadlocks waiting for the second (for even)
//    because the reader only pushed one tile and already popped it.
//    Separate CBs = two tiles can be in flight simultaneously.
//
//  CB ownership contract (ALL explicit — no hidden pops):
//    Every cb_wait_front / cb_pop_front is visible at call site.
//
//  FPU used for ALL butterfly arithmetic (add/sub/mul).
//  SFPU used ONLY for IFFT negation (once per tile, not per stage).
//
//  Double buffering:
//    Input CBs  (c_0..c_5):    depth=2, reader prefetches tile i+1
//    Output CBs (c_16..c_19):  depth=2, writer drains tile i
//    Twiddle CB (c_4/c_5):     depth=2, reader prefetches next tile
//    Inter-stage (c_6..c_7,
//                 c_10..c_15): depth=1 per buffer (ping-pong covers it)
//
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

// ── Operation tags ──────────────────────────────────────────────────
constexpr uint32_t ADD = 0;
constexpr uint32_t SUB = 1;
constexpr uint32_t MUL = 2;

// ── CB index map ────────────────────────────────────────────────────
//
//  INPUT CBs  (reader RISCV_0 writes, compute reads)
//    c_0  cb_even_r      stage-0 even real     depth=2
//    c_1  cb_even_i      stage-0 even imag     depth=2
//    c_2  cb_odd_r       stage-0 odd  real     depth=2
//    c_3  cb_odd_i       stage-0 odd  imag     depth=2
//    c_4  cb_tw_r        twiddle cos(θ)        depth=2
//    c_5  cb_tw_i        twiddle sin(θ)        depth=2
//
//  INTER-STAGE PING-PONG  (compute writes and reads, no DRAM)
//    c_10 cb_ping_even_r  ping even real       depth=1
//    c_11 cb_ping_even_i  ping even imag       depth=1
//    c_12 cb_ping_odd_r   ping odd  real       depth=1
//    c_13 cb_ping_odd_i   ping odd  imag       depth=1
//    c_14 cb_pong_even_r  pong even real       depth=1
//    c_15 cb_pong_even_i  pong even imag       depth=1
//    c_6  cb_pong_odd_r   pong odd  real       depth=1
//    c_7  cb_pong_odd_i   pong odd  imag       depth=1
//
//  OUTPUT CBs  (compute writes, writer RISCV_1 drains to DRAM)
//    c_16 cb_out0_r      X[k]     real         depth=2
//    c_17 cb_out0_i      X[k]     imag         depth=2
//    c_18 cb_out1_r      X[k+N/2] real         depth=2
//    c_19 cb_out1_i      X[k+N/2] imag         depth=2
//
//  INTERMEDIATE CBs  (compute internal — never seen by reader/writer)
//    c_20 cb_tmp0        scratch product        depth=1
//    c_21 cb_tmp1        scratch product        depth=1
//    c_22 cb_tw_odd_r    W·O[k] real            depth=1
//    c_23 cb_tw_odd_i    W·O[k] imag            depth=1
//    c_24 cb_neg_tw_i    -sin(θ) IFFT           depth=1


// ═══════════════════════════════════════════════════════════════════
//  FPU BINARY OP
//  Contract: caller has already cb_wait_front on both inputs.
//            caller will cb_pop_front after use.
//            This function only writes to cb_tgt.
// ═══════════════════════════════════════════════════════════════════
template <uint32_t OP>
FORCE_INLINE void fpu_op(
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t cb_tgt)
{
    tile_regs_acquire();

    if constexpr (OP == ADD) {
        add_tiles_init(cb_a, cb_b);
        add_tiles(cb_a, cb_b, 0, 0, 0);
    } else if constexpr (OP == SUB) {
        sub_tiles_init(cb_a, cb_b);
        sub_tiles(cb_a, cb_b, 0, 0, 0);
    } else if constexpr (OP == MUL) {
        mul_tiles_init(cb_a, cb_b);
        mul_tiles(cb_a, cb_b, 0, 0, 0);
    }

    tile_regs_commit();

    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  SFPU NEGATE
//  Contract: caller must cb_wait_front(cb_in) before calling.
//            caller must cb_pop_front(cb_in) after calling.
//            This function only writes to cb_tgt.
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void sfpu_neg(uint32_t cb_in, uint32_t cb_tgt)
{
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    negative_tile_init();
    negative_tile(0);
    tile_regs_commit();

    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  COMPLEX MULTIPLY: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
//
//  Contract (ALL inputs pre-waited by caller):
//    BORROWS:  cb_a_r, cb_a_i, cb_b_r, cb_b_i
//              caller waited; caller will pop after this returns
//    PRODUCES: cb_out_r, cb_out_i (pushed; caller must wait+pop)
//    OWNS tmp: cb_tmp0, cb_tmp1 (produced and popped internally)
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void complex_mul(
    uint32_t cb_a_r,   uint32_t cb_a_i,
    uint32_t cb_b_r,   uint32_t cb_b_i,
    uint32_t cb_out_r, uint32_t cb_out_i,
    uint32_t cb_tmp0,  uint32_t cb_tmp1)
{
    // Real part: ac - bd
    fpu_op<MUL>(cb_a_r, cb_b_r, cb_tmp0);          // tmp0 = ac
    fpu_op<MUL>(cb_a_i, cb_b_i, cb_tmp1);          // tmp1 = bd

    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    fpu_op<SUB>(cb_tmp0, cb_tmp1, cb_out_r);        // out_r = ac - bd
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);

    // Imaginary part: ad + bc
    fpu_op<MUL>(cb_a_r, cb_b_i, cb_tmp0);          // tmp0 = ad
    fpu_op<MUL>(cb_a_i, cb_b_r, cb_tmp1);          // tmp1 = bc

    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    fpu_op<ADD>(cb_tmp0, cb_tmp1, cb_out_i);        // out_i = ad + bc
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  FFT BUTTERFLY
//  X[k]     = E[k] + W·O[k]
//  X[k+N/2] = E[k] - W·O[k]
//
//  Ownership contract:
//    OWNS (waits + pops internally):
//      cb_even_r, cb_even_i      even input (separate CB from odd)
//      cb_odd_r,  cb_odd_i       odd  input (separate CB from even)
//      cb_tw_odd_r, cb_tw_odd_i  W·O[k] intermediate
//    BORROWS (caller waited before call; caller pops after):
//      cb_tw_r, cb_tw_i          twiddle factors
//    PRODUCES (caller must wait + pop):
//      cb_out0_r/i  → X[k]
//      cb_out1_r/i  → X[k+N/2]
//    OWNS intermediates (via complex_mul):
//      cb_tmp0, cb_tmp1
//
//  CRITICAL: cb_even_r/i and cb_odd_r/i MUST be different CBs.
//    butterfly() consumes odd first (complex_mul pops it), then
//    waits for even. If they share a CB the second wait deadlocks.
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void butterfly(
    uint32_t cb_even_r,    uint32_t cb_even_i,
    uint32_t cb_odd_r,     uint32_t cb_odd_i,
    uint32_t cb_tw_r,      uint32_t cb_tw_i,
    uint32_t cb_out0_r,    uint32_t cb_out0_i,
    uint32_t cb_out1_r,    uint32_t cb_out1_i,
    uint32_t cb_tmp0,      uint32_t cb_tmp1,
    uint32_t cb_tw_odd_r,  uint32_t cb_tw_odd_i)
{
    // Step 1: W·O[k]
    // cb_tw_r, cb_tw_i  — pre-waited by caller (twiddle borrows)
    // cb_odd_r, cb_odd_i — waited here (we own them)
    cb_wait_front(cb_odd_r, 1);
    cb_wait_front(cb_odd_i, 1);

    complex_mul(
        cb_odd_r,    cb_odd_i,
        cb_tw_r,     cb_tw_i,
        cb_tw_odd_r, cb_tw_odd_i,
        cb_tmp0,     cb_tmp1
    );

    // Done with odd inputs — pop them
    cb_pop_front(cb_odd_r, 1);
    cb_pop_front(cb_odd_i, 1);
    // NOTE: cb_tw_r, cb_tw_i NOT popped — caller owns them

    // Step 2: butterfly ADD and SUB
    // Now wait for even — safe because even is a DIFFERENT CB from odd
    cb_wait_front(cb_even_r,   1);
    cb_wait_front(cb_even_i,   1);
    cb_wait_front(cb_tw_odd_r, 1);
    cb_wait_front(cb_tw_odd_i, 1);

    // X[k] = E + W·O
    fpu_op<ADD>(cb_even_r, cb_tw_odd_r, cb_out0_r);
    fpu_op<ADD>(cb_even_i, cb_tw_odd_i, cb_out0_i);

    // X[k+N/2] = E - W·O
    fpu_op<SUB>(cb_even_r, cb_tw_odd_r, cb_out1_r);
    fpu_op<SUB>(cb_even_i, cb_tw_odd_i, cb_out1_i);

    cb_pop_front(cb_even_r,   1);
    cb_pop_front(cb_even_i,   1);
    cb_pop_front(cb_tw_odd_r, 1);
    cb_pop_front(cb_tw_odd_i, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  PROCESS ONE STAGE
//
//  Parameters:
//    cb_even_r / cb_even_i   — even input (SEPARATE from odd)
//    cb_odd_r  / cb_odd_i    — odd  input (SEPARATE from even)
//    cb_tw_r   / cb_tw_i     — twiddle CBs for this stage
//    cb_out0_r / cb_out0_i   — X[k]     destination
//    cb_out1_r / cb_out1_i   — X[k+N/2] destination
//    num_tiles               — tiles to process this stage
//    is_ifft                 — true: negate twiddle imag before butterfly
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
            // ── IFFT: negate twiddle imag (conjugate twiddle) ──────
            cb_wait_front(cb_tw_i, 1);
            sfpu_neg(cb_tw_i, cb_neg_tw_i);
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
            // ── Forward FFT: use twiddle directly ─────────────────
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
//  Runtime args (from host):
//    0: direction        0=forward, 1=inverse
//    1: num_stages       log2(N)
//    2: tiles_per_stage  N/2 / TILE_SIZE  (butterflies per stage)
// ═══════════════════════════════════════════════════════════════════
void kernel_main() {
    const uint32_t direction       = get_arg_val<uint32_t>(0);
    const uint32_t num_stages      = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(2);

    // ── CB index declarations ──────────────────────────────────────

    // Stage 0 inputs (from DRAM via reader)
    constexpr auto cb_in_even_r = tt::CBIndex::c_0;   // stage-0 even real
    constexpr auto cb_in_even_i = tt::CBIndex::c_1;   // stage-0 even imag
    constexpr auto cb_in_odd_r  = tt::CBIndex::c_2;   // stage-0 odd  real
    constexpr auto cb_in_odd_i  = tt::CBIndex::c_3;   // stage-0 odd  imag

    // Twiddle factors (reader streams one stage slice at a time)
    constexpr auto cb_tw_r      = tt::CBIndex::c_4;
    constexpr auto cb_tw_i      = tt::CBIndex::c_5;

    // Ping inter-stage buffers — even and odd SEPARATE
    constexpr auto cb_ping_even_r = tt::CBIndex::c_10;
    constexpr auto cb_ping_even_i = tt::CBIndex::c_11;
    constexpr auto cb_ping_odd_r  = tt::CBIndex::c_12;
    constexpr auto cb_ping_odd_i  = tt::CBIndex::c_13;

    // Pong inter-stage buffers — even and odd SEPARATE
    // Using c_14/c_15/c_6/c_7 — safe: c_6/c_7 not used by reader/writer
    constexpr auto cb_pong_even_r = tt::CBIndex::c_14;
    constexpr auto cb_pong_even_i = tt::CBIndex::c_15;
    constexpr auto cb_pong_odd_r  = tt::CBIndex::c_6;
    constexpr auto cb_pong_odd_i  = tt::CBIndex::c_7;

    // Final stage outputs (writer drains to DRAM)
    constexpr auto cb_out0_r = tt::CBIndex::c_16;
    constexpr auto cb_out0_i = tt::CBIndex::c_17;
    constexpr auto cb_out1_r = tt::CBIndex::c_18;
    constexpr auto cb_out1_i = tt::CBIndex::c_19;

    // Scratch intermediates (compute internal, depth=1)
    constexpr auto cb_tmp0     = tt::CBIndex::c_20;
    constexpr auto cb_tmp1     = tt::CBIndex::c_21;
    constexpr auto cb_tw_odd_r = tt::CBIndex::c_22;
    constexpr auto cb_tw_odd_i = tt::CBIndex::c_23;
    constexpr auto cb_neg_tw_i = tt::CBIndex::c_24;

    // ── Common compute init ────────────────────────────────────────
    unary_op_init_common(cb_in_even_r, cb_out0_r);
    binary_op_init_common(cb_in_even_r, cb_in_odd_r, cb_out0_r);
    copy_tile_to_dst_init_short(cb_in_even_r);

    const bool is_ifft = (direction == 1);

    // ══════════════════════════════════════════════════════════════
    //  STAGE LOOP
    //
    //  Stage 0:
    //    src_even = c_0/c_1  (from DRAM)
    //    src_odd  = c_2/c_3  (from DRAM)
    //    dst_even = ping_even (c_10/c_11)
    //    dst_odd  = ping_odd  (c_12/c_13)
    //
    //  Stage 1 (odd):
    //    src_even = ping_even (c_10/c_11)
    //    src_odd  = ping_odd  (c_12/c_13)
    //    dst_even = pong_even (c_14/c_15)
    //    dst_odd  = pong_odd  (c_6/c_7)
    //
    //  Stage 2 (even >= 2):
    //    src_even = pong_even (c_14/c_15)
    //    src_odd  = pong_odd  (c_6/c_7)
    //    dst_even = ping_even (c_10/c_11)
    //    dst_odd  = ping_odd  (c_12/c_13)
    //
    //  Stage N-1 (last):
    //    dst_even = c_16/c_17  (to DRAM)
    //    dst_odd  = c_18/c_19  (to DRAM)
    //
    //  NOTE: dst_even feeds cb_out0 (X[k])
    //        dst_odd  feeds cb_out1 (X[k+N/2])
    //        For inter-stage: butterfly ADD result → even CB
    //                         butterfly SUB result → odd  CB
    // ══════════════════════════════════════════════════════════════

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        // ── Select source CBs ──────────────────────────────────────
        uint32_t src_even_r, src_even_i;
        uint32_t src_odd_r,  src_odd_i;

        if (stage == 0) {
            src_even_r = cb_in_even_r;
            src_even_i = cb_in_even_i;
            src_odd_r  = cb_in_odd_r;
            src_odd_i  = cb_in_odd_i;
        } else if ((stage & 1) == 1) {
            // Odd stage: read from ping
            src_even_r = cb_ping_even_r;
            src_even_i = cb_ping_even_i;
            src_odd_r  = cb_ping_odd_r;
            src_odd_i  = cb_ping_odd_i;
        } else {
            // Even stage (>= 2): read from pong
            src_even_r = cb_pong_even_r;
            src_even_i = cb_pong_even_i;
            src_odd_r  = cb_pong_odd_r;
            src_odd_i  = cb_pong_odd_i;
        }

        // ── Select destination CBs ─────────────────────────────────
        uint32_t dst0_r, dst0_i;   // X[k]     = even output → out0
        uint32_t dst1_r, dst1_i;   // X[k+N/2] = odd  output → out1

        if (stage == num_stages - 1) {
            // Last stage → DRAM output CBs
            dst0_r = cb_out0_r;
            dst0_i = cb_out0_i;
            dst1_r = cb_out1_r;
            dst1_i = cb_out1_i;
        } else if ((stage & 1) == 0) {
            // Even stage (including 0) → write to ping
            dst0_r = cb_ping_even_r;
            dst0_i = cb_ping_even_i;
            dst1_r = cb_ping_odd_r;
            dst1_i = cb_ping_odd_i;
        } else {
            // Odd stage → write to pong
            dst0_r = cb_pong_even_r;
            dst0_i = cb_pong_even_i;
            dst1_r = cb_pong_odd_r;
            dst1_i = cb_pong_odd_i;
        }

        // ── Process all tiles for this stage ──────────────────────
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