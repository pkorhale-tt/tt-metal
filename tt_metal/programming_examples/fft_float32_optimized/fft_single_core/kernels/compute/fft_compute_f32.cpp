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
//    Stage 0:  DRAM input  ──► [butterfly] ──► L1 ping buffer
//    Stage 1:  L1 ping     ──► [butterfly] ──► L1 pong buffer
//    Stage 2:  L1 pong     ──► [butterfly] ──► L1 ping buffer
//    ...
//    Stage N-1:L1 ping/pong──► [butterfly] ──► DRAM output
//
//  Ping-pong L1 inter-stage buffers:
//    data_ping_r / data_ping_i  (c_10 / c_11)
//    data_pong_r / data_pong_i  (c_12 / c_13)
//    Alternates each stage — zero DRAM between stages.
//
//  CB ownership contract (ALL explicit — no hidden pops):
//    Every cb_wait_front / cb_pop_front is visible at call site.
//    No CB_OP_IN=true used anywhere in the hot path.
//
//  FPU used for ALL butterfly arithmetic (add/sub/mul).
//  SFPU used ONLY for IFFT negation (once per stage, not per tile).
//
//  Double buffering:
//    Input CBs (c_0..c_5):   depth=2, reader prefetches tile i+1
//                            while compute processes tile i.
//    Output CBs (c_16..c_19):depth=2, writer drains tile i
//                            while compute produces tile i+1.
//    Twiddle CB (c_6/c_7):   depth=2, reader prefetches next tile.
//    Inter-stage (c_10..c_13):depth=1 per buffer (ping-pong covers it)
//
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"


// ── Operation tags ──────────────────────────────────────────────────
constexpr uint32_t ADD = 0;
constexpr uint32_t SUB = 1;
constexpr uint32_t MUL = 2;
constexpr uint32_t NEG = 3;

// ── CB index map ────────────────────────────────────────────────────
//
//  INPUT CBs  (reader RISCV_0 writes, compute reads)
//    c_0  cb_even_r      even real        depth=2 double-buffer
//    c_1  cb_even_i      even imag        depth=2 double-buffer
//    c_2  cb_odd_r       odd  real        depth=2 double-buffer
//    c_3  cb_odd_i       odd  imag        depth=2 double-buffer
//    c_4  cb_tw_r        twiddle cos(θ)  depth=2 double-buffer
//    c_5  cb_tw_i        twiddle sin(θ)  depth=2 double-buffer
//
//  INTER-STAGE PING-PONG  (compute writes and reads, no DRAM)
//    c_10 cb_ping_r      ping real        depth=1 (ping-pong covers it)
//    c_11 cb_ping_i      ping imag        depth=1
//    c_12 cb_pong_r      pong real        depth=1
//    c_13 cb_pong_i      pong imag        depth=1
//
//  OUTPUT CBs  (compute writes, writer RISCV_1 drains to DRAM)
//    c_16 cb_out0_r      X[k]     real    depth=2 double-buffer
//    c_17 cb_out0_i      X[k]     imag    depth=2 double-buffer
//    c_18 cb_out1_r      X[k+N/2] real    depth=2 double-buffer
//    c_19 cb_out1_i      X[k+N/2] imag    depth=2 double-buffer
//
//  INTERMEDIATE CBs  (compute internal — never seen by reader/writer)
//    c_20 cb_tmp0        scratch product  depth=1 transient
//    c_21 cb_tmp1        scratch product  depth=1 transient
//    c_22 cb_tw_odd_r    W·O[k] real      depth=1 transient
//    c_23 cb_tw_odd_i    W·O[k] imag      depth=1 transient
//    c_24 cb_neg_tw_i    -sin(θ) IFFT     depth=1 transient


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
//
//  FPU path only — 6 tile operations, all 1024-wide.
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void complex_mul(
    uint32_t cb_a_r,   uint32_t cb_a_i,
    uint32_t cb_b_r,   uint32_t cb_b_i,
    uint32_t cb_out_r, uint32_t cb_out_i,
    uint32_t cb_tmp0,  uint32_t cb_tmp1)
{
    // ── Real part: ac - bd ─────────────────────────────────────────
    fpu_op<MUL>(cb_a_r, cb_b_r, cb_tmp0);          // tmp0 = ac
    fpu_op<MUL>(cb_a_i, cb_b_i, cb_tmp1);          // tmp1 = bd

    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    fpu_op<SUB>(cb_tmp0, cb_tmp1, cb_out_r);        // out_r = ac - bd
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);

    // ── Imaginary part: ad + bc ────────────────────────────────────
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
//      cb_even_r, cb_even_i    butterfly even input
//      cb_odd_r,  cb_odd_i     butterfly odd input
//      cb_tw_odd_r, cb_tw_odd_i  W·O[k] intermediate
//    BORROWS (caller waited before call; caller pops after):
//      cb_tw_r, cb_tw_i        twiddle factors
//    PRODUCES (caller must wait + pop):
//      cb_out0_r/i, cb_out1_r/i
//    OWNS intermediates (via complex_mul):
//      cb_tmp0, cb_tmp1
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

    // Done with odd inputs
    cb_pop_front(cb_odd_r, 1);
    cb_pop_front(cb_odd_i, 1);
    // NOTE: cb_tw_r, cb_tw_i NOT popped — caller owns them

    // Step 2: butterfly ADD and SUB
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
//    cb_in_r / cb_in_i     — source data CBs (DRAM-backed or ping/pong)
//    cb_tw_r / cb_tw_i     — twiddle CBs for this stage (from DRAM)
//    cb_out_r / cb_out_i   — destination CBs (DRAM-backed or ping/pong)
//    num_tiles             — tiles to process this stage
//    is_ifft               — true: negate twiddle imag before butterfly
//    is_first_stage        — source is external input CBs (c_0..c_3)
//    is_last_stage         — destination is external output CBs (c_16..c_19)
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void process_stage(
    uint32_t cb_in_r,   uint32_t cb_in_i,
    uint32_t cb_tw_r,   uint32_t cb_tw_i,
    uint32_t cb_out0_r, uint32_t cb_out0_i,
    uint32_t cb_out1_r, uint32_t cb_out1_i,
    uint32_t cb_tmp0,   uint32_t cb_tmp1,
    uint32_t cb_tw_odd_r, uint32_t cb_tw_odd_i,
    uint32_t cb_neg_tw_i,
    uint32_t num_tiles,
    bool     is_ifft)
{
    // For IFFT: negate the twiddle imaginary component ONCE
    // before the tile loop. We produce one negated-twiddle tile
    // per input tile, consuming the original cb_tw_i.
    //
    // For forward FFT: cb_tw_i is used directly.

    for (uint32_t t = 0; t < num_tiles; t++) {

        if (is_ifft) {
            // ── IFFT: negate twiddle imag ──────────────────────────
            // Explicit wait → negate → explicit pop
            // No hidden behavior.
            cb_wait_front(cb_tw_i, 1);
            sfpu_neg(cb_tw_i, cb_neg_tw_i);
            cb_pop_front(cb_tw_i, 1);

            // Wait on twiddle real — borrowed by butterfly
            cb_wait_front(cb_tw_r, 1);
            // Wait on negated twiddle — borrowed by butterfly
            cb_wait_front(cb_neg_tw_i, 1);

            butterfly(
                cb_in_r,     cb_in_i,
                cb_in_r,     cb_in_i,     // odd = second half (same CB,
                                          // reader sends even then odd)
                cb_tw_r,     cb_neg_tw_i, // conjugated twiddle
                cb_out0_r,   cb_out0_i,
                cb_out1_r,   cb_out1_i,
                cb_tmp0,     cb_tmp1,
                cb_tw_odd_r, cb_tw_odd_i
            );

            // Pop borrowed twiddles (butterfly did NOT pop them)
            cb_pop_front(cb_tw_r,     1);
            cb_pop_front(cb_neg_tw_i, 1);

        } else {
            // ── Forward FFT: use twiddle directly ─────────────────
            cb_wait_front(cb_tw_r, 1);
            cb_wait_front(cb_tw_i, 1);

            butterfly(
                cb_in_r,     cb_in_i,
                cb_in_r,     cb_in_i,
                cb_tw_r,     cb_tw_i,
                cb_out0_r,   cb_out0_i,
                cb_out1_r,   cb_out1_i,
                cb_tmp0,     cb_tmp1,
                cb_tw_odd_r, cb_tw_odd_i
            );

            // Pop borrowed twiddles
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
    constexpr auto cb_in0_r    = tt::CBIndex::c_0;   // even real
    constexpr auto cb_in0_i    = tt::CBIndex::c_1;   // even imag
    constexpr auto cb_in1_r    = tt::CBIndex::c_2;   // odd  real
    constexpr auto cb_in1_i    = tt::CBIndex::c_3;   // odd  imag

    // Twiddle factors (reader streams one stage slice at a time)
    constexpr auto cb_tw_r     = tt::CBIndex::c_4;
    constexpr auto cb_tw_i     = tt::CBIndex::c_5;

    // Ping-pong inter-stage buffers (compute owns both sides)
    constexpr auto cb_ping_r   = tt::CBIndex::c_10;
    constexpr auto cb_ping_i   = tt::CBIndex::c_11;
    constexpr auto cb_pong_r   = tt::CBIndex::c_12;
    constexpr auto cb_pong_i   = tt::CBIndex::c_13;

    // Final stage outputs (writer drains to DRAM)
    constexpr auto cb_out0_r   = tt::CBIndex::c_16;
    constexpr auto cb_out0_i   = tt::CBIndex::c_17;
    constexpr auto cb_out1_r   = tt::CBIndex::c_18;
    constexpr auto cb_out1_i   = tt::CBIndex::c_19;

    // Scratch intermediates (compute internal, depth=1)
    constexpr auto cb_tmp0     = tt::CBIndex::c_20;
    constexpr auto cb_tmp1     = tt::CBIndex::c_21;
    constexpr auto cb_tw_odd_r = tt::CBIndex::c_22;
    constexpr auto cb_tw_odd_i = tt::CBIndex::c_23;
    constexpr auto cb_neg_tw_i = tt::CBIndex::c_24;

    // ── Common compute init ────────────────────────────────────────
    unary_op_init_common(cb_in0_r, cb_out0_r);
    binary_op_init_common(cb_in0_r, cb_in1_r, cb_out0_r);
    copy_tile_to_dst_init_short(cb_in0_r);

    const bool is_ifft = (direction == 1);

    // ══════════════════════════════════════════════════════════════
    //  STAGE LOOP — ALL STAGES RUN ON DEVICE
    //
    //  Stage 0:    input  = external CBs (c_0..c_3) from DRAM
    //              output = ping buffer  (c_10..c_11)
    //
    //  Stage 1..N-2: ping-pong alternates
    //              odd  stage: ping → pong
    //              even stage: pong → ping
    //
    //  Stage N-1:  input  = ping or pong (depends on parity)
    //              output = external CBs (c_16..c_19) to DRAM
    //
    //  Ping-pong eliminates ALL DRAM access between stages.
    //  Twiddle CB is re-streamed per stage by reader.
    // ══════════════════════════════════════════════════════════════

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        // ── Select source CBs ──────────────────────────────────────
        // Stage 0:     read from external input CBs
        // Stage 1+:    read from ping (odd stage) or pong (even stage)
        uint32_t src_r, src_i;
        if (stage == 0) {
            src_r = cb_in0_r;
            src_i = cb_in0_i;
        } else if ((stage & 1) == 1) {
            // Odd stage: read from ping
            src_r = cb_ping_r;
            src_i = cb_ping_i;
        } else {
            // Even stage (>=2): read from pong
            src_r = cb_pong_r;
            src_i = cb_pong_i;
        }

        // ── Select destination CBs ─────────────────────────────────
        // Last stage:  write to external output CBs (writer → DRAM)
        // All others:  write to ping or pong
        uint32_t dst0_r, dst0_i, dst1_r, dst1_i;
        if (stage == num_stages - 1) {
            dst0_r = cb_out0_r;
            dst0_i = cb_out0_i;
            dst1_r = cb_out1_r;
            dst1_i = cb_out1_i;
        } else if ((stage & 1) == 0) {
            // Even stage: write to ping
            dst0_r = cb_ping_r;
            dst0_i = cb_ping_i;
            dst1_r = cb_ping_r;
            dst1_i = cb_ping_i;
        } else {
            // Odd stage: write to pong
            dst0_r = cb_pong_r;
            dst0_i = cb_pong_i;
            dst1_r = cb_pong_r;
            dst1_i = cb_pong_i;
        }

        // ── Process all tiles for this stage ──────────────────────
        process_stage(
            src_r,      src_i,
            cb_tw_r,    cb_tw_i,
            dst0_r,     dst0_i,
            dst1_r,     dst1_i,
            cb_tmp0,    cb_tmp1,
            cb_tw_odd_r, cb_tw_odd_i,
            cb_neg_tw_i,
            tiles_per_stage,
            is_ifft
        );
    }
}