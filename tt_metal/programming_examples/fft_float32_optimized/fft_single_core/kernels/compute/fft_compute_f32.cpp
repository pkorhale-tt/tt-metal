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
//    Stage 0:  DRAM input    ──► [butterfly] ──► ping_even + ping_odd
//    Stage 1:  ping_even/odd ──► [butterfly] ──► pong_even + pong_odd
//    Stage 2:  pong_even/odd ──► [butterfly] ──► ping_even + ping_odd
//    ...
//    Stage N-1:ping/pong     ──► [butterfly] ──► DRAM output
//
//  SEPARATE even/odd CBs at every inter-stage level:
//    ping_even_r/i  (c_10/c_11)   ping_odd_r/i  (c_12/c_13)
//    pong_even_r/i  (c_14/c_15)   pong_odd_r/i  (c_6/c_7)
//
//  WHY separate: butterfly needs two tiles simultaneously (even + odd).
//  Sharing one CB causes a deadlock — odd consumed first, even wait
//  never resolves because the CB is already empty.
//
//  FPU init strategy:
//    binary_op_init_common() called ONCE at kernel start.
//    Per-op inits (add_tiles_init etc.) called ONLY when the operation
//    changes from the previous call — tracked by `current_op`.
//    This avoids corrupting unpacker state from repeated re-inits
//    with different CB arguments across complex_mul / butterfly.
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
constexpr uint32_t OP_NONE = 0;
constexpr uint32_t OP_ADD  = 1;
constexpr uint32_t OP_SUB  = 2;
constexpr uint32_t OP_MUL  = 3;

// ── CB index map ────────────────────────────────────────────────────
//
//  INPUT CBs
//    c_0  cb_even_r   stage-0 even real   depth=2
//    c_1  cb_even_i   stage-0 even imag   depth=2
//    c_2  cb_odd_r    stage-0 odd  real   depth=2
//    c_3  cb_odd_i    stage-0 odd  imag   depth=2
//    c_4  cb_tw_r     twiddle cos          depth=2
//    c_5  cb_tw_i     twiddle sin          depth=2
//
//  INTER-STAGE PING (compute owns)
//    c_10 ping_even_r   depth=tiles_per_stage
//    c_11 ping_even_i   depth=tiles_per_stage
//    c_12 ping_odd_r    depth=tiles_per_stage
//    c_13 ping_odd_i    depth=tiles_per_stage
//
//  INTER-STAGE PONG (compute owns)
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
//  SCRATCH (compute internal)
//    c_20 tmp0        depth=1
//    c_21 tmp1        depth=1
//    c_22 tw_odd_r    depth=1
//    c_23 tw_odd_i    depth=1
//    c_24 neg_tw_i    depth=1


// ═══════════════════════════════════════════════════════════════════
//  FPU BINARY OP  — init-guarded
//
//  current_op tracks which op was last initialised. We only call
//  *_tiles_init() when switching ops, avoiding repeated re-inits
//  with changing CB args that corrupt unpacker state.
//
//  Contract:
//    Caller has cb_wait_front on cb_a and cb_b.
//    Caller will cb_pop_front after this returns.
//    This function writes one tile to cb_tgt.
// ═══════════════════════════════════════════════════════════════════
template <uint32_t OP>
FORCE_INLINE void fpu_op(
    uint32_t  cb_a,
    uint32_t  cb_b,
    uint32_t  cb_tgt,
    uint32_t& current_op)
{
    if (current_op != OP) {
        if constexpr (OP == OP_ADD) {
            add_tiles_init(cb_a, cb_b);
        } else if constexpr (OP == OP_SUB) {
            sub_tiles_init(cb_a, cb_b);
        } else if constexpr (OP == OP_MUL) {
            mul_tiles_init(cb_a, cb_b);
        }
        current_op = OP;
    }

    tile_regs_acquire();
    if constexpr (OP == OP_ADD) {
        add_tiles(cb_a, cb_b, 0, 0, 0);
    } else if constexpr (OP == OP_SUB) {
        sub_tiles(cb_a, cb_b, 0, 0, 0);
    } else if constexpr (OP == OP_MUL) {
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
//  Caller must cb_wait_front(cb_in) before and cb_pop_front after.
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
//  BORROWS:  cb_a_r, cb_a_i, cb_b_r, cb_b_i  (caller waited, will pop)
//  PRODUCES: cb_out_r, cb_out_i               (caller must wait+pop)
//  OWNS tmp: cb_tmp0, cb_tmp1                 (popped internally)
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void complex_mul(
    uint32_t  cb_a_r,   uint32_t cb_a_i,
    uint32_t  cb_b_r,   uint32_t cb_b_i,
    uint32_t  cb_out_r, uint32_t cb_out_i,
    uint32_t  cb_tmp0,  uint32_t cb_tmp1,
    uint32_t& current_op)
{
    // Real: ac - bd
    fpu_op<OP_MUL>(cb_a_r, cb_b_r, cb_tmp0, current_op);   // tmp0 = ac
    fpu_op<OP_MUL>(cb_a_i, cb_b_i, cb_tmp1, current_op);   // tmp1 = bd

    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    fpu_op<OP_SUB>(cb_tmp0, cb_tmp1, cb_out_r, current_op); // out_r = ac-bd
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);

    // Imag: ad + bc
    fpu_op<OP_MUL>(cb_a_r, cb_b_i, cb_tmp0, current_op);   // tmp0 = ad
    fpu_op<OP_MUL>(cb_a_i, cb_b_r, cb_tmp1, current_op);   // tmp1 = bc

    cb_wait_front(cb_tmp0, 1);
    cb_wait_front(cb_tmp1, 1);
    fpu_op<OP_ADD>(cb_tmp0, cb_tmp1, cb_out_i, current_op); // out_i = ad+bc
    cb_pop_front(cb_tmp0, 1);
    cb_pop_front(cb_tmp1, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  FFT BUTTERFLY
//  X[k]     = E[k] + W·O[k]
//  X[k+N/2] = E[k] - W·O[k]
//
//  CRITICAL: cb_even_r/i and cb_odd_r/i MUST be different CBs.
//
//  OWNS (waits + pops): cb_even_r/i, cb_odd_r/i, cb_tw_odd_r/i
//  BORROWS (caller waited, caller pops): cb_tw_r, cb_tw_i
//  PRODUCES: cb_out0_r/i, cb_out1_r/i
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void butterfly(
    uint32_t  cb_even_r,   uint32_t cb_even_i,
    uint32_t  cb_odd_r,    uint32_t cb_odd_i,
    uint32_t  cb_tw_r,     uint32_t cb_tw_i,
    uint32_t  cb_out0_r,   uint32_t cb_out0_i,
    uint32_t  cb_out1_r,   uint32_t cb_out1_i,
    uint32_t  cb_tmp0,     uint32_t cb_tmp1,
    uint32_t  cb_tw_odd_r, uint32_t cb_tw_odd_i,
    uint32_t& current_op)
{
    // Step 1: W·O[k]  — consume odd input
    cb_wait_front(cb_odd_r, 1);
    cb_wait_front(cb_odd_i, 1);

    complex_mul(
        cb_odd_r,    cb_odd_i,
        cb_tw_r,     cb_tw_i,
        cb_tw_odd_r, cb_tw_odd_i,
        cb_tmp0,     cb_tmp1,
        current_op
    );

    cb_pop_front(cb_odd_r, 1);
    cb_pop_front(cb_odd_i, 1);
    // tw_r, tw_i NOT popped — caller owns them

    // Step 2: butterfly ADD / SUB
    // Safe to wait on even here — it is a DIFFERENT CB from odd
    cb_wait_front(cb_even_r,   1);
    cb_wait_front(cb_even_i,   1);
    cb_wait_front(cb_tw_odd_r, 1);
    cb_wait_front(cb_tw_odd_i, 1);

    fpu_op<OP_ADD>(cb_even_r, cb_tw_odd_r, cb_out0_r, current_op); // X[k] real
    fpu_op<OP_ADD>(cb_even_i, cb_tw_odd_i, cb_out0_i, current_op); // X[k] imag

    fpu_op<OP_SUB>(cb_even_r, cb_tw_odd_r, cb_out1_r, current_op); // X[k+N/2] real
    fpu_op<OP_SUB>(cb_even_i, cb_tw_odd_i, cb_out1_i, current_op); // X[k+N/2] imag

    cb_pop_front(cb_even_r,   1);
    cb_pop_front(cb_even_i,   1);
    cb_pop_front(cb_tw_odd_r, 1);
    cb_pop_front(cb_tw_odd_i, 1);
}

// ═══════════════════════════════════════════════════════════════════
//  PROCESS ONE STAGE
// ═══════════════════════════════════════════════════════════════════
FORCE_INLINE void process_stage(
    uint32_t  cb_even_r,   uint32_t cb_even_i,
    uint32_t  cb_odd_r,    uint32_t cb_odd_i,
    uint32_t  cb_tw_r,     uint32_t cb_tw_i,
    uint32_t  cb_out0_r,   uint32_t cb_out0_i,
    uint32_t  cb_out1_r,   uint32_t cb_out1_i,
    uint32_t  cb_tmp0,     uint32_t cb_tmp1,
    uint32_t  cb_tw_odd_r, uint32_t cb_tw_odd_i,
    uint32_t  cb_neg_tw_i,
    uint32_t  num_tiles,
    bool      is_ifft,
    uint32_t& current_op)
{
    for (uint32_t t = 0; t < num_tiles; t++) {

        if (is_ifft) {
            // Conjugate twiddle: negate imaginary part
            cb_wait_front(cb_tw_i, 1);
            sfpu_neg(cb_tw_i, cb_neg_tw_i);
            cb_pop_front(cb_tw_i, 1);

            // sfpu_neg uses SFPU path — FPU op state is now unknown.
            // Force re-init on the next fpu_op call.
            current_op = OP_NONE;

            cb_wait_front(cb_tw_r,     1);
            cb_wait_front(cb_neg_tw_i, 1);

            butterfly(
                cb_even_r,    cb_even_i,
                cb_odd_r,     cb_odd_i,
                cb_tw_r,      cb_neg_tw_i,
                cb_out0_r,    cb_out0_i,
                cb_out1_r,    cb_out1_i,
                cb_tmp0,      cb_tmp1,
                cb_tw_odd_r,  cb_tw_odd_i,
                current_op
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
                cb_tw_odd_r,  cb_tw_odd_i,
                current_op
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

    // ── One-time compute init ──────────────────────────────────────
    // binary_op_init_common sets up unpacker for both src CBs.
    // Per-op inits are done lazily inside fpu_op() via current_op.
    binary_op_init_common(cb_in_even_r, cb_in_odd_r, cb_out0_r);
    unary_op_init_common(cb_in_even_r, cb_out0_r);
    copy_tile_to_dst_init_short(cb_in_even_r);

    const bool is_ifft = (direction == 1);

    // Tracks the last FPU op initialised.
    // Prevents re-init with changing CB args — avoids unpacker corruption.
    uint32_t current_op = OP_NONE;

    // ══════════════════════════════════════════════════════════════
    //  STAGE LOOP
    //
    //  Stage routing:
    //
    //  stage  | src_even        src_odd       | dst X[k]      dst X[k+N/2]
    //  -------+---------------------------------+---------------------------
    //  0      | c_0/c_1         c_2/c_3       | ping_even     ping_odd
    //  1(odd) | ping_even       ping_odd      | pong_even     pong_odd
    //  2(even)| pong_even       pong_odd      | ping_even     ping_odd
    //  ...
    //  last   | (ping/pong)     (ping/pong)   | c_16/c_17     c_18/c_19
    //
    //  Special: if num_stages==1, stage 0 IS the last stage.
    //  The last-stage check takes priority over even/odd routing.
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
            // Last stage → DRAM output (writer drains to DRAM)
            dst0_r = cb_out0_r;      dst0_i = cb_out0_i;
            dst1_r = cb_out1_r;      dst1_i = cb_out1_i;
        } else if ((stage & 1) == 0) {
            // Even stage → ping
            dst0_r = cb_ping_even_r; dst0_i = cb_ping_even_i;
            dst1_r = cb_ping_odd_r;  dst1_i = cb_ping_odd_i;
        } else {
            // Odd stage → pong
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
            is_ifft,
            current_op
        );
    }
}