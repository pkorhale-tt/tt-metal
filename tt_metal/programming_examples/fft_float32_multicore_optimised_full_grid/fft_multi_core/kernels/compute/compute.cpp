// fft_compute_f32.cpp  — MULTICORE: per-core butterfly kernel  [OPTIMISED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS vs. previous version
// ══════════════════════════════════════════════════════════════════════
//
//  OPT-1  Hoist *_tiles_init outside the per-tile loop.
//  ─────────────────────────────────────────────────────────────────────
//  Previously each butterfly tile triggered 9 FPU reconfiguration calls
//  (mul/sub/add variants).  For tiles_per_stage = T that is 9·T calls;
//  now it is exactly 9 per stage regardless of T.
//
//  How: split each logical phase (e.g. "multiply all T even-twiddle
//  tiles") into:
//    1. one init call before the tile loop
//    2. T × (acquire → op → commit/wait → pack → release)
//  This is the standard "batch tile" pattern used in tt-metal matmul.
//
//  OPT-2  Batch cb_wait_front / cb_pop_front at stage boundaries.
//  ─────────────────────────────────────────────────────────────────────
//  Previously each tile did a separate wait.  Now the kernel waits for
//  all T tiles in one call before entering any loop, and pops all T
//  at once after the phase completes.  This removes T–1 redundant
//  barrier-check polls per stage.
//
//  OPT-3  Reuse even_r/even_i CB tiles for both out0 and out1 phases.
//  ─────────────────────────────────────────────────────────────────────
//  even_r tiles are read twice (for out0_r = even+t and out1_r = even-t).
//  We delay cb_pop_front(cb_even_r) until after both phases G and I,
//  and similarly for even_i.  This avoids a second cb_wait_front and
//  the same for tw_odd_r/tw_odd_i.
//
// CB map (unchanged):
//   0  cb_even_r    stage input even real
//   1  cb_even_i    stage input even imag
//   2  cb_odd_r     stage input odd  real
//   3  cb_odd_i     stage input odd  imag
//   4  cb_tw_r      expanded twiddle real  (reader fills per stage)
//   5  cb_tw_i      expanded twiddle imag
//  16  cb_out0_r    butterfly sum real
//  17  cb_out0_i    butterfly sum imag
//  18  cb_out1_r    butterfly diff real
//  19  cb_out1_i    butterfly diff imag
//  20  cb_tmp0      scratch
//  21  cb_tmp1      scratch
//  22  cb_tw_odd_r  W*odd real
//  23  cb_tw_odd_i  W*odd imag
//
// Args:
//   0  num_stages       log2(N_row)
//   1  tiles_per_stage  tiles this core processes per butterfly stage
//                       = rows_this × tiles_per_row   (host must pass this)

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

// ── Helper: run one compute phase over T tiles ────────────────────────────
//
// Caller is responsible for:
//   • cb_wait_front(src_a, T) and cb_wait_front(src_b, T) BEFORE calling
//   • cb_reserve_back(dst, T) BEFORE calling
//   • cb_push_back(dst, T) AFTER calling
//   • cb_pop_front(src_a, T) and cb_pop_front(src_b, T) AFTER (if desired)
//
// We keep this as inline code (not a function) to avoid function-call
// overhead on the embedded RISC-V, using a local macro.
//
// PHASE_MUL / PHASE_ADD / PHASE_SUB each expand to:
//   init_fn(src_a, src_b, dst);
//   for t in [0, T):
//       acquire → op(src_a, src_b, t, t, 0) → commit/wait → pack → release
//
// The FPU init is called ONCE for the whole batch of T tiles.

#define BATCH_BINARY_OP(init_fn, op_fn, src_a, src_b, dst_cb, T)  \
    do {                                                            \
        init_fn((src_a), (src_b), (dst_cb));                       \
        for (uint32_t _t = 0; _t < (T); _t++) {                   \
            tile_regs_acquire();                                    \
            op_fn((src_a), (src_b), _t, _t, 0);                   \
            tile_regs_commit();                                     \
            tile_regs_wait();                                       \
            pack_tile(0, (dst_cb));                                 \
            tile_regs_release();                                    \
        }                                                           \
    } while (0)

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    // tiles_per_stage = rows_this × tiles_per_row
    // (host must pass this correctly for multi-row cores)

    constexpr uint32_t cb_even_r   = 0;
    constexpr uint32_t cb_even_i   = 1;
    constexpr uint32_t cb_odd_r    = 2;
    constexpr uint32_t cb_odd_i    = 3;
    constexpr uint32_t cb_tw_r     = 4;
    constexpr uint32_t cb_tw_i     = 5;
    constexpr uint32_t cb_out0_r   = 16;
    constexpr uint32_t cb_out0_i   = 17;
    constexpr uint32_t cb_out1_r   = 18;
    constexpr uint32_t cb_out1_i   = 19;
    constexpr uint32_t cb_tmp0     = 20;
    constexpr uint32_t cb_tmp1     = 21;
    constexpr uint32_t cb_tw_odd_r = 22;
    constexpr uint32_t cb_tw_odd_i = 23;

    const uint32_t T = tiles_per_stage;

    // One-time init for the CB data-format configuration.
    // Per-phase inits below only need to set the operation type.
    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {

        // ── OPT-2: batch wait for all T tiles up front ────────────────
        cb_wait_front(cb_tw_r,   T);
        cb_wait_front(cb_tw_i,   T);
        cb_wait_front(cb_odd_r,  T);
        cb_wait_front(cb_odd_i,  T);
        cb_wait_front(cb_even_r, T);
        cb_wait_front(cb_even_i, T);

        // ── Phase A: tmp0 = tw_r × odd_r ─────────────────────────────
        // OPT-1: mul_tiles_init called ONCE for T tiles
        cb_reserve_back(cb_tmp0, T);
        BATCH_BINARY_OP(mul_tiles_init, mul_tiles, cb_tw_r, cb_odd_r, cb_tmp0, T);
        cb_push_back(cb_tmp0, T);

        // ── Phase B: tmp1 = tw_i × odd_i ─────────────────────────────
        cb_reserve_back(cb_tmp1, T);
        BATCH_BINARY_OP(mul_tiles_init, mul_tiles, cb_tw_i, cb_odd_i, cb_tmp1, T);
        cb_push_back(cb_tmp1, T);

        // ── Phase C: tw_odd_r = tmp0 − tmp1 ──────────────────────────
        cb_wait_front(cb_tmp0, T);
        cb_wait_front(cb_tmp1, T);
        cb_reserve_back(cb_tw_odd_r, T);
        BATCH_BINARY_OP(sub_tiles_init, sub_tiles, cb_tmp0, cb_tmp1, cb_tw_odd_r, T);
        cb_push_back(cb_tw_odd_r, T);
        cb_pop_front(cb_tmp0, T);
        cb_pop_front(cb_tmp1, T);

        // ── Phase D: tmp0 = tw_r × odd_i ─────────────────────────────
        cb_reserve_back(cb_tmp0, T);
        BATCH_BINARY_OP(mul_tiles_init, mul_tiles, cb_tw_r, cb_odd_i, cb_tmp0, T);
        cb_push_back(cb_tmp0, T);

        // ── Phase E: tmp1 = tw_i × odd_r ─────────────────────────────
        cb_reserve_back(cb_tmp1, T);
        BATCH_BINARY_OP(mul_tiles_init, mul_tiles, cb_tw_i, cb_odd_r, cb_tmp1, T);
        cb_push_back(cb_tmp1, T);

        // ── Phase F: tw_odd_i = tmp0 + tmp1 ──────────────────────────
        cb_wait_front(cb_tmp0, T);
        cb_wait_front(cb_tmp1, T);
        cb_reserve_back(cb_tw_odd_i, T);
        BATCH_BINARY_OP(add_tiles_init, add_tiles, cb_tmp0, cb_tmp1, cb_tw_odd_i, T);
        cb_push_back(cb_tw_odd_i, T);
        cb_pop_front(cb_tmp0, T);
        cb_pop_front(cb_tmp1, T);

        // Twiddle and odd inputs are fully consumed — pop them now.
        cb_pop_front(cb_tw_r,  T);
        cb_pop_front(cb_tw_i,  T);
        cb_pop_front(cb_odd_r, T);
        cb_pop_front(cb_odd_i, T);

        cb_wait_front(cb_tw_odd_r, T);
        cb_wait_front(cb_tw_odd_i, T);

        // ── Phase G: out0_r = even_r + tw_odd_r ──────────────────────
        // OPT-3: do NOT pop even_r yet; Phase I will reuse it.
        cb_reserve_back(cb_out0_r, T);
        BATCH_BINARY_OP(add_tiles_init, add_tiles, cb_even_r, cb_tw_odd_r, cb_out0_r, T);
        cb_push_back(cb_out0_r, T);

        // ── Phase H: out0_i = even_i + tw_odd_i ──────────────────────
        // OPT-3: do NOT pop even_i yet; Phase J will reuse it.
        cb_reserve_back(cb_out0_i, T);
        BATCH_BINARY_OP(add_tiles_init, add_tiles, cb_even_i, cb_tw_odd_i, cb_out0_i, T);
        cb_push_back(cb_out0_i, T);

        // ── Phase I: out1_r = even_r − tw_odd_r ──────────────────────
        // Now pop even_r and tw_odd_r (both fully consumed after this).
        cb_reserve_back(cb_out1_r, T);
        BATCH_BINARY_OP(sub_tiles_init, sub_tiles, cb_even_r, cb_tw_odd_r, cb_out1_r, T);
        cb_push_back(cb_out1_r, T);
        cb_pop_front(cb_even_r,   T);
        cb_pop_front(cb_tw_odd_r, T);

        // ── Phase J: out1_i = even_i − tw_odd_i ──────────────────────
        cb_reserve_back(cb_out1_i, T);
        BATCH_BINARY_OP(sub_tiles_init, sub_tiles, cb_even_i, cb_tw_odd_i, cb_out1_i, T);
        cb_push_back(cb_out1_i, T);
        cb_pop_front(cb_even_i,   T);
        cb_pop_front(cb_tw_odd_i, T);
    }
}