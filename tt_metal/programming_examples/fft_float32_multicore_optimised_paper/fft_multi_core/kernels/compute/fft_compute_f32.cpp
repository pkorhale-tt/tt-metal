// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FFT COMPUTE KERNEL — Paper-Aligned Implementation
// Ref: "Exploring Fast Fourier Transforms on the Tenstorrent Wormhole"
//      Brown, Davies, Le Clair (arXiv:2506.15437v1)
//
// Design (Fig. 3 / Listing 1.2 of paper):
//   - SFPU vector unit performs all arithmetic (not FPU/matrix unit)
//   - Twiddle factors pre-computed on init, stored in SRAM via CBs
//   - Decoupled: compute blocks on CB availability from data-mover cores
//   - Chunked: domain split into tiles so reader/compute/writer overlap
//   - Single-copy: writer reorders directly to next step's layout
//
// Circular Buffer layout:
//   cb_data0_r/i (0,1) — LHS (even) real/imag  [double-buffered]
//   cb_data1_r/i (2,3) — RHS (odd)  real/imag  [double-buffered]
//   cb_twiddle_r/i (4,5) — twiddle factors      [pre-loaded once]
//   cb_f0 (6), cb_f1 (7) — intermediate f0, f1  [scratch, depth=1]
//   cb_out0_r/i (16,17) — upper butterfly output
//   cb_out1_r/i (18,19) — lower butterfly output

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/common.h"

// ─────────────────────────────────────────────────────────────────────────────
// SFPU binary helper — mirrors paper's maths_sfpu_op<OPERATION>() (Listing 1.3)
//
// Acquires dst lock, copies tiles from two input CBs into dst[0] and dst[1],
// executes the binary op, packs result to output CB, then releases dst lock.
//
// Template params:
//   OP          — one of: ADD_BINARY, SUB_BINARY, MUL_BINARY
//   POP_IN1     — pop cb_in1 after use (for intermediate CBs)
//   POP_IN2     — pop cb_in2 after use
// ─────────────────────────────────────────────────────────────────────────────
enum BinOp { ADD_BINARY, SUB_BINARY, MUL_BINARY };

template <BinOp OP, bool POP_IN1 = false, bool POP_IN2 = false>
ALWI void sfpu_binary(uint32_t cb_in1, uint32_t cb_in2, uint32_t cb_out,
                      uint32_t in1_tile = 0, uint32_t in2_tile = 0) {
    // Wait for inputs if they may not be ready yet
    // (Caller is responsible for cb_wait_front before calling)

    // Acquire dst register (Half mode: 8 segments available)
    acquire_dst(tt::DstMode::Half);

    // Copy input tiles from SRAM into srcA/srcB → dst[0], dst[1]
    copy_tile(cb_in1, in1_tile, 0);
    copy_tile(cb_in2, in2_tile, 1);

    // Execute operation on dst segments 0 and 1; result lands in segment 0
    if constexpr (OP == ADD_BINARY) {
        add_tiles(cb_in1, cb_in2, in1_tile, in2_tile, 0);
    } else if constexpr (OP == SUB_BINARY) {
        sub_tiles(cb_in1, cb_in2, in1_tile, in2_tile, 0);
    } else if constexpr (OP == MUL_BINARY) {
        mul_tiles(cb_in1, cb_in2, in1_tile, in2_tile, 0);
    }

    // Reserve output page, pack result, make available
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    release_dst(tt::DstMode::Half);

    // Optionally free intermediate input pages
    if constexpr (POP_IN1) cb_pop_front(cb_in1, 1);
    if constexpr (POP_IN2) cb_pop_front(cb_in2, 1);

    cb_push_back(cb_out, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterfly for one tile-chunk (paper Listing 1.2 expanded)
//
// Computes (for each element pair in the tile):
//   f0 = data1_r * twiddle_r  -  data1_i * twiddle_i
//   f1 = data1_r * twiddle_i  +  data1_i * twiddle_r
//
//   out0_r = data0_r + f0    out0_i = data0_i + f1
//   out1_r = data0_r - f0    out1_i = data0_i - f1
// ─────────────────────────────────────────────────────────────────────────────
ALWI void butterfly_tile(
    uint32_t cb_data0_r, uint32_t cb_data0_i,
    uint32_t cb_data1_r, uint32_t cb_data1_i,
    uint32_t cb_tw_r,    uint32_t cb_tw_i,
    uint32_t cb_int0,    uint32_t cb_int1,
    uint32_t cb_f0,      uint32_t cb_f1,
    uint32_t cb_out0_r,  uint32_t cb_out0_i,
    uint32_t cb_out1_r,  uint32_t cb_out1_i)
{
    // ── Compute f0 = data1_r * tw_r - data1_i * tw_i ──────────────────────
    // int0 = data1_r * tw_r
    sfpu_binary<MUL_BINARY>(cb_data1_r, cb_tw_r,  cb_int0);
    // int1 = data1_i * tw_i  (pop both intermediates after use)
    sfpu_binary<MUL_BINARY>(cb_data1_i, cb_tw_i,  cb_int1);
    // f0 = int0 - int1  (pop both intermediates after use)
    sfpu_binary<SUB_BINARY, /*pop_in1=*/true, /*pop_in2=*/true>(cb_int0, cb_int1, cb_f0);

    // ── Compute f1 = data1_r * tw_i + data1_i * tw_r ──────────────────────
    sfpu_binary<MUL_BINARY>(cb_data1_r, cb_tw_i,  cb_int0);
    sfpu_binary<MUL_BINARY>(cb_data1_i, cb_tw_r,  cb_int1);
    sfpu_binary<ADD_BINARY, true, true>(cb_int0, cb_int1, cb_f1);

    // ── Wait for data0 (LHS) which arrives after data1 per paper Listing 1.2 ─
    cb_wait_front(cb_data0_r, 1);
    cb_wait_front(cb_data0_i, 1);

    // ── Apply butterfly (pop f0, f1 after producing all four outputs) ───────
    // out0_r = data0_r + f0
    sfpu_binary<ADD_BINARY>(cb_data0_r, cb_f0, cb_out0_r);
    // out0_i = data0_i + f1
    sfpu_binary<ADD_BINARY>(cb_data0_i, cb_f1, cb_out0_i);
    // out1_r = data0_r - f0  (pop f0 now)
    sfpu_binary<SUB_BINARY, false, /*pop_f0=*/true>(cb_data0_r, cb_f0, cb_out1_r);
    // out1_i = data0_i - f1  (pop f1 now)
    sfpu_binary<SUB_BINARY, false, /*pop_f1=*/true>(cb_data0_i, cb_f1, cb_out1_i);

    // ── Free consumed input pages (enables CB memory reuse for next chunk) ──
    cb_pop_front(cb_data0_r, 1);
    cb_pop_front(cb_data0_i, 1);
    cb_pop_front(cb_data1_r, 1);
    cb_pop_front(cb_data1_i, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN KERNEL ENTRY POINT
// ─────────────────────────────────────────────────────────────────────────────
void MAIN {
    // ── Runtime arguments ──────────────────────────────────────────────────
    const uint32_t num_steps       = get_arg_val<uint32_t>(0);  // = log2(N)
    const uint32_t tiles_per_chunk = get_arg_val<uint32_t>(1);  // chunked domain
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(2);

    // ── CB indices (must match reader/writer kernels) ──────────────────────
    constexpr uint32_t cb_data0_r = 0;   // LHS real
    constexpr uint32_t cb_data0_i = 1;   // LHS imag
    constexpr uint32_t cb_data1_r = 2;   // RHS real
    constexpr uint32_t cb_data1_i = 3;   // RHS imag
    constexpr uint32_t cb_tw_r    = 4;   // twiddle real
    constexpr uint32_t cb_tw_i    = 5;   // twiddle imag
    constexpr uint32_t cb_int0    = 6;   // intermediate product (scratch)
    constexpr uint32_t cb_int1    = 7;   // intermediate product (scratch)
    constexpr uint32_t cb_f0      = 8;   // f0 result (scratch)
    constexpr uint32_t cb_f1      = 9;   // f1 result (scratch)
    constexpr uint32_t cb_out0_r  = 16;  // upper butterfly real
    constexpr uint32_t cb_out0_i  = 17;  // upper butterfly imag
    constexpr uint32_t cb_out1_r  = 18;  // lower butterfly real
    constexpr uint32_t cb_out1_i  = 19;  // lower butterfly imag

    // ── Initialize SFPU ops once (not per iteration — paper optimization) ──
    // Paper: "twiddle factors are calculated by the compute engine on
    //         initialisation and stored in SRAM"
    binary_op_init_common(cb_data0_r, cb_data1_r, cb_out0_r);
    mul_tiles_init();
    add_tiles_init();
    sub_tiles_init();
    copy_tile_to_dst_init_short();

    // ── Main loop: rows × steps × chunks ──────────────────────────────────
    // The "chunked" optimization (Table 1: "Chunked" → 9.38 ms vs 14.39 ms)
    // splits the domain into segments so reader/compute/writer overlap.
    for (uint32_t row = 0; row < rows_per_core; row++) {
        for (uint32_t step = 0; step < num_steps; step++) {
            for (uint32_t chunk = 0; chunk < tiles_per_chunk; chunk++) {

                // Wait for RHS (data1) and twiddle tiles — reader pre-loads these
                // Paper Listing 1.2 lines 2-3: wait on data1 first
                cb_wait_front(cb_data1_r, 1);
                cb_wait_front(cb_data1_i, 1);
                cb_wait_front(cb_tw_r,    1);
                cb_wait_front(cb_tw_i,    1);
                // Note: data0 (LHS) is waited inside butterfly_tile after f0/f1
                // are computed, matching paper lines 15-16

                // Compute one butterfly tile (see butterfly_tile above)
                butterfly_tile(
                    cb_data0_r, cb_data0_i,
                    cb_data1_r, cb_data1_i,
                    cb_tw_r,    cb_tw_i,
                    cb_int0,    cb_int1,
                    cb_f0,      cb_f1,
                    cb_out0_r,  cb_out0_i,
                    cb_out1_r,  cb_out1_i
                );

                // Pop twiddle pages (twiddles are re-loaded each step by reader)
                cb_pop_front(cb_tw_r, 1);
                cb_pop_front(cb_tw_i, 1);
            }
            // Writer kernel consumes cb_out* after each step and either:
            //   - Reorders to next step's layout → feeds back into cb_data0/1
            //     (single-copy optimization, paper Fig. 5)
            //   - Writes to DRAM on the final step
        }
    }
}