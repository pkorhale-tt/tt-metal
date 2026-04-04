// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0
//
// fft_compute_f32.cpp  –  EXACT match to paper Listing 1.2 / 1.3
//
// The paper (Section 4, Listing 1.2) operates step-by-step, waiting on
// cb_data1_{r,i} first (the "odd" / RHS element), computing f0 and f1 via
// the SFPU, then waiting on cb_data0_{r,i} (the "even" / LHS element) and
// applying f0/f1 to produce the two output pairs.
//
// Listing 1.3 shows the maths helper: tile_regs_acquire → copy_tile (both
// inputs into dst segments 0 and 1) → op on dst → tile_regs_commit →
// pack_tile → tile_regs_release.  The *_init calls are the bug-fix: they
// must be OUTSIDE the acquire/release pair.
//
// CB mapping (must match reader and host):
//   cb_data0_r = 0   (even / LHS, real)
//   cb_data0_i = 1   (even / LHS, imaginary)
//   cb_data1_r = 2   (odd  / RHS, real)
//   cb_data1_i = 3   (odd  / RHS, imaginary)
//   cb_twiddle_r = 4
//   cb_twiddle_i = 5
//   cb_out0_r = 16   (result data0, real)
//   cb_out0_i = 17   (result data0, imaginary)
//   cb_out1_r = 18   (result data1, real)
//   cb_out1_i = 19   (result data1, imaginary)
//   cb_int0 = 20     (intermediate f0/f1 scratch)
//   cb_int1 = 21
//   cb_f0   = 22
//   cb_f1   = 23
//
// Kernel args:
//   arg[0] = num_steps   (= log2(N), the outer loop count)
//   arg[1] = num_chunks  (inner loop count – number of CB page pairs per step)

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/copy_dest_values.h"

// ---------------------------------------------------------------------------
// maths_sfpu_op helpers  –  faithful to paper Listing 1.3
//
// Three concrete functions (one per operation) avoid the two-phase lookup
// problem that arises when add/sub/mul_binary_tile are called inside a
// function template: those ALWI functions have no template-dependent
// arguments, so the RISC-V g++ rejects forward declarations without bodies.
// Using concrete functions means all names are resolved at definition time.
//
// Each helper:
//   1. Optionally waits on / pops input CBs (paper lines 4-5, 19-20).
//   2. Calls copy_tile_to_dst_init_short OUTSIDE acquire (deadlock fix).
//   3. Acquires dst lock → copies both inputs into dst[0] and dst[1] →
//      performs the operation → commits (paper lines 6-18).
//   4. Packs dst[0] to cb_tgt → releases (paper lines 22-25).
// ---------------------------------------------------------------------------

// Multiply: cb_tgt = cb_in_1 * cb_in_2
inline void maths_sfpu_mul(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    copy_tile_to_dst_init_short(cb_in_1);   // OUTSIDE acquire (deadlock fix)
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();
    copy_tile(cb_in_1, 0, 0);
    copy_tile(cb_in_2, 0, 1);
    mul_binary_tile(0, 1);
    tile_regs_commit();
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

// Subtract: cb_tgt = cb_in_1 - cb_in_2
inline void maths_sfpu_sub(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    copy_tile_to_dst_init_short(cb_in_1);
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();
    copy_tile(cb_in_1, 0, 0);
    copy_tile(cb_in_2, 0, 1);
    sub_binary_tile(0, 1);
    tile_regs_commit();
    if (pop_in1) cb_pop_front(cb_in_1, 1);
    if (pop_in2) cb_pop_front(cb_in_2, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

// Add: cb_tgt = cb_in_1 + cb_in_2
inline void maths_sfpu_add(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt,
                            bool pop_in1 = false, bool pop_in2 = false) {
    copy_tile_to_dst_init_short(cb_in_1);
    cb_reserve_back(cb_tgt, 1);
    tile_regs_acquire();
    copy_tile(cb_in_1, 0, 0);
    copy_tile(cb_in_2, 0, 1);
    add_binary_tile(0, 1);
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

    // CB indices – must match reader, writer and host exactly.
    constexpr uint32_t cb_data0_r   = 0;
    constexpr uint32_t cb_data0_i   = 1;
    constexpr uint32_t cb_data1_r   = 2;
    constexpr uint32_t cb_data1_i   = 3;
    constexpr uint32_t cb_twiddle_r = 4;
    constexpr uint32_t cb_twiddle_i = 5;

    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    constexpr uint32_t cb_int0 = 20;
    constexpr uint32_t cb_int1 = 21;
    constexpr uint32_t cb_f0   = 22;
    constexpr uint32_t cb_f1   = 23;

    // Outer loop: one iteration per FFT step (= log2 N steps).
    for (uint32_t step = 0; step < num_steps; ++step) {

        // Inner loop: one iteration per chunk of data within the step.
        // The reader feeds one page per CB per chunk; the writer drains
        // one page per output CB per chunk.
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {

            // ------------------------------------------------------------------
            // Paper Listing 1.2, lines 2-3:
            // Wait for the RHS (odd / data1) real and imaginary pages first.
            // ------------------------------------------------------------------
            cb_wait_front(cb_data1_r, 1);
            cb_wait_front(cb_data1_i, 1);

            // ------------------------------------------------------------------
            // Paper Listing 1.2, lines 6-8:  f0 = data1_r*tw_r - data1_i*tw_i
            // ------------------------------------------------------------------

            // int0 = data1_r * twiddle_r
            maths_sfpu_mul(cb_data1_r, cb_twiddle_r, cb_int0);

            // int1 = data1_i * twiddle_i
            maths_sfpu_mul(cb_data1_i, cb_twiddle_i, cb_int1);

            // f0 = int0 - int1  (pops both int CBs)
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            maths_sfpu_sub(cb_int0, cb_int1, cb_f0, true, true);

            // ------------------------------------------------------------------
            // Paper Listing 1.2, lines 11-13: f1 = data1_r*tw_i + data1_i*tw_r
            // ------------------------------------------------------------------

            // int0 = data1_r * twiddle_i
            maths_sfpu_mul(cb_data1_r, cb_twiddle_i, cb_int0);

            // int1 = data1_i * twiddle_r
            maths_sfpu_mul(cb_data1_i, cb_twiddle_r, cb_int1);

            // f1 = int0 + int1  (pops both int CBs)
            cb_wait_front(cb_int0, 1);
            cb_wait_front(cb_int1, 1);
            maths_sfpu_add(cb_int0, cb_int1, cb_f1, true, true);

            // ------------------------------------------------------------------
            // Paper Listing 1.2, lines 15-16:
            // Now wait for LHS (even / data0) real and imaginary pages.
            // ------------------------------------------------------------------
            cb_wait_front(cb_data0_r, 1);
            cb_wait_front(cb_data0_i, 1);

            // ------------------------------------------------------------------
            // Paper Listing 1.2, lines 18-25:
            // out1 = data0 - f  (data1 index result)
            // out0 = data0 + f  (data0 index result)
            // ------------------------------------------------------------------

            // out1_r = data0_r - f0
            cb_wait_front(cb_f0, 1);
            maths_sfpu_sub(cb_data0_r, cb_f0, cb_out1_r);

            // out1_i = data0_i - f1
            cb_wait_front(cb_f1, 1);
            maths_sfpu_sub(cb_data0_i, cb_f1, cb_out1_i);

            // out0_r = data0_r + f0
            cb_wait_front(cb_f0, 1);
            maths_sfpu_add(cb_data0_r, cb_f0, cb_out0_r);

            // out0_i = data0_i + f1
            cb_wait_front(cb_f1, 1);
            maths_sfpu_add(cb_data0_i, cb_f1, cb_out0_i);

            // ------------------------------------------------------------------
            // Paper Listing 1.2, lines 27-30:  pop all input pages.
            // ------------------------------------------------------------------
            cb_pop_front(cb_data0_r,   1);
            cb_pop_front(cb_data0_i,   1);
            cb_pop_front(cb_data1_r,   1);
            cb_pop_front(cb_data1_i,   1);
            cb_pop_front(cb_f0,        1);
            cb_pop_front(cb_f1,        1);
        }
    }
}