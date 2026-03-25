// compute.cpp — Tensix compute kernel
// Wormhole 1-D Cooley-Tukey FFT (decimation-in-time, radix-2)
// ═══════════════════════════════════════════════════════════════════════
//
//  Per butterfly tile iteration:
//    W * odd = (tw_r*odd_r - tw_i*odd_i) + j*(tw_r*odd_i + tw_i*odd_r)
//    out_even = even + W*odd
//    out_odd  = even - W*odd
//
//  CB map (depths set by host):
//    Stage 0 inputs:    CB 0 even_r, CB 1 even_i, CB 2 odd_r, CB 3 odd_i
//    Stage 1+ inputs:   CB 0 even_r, CB 1 even_i, CB 2 odd_r, CB 3 odd_i
//                       (writer feeds back here after each intermediate stage)
//    Twiddles:          CB 4 tw_r,   CB 5 tw_i   (all stages, from reader)
//    Outputs:           CB 6 out_even_r, CB 7 out_even_i
//                       CB 8 out_odd_r,  CB 9 out_odd_i
//    Scratch (depth=1): CB 10, 11, 12, 13
//
//  Stack discipline: no VLA, no large locals. All scratch via depth-1 CBs.
//  Each scratch CB is always popped before it is reserved again.
//
//  Argument map:
//    [0] num_stages
//    [1] tiles_per_stage   (= tiles_per_row, one tile per butterfly group)
//    [2] rows_per_core

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// ---------------------------------------------------------------------------
// Helpers: compute and pack into dst_cb.  Inputs not popped — caller manages.
// ---------------------------------------------------------------------------
ALWI void mul_into(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    copy_tile_init(a); copy_tile(a, 0, 0);
    copy_tile_init(b); copy_tile(b, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(dst);
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

ALWI void add_into(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    copy_tile_init(a); copy_tile(a, 0, 0);
    copy_tile_init(b); copy_tile(b, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(dst);
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

ALWI void sub_into(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    copy_tile_init(a); copy_tile(a, 0, 0);
    copy_tile_init(b); copy_tile(b, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    tile_regs_commit(); tile_regs_wait();
    pack_reconfig_data_format(dst);
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(2);

    constexpr uint32_t CB_EVEN_R    = 0;
    constexpr uint32_t CB_EVEN_I    = 1;
    constexpr uint32_t CB_ODD_R     = 2;
    constexpr uint32_t CB_ODD_I     = 3;
    constexpr uint32_t CB_TW_R      = 4;
    constexpr uint32_t CB_TW_I      = 5;
    constexpr uint32_t CB_OUT_EVEN_R = 6;
    constexpr uint32_t CB_OUT_EVEN_I = 7;
    constexpr uint32_t CB_OUT_ODD_R  = 8;
    constexpr uint32_t CB_OUT_ODD_I  = 9;
    // Scratch: depth=1 each, popped immediately after use
    constexpr uint32_t CB_S0 = 10;
    constexpr uint32_t CB_S1 = 11;
    constexpr uint32_t CB_S2 = 12;  // W_real = tw_r*odd_r - tw_i*odd_i
    constexpr uint32_t CB_S3 = 13;  // W_imag = tw_r*odd_i + tw_i*odd_r

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) return;

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            // CB_EVEN_R/I and CB_ODD_R/I are always 0-3:
            // - stage 0: filled by reader from DRAM
            // - stage 1+: filled by writer feedback (same CB indices)
            // The writer pops output CBs 6-9 and pushes into 0-3 for the
            // next stage, so the CB indices are the same throughout.

            for (uint32_t t = 0; t < tiles_per_stage; ++t) {
                // ── Wait for all inputs ───────────────────────────────
                cb_wait_front(CB_EVEN_R, 1);
                cb_wait_front(CB_EVEN_I, 1);
                cb_wait_front(CB_ODD_R,  1);
                cb_wait_front(CB_ODD_I,  1);
                cb_wait_front(CB_TW_R,   1);
                cb_wait_front(CB_TW_I,   1);

                // ── Complex multiply: W * odd ─────────────────────────
                //
                // real: tw_r*odd_r → S0,  tw_i*odd_i → S1,  S2 = S0 - S1
                mul_into(CB_TW_R, CB_ODD_R, CB_S0);
                cb_wait_front(CB_S0, 1);
                mul_into(CB_TW_I, CB_ODD_I, CB_S1);
                cb_wait_front(CB_S1, 1);
                sub_into(CB_S0, CB_S1, CB_S2);   // S2 = W_real * odd
                cb_pop_front(CB_S0, 1);
                cb_pop_front(CB_S1, 1);

                // imag: tw_r*odd_i → S0,  tw_i*odd_r → S1,  S3 = S0 + S1
                mul_into(CB_TW_R, CB_ODD_I, CB_S0);
                cb_wait_front(CB_S0, 1);
                mul_into(CB_TW_I, CB_ODD_R, CB_S1);
                cb_wait_front(CB_S1, 1);
                add_into(CB_S0, CB_S1, CB_S3);   // S3 = W_imag * odd
                cb_pop_front(CB_S0, 1);
                cb_pop_front(CB_S1, 1);

                // Twiddle and odd consumed — pop now.
                cb_pop_front(CB_TW_R,  1);
                cb_pop_front(CB_TW_I,  1);
                cb_pop_front(CB_ODD_R, 1);
                cb_pop_front(CB_ODD_I, 1);

                // ── Butterfly ────────────────────────────────────────
                // Wait for W*odd results (S2, S3 were computed above).
                cb_wait_front(CB_S2, 1);
                cb_wait_front(CB_S3, 1);

                // out_even_r = even_r + S2
                add_into(CB_EVEN_R, CB_S2, CB_OUT_EVEN_R);
                // out_even_i = even_i + S3
                add_into(CB_EVEN_I, CB_S3, CB_OUT_EVEN_I);
                // out_odd_r  = even_r - S2
                sub_into(CB_EVEN_R, CB_S2, CB_OUT_ODD_R);
                // out_odd_i  = even_i - S3
                sub_into(CB_EVEN_I, CB_S3, CB_OUT_ODD_I);

                // Even inputs and W*odd results consumed.
                cb_pop_front(CB_EVEN_R, 1);
                cb_pop_front(CB_EVEN_I, 1);
                cb_pop_front(CB_S2,     1);
                cb_pop_front(CB_S3,     1);

                // Output CBs 6-9 are now filled; writer will drain them.
            }
        }
    }
}