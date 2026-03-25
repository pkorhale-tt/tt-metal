// compute.cpp — Tensix compute kernel  (Wormhole 1-D FFT, DIT radix-2)
//
// FIXES vs compute_debug.cpp:
//   1. The cb_wait_front(CB_T2,1) and cb_wait_front(CB_T3,1) before the
//      butterfly were already present in the original — left in place.
//   2. Removed the redundant second pair of cb_wait_front(CB_T2/T3) that
//      appeared after the butterfly addCb/subCb calls (they were no-ops
//      since T2/T3 were already front-valid, but added confusion).
//   3. Added explicit tile_regs_release() guards so the register file is
//      always cleanly released before the next acquire, preventing hangs
//      on back-to-back math ops.
//   4. cb_pop_front ordering: pop T2/T3 AFTER all four butterfly ops are
//      done (they were already correct; made explicit with a comment).
//
// CB map:
//   0  even_r   → compute (reader stage-0 / writer feedback stage 1+)
//   1  even_i
//   2  odd_r
//   3  odd_i
//   4  tw_r     → compute (reader, 1 tile per butterfly)
//   5  tw_i
//   6  out_even_r  compute → writer
//   7  out_even_i
//   8  out_odd_r
//   9  out_odd_i
//  10–13  scratch, depth=1
//
// Args: [0] num_stages  [1] tiles_per_stage  [2] rows_per_core

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// ── Helpers ───────────────────────────────────────────────────────────
// Each helper: reserve 1 slot → acquire regs → load a,b → op → commit/wait
// → pack → release → push.  Depth-1 scratch CBs serialise naturally.

inline void mulCb(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    copy_tile_init(a); copy_tile(a, 0, 0);
    copy_tile_init(b); copy_tile(b, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(dst);
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

inline void addCb(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    copy_tile_init(a); copy_tile(a, 0, 0);
    copy_tile_init(b); copy_tile(b, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(dst);
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

inline void subCb(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    copy_tile_init(a); copy_tile(a, 0, 0);
    copy_tile_init(b); copy_tile(b, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(dst);
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

// ── Kernel ────────────────────────────────────────────────────────────
void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(2);

    constexpr uint32_t CB_ER  = 0;
    constexpr uint32_t CB_EI  = 1;
    constexpr uint32_t CB_OR  = 2;
    constexpr uint32_t CB_OI  = 3;
    constexpr uint32_t CB_TWR = 4;
    constexpr uint32_t CB_TWI = 5;
    constexpr uint32_t CB_OER = 6;
    constexpr uint32_t CB_OEI = 7;
    constexpr uint32_t CB_OOR = 8;
    constexpr uint32_t CB_OOI = 9;
    constexpr uint32_t CB_T0  = 10;
    constexpr uint32_t CB_T1  = 11;
    constexpr uint32_t CB_T2  = 12;  // W_real = tw_r*odd_r − tw_i*odd_i
    constexpr uint32_t CB_T3  = 13;  // W_imag = tw_r*odd_i + tw_i*odd_r

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) return;

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            for (uint32_t t = 0; t < tiles_per_stage; ++t) {

                // ── Wait for all 6 inputs ────────────────────────────
                cb_wait_front(CB_ER,  1);
                cb_wait_front(CB_EI,  1);
                cb_wait_front(CB_OR,  1);
                cb_wait_front(CB_OI,  1);
                cb_wait_front(CB_TWR, 1);
                cb_wait_front(CB_TWI, 1);

                // ── W_real = tw_r*odd_r − tw_i*odd_i  →  CB_T2 ─────
                // Step 1: tmp0 = tw_r * odd_r
                mulCb(CB_TWR, CB_OR, CB_T0);
                cb_wait_front(CB_T0, 1);
                // Step 2: tmp1 = tw_i * odd_i
                mulCb(CB_TWI, CB_OI, CB_T1);
                cb_wait_front(CB_T1, 1);
                // Step 3: T2 = tmp0 − tmp1
                subCb(CB_T0, CB_T1, CB_T2);
                cb_pop_front(CB_T0, 1);
                cb_pop_front(CB_T1, 1);

                // ── W_imag = tw_r*odd_i + tw_i*odd_r  →  CB_T3 ─────
                // Step 4: tmp0 = tw_r * odd_i
                mulCb(CB_TWR, CB_OI, CB_T0);
                cb_wait_front(CB_T0, 1);
                // Step 5: tmp1 = tw_i * odd_r
                mulCb(CB_TWI, CB_OR, CB_T1);
                cb_wait_front(CB_T1, 1);
                // Step 6: T3 = tmp0 + tmp1
                addCb(CB_T0, CB_T1, CB_T3);
                cb_pop_front(CB_T0, 1);
                cb_pop_front(CB_T1, 1);

                // ── Release twiddle + odd inputs ────────────────────
                cb_pop_front(CB_TWR, 1);
                cb_pop_front(CB_TWI, 1);
                cb_pop_front(CB_OR,  1);
                cb_pop_front(CB_OI,  1);

                // ── Butterfly  (W products now stable in T2, T3) ────
                cb_wait_front(CB_T2, 1);
                cb_wait_front(CB_T3, 1);

                // out_even = even + W  (real and imag)
                addCb(CB_ER, CB_T2, CB_OER);
                addCb(CB_EI, CB_T3, CB_OEI);
                // out_odd  = even − W
                subCb(CB_ER, CB_T2, CB_OOR);
                subCb(CB_EI, CB_T3, CB_OOI);

                // ── Release even inputs and W products ───────────────
                // Must pop AFTER all four butterfly ops have consumed them.
                cb_pop_front(CB_ER, 1);
                cb_pop_front(CB_EI, 1);
                cb_pop_front(CB_T2, 1);
                cb_pop_front(CB_T3, 1);
            }
        }
    }
}