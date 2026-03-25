// compute.cpp — Tensix compute kernel
// Wormhole 1-D FFT  (DIT radix-2)
// CB map:
//   0  even_r input   (stage 0: reader,  stage 1+: writer feedback)
//   1  even_i input
//   2  odd_r  input
//   3  odd_i  input
//   4  tw_r   twiddle real  (reader, all stages)
//   5  tw_i   twiddle imag
//   6  out_even_r  → writer
//   7  out_even_i  → writer
//   8  out_odd_r   → writer
//   9  out_odd_i   → writer
//  10-13  scratch depth=1
// Args: [0] num_stages  [1] tiles_per_stage  [2] rows_per_core

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

inline void mulCb(uint32_t a, uint32_t b, uint32_t dst) {
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

inline void addCb(uint32_t a, uint32_t b, uint32_t dst) {
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

inline void subCb(uint32_t a, uint32_t b, uint32_t dst) {
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

    // Stage-0 inputs from reader; stage 1+ feedback from writer — same CB indices
    constexpr uint32_t CB_ER  = 0;   // even real
    constexpr uint32_t CB_EI  = 1;   // even imag
    constexpr uint32_t CB_OR  = 2;   // odd  real
    constexpr uint32_t CB_OI  = 3;   // odd  imag
    constexpr uint32_t CB_TWR = 4;   // twiddle real
    constexpr uint32_t CB_TWI = 5;   // twiddle imag
    constexpr uint32_t CB_OER = 6;   // output even real
    constexpr uint32_t CB_OEI = 7;   // output even imag
    constexpr uint32_t CB_OOR = 8;   // output odd  real
    constexpr uint32_t CB_OOI = 9;   // output odd  imag
    constexpr uint32_t CB_T0  = 10;  // scratch
    constexpr uint32_t CB_T1  = 11;  // scratch
    constexpr uint32_t CB_T2  = 12;  // scratch: W_real = tw_r*odd_r - tw_i*odd_i
    constexpr uint32_t CB_T3  = 13;  // scratch: W_imag = tw_r*odd_i + tw_i*odd_r

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) return;

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            for (uint32_t t = 0; t < tiles_per_stage; ++t) {

                // Wait for all 6 inputs
                cb_wait_front(CB_ER,  1);
                cb_wait_front(CB_EI,  1);
                cb_wait_front(CB_OR,  1);
                cb_wait_front(CB_OI,  1);
                cb_wait_front(CB_TWR, 1);
                cb_wait_front(CB_TWI, 1);

                // W_real = tw_r*odd_r - tw_i*odd_i → CB_T2
                mulCb(CB_TWR, CB_OR, CB_T0);
                cb_wait_front(CB_T0, 1);
                mulCb(CB_TWI, CB_OI, CB_T1);
                cb_wait_front(CB_T1, 1);
                subCb(CB_T0, CB_T1, CB_T2);
                cb_pop_front(CB_T0, 1);
                cb_pop_front(CB_T1, 1);

                // W_imag = tw_r*odd_i + tw_i*odd_r → CB_T3
                mulCb(CB_TWR, CB_OI, CB_T0);
                cb_wait_front(CB_T0, 1);
                mulCb(CB_TWI, CB_OR, CB_T1);
                cb_wait_front(CB_T1, 1);
                addCb(CB_T0, CB_T1, CB_T3);
                cb_pop_front(CB_T0, 1);
                cb_pop_front(CB_T1, 1);

                // Done with twiddles and odd inputs
                cb_pop_front(CB_TWR, 1);
                cb_pop_front(CB_TWI, 1);
                cb_pop_front(CB_OR,  1);
                cb_pop_front(CB_OI,  1);

                // Butterfly — W products now ready in CB_T2, CB_T3
                cb_wait_front(CB_T2, 1);
                cb_wait_front(CB_T3, 1);

                // out_even = even + W*odd
                addCb(CB_ER, CB_T2, CB_OER);
                addCb(CB_EI, CB_T3, CB_OEI);
                // out_odd  = even - W*odd
                subCb(CB_ER, CB_T2, CB_OOR);
                subCb(CB_EI, CB_T3, CB_OOI);

                // Done with even inputs and W products
                cb_pop_front(CB_ER, 1);
                cb_pop_front(CB_EI, 1);
                cb_pop_front(CB_T2, 1);
                cb_pop_front(CB_T3, 1);
            }
        }
    }
}