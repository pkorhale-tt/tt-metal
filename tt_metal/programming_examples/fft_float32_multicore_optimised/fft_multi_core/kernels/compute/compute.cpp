// compute.cpp — Tensix compute kernel (Wormhole 1-D FFT, DIT radix-2)
//
// Uses FPU tile ops directly on circular buffers. This avoids the fragile
// scratch-CB round-trip through copy_tile/mul_binary_tile/add_binary_tile that
// was producing incorrect zero outputs.

#include <cstdint>
#include "api/compute/eltwise_binary.h"

inline void binaryOpToCbMul(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    mul_tiles_init(a, b);
    mul_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

inline void binaryOpToCbAdd(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    add_tiles_init(a, b, false);
    add_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

inline void binaryOpToCbSub(uint32_t a, uint32_t b, uint32_t dst) {
    cb_reserve_back(dst, 1);
    tile_regs_acquire();
    sub_tiles_init(a, b, false);
    sub_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dst);
    tile_regs_release();
    cb_push_back(dst, 1);
}

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
    constexpr uint32_t CB_T2  = 12;
    constexpr uint32_t CB_T3  = 13;

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) return;

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            for (uint32_t t = 0; t < tiles_per_stage; ++t) {
                cb_wait_front(CB_ER,  1);
                cb_wait_front(CB_EI,  1);
                cb_wait_front(CB_OR,  1);
                cb_wait_front(CB_OI,  1);
                cb_wait_front(CB_TWR, 1);
                cb_wait_front(CB_TWI, 1);

                // W_real = tw_r*odd_r - tw_i*odd_i
                binaryOpToCbMul(CB_TWR, CB_OR, CB_T0);
                binaryOpToCbMul(CB_TWI, CB_OI, CB_T1);
                cb_wait_front(CB_T0, 1);
                cb_wait_front(CB_T1, 1);
                binaryOpToCbSub(CB_T0, CB_T1, CB_T2);
                cb_pop_front(CB_T0, 1);
                cb_pop_front(CB_T1, 1);

                // W_imag = tw_r*odd_i + tw_i*odd_r
                binaryOpToCbMul(CB_TWR, CB_OI, CB_T0);
                binaryOpToCbMul(CB_TWI, CB_OR, CB_T1);
                cb_wait_front(CB_T0, 1);
                cb_wait_front(CB_T1, 1);
                binaryOpToCbAdd(CB_T0, CB_T1, CB_T3);
                cb_pop_front(CB_T0, 1);
                cb_pop_front(CB_T1, 1);

                cb_pop_front(CB_TWR, 1);
                cb_pop_front(CB_TWI, 1);
                cb_pop_front(CB_OR,  1);
                cb_pop_front(CB_OI,  1);

                cb_wait_front(CB_T2, 1);
                cb_wait_front(CB_T3, 1);

                binaryOpToCbAdd(CB_ER, CB_T2, CB_OER);
                binaryOpToCbAdd(CB_EI, CB_T3, CB_OEI);
                binaryOpToCbSub(CB_ER, CB_T2, CB_OOR);
                binaryOpToCbSub(CB_EI, CB_T3, CB_OOI);

                cb_pop_front(CB_ER, 1);
                cb_pop_front(CB_EI, 1);
                cb_pop_front(CB_T2, 1);
                cb_pop_front(CB_T3, 1);
            }
        }
    }
}
