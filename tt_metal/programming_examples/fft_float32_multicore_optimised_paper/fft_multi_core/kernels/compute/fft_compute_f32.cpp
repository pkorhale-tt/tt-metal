// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    const uint32_t numStages = get_arg_val<uint32_t>(0);
    const uint32_t tilesPerRow = get_arg_val<uint32_t>(1);

    constexpr uint32_t cbEvenR = 0;
    constexpr uint32_t cbEvenI = 1;
    constexpr uint32_t cbOddR  = 2;
    constexpr uint32_t cbOddI  = 3;
    constexpr uint32_t cbTwR   = 4;
    constexpr uint32_t cbTwI   = 5;

    constexpr uint32_t cbOut0R = 16;
    constexpr uint32_t cbOut0I = 17;
    constexpr uint32_t cbOut1R = 18;
    constexpr uint32_t cbOut1I = 19;

    constexpr uint32_t cbTmpR  = 20;
    constexpr uint32_t cbTmpI  = 21;

    binary_op_init_common(cbEvenR, cbOddR, cbOut0R);
    add_tiles_init();
    sub_tiles_init();
    mul_tiles_init();

    for (uint32_t stage = 0; stage < numStages; ++stage) {
        for (uint32_t t = 0; t < tilesPerRow; ++t) {
            cb_wait_front(cbEvenR, 1);
            cb_wait_front(cbEvenI, 1);
            cb_wait_front(cbOddR, 1);
            cb_wait_front(cbOddI, 1);
            cb_wait_front(cbTwR, 1);
            cb_wait_front(cbTwI, 1);

            cb_reserve_back(cbTmpR, 1);
            cb_reserve_back(cbTmpI, 1);
            cb_reserve_back(cbOut0R, 1);
            cb_reserve_back(cbOut0I, 1);
            cb_reserve_back(cbOut1R, 1);
            cb_reserve_back(cbOut1I, 1);

            tile_regs_acquire();

            // tmp_r = tw_r * odd_r - tw_i * odd_i
            mul_tiles(cbTwR, cbOddR, 0, 0, 0);
            mul_tiles(cbTwI, cbOddI, 0, 0, 1);
            sub_tiles(0, 1, 0, 0, 2);

            // tmp_i = tw_r * odd_i + tw_i * odd_r
            mul_tiles(cbTwR, cbOddI, 0, 0, 3);
            mul_tiles(cbTwI, cbOddR, 0, 0, 4);
            add_tiles(3, 4, 0, 0, 5);

            pack_tile(2, cbTmpR);
            pack_tile(5, cbTmpI);

            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();

            cb_push_back(cbTmpR, 1);
            cb_push_back(cbTmpI, 1);

            cb_wait_front(cbTmpR, 1);
            cb_wait_front(cbTmpI, 1);

            tile_regs_acquire();

            // out0 = even + tmp
            add_tiles(cbEvenR, cbTmpR, 0, 0, 6);
            add_tiles(cbEvenI, cbTmpI, 0, 0, 7);

            // out1 = even - tmp
            sub_tiles(cbEvenR, cbTmpR, 0, 0, 8);
            sub_tiles(cbEvenI, cbTmpI, 0, 0, 9);

            pack_tile(6, cbOut0R);
            pack_tile(7, cbOut0I);
            pack_tile(8, cbOut1R);
            pack_tile(9, cbOut1I);

            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();

            cb_push_back(cbOut0R, 1);
            cb_push_back(cbOut0I, 1);
            cb_push_back(cbOut1R, 1);
            cb_push_back(cbOut1I, 1);

            cb_pop_front(cbEvenR, 1);
            cb_pop_front(cbEvenI, 1);
            cb_pop_front(cbOddR, 1);
            cb_pop_front(cbOddI, 1);
            cb_pop_front(cbTwR, 1);
            cb_pop_front(cbTwI, 1);
            cb_pop_front(cbTmpR, 1);
            cb_pop_front(cbTmpI, 1);
        }
    }
}