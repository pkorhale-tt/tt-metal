// writer_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_BYTES = 32 * 32 * sizeof(float);  // 4096

void kernel_main() {

    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    if (num_tiles == 0) return;

    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t off = t * TILE_BYTES;

        cb_wait_front(cb_out0_r, 1);
        cb_wait_front(cb_out0_i, 1);
        cb_wait_front(cb_out1_r, 1);
        cb_wait_front(cb_out1_i, 1);

        noc_async_write(get_read_ptr(cb_out0_r), get_noc_addr(out0_r_addr + off), TILE_BYTES);
        noc_async_write(get_read_ptr(cb_out0_i), get_noc_addr(out0_i_addr + off), TILE_BYTES);
        noc_async_write(get_read_ptr(cb_out1_r), get_noc_addr(out1_r_addr + off), TILE_BYTES);
        noc_async_write(get_read_ptr(cb_out1_i), get_noc_addr(out1_i_addr + off), TILE_BYTES);

        noc_async_write_barrier();

        cb_pop_front(cb_out0_r, 1);
        cb_pop_front(cb_out0_i, 1);
        cb_pop_front(cb_out1_r, 1);
        cb_pop_front(cb_out1_i, 1);
    }
}