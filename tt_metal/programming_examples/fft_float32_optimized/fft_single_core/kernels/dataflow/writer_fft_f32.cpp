// writer_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);

    // CB indices
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    if (num_tiles == 0) {
        return;
    }

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(cb_out0_r, 1);
        cb_wait_front(cb_out0_i, 1);
        cb_wait_front(cb_out1_r, 1);
        cb_wait_front(cb_out1_i, 1);

        noc_async_write_tile(t, out0_r_gen, get_read_ptr(cb_out0_r));
        noc_async_write_tile(t, out0_i_gen, get_read_ptr(cb_out0_i));
        noc_async_write_tile(t, out1_r_gen, get_read_ptr(cb_out1_r));
        noc_async_write_tile(t, out1_i_gen, get_read_ptr(cb_out1_i));

        noc_async_write_barrier();

        cb_pop_front(cb_out0_r, 1);
        cb_pop_front(cb_out0_i, 1);
        cb_pop_front(cb_out1_r, 1);
        cb_pop_front(cb_out1_i, 1);
    }
}