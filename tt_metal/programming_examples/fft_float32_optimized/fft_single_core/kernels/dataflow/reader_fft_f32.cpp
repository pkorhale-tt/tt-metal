// reader_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {

    const uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tw_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(6);
    const uint32_t num_stages  = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);

    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr, .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr, .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_r_gen = {
        .bank_base_address = odd_r_addr,  .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_i_gen = {
        .bank_base_address = odd_i_addr,  .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr,   .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr,   .page_size = tile_bytes, .data_format = data_format };

    if (num_tiles == 0 || num_stages == 0) return;

    // ── Phase 1: stage 0 data + twiddles ──────────────────────────
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_reserve_back(cb_tw_r,   1);
        cb_reserve_back(cb_tw_i,   1);
        cb_reserve_back(cb_odd_r,  1);
        cb_reserve_back(cb_odd_i,  1);
        cb_reserve_back(cb_even_r, 1);
        cb_reserve_back(cb_even_i, 1);

        noc_async_read_tile(t, tw_r_gen,  get_write_ptr(cb_tw_r));
        noc_async_read_tile(t, tw_i_gen,  get_write_ptr(cb_tw_i));
        noc_async_read_tile(t, odd_r_gen, get_write_ptr(cb_odd_r));
        noc_async_read_tile(t, odd_i_gen, get_write_ptr(cb_odd_i));
        noc_async_read_tile(t, even_r_gen,get_write_ptr(cb_even_r));
        noc_async_read_tile(t, even_i_gen,get_write_ptr(cb_even_i));
        noc_async_read_barrier();

        cb_push_back(cb_tw_r,   1);
        cb_push_back(cb_tw_i,   1);
        cb_push_back(cb_odd_r,  1);
        cb_push_back(cb_odd_i,  1);
        cb_push_back(cb_even_r, 1);
        cb_push_back(cb_even_i, 1);
    }

    // ── Phase 2: twiddles for stages 1..num_stages-1 ──────────────
    for (uint32_t stage = 1; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t global_tile = stage * num_tiles + t;

            cb_reserve_back(cb_tw_r, 1);
            cb_reserve_back(cb_tw_i, 1);

            noc_async_read_tile(global_tile, tw_r_gen, get_write_ptr(cb_tw_r));
            noc_async_read_tile(global_tile, tw_i_gen, get_write_ptr(cb_tw_i));
            noc_async_read_barrier();

            cb_push_back(cb_tw_r, 1);
            cb_push_back(cb_tw_i, 1);
        }
    }
}