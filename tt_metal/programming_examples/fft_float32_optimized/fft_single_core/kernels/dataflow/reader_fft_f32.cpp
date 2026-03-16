// reader_fft_f32.cpp  — OPTIMAL
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Single DRAM upload: stage-0 even/odd inputs + ALL twiddles for all stages.
// After this one burst the reader is done. All inter-stage data movement
// happens entirely in L1 (writer kernel), so DRAM is never touched again
// until the final result write.
//
// NOC usage: all reads issued in one burst, one barrier for all of them.
// CB map: 0=even_r, 1=even_i, 2=odd_r, 3=odd_i, 4=tw_r, 5=tw_i

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tw_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(6);  // tiles_per_stage
    const uint32_t num_stages  = get_arg_val<uint32_t>(7);  // log2N

    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    const uint32_t total_tw      = num_stages * num_tiles;

    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_r_gen  = {
        .bank_base_address = odd_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_i_gen  = {
        .bank_base_address = odd_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_r_gen   = {
        .bank_base_address = tw_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_i_gen   = {
        .bank_base_address = tw_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    if (num_tiles == 0 || num_stages == 0) return;

    // Reserve ALL slots upfront — one contiguous reservation per CB.
    // This lets the NOC saturate without any CB stall in the middle.
    cb_reserve_back(cb_even_r, num_tiles);
    cb_reserve_back(cb_even_i, num_tiles);
    cb_reserve_back(cb_odd_r,  num_tiles);
    cb_reserve_back(cb_odd_i,  num_tiles);
    cb_reserve_back(cb_tw_r,   total_tw);
    cb_reserve_back(cb_tw_i,   total_tw);

    // Issue ALL reads in one burst — no barrier inside any loop.
    for (uint32_t t = 0; t < num_tiles; t++) {
        noc_async_read_tile(t, even_r_gen, get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(t, even_i_gen, get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(t, odd_r_gen,  get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(t, odd_i_gen,  get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }
    for (uint32_t t = 0; t < total_tw; t++) {
        noc_async_read_tile(t, tw_r_gen, get_write_ptr(cb_tw_r) + t * tile_bytes);
        noc_async_read_tile(t, tw_i_gen, get_write_ptr(cb_tw_i) + t * tile_bytes);
    }

    // ONE barrier for every read issued above.
    noc_async_read_barrier();

    // Signal compute that all data is ready in L1.
    cb_push_back(cb_even_r, num_tiles);
    cb_push_back(cb_even_i, num_tiles);
    cb_push_back(cb_odd_r,  num_tiles);
    cb_push_back(cb_odd_i,  num_tiles);
    cb_push_back(cb_tw_r,   total_tw);
    cb_push_back(cb_tw_i,   total_tw);
    // Reader is done. All inter-stage data movement is L1-to-L1,
    // handled by the writer after each butterfly stage.
}