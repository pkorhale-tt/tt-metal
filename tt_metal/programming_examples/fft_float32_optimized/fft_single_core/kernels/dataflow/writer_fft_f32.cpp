// writer_fft_f32.cpp  — CORRECTED (pre-staged approach)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The writer drains all num_stages * num_tiles outputs from CBs 16-19.
// For intermediate stages (0..num_stages-2) the compute output is discarded
// by the host (it pre-computed the correct next-stage inputs anyway).
// For the last stage the outputs are the final FFT result.
// The writer does not distinguish between stages — it just drains all tiles.
// The host knows which tiles contain the final result.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t out_r2_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_i2_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);
    const uint32_t num_stages  = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out_r  = 16;
    constexpr uint32_t cb_out_i  = 17;
    constexpr uint32_t cb_out_r2 = 18;
    constexpr uint32_t cb_out_i2 = 19;

    const uint32_t tile_bytes    = get_tile_size(cb_out_r);
    const DataFormat data_format = get_dataformat(cb_out_r);

    const InterleavedAddrGenFast<true> out_r_gen = {
        .bank_base_address = out_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out_i_gen = {
        .bank_base_address = out_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out_r2_gen = {
        .bank_base_address = out_r2_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out_i2_gen = {
        .bank_base_address = out_i2_addr,
        .page_size = tile_bytes, .data_format = data_format };

    if (num_tiles == 0) return;

    const uint32_t total_tiles = num_stages * num_tiles;

    // Drain all stage outputs — only the last num_tiles tiles are the FFT result.
    // We write all of them to DRAM (only the last ones are read back by the host).
    cb_wait_front(cb_out_r,  total_tiles);
    cb_wait_front(cb_out_i,  total_tiles);
    cb_wait_front(cb_out_r2, total_tiles);
    cb_wait_front(cb_out_i2, total_tiles);

    for (uint32_t t = 0; t < total_tiles; t++) {
        noc_async_write_tile(t, out_r_gen,
            get_read_ptr(cb_out_r)  + t * tile_bytes);
        noc_async_write_tile(t, out_i_gen,
            get_read_ptr(cb_out_i)  + t * tile_bytes);
        noc_async_write_tile(t, out_r2_gen,
            get_read_ptr(cb_out_r2) + t * tile_bytes);
        noc_async_write_tile(t, out_i2_gen,
            get_read_ptr(cb_out_i2) + t * tile_bytes);
    }

    noc_async_write_barrier();

    cb_pop_front(cb_out_r,  total_tiles);
    cb_pop_front(cb_out_i,  total_tiles);
    cb_pop_front(cb_out_r2, total_tiles);
    cb_pop_front(cb_out_i2, total_tiles);
}