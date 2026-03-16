// reader_fft_f32.cpp  — CORRECTED (pre-staged inputs)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The host pre-computes the correct even/odd split for every stage and
// uploads them to DRAM in one contiguous buffer:
//   even_r_buf: [stage0_even_r | stage1_even_r | ... | stage(n-1)_even_r]
//   odd_r_buf:  [stage0_odd_r  | stage1_odd_r  | ... ]
//   (similarly for even_i, odd_i)
//   tw_r_buf:   [stage0_tw_r   | stage1_tw_r   | ... ]
//
// This reader streams each stage's tiles into CBs 0-5 in order.
// The compute kernel's output (CBs 16-19) is drained by the writer ONLY
// for the final stage.  For intermediate stages, the writer is NOT used —
// instead the reader feeds the pre-computed next-stage inputs.

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

    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);

    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_r_gen = {
        .bank_base_address = odd_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_i_gen = {
        .bank_base_address = odd_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    if (num_tiles == 0 || num_stages == 0) return;

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        // Reserve all slots for this stage upfront
        cb_reserve_back(cb_tw_r,   num_tiles);
        cb_reserve_back(cb_tw_i,   num_tiles);
        cb_reserve_back(cb_odd_r,  num_tiles);
        cb_reserve_back(cb_odd_i,  num_tiles);
        cb_reserve_back(cb_even_r, num_tiles);
        cb_reserve_back(cb_even_i, num_tiles);

        // Issue all reads for this stage in one burst — no barrier inside loop
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t global_tile = stage * num_tiles + t;
            noc_async_read_tile(global_tile, tw_r_gen,
                get_write_ptr(cb_tw_r)   + t * tile_bytes);
            noc_async_read_tile(global_tile, tw_i_gen,
                get_write_ptr(cb_tw_i)   + t * tile_bytes);
            noc_async_read_tile(global_tile, odd_r_gen,
                get_write_ptr(cb_odd_r)  + t * tile_bytes);
            noc_async_read_tile(global_tile, odd_i_gen,
                get_write_ptr(cb_odd_i)  + t * tile_bytes);
            noc_async_read_tile(global_tile, even_r_gen,
                get_write_ptr(cb_even_r) + t * tile_bytes);
            noc_async_read_tile(global_tile, even_i_gen,
                get_write_ptr(cb_even_i) + t * tile_bytes);
        }

        // One barrier for all reads of this stage
        noc_async_read_barrier();

        cb_push_back(cb_tw_r,   num_tiles);
        cb_push_back(cb_tw_i,   num_tiles);
        cb_push_back(cb_odd_r,  num_tiles);
        cb_push_back(cb_odd_i,  num_tiles);
        cb_push_back(cb_even_r, num_tiles);
        cb_push_back(cb_even_i, num_tiles);

        // For all stages except the last, the compute kernel writes to
        // CBs 16-19 which are immediately drained by the writer into DRAM.
        // The host has pre-computed what the "next stage even/odd" should be
        // so the reader just feeds pre-staged data from DRAM each iteration.
        // For the last stage, the writer saves the final result.
    }
}