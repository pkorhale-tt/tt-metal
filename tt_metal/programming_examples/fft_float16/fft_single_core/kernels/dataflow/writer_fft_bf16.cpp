// writer_fft_bf16.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t start_tile = get_arg_val<uint32_t>(5);
    
    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;
    
    uint32_t tile_bytes = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);
    
    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr, .page_size = tile_bytes, .data_format = data_format
    };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr, .page_size = tile_bytes, .data_format = data_format
    };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr, .page_size = tile_bytes, .data_format = data_format
    };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr, .page_size = tile_bytes, .data_format = data_format
    };
    
    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_idx = start_tile + i;
        
        cb_wait_front(cb_out0_r, 1);
        uint32_t l1_out0_r = get_read_ptr(cb_out0_r);
        noc_async_write_tile(tile_idx, out0_r_gen, l1_out0_r);
        noc_async_write_barrier();
        cb_pop_front(cb_out0_r, 1);
        
        cb_wait_front(cb_out0_i, 1);
        uint32_t l1_out0_i = get_read_ptr(cb_out0_i);
        noc_async_write_tile(tile_idx, out0_i_gen, l1_out0_i);
        noc_async_write_barrier();
        cb_pop_front(cb_out0_i, 1);
        
        cb_wait_front(cb_out1_r, 1);
        uint32_t l1_out1_r = get_read_ptr(cb_out1_r);
        noc_async_write_tile(tile_idx, out1_r_gen, l1_out1_r);
        noc_async_write_barrier();
        cb_pop_front(cb_out1_r, 1);
        
        cb_wait_front(cb_out1_i, 1);
        uint32_t l1_out1_i = get_read_ptr(cb_out1_i);
        noc_async_write_tile(tile_idx, out1_i_gen, l1_out1_i);
        noc_async_write_barrier();
        cb_pop_front(cb_out1_i, 1);
    }
}
