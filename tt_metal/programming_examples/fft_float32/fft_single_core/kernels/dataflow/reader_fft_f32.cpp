// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_r_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_i_addr = get_arg_val<uint32_t>(1);
    uint32_t src1_r_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_i_addr = get_arg_val<uint32_t>(3);
    uint32_t tw_r_addr = get_arg_val<uint32_t>(4);
    uint32_t tw_i_addr = get_arg_val<uint32_t>(5);
    uint32_t num_tiles = get_arg_val<uint32_t>(6);
    
    constexpr uint32_t cb_in0_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_in0_i = tt::CBIndex::c_1;
    constexpr uint32_t cb_in1_r = tt::CBIndex::c_2;
    constexpr uint32_t cb_in1_i = tt::CBIndex::c_3;
    constexpr uint32_t cb_tw_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i = tt::CBIndex::c_5;
    
    uint32_t tile_bytes = get_tile_size(cb_in0_r);
    
    const InterleavedAddrGenFast<true> s0_r = {
        .bank_base_address = src0_r_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float32
    };
    const InterleavedAddrGenFast<true> s0_i = {
        .bank_base_address = src0_i_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float32
    };
    const InterleavedAddrGenFast<true> s1_r = {
        .bank_base_address = src1_r_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float32
    };
    const InterleavedAddrGenFast<true> s1_i = {
        .bank_base_address = src1_i_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float32
    };
    const InterleavedAddrGenFast<true> tw_r = {
        .bank_base_address = tw_r_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float32
    };
    const InterleavedAddrGenFast<true> tw_i = {
        .bank_base_address = tw_i_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Float32
    };
    
    for (uint32_t i = 0; i < num_tiles; i++) {
        // LHS
        cb_reserve_back(cb_in0_r, 1);
        cb_reserve_back(cb_in0_i, 1);
        uint32_t l1_in0_r = get_write_ptr(cb_in0_r);
        uint32_t l1_in0_i = get_write_ptr(cb_in0_i);
        noc_async_read_tile(i, s0_r, l1_in0_r);
        noc_async_read_tile(i, s0_i, l1_in0_i);
        noc_async_read_barrier();
        cb_push_back(cb_in0_r, 1);
        cb_push_back(cb_in0_i, 1);
        
        // RHS
        cb_reserve_back(cb_in1_r, 1);
        cb_reserve_back(cb_in1_i, 1);
        uint32_t l1_in1_r = get_write_ptr(cb_in1_r);
        uint32_t l1_in1_i = get_write_ptr(cb_in1_i);
        noc_async_read_tile(i, s1_r, l1_in1_r);
        noc_async_read_tile(i, s1_i, l1_in1_i);
        noc_async_read_barrier();
        cb_push_back(cb_in1_r, 1);
        cb_push_back(cb_in1_i, 1);
        
        // Twiddles
        cb_reserve_back(cb_tw_r, 1);
        cb_reserve_back(cb_tw_i, 1);
        uint32_t l1_tw_r = get_write_ptr(cb_tw_r);
        uint32_t l1_tw_i = get_write_ptr(cb_tw_i);
        noc_async_read_tile(i, tw_r, l1_tw_r);
        noc_async_read_tile(i, tw_i, l1_tw_i);
        noc_async_read_barrier();
        cb_push_back(cb_tw_r, 1);
        cb_push_back(cb_tw_i, 1);
    }
}