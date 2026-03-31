// reader_fft_1d_64core.cpp - FIXED
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr     = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr     = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr      = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr      = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr  = get_arg_val<uint32_t>(4);
    const uint32_t compact_i_addr  = get_arg_val<uint32_t>(5);
    const uint32_t local_tiles     = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset     = get_arg_val<uint32_t>(7);
    const uint32_t num_stages      = get_arg_val<uint32_t>(8);
    const uint32_t half_N          = get_arg_val<uint32_t>(9);
    const uint32_t local_half      = get_arg_val<uint32_t>(10);
    const uint32_t core_elem_base  = get_arg_val<uint32_t>(11);
    const uint32_t core_id         = get_arg_val<uint32_t>(12);
    const uint32_t num_cores       = get_arg_val<uint32_t>(13);
    const uint32_t log2_cores      = get_arg_val<uint32_t>(14);
    const uint32_t local_stages    = get_arg_val<uint32_t>(15);
    
    constexpr uint32_t cb_even_r     = 0;
    constexpr uint32_t cb_even_i     = 1;
    constexpr uint32_t cb_odd_r      = 2;
    constexpr uint32_t cb_odd_i      = 3;
    constexpr uint32_t cb_tw_r       = 4;
    constexpr uint32_t cb_tw_i       = 5;
    constexpr uint32_t cb_compact_r  = 10;
    constexpr uint32_t cb_compact_i  = 11;
    
    const uint32_t tile_bytes = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    constexpr uint32_t ELEM = sizeof(float);
    
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
    
    // ─────── Phase 1: Load initial data ───────
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);
    
    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t global_t = tile_offset + t;
        noc_async_read_tile(global_t, even_r_gen,
            get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(global_t, even_i_gen,
            get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(global_t, odd_r_gen,
            get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(global_t, odd_i_gen,
            get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }
    
    // Load compact twiddle table
    const uint32_t compact_bytes = half_N * ELEM;
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read(compact_r_addr, get_write_ptr(cb_compact_r), compact_bytes);
    noc_async_read(compact_i_addr, get_write_ptr(cb_compact_i), compact_bytes);
    
    noc_async_read_barrier();
    
    cb_push_back(cb_even_r, local_tiles);
    cb_push_back(cb_even_i, local_tiles);
    cb_push_back(cb_odd_r,  local_tiles);
    cb_push_back(cb_odd_i,  local_tiles);
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);
    
    // Get compact twiddle pointers
    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);
    
    // ─────── Phase 2: Generate twiddles for ALL stages (FIXED) ───────
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;
        const uint32_t half_m_mask = half_m - 1u;
        
        cb_reserve_back(cb_tw_r, local_tiles);
        cb_reserve_back(cb_tw_i, local_tiles);
        
        const uint32_t dst_r = get_write_ptr(cb_tw_r);
        const uint32_t dst_i = get_write_ptr(cb_tw_i);
        
        // FIXED: Use consistent global indexing for all stages
        for (uint32_t lp = 0; lp < local_half; lp++) {
            // Global element index (contiguous distribution)
            uint32_t global_elem = core_elem_base + lp;
            
            // Twiddle index based on position within butterfly group
            uint32_t j   = global_elem & half_m_mask;
            uint32_t idx = j * N_over_m;
            
            // Clamp to valid range
            if (idx >= half_N) idx = 0;
            
            uint32_t raw_r = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_r_base + idx * ELEM);
            uint32_t raw_i = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_i_base + idx * ELEM);
            
            *reinterpret_cast<volatile uint32_t*>(dst_r + lp * ELEM) = raw_r;
            *reinterpret_cast<volatile uint32_t*>(dst_i + lp * ELEM) = raw_i;
        }
        
        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }
    
    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}