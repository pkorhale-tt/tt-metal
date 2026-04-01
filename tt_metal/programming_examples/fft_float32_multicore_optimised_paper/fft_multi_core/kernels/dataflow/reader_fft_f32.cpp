// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PRODUCTION FFT READER KERNEL
// Optimization: Pre-computed twiddle tiles (no scatter)
// Paper: "twiddle factors calculated on initialization"

#include <cstdint>
#include "dataflow_api.h"

void MAIN {
    // Runtime arguments
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t twiddle_r_addr = get_arg_val<uint32_t>(4);  // Pre-computed
    const uint32_t twiddle_i_addr = get_arg_val<uint32_t>(5);  // Pre-computed
    const uint32_t tiles_per_row  = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(9);
    
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;
    
    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    
    // ═══════════════════════════════════════════════════════════
    // Address generators for NOC operations
    // ═══════════════════════════════════════════════════════════
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
        .bank_base_address = twiddle_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = twiddle_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    
    // ═══════════════════════════════════════════════════════════
    // Main loop: Process rows assigned to this core
    // ═══════════════════════════════════════════════════════════
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        
        // ═══════════════════════════════════════════════════════
        // Stage 0: Load initial even/odd data from DRAM
        // Paper: "data movement from compute" - decoupled cores
        // ═══════════════════════════════════════════════════════
        cb_reserve_back(cb_even_r, tiles_per_row);
        cb_reserve_back(cb_even_i, tiles_per_row);
        cb_reserve_back(cb_odd_r,  tiles_per_row);
        cb_reserve_back(cb_odd_i,  tiles_per_row);
        
        uint32_t even_r_ptr = get_write_ptr(cb_even_r);
        uint32_t even_i_ptr = get_write_ptr(cb_even_i);
        uint32_t odd_r_ptr  = get_write_ptr(cb_odd_r);
        uint32_t odd_i_ptr  = get_write_ptr(cb_odd_i);
        
        // OPTIMIZATION: Batch NOC reads (not one-by-one)
        for (uint32_t t = 0; t < tiles_per_row; t++) {
            const uint32_t tile_id = row_tile_base + t;
            
            noc_async_read_tile(tile_id, even_r_gen, even_r_ptr);
            noc_async_read_tile(tile_id, even_i_gen, even_i_ptr);
            noc_async_read_tile(tile_id, odd_r_gen,  odd_r_ptr);
            noc_async_read_tile(tile_id, odd_i_gen,  odd_i_ptr);
            
            even_r_ptr += tile_bytes;
            even_i_ptr += tile_bytes;
            odd_r_ptr  += tile_bytes;
            odd_i_ptr  += tile_bytes;
        }
        
        noc_async_read_barrier();  // Single barrier for all reads
        
        cb_push_back(cb_even_r, tiles_per_row);
        cb_push_back(cb_even_i, tiles_per_row);
        cb_push_back(cb_odd_r,  tiles_per_row);
        cb_push_back(cb_odd_i,  tiles_per_row);
        
        // ═══════════════════════════════════════════════════════
        // All stages: Load pre-computed twiddle tiles
        // CRITICAL OPTIMIZATION: No scatter - bulk NOC read
        // Paper insight: This eliminates the 2x bottleneck
        // ═══════════════════════════════════════════════════════
        for (uint32_t stage = 0; stage < num_stages; stage++) {
            // Calculate twiddle tile index for this stage and row
            const uint32_t twiddle_tile_base = 
                (stage * rows_per_core + row) * tiles_per_row;
            
            cb_reserve_back(cb_tw_r, tiles_per_row);
            cb_reserve_back(cb_tw_i, tiles_per_row);
            
            uint32_t tw_r_ptr = get_write_ptr(cb_tw_r);
            uint32_t tw_i_ptr = get_write_ptr(cb_tw_i);
            
            // Bulk read pre-computed twiddle tiles
            for (uint32_t t = 0; t < tiles_per_row; t++) {
                const uint32_t tw_tile_id = twiddle_tile_base + t;
                
                noc_async_read_tile(tw_tile_id, tw_r_gen, tw_r_ptr);
                noc_async_read_tile(tw_tile_id, tw_i_gen, tw_i_ptr);
                
                tw_r_ptr += tile_bytes;
                tw_i_ptr += tile_bytes;
            }
            
            noc_async_read_barrier();
            
            cb_push_back(cb_tw_r, tiles_per_row);
            cb_push_back(cb_tw_i, tiles_per_row);
        }
    }
}