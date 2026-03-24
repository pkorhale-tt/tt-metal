// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PRODUCTION FFT WRITER KERNEL
// Optimization: Minimize shuffle overhead with DMA operations

#include <cstdint>
#include "dataflow_api.h"

void MAIN {
    const uint32_t out0_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr   = get_arg_val<uint32_t>(3);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t tile_offset   = get_arg_val<uint32_t>(6);
    const uint32_t rows_per_core = get_arg_val<uint32_t>(7);
    
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;   // For shuffle
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    
    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);
    
    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    
    // ═══════════════════════════════════════════════════════════
    // Helper: 128-bit DMA copy (4 floats at a time)
    // Paper optimization: "128-bit copies"
    // ═══════════════════════════════════════════════════════════
    auto copy_128bit = [](uint32_t dst, uint32_t src, uint32_t num_elements) {
        constexpr uint32_t CHUNK = 4;  // 4 floats = 128 bits
        for (uint32_t i = 0; i < num_elements; i += CHUNK) {
            uint64_t data_lo = *reinterpret_cast<volatile uint64_t*>(src + i * 4);
            uint64_t data_hi = *reinterpret_cast<volatile uint64_t*>(src + (i+2) * 4);
            *reinterpret_cast<volatile uint64_t*>(dst + i * 4) = data_lo;
            *reinterpret_cast<volatile uint64_t*>(dst + (i+2) * 4) = data_hi;
        }
    };
    
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        
        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const bool is_last_stage = (stage == num_stages - 1);
            
            cb_wait_front(cb_out0_r, tiles_per_row);
            cb_wait_front(cb_out0_i, tiles_per_row);
            cb_wait_front(cb_out1_r, tiles_per_row);
            cb_wait_front(cb_out1_i, tiles_per_row);
            
            const uint32_t src0_r = get_read_ptr(cb_out0_r);
            const uint32_t src0_i = get_read_ptr(cb_out0_i);
            const uint32_t src1_r = get_read_ptr(cb_out1_r);
            const uint32_t src1_i = get_read_ptr(cb_out1_i);
            
            if (is_last_stage) {
                // ═══════════════════════════════════════════════
                // Final stage: Write results to DRAM
                // ═══════════════════════════════════════════════
                for (uint32_t t = 0; t < tiles_per_row; t++) {
                    const uint32_t tile_id = row_tile_base + t;
                    
                    noc_async_write_tile(tile_id, out0_r_gen, src0_r + t * tile_bytes);
                    noc_async_write_tile(tile_id, out0_i_gen, src0_i + t * tile_bytes);
                    noc_async_write_tile(tile_id, out1_r_gen, src1_r + t * tile_bytes);
                    noc_async_write_tile(tile_id, out1_i_gen, src1_i + t * tile_bytes);
                }
                
                noc_async_write_barrier();
                
                cb_pop_front(cb_out0_r, tiles_per_row);
                cb_pop_front(cb_out0_i, tiles_per_row);
                cb_pop_front(cb_out1_r, tiles_per_row);
                cb_pop_front(cb_out1_i, tiles_per_row);
                
            } else {
                // ═══════════════════════════════════════════════
                // Intermediate stage: Shuffle for next butterfly
                // OPTIMIZATION: Use 128-bit DMA instead of scalar
                // ═══════════════════════════════════════════════
                
                cb_pop_front(cb_out0_r, tiles_per_row);
                cb_pop_front(cb_out0_i, tiles_per_row);
                cb_pop_front(cb_out1_r, tiles_per_row);
                cb_pop_front(cb_out1_i, tiles_per_row);
                
                cb_reserve_back(cb_even_r, tiles_per_row);
                cb_reserve_back(cb_even_i, tiles_per_row);
                cb_reserve_back(cb_odd_r,  tiles_per_row);
                cb_reserve_back(cb_odd_i,  tiles_per_row);
                
                const uint32_t dst_even_r = get_write_ptr(cb_even_r);
                const uint32_t dst_even_i = get_write_ptr(cb_even_i);
                const uint32_t dst_odd_r  = get_write_ptr(cb_odd_r);
                const uint32_t dst_odd_i  = get_write_ptr(cb_odd_i);
                
                const uint32_t elements_per_tile = tile_bytes / sizeof(float);
                const uint32_t total_elements = tiles_per_row * elements_per_tile;
                
                // Use 128-bit chunked copy
                copy_128bit(dst_even_r, src0_r, total_elements);
                copy_128bit(dst_even_i, src0_i, total_elements);
                copy_128bit(dst_odd_r,  src1_r, total_elements);
                copy_128bit(dst_odd_i,  src1_i, total_elements);
                
                cb_push_back(cb_even_r, tiles_per_row);
                cb_push_back(cb_even_i, tiles_per_row);
                cb_push_back(cb_odd_r,  tiles_per_row);
                cb_push_back(cb_odd_i,  tiles_per_row);
            }
        }
    }
}