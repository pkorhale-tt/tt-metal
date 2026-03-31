// writer_fft_1d_64core.cpp - FIXED with cross-core exchange
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr    = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr    = get_arg_val<uint32_t>(3);
    const uint32_t local_tiles    = get_arg_val<uint32_t>(4);
    const uint32_t num_stages     = get_arg_val<uint32_t>(5);
    const uint32_t local_half     = get_arg_val<uint32_t>(6);
    const uint32_t half_N         = get_arg_val<uint32_t>(7);
    const uint32_t num_cores      = get_arg_val<uint32_t>(8);
    const uint32_t core_id        = get_arg_val<uint32_t>(9);
    const uint32_t log2_cores     = get_arg_val<uint32_t>(10);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(11);
    const uint32_t core_elem_base = get_arg_val<uint32_t>(12);
    const uint32_t local_stages   = get_arg_val<uint32_t>(13);
    
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_recv_r = 24;
    constexpr uint32_t cb_recv_i = 25;
    constexpr uint32_t cb_sync   = 28;
    
    const uint32_t tile_bytes = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);
    constexpr uint32_t ELEM = sizeof(float);
    constexpr uint32_t TILE_SIZE = 1024;
    
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
    
    auto rd = [](uint32_t addr) -> float {
        uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
        float v; __builtin_memcpy(&v, &raw, 4); return v;
    };
    
    auto wr = [](uint32_t addr, float v) {
        uint32_t raw; __builtin_memcpy(&raw, &v, 4);
        *reinterpret_cast<volatile uint32_t*>(addr) = raw;
    };
    
    // Convert core_id to NOC coordinates (8x8 grid)
    auto get_noc_coords = [](uint32_t cid) -> uint64_t {
        uint32_t x = cid % 8;
        uint32_t y = cid / 8;
        return NOC_XY_ADDR(x, y, 0);
    };
    
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);
        const bool is_cross_core = (stage >= local_stages);
        
        cb_wait_front(cb_out0_r, local_tiles);
        cb_wait_front(cb_out0_i, local_tiles);
        cb_wait_front(cb_out1_r, local_tiles);
        cb_wait_front(cb_out1_i, local_tiles);
        
        const uint32_t src0r = get_read_ptr(cb_out0_r);
        const uint32_t src0i = get_read_ptr(cb_out0_i);
        const uint32_t src1r = get_read_ptr(cb_out1_r);
        const uint32_t src1i = get_read_ptr(cb_out1_i);
        
        if (is_last) {
            // ═══════════════════════════════════════════════════
            // FINAL STAGE: Write to DRAM
            // ═══════════════════════════════════════════════════
            for (uint32_t t = 0; t < local_tiles; t++) {
                uint32_t gt = tile_offset + t;
                noc_async_write_tile(gt, out0_r_gen, src0r + t * tile_bytes);
                noc_async_write_tile(gt, out0_i_gen, src0i + t * tile_bytes);
                noc_async_write_tile(gt, out1_r_gen, src1r + t * tile_bytes);
                noc_async_write_tile(gt, out1_i_gen, src1i + t * tile_bytes);
            }
            noc_async_write_barrier();
            
            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);
            
        } else if (is_cross_core) {
            // ═══════════════════════════════════════════════════
            // CROSS-CORE STAGE: Exchange data with partner
            // ═══════════════════════════════════════════════════
            const uint32_t stage_bit = stage - local_stages;
            const uint32_t partner_core = core_id ^ (1u << stage_bit);
            const uint64_t partner_noc = get_noc_coords(partner_core);
            
            const uint32_t half_size = local_half / 2;
            const uint32_t half_bytes = half_size * ELEM;
            
            // Determine which half to keep and which to exchange
            const bool keep_lower = ((core_id >> stage_bit) & 1) == 0;
            
            // Reserve receive buffers
            cb_reserve_back(cb_recv_r, local_tiles);
            cb_reserve_back(cb_recv_i, local_tiles);
            
            const uint32_t recv_r = get_write_ptr(cb_recv_r);
            const uint32_t recv_i = get_write_ptr(cb_recv_i);
            
            if (keep_lower) {
                // Keep out0 (lower half), exchange out1 (upper half)
                // Send upper half of out1 to partner
                noc_async_write(src1r + half_size * ELEM, 
                               partner_noc | recv_r,
                               half_bytes);
                noc_async_write(src1i + half_size * ELEM,
                               partner_noc | recv_i,
                               half_bytes);
                
                // Receive partner's lower half of out1
                noc_async_read(partner_noc | (src1r + 0),
                              recv_r + half_size * ELEM,
                              half_bytes);
                noc_async_read(partner_noc | (src1i + 0),
                              recv_i + half_size * ELEM,
                              half_bytes);
                
            } else {
                // Keep out1 (upper half), exchange out0 (lower half)
                // Send lower half of out0 to partner
                noc_async_write(src0r,
                               partner_noc | recv_r,
                               half_bytes);
                noc_async_write(src0i,
                               partner_noc | recv_i,
                               half_bytes);
                
                // Receive partner's upper half of out0
                noc_async_read(partner_noc | (src0r + half_size * ELEM),
                              recv_r + half_size * ELEM,
                              half_bytes);
                noc_async_read(partner_noc | (src0i + half_size * ELEM),
                              recv_i + half_size * ELEM,
                              half_bytes);
            }
            
            noc_async_read_barrier();
            noc_async_write_barrier();
            
            // Synchronization barrier
            volatile uint32_t* sync_ptr = reinterpret_cast<volatile uint32_t*>(
                get_write_ptr(cb_sync));
            *sync_ptr = 1;
            noc_semaphore_inc(partner_noc | (uint32_t)sync_ptr, 1);
            while (*sync_ptr < 2) {}  // Wait for partner
            *sync_ptr = 0;  // Reset
            
            cb_push_back(cb_recv_r, local_tiles);
            cb_push_back(cb_recv_i, local_tiles);
            
            // Prepare next stage inputs
            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r, local_tiles);
            cb_reserve_back(cb_odd_i, local_tiles);
            
            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);
            
            cb_wait_front(cb_recv_r, local_tiles);
            cb_wait_front(cb_recv_i, local_tiles);
            
            // Interleave local and received data for next stage
            const uint32_t m = 1u << (stage + 1);
            for (uint32_t i = 0; i < half_size; i++) {
                uint32_t dst_idx = i * 2;
                
                if (keep_lower) {
                    // Even: local out0, Odd: received
                    wr(dst_er + dst_idx * ELEM, rd(src0r + i * ELEM));
                    wr(dst_ei + dst_idx * ELEM, rd(src0i + i * ELEM));
                    wr(dst_or + dst_idx * ELEM, rd(recv_r + i * ELEM));
                    wr(dst_oi + dst_idx * ELEM, rd(recv_i + i * ELEM));
                } else {
                    // Even: received, Odd: local out1
                    wr(dst_er + dst_idx * ELEM, rd(recv_r + i * ELEM));
                    wr(dst_ei + dst_idx * ELEM, rd(recv_i + i * ELEM));
                    wr(dst_or + dst_idx * ELEM, rd(src1r + i * ELEM));
                    wr(dst_oi + dst_idx * ELEM, rd(src1i + i * ELEM));
                }
            }
            
            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);
            cb_pop_front(cb_recv_r, local_tiles);
            cb_pop_front(cb_recv_i, local_tiles);
            
            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r, local_tiles);
            cb_push_back(cb_odd_i, local_tiles);
            
        } else {
            // ═══════════════════════════════════════════════════
            // LOCAL STAGE: Shuffle within core
            // ═══════════════════════════════════════════════════
            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            
            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r, local_tiles);
            cb_reserve_back(cb_odd_i, local_tiles);
            
            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);
            
            // Improved shuffle logic
            const uint32_t num_groups = local_half / m;
            const uint32_t log2m = stage + 1;
            const uint32_t m_mask = m - 1;
            
            for (uint32_t g = 0; g < num_groups; g++) {
                for (uint32_t j = 0; j < half_m; j++) {
                    uint32_t dst_idx = g * m + j;
                    
                    // Map to source in out0/out1
                    uint32_t global_pos = core_elem_base + dst_idx;
                    uint32_t old_group = global_pos >> log2m;
                    uint32_t offset = global_pos & m_mask;
                    
                    uint32_t src_idx = old_group * half_m + offset;
                    uint32_t local_src = src_idx >= core_elem_base ? 
                                        src_idx - core_elem_base : 0;
                    
                    if (local_src < local_half) {
                        bool from_out0 = (offset < half_m);
                        uint32_t srcr = from_out0 ? src0r : src1r;
                        uint32_t srci = from_out0 ? src0i : src1i;
                        
                        wr(dst_er + dst_idx * ELEM, rd(srcr + local_src * ELEM));
                        wr(dst_ei + dst_idx * ELEM, rd(srci + local_src * ELEM));
                    }
                    
                    // Odd part (offset by half_m in global space)
                    dst_idx = g * m + half_m + j;
                    global_pos = core_elem_base + dst_idx;
                    old_group = global_pos >> log2m;
                    offset = global_pos & m_mask;
                    
                    src_idx = old_group * half_m + offset;
                    local_src = src_idx >= core_elem_base ? 
                               src_idx - core_elem_base : 0;
                    
                    if (local_src < local_half) {
                        bool from_out0 = (offset < half_m);
                        uint32_t srcr = from_out0 ? src0r : src1r;
                        uint32_t srci = from_out0 ? src0i : src1i;
                        
                        wr(dst_or + (g * half_m + j) * ELEM, 
                           rd(srcr + local_src * ELEM));
                        wr(dst_oi + (g * half_m + j) * ELEM,
                           rd(srci + local_src * ELEM));
                    }
                }
            }
            
            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);
            
            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r, local_tiles);
            cb_push_back(cb_odd_i, local_tiles);
        }
    }
}