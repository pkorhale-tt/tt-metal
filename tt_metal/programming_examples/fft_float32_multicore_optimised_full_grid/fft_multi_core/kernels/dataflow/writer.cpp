// writer.cpp - Simplified writer (local stages only, no cross-core)
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
    
    const uint32_t tile_bytes = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);
    constexpr uint32_t ELEM = sizeof(float);
    
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
    
    auto safe_local_idx = [&](uint32_t global_idx) -> uint32_t {
        if (global_idx < core_elem_base) return 0u;
        return global_idx - core_elem_base;
    };
    
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);
        
        cb_wait_front(cb_out0_r, local_tiles);
        cb_wait_front(cb_out0_i, local_tiles);
        cb_wait_front(cb_out1_r, local_tiles);
        cb_wait_front(cb_out1_i, local_tiles);
        
        const uint32_t src0r = get_read_ptr(cb_out0_r);
        const uint32_t src0i = get_read_ptr(cb_out0_i);
        const uint32_t src1r = get_read_ptr(cb_out1_r);
        const uint32_t src1i = get_read_ptr(cb_out1_i);
        
        if (is_last) {
            // Final stage: Write to DRAM
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
            
        } else {
            // Shuffle stage (local only)
            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = (half_m2 <= local_half) ? local_half / half_m2 : 0u;
            
            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);
            
            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);
            
            const uint32_t log2m  = stage + 1;
            const uint32_t m_mask = m - 1u;
            
            uint32_t dst = 0;
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                const uint32_t local_base_e = g2 * m2;
                const uint32_t local_base_o = local_base_e + half_m2;
                
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                    // Even
                    {
                        uint32_t f      = core_elem_base + local_base_e + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx = safe_local_idx(global_idx);
                        
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_er + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_ei + dst*ELEM, rd(srci + local_idx*ELEM));
                    }
                    // Odd
                    {
                        uint32_t f      = core_elem_base + local_base_o + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx = safe_local_idx(global_idx);
                        
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_or + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_oi + dst*ELEM, rd(srci + local_idx*ELEM));
                    }
                    dst++;
                }
            }
            
            if (G2 == 0) {
                for (uint32_t lp = 0; lp < local_half; lp++) {
                    wr(dst_er + lp*ELEM, rd(src0r + lp*ELEM));
                    wr(dst_ei + lp*ELEM, rd(src0i + lp*ELEM));
                    wr(dst_or + lp*ELEM, rd(src1r + lp*ELEM));
                    wr(dst_oi + lp*ELEM, rd(src1i + lp*ELEM));
                }
            }
            
            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);
            
            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r,  local_tiles);
            cb_push_back(cb_odd_i,  local_tiles);
        }
    }
}