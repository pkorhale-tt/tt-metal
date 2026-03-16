// writer_fft_f32_opt.cpp — NOC-ACCELERATED L1 SHUFFLE
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);
    const uint32_t num_stages  = get_arg_val<uint32_t>(5);
    const uint32_t half_N      = get_arg_val<uint32_t>(6);
    
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
    
    if (num_tiles == 0) return;
    
    constexpr uint64_t noc_xy = uint64_t(NOC_XY_ENCODING(DEVICE_NOC_X, DEVICE_NOC_Y));
    
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);
        
        cb_wait_front(cb_out0_r, num_tiles);
        cb_wait_front(cb_out0_i, num_tiles);
        cb_wait_front(cb_out1_r, num_tiles);
        cb_wait_front(cb_out1_i, num_tiles);
        
        if (is_last) {
            // ══════════════════════════════════════════════════════
            // LAST STAGE: DRAM Write
            // ══════════════════════════════════════════════════════
            for (uint32_t t = 0; t < num_tiles; t++) {
                noc_async_write_tile(t, out0_r_gen, get_read_ptr(cb_out0_r) + t * tile_bytes);
                noc_async_write_tile(t, out0_i_gen, get_read_ptr(cb_out0_i) + t * tile_bytes);
                noc_async_write_tile(t, out1_r_gen, get_read_ptr(cb_out1_r) + t * tile_bytes);
                noc_async_write_tile(t, out1_i_gen, get_read_ptr(cb_out1_i) + t * tile_bytes);
            }
            noc_async_write_barrier();
            
            cb_pop_front(cb_out0_r, num_tiles);
            cb_pop_front(cb_out0_i, num_tiles);
            cb_pop_front(cb_out1_r, num_tiles);
            cb_pop_front(cb_out1_i, num_tiles);
            
        } else {
            // ══════════════════════════════════════════════════════
            // INTERMEDIATE STAGES: L1-to-L1 NOC Shuffle
            // ══════════════════════════════════════════════════════
            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = half_N / half_m2;
            
            const uint32_t src0r = get_read_ptr(cb_out0_r);
            const uint32_t src0i = get_read_ptr(cb_out0_i);
            const uint32_t src1r = get_read_ptr(cb_out1_r);
            const uint32_t src1i = get_read_ptr(cb_out1_i);
            
            cb_reserve_back(cb_even_r, num_tiles);
            cb_reserve_back(cb_even_i, num_tiles);
            cb_reserve_back(cb_odd_r,  num_tiles);
            cb_reserve_back(cb_odd_i,  num_tiles);
            
            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);
            
            // ──────────────────────────────────────────────────────
            // OPTIMIZATION: NOC writes with proper barriers
            // ──────────────────────────────────────────────────────
            const uint32_t log2m  = stage + 1;
            const uint32_t m_mask = m - 1u;
            constexpr uint32_t ELEM = sizeof(float);
            
            uint32_t dst = 0;
            
            // Issue all EVEN NOC writes first
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                const uint32_t base_e = g2 * m2;
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                    uint32_t f      = base_e + j2;
                    uint32_t g_old  = f >> log2m;
                    uint32_t offset = f & m_mask;
                    
                    uint32_t idx = g_old * half_m + ((offset < half_m) ? offset : (offset - half_m));
                    uint32_t srcr = (offset < half_m) ? src0r : src1r;
                    uint32_t srci = (offset < half_m) ? src0i : src1i;
                    
                    // L1-to-L1 NOC write (same core)
                    noc_async_write(srcr + idx * ELEM, noc_xy | (dst_er + dst * ELEM), ELEM);
                    noc_async_write(srci + idx * ELEM, noc_xy | (dst_ei + dst * ELEM), ELEM);
                    
                    dst++;
                }
            }
            
            // Barrier: all EVEN writes complete before ODD starts
            noc_async_write_barrier();
            
            dst = 0;
            
            // Issue all ODD NOC writes
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                const uint32_t base_o = g2 * m2 + half_m2;
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                    uint32_t f      = base_o + j2;
                    uint32_t g_old  = f >> log2m;
                    uint32_t offset = f & m_mask;
                    
                    uint32_t idx = g_old * half_m + ((offset < half_m) ? offset : (offset - half_m));
                    uint32_t srcr = (offset < half_m) ? src0r : src1r;
                    uint32_t srci = (offset < half_m) ? src0i : src1i;
                    
                    noc_async_write(srcr + idx * ELEM, noc_xy | (dst_or + dst * ELEM), ELEM);
                    noc_async_write(srci + idx * ELEM, noc_xy | (dst_oi + dst * ELEM), ELEM);
                    
                    dst++;
                }
            }
            
            // Final barrier: all ODD writes complete
            noc_async_write_barrier();
            
            cb_pop_front(cb_out0_r, num_tiles);
            cb_pop_front(cb_out0_i, num_tiles);
            cb_pop_front(cb_out1_r, num_tiles);
            cb_pop_front(cb_out1_i, num_tiles);
            
            cb_push_back(cb_even_r, num_tiles);
            cb_push_back(cb_even_i, num_tiles);
            cb_push_back(cb_odd_r,  num_tiles);
            cb_push_back(cb_odd_i,  num_tiles);
        }
    }
}