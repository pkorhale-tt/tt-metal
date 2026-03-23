// writer_fft_f32.cpp — MULTICORE writer (PORTABLE FIX)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

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
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;
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

    if (local_tiles == 0 || rows_per_core == 0) return;

    auto copy_floats = [](volatile float* dst, volatile float* src, uint32_t count) {
        for (uint32_t i = 0; i < count; i++) dst[i] = src[i];
    };

    // Outer loop over rows
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_offset = tile_offset + row * local_tiles;
        const uint32_t row_elem_base   = row_tile_offset * (tile_bytes / sizeof(float));

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
                for (uint32_t t = 0; t < local_tiles; t++) {
                    const uint32_t gt = row_tile_offset + t;
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
                // Shuffle for next stage
                const uint32_t m       = 1u << (stage + 1);
                const uint32_t half_m  = m >> 1;
                const uint32_t m2      = m << 1;
                const uint32_t half_m2 = m2 >> 1;
                const uint32_t G2      = (half_m2 <= local_half) ? local_half / half_m2 : 0u;

                cb_reserve_back(cb_even_r, local_tiles);
                cb_reserve_back(cb_even_i, local_tiles);
                cb_reserve_back(cb_odd_r,  local_tiles);
                cb_reserve_back(cb_odd_i,  local_tiles);

                volatile float* dst_er = reinterpret_cast<volatile float*>(get_write_ptr(cb_even_r));
                volatile float* dst_ei = reinterpret_cast<volatile float*>(get_write_ptr(cb_even_i));
                volatile float* dst_or = reinterpret_cast<volatile float*>(get_write_ptr(cb_odd_r));
                volatile float* dst_oi = reinterpret_cast<volatile float*>(get_write_ptr(cb_odd_i));
                volatile float* s0r = reinterpret_cast<volatile float*>(src0r);
                volatile float* s0i = reinterpret_cast<volatile float*>(src0i);
                volatile float* s1r = reinterpret_cast<volatile float*>(src1r);
                volatile float* s1i = reinterpret_cast<volatile float*>(src1i);

                if (G2 > 0) {
                    const uint32_t log2m  = stage + 1;
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst = 0;

                    for (uint32_t g2 = 0; g2 < G2; g2++) {
                        const uint32_t lb_e = g2 * m2;
                        const uint32_t lb_o = lb_e + half_m2;

                        // Block A
                        {
                            uint32_t f0    = row_elem_base + lb_e;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + off;
                            if (ss >= row_elem_base) {
                                uint32_t ls = ss - row_elem_base;
                                copy_floats(&dst_er[dst], &s0r[ls], half_m);
                                copy_floats(&dst_ei[dst], &s0i[ls], half_m);
                            }
                        }
                        // Block B
                        {
                            uint32_t f0    = row_elem_base + lb_e + half_m;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + (off - half_m);
                            if (ss >= row_elem_base) {
                                uint32_t ls = ss - row_elem_base;
                                copy_floats(&dst_er[dst + half_m], &s1r[ls], half_m);
                                copy_floats(&dst_ei[dst + half_m], &s1i[ls], half_m);
                            }
                        }
                        // Block C
                        {
                            uint32_t f0    = row_elem_base + lb_o;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + off;
                            if (ss >= row_elem_base) {
                                uint32_t ls = ss - row_elem_base;
                                copy_floats(&dst_or[dst], &s0r[ls], half_m);
                                copy_floats(&dst_oi[dst], &s0i[ls], half_m);
                            }
                        }
                        // Block D
                        {
                            uint32_t f0    = row_elem_base + lb_o + half_m;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + (off - half_m);
                            if (ss >= row_elem_base) {
                                uint32_t ls = ss - row_elem_base;
                                copy_floats(&dst_or[dst + half_m], &s1r[ls], half_m);
                                copy_floats(&dst_oi[dst + half_m], &s1i[ls], half_m);
                            }
                        }
                        dst += half_m2;
                    }
                } else {
                    copy_floats(dst_er, s0r, local_half);
                    copy_floats(dst_ei, s0i, local_half);
                    copy_floats(dst_or, s1r, local_half);
                    copy_floats(dst_oi, s1i, local_half);
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
}