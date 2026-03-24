// writer_fft_f32.cpp — FIXED v5
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  ROOT CAUSE OF HANG (N=2 / any small N)
// ══════════════════════════════════════════════════════════════════════
//
//  v4 allocated the inter-stage shuffle buffer on the RISC-V stack:
//    uint32_t scratch_er[elems_per_batch];   // 1024 uint32_t = 4 KB
//    uint32_t scratch_ei[...];               // × 4 arrays = 16 KB
//    uint32_t scratch_or[...];
//    uint32_t scratch_oi[...];
//
//  The RISC-V dataflow processor has ~4 KB of usable stack.
//  Allocating 16 KB silently overwrites adjacent memory and hangs.
//
//  FIX: replace stack VLAs with four dedicated depth-1 L1 scratch CBs
//  (indices 12-15).  Their write pointers are used as flat byte arrays.
//  No push/pop — purely memory-mapped L1 access.
//
//  HOST CHANGE REQUIRED (add to the per-core CB setup):
//    create_cb(prog, cc, 12, 1, TILE_BYTES);  // scr even-r
//    create_cb(prog, cc, 13, 1, TILE_BYTES);  // scr even-i
//    create_cb(prog, cc, 14, 1, TILE_BYTES);  // scr odd-r
//    create_cb(prog, cc, 15, 1, TILE_BYTES);  // scr odd-i
//
// ══════════════════════════════════════════════════════════════════════
//  DEADLOCK-FREE PROTOCOL (two-pass, same as v4)
// ══════════════════════════════════════════════════════════════════════
//
//  Intermediate stages only:
//    Pass 1 (drain):  wait CB 16-19, shuffle into L1 scratch, pop CB 16-19.
//    Pass 2 (fill):   reserve CB 6-9, copy from scratch, push.
//
//  Popping CB 16-19 before reserving CB 6-9 breaks the circular deadlock.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  out0_r_addr
//  [1]  out0_i_addr
//  [2]  out1_r_addr
//  [3]  out1_i_addr
//  [4]  num_tiles      (= tiles_per_row)
//  [5]  num_stages
//  [6]  half_N
//  [7-10] padding
//  [11] tile_offset
//  [12] padding
//  [13] rows_per_core
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr   = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles     = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t half_N        = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);
    const uint32_t rows_per_core = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_out0_r      = 16;
    constexpr uint32_t cb_out0_i      = 17;
    constexpr uint32_t cb_out1_r      = 18;
    constexpr uint32_t cb_out1_i      = 19;

    constexpr uint32_t cb_next_even_r = 6;
    constexpr uint32_t cb_next_even_i = 7;
    constexpr uint32_t cb_next_odd_r  = 8;
    constexpr uint32_t cb_next_odd_i  = 9;

    // L1 scratch CBs — used as plain memory, never pushed/popped.
    // Must be created in the host with depth=1.
    constexpr uint32_t cb_scratch_er  = 12;
    constexpr uint32_t cb_scratch_ei  = 13;
    constexpr uint32_t cb_scratch_or  = 14;
    constexpr uint32_t cb_scratch_oi  = 15;

    if (num_tiles == 0 || num_stages == 0 || rows_per_core == 0) return;

    const uint32_t tile_bytes      = get_tile_size(cb_out0_r);
    const DataFormat data_format   = get_dataformat(cb_out0_r);
    constexpr uint32_t ELEM        = sizeof(float);
    const uint32_t elems_per_batch = num_tiles * (tile_bytes / ELEM);

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

    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // Stable L1 scratch pointers — write pointers of depth-1 CBs.
    // These never change because we never push/pop the scratch CBs.
    const uint32_t scr_er = get_write_ptr(cb_scratch_er);
    const uint32_t scr_ei = get_write_ptr(cb_scratch_ei);
    const uint32_t scr_or = get_write_ptr(cb_scratch_or);
    const uint32_t scr_oi = get_write_ptr(cb_scratch_oi);

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * num_tiles;

        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const bool is_last = (stage == num_stages - 1);

            cb_wait_front(cb_out0_r, num_tiles);
            cb_wait_front(cb_out0_i, num_tiles);
            cb_wait_front(cb_out1_r, num_tiles);
            cb_wait_front(cb_out1_i, num_tiles);

            if (is_last) {
                // ── Final stage: write to DRAM ────────────────────────
                const uint32_t src0r = get_read_ptr(cb_out0_r);
                const uint32_t src0i = get_read_ptr(cb_out0_i);
                const uint32_t src1r = get_read_ptr(cb_out1_r);
                const uint32_t src1i = get_read_ptr(cb_out1_i);

                for (uint32_t t = 0; t < num_tiles; ++t) {
                    noc_async_write_tile(row_tile_base + t, out0_r_gen,
                                         src0r + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out0_i_gen,
                                         src0i + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_r_gen,
                                         src1r + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_i_gen,
                                         src1i + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);

            } else {
                // ── Intermediate stage: shuffle → CB 6-9 ─────────────
                //
                // Pass 1 — drain: shuffle compute outputs into L1 scratch,
                //   then pop CB 16-19 BEFORE reserving CB 6-9.
                //
                const uint32_t m       = 1u << (stage + 1);
                const uint32_t half_m  = m >> 1;
                const uint32_t m2      = m << 1;
                const uint32_t half_m2 = m2 >> 1;
                const uint32_t G2      =
                    (half_N >= half_m2) ? (half_N / half_m2) : 0u;

                const uint32_t src0r = get_read_ptr(cb_out0_r);
                const uint32_t src0i = get_read_ptr(cb_out0_i);
                const uint32_t src1r = get_read_ptr(cb_out1_r);
                const uint32_t src1i = get_read_ptr(cb_out1_i);

                // Zero-fill scratch (clears unused tile tail).
                for (uint32_t lp = 0; lp < elems_per_batch; ++lp) {
                    const uint32_t off = lp * ELEM;
                    wr32(scr_er + off, 0u);
                    wr32(scr_ei + off, 0u);
                    wr32(scr_or + off, 0u);
                    wr32(scr_oi + off, 0u);
                }

                if (G2 != 0) {
                    const uint32_t log2m  = stage + 1;
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst = 0;

                    for (uint32_t g2 = 0; g2 < G2; ++g2) {
                        const uint32_t base_e = g2 * m2;
                        const uint32_t base_o = base_e + half_m2;

                        for (uint32_t j2 = 0; j2 < half_m2; ++j2) {
                            // Even element
                            {
                                const uint32_t f      = base_e + j2;
                                const uint32_t g_old  = f >> log2m;
                                const uint32_t offset = f & m_mask;
                                uint32_t idx, srcr, srci;
                                if (offset < half_m) {
                                    idx  = g_old * half_m + offset;
                                    srcr = src0r; srci = src0i;
                                } else {
                                    idx  = g_old * half_m + (offset - half_m);
                                    srcr = src1r; srci = src1i;
                                }
                                wr32(scr_er + dst * ELEM, rd32(srcr + idx * ELEM));
                                wr32(scr_ei + dst * ELEM, rd32(srci + idx * ELEM));
                            }
                            // Odd element
                            {
                                const uint32_t f      = base_o + j2;
                                const uint32_t g_old  = f >> log2m;
                                const uint32_t offset = f & m_mask;
                                uint32_t idx, srcr, srci;
                                if (offset < half_m) {
                                    idx  = g_old * half_m + offset;
                                    srcr = src0r; srci = src0i;
                                } else {
                                    idx  = g_old * half_m + (offset - half_m);
                                    srcr = src1r; srci = src1i;
                                }
                                wr32(scr_or + dst * ELEM, rd32(srcr + idx * ELEM));
                                wr32(scr_oi + dst * ELEM, rd32(srci + idx * ELEM));
                            }
                            ++dst;
                        }
                    }
                }

                // Pop CB 16-19 NOW — before reserving CB 6-9.
                // This is what prevents the circular deadlock.
                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);

                // Pass 2 — fill: copy from L1 scratch into CB 6-9.
                cb_reserve_back(cb_next_even_r, num_tiles);
                cb_reserve_back(cb_next_even_i, num_tiles);
                cb_reserve_back(cb_next_odd_r,  num_tiles);
                cb_reserve_back(cb_next_odd_i,  num_tiles);

                const uint32_t dst_er = get_write_ptr(cb_next_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_next_even_i);
                const uint32_t dst_or = get_write_ptr(cb_next_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_next_odd_i);

                for (uint32_t lp = 0; lp < elems_per_batch; ++lp) {
                    const uint32_t off = lp * ELEM;
                    wr32(dst_er + off, rd32(scr_er + off));
                    wr32(dst_ei + off, rd32(scr_ei + off));
                    wr32(dst_or + off, rd32(scr_or + off));
                    wr32(dst_oi + off, rd32(scr_oi + off));
                }

                cb_push_back(cb_next_even_r, num_tiles);
                cb_push_back(cb_next_even_i, num_tiles);
                cb_push_back(cb_next_odd_r,  num_tiles);
                cb_push_back(cb_next_odd_i,  num_tiles);
            }
        }
    }
}