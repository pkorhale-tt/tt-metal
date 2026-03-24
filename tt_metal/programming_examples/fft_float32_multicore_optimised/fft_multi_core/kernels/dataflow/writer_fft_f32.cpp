// writer_fft_f32.cpp — MULTICORE row-aware writer (FIXED v2)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  CHANGES vs previous version
// ══════════════════════════════════════════════════════════════════════
//
//  BUG FIX (CB index mismatch — inter-stage shuffle destination)
//  ─────────────────────────────────────────────────────────────────────
//  Previous: writer shuffled butterfly output into CB 0-3
//            (cb_even_r=0, cb_even_i=1, cb_odd_r=2, cb_odd_i=3)
//
//  Problem:  the compute kernel reads stage>=1 inputs from CB 6-9
//            (cb_next_even_r=6, cb_next_even_i=7,
//             cb_next_odd_r=8,  cb_next_odd_i=9)
//            so compute would stall forever at stage 1+ waiting on CB 6-9
//            while writer was filling CB 0-3 instead.
//            This was THE deadlock — CB 6-9 was never written after stage 0.
//
//  Fix:      writer now pushes shuffle data into CB 6-9, matching the
//            compute kernel's cb_next_* indices.
//
//  Note:     CB 0-3 is now exclusively owned by the reader (stage-0 input)
//            and the compute kernel (stage-0 read).  The writer never
//            touches CB 0-3.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP (unchanged)
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  out0_r_addr
//  [1]  out0_i_addr
//  [2]  out1_r_addr
//  [3]  out1_i_addr
//  [4]  num_tiles      (tiles_per_row)
//  [5]  num_stages     (log2_row)
//  [6]  half_N         (N_row/2)
//  [7]  (unused padding)
//  [8]  (unused padding)
//  [9]  (unused padding)
//  [10] (unused padding)
//  [11] tile_offset    (starting tile index for this core)
//  [12] (unused padding)
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
    const uint32_t num_tiles     = get_arg_val<uint32_t>(4);   // tiles_per_row
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);   // log2_row
    const uint32_t half_N        = get_arg_val<uint32_t>(6);   // N_row/2
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);  // starting tile for this core
    const uint32_t rows_per_core = get_arg_val<uint32_t>(13);  // rows handled by this core

    // ── CB indices ────────────────────────────────────────────────────
    //
    // Output CBs: compute produces here, writer consumes.
    //
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    // Shuffle destination CBs: writer produces here, compute consumes
    // on the next stage (cb_next_even_r=6 … cb_next_odd_i=9).
    //
    // FIX: was 0,1,2,3 — must be 6,7,8,9 to match compute's
    //      cb_next_even_r/i and cb_next_odd_r/i.
    //
    constexpr uint32_t cb_next_even_r = 6;   // FIX: was 0
    constexpr uint32_t cb_next_even_i = 7;   // FIX: was 1
    constexpr uint32_t cb_next_odd_r  = 8;   // FIX: was 2
    constexpr uint32_t cb_next_odd_i  = 9;   // FIX: was 3

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

    if (num_tiles == 0 || num_stages == 0 || rows_per_core == 0) {
        return;
    }

    constexpr uint32_t ELEM = sizeof(float);

    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * num_tiles;

        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const bool is_last = (stage == num_stages - 1);

            cb_wait_front(cb_out0_r, num_tiles);
            cb_wait_front(cb_out0_i, num_tiles);
            cb_wait_front(cb_out1_r, num_tiles);
            cb_wait_front(cb_out1_i, num_tiles);

            if (is_last) {
                // ── Final stage: write butterfly outputs to DRAM ─────
                for (uint32_t t = 0; t < num_tiles; t++) {
                    noc_async_write_tile(row_tile_base + t, out0_r_gen,
                        get_read_ptr(cb_out0_r) + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out0_i_gen,
                        get_read_ptr(cb_out0_i) + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_r_gen,
                        get_read_ptr(cb_out1_r) + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_i_gen,
                        get_read_ptr(cb_out1_i) + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);

            } else {
                // ── Intermediate stage: shuffle butterfly outputs into
                //    CB 6-9 (cb_next_even/odd r/i) for the next compute
                //    stage.
                // ────────────────────────────────────────────────────

                const uint32_t m       = 1u << (stage + 1);
                const uint32_t half_m  = m >> 1;
                const uint32_t m2      = m << 1;
                const uint32_t half_m2 = m2 >> 1;
                const uint32_t G2      = (half_N >= half_m2) ? (half_N / half_m2) : 0u;

                if (G2 == 0) {
                    cb_pop_front(cb_out0_r, num_tiles);
                    cb_pop_front(cb_out0_i, num_tiles);
                    cb_pop_front(cb_out1_r, num_tiles);
                    cb_pop_front(cb_out1_i, num_tiles);
                    continue;
                }

                // Capture source pointers before popping (popping only
                // moves the read-pointer; the data remains valid until
                // the next reserve overwrites it, but getting the pointer
                // first is the safe pattern).
                const uint32_t src0r = get_read_ptr(cb_out0_r);
                const uint32_t src0i = get_read_ptr(cb_out0_i);
                const uint32_t src1r = get_read_ptr(cb_out1_r);
                const uint32_t src1i = get_read_ptr(cb_out1_i);

                // Free the output CBs FIRST so compute is unblocked and
                // can start producing the next stage's output immediately
                // while we fill the shuffle destination below.
                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);

                // Now claim space in the next-stage input CBs (6-9).
                // This may block if compute hasn't yet drained them from
                // a previous iteration, which is the correct back-pressure.
                cb_reserve_back(cb_next_even_r, num_tiles);
                cb_reserve_back(cb_next_even_i, num_tiles);
                cb_reserve_back(cb_next_odd_r,  num_tiles);
                cb_reserve_back(cb_next_odd_i,  num_tiles);

                const uint32_t dst_er = get_write_ptr(cb_next_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_next_even_i);
                const uint32_t dst_or = get_write_ptr(cb_next_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_next_odd_i);

                auto rd = [](uint32_t addr) -> float {
                    uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
                    float v = 0.0f;
                    __builtin_memcpy(&v, &raw, sizeof(float));
                    return v;
                };
                auto wr = [](uint32_t addr, float v) {
                    uint32_t raw = 0u;
                    __builtin_memcpy(&raw, &v, sizeof(float));
                    *reinterpret_cast<volatile uint32_t*>(addr) = raw;
                };

                const uint32_t log2m  = stage + 1;
                const uint32_t m_mask = m - 1u;

                uint32_t dst = 0;
                for (uint32_t g2 = 0; g2 < G2; g2++) {
                    const uint32_t base_e = g2 * m2;
                    const uint32_t base_o = base_e + half_m2;
                    for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                        // Even butterfly output element
                        {
                            uint32_t f      = base_e + j2;
                            uint32_t g_old  = f >> log2m;
                            uint32_t offset = f & m_mask;
                            uint32_t idx, srcr, srci;
                            if (offset < half_m) {
                                idx = g_old * half_m + offset;
                                srcr = src0r; srci = src0i;
                            } else {
                                idx = g_old * half_m + (offset - half_m);
                                srcr = src1r; srci = src1i;
                            }
                            wr(dst_er + dst * ELEM, rd(srcr + idx * ELEM));
                            wr(dst_ei + dst * ELEM, rd(srci + idx * ELEM));
                        }

                        // Odd butterfly output element
                        {
                            uint32_t f      = base_o + j2;
                            uint32_t g_old  = f >> log2m;
                            uint32_t offset = f & m_mask;
                            uint32_t idx, srcr, srci;
                            if (offset < half_m) {
                                idx = g_old * half_m + offset;
                                srcr = src0r; srci = src0i;
                            } else {
                                idx = g_old * half_m + (offset - half_m);
                                srcr = src1r; srci = src1i;
                            }
                            wr(dst_or + dst * ELEM, rd(srcr + idx * ELEM));
                            wr(dst_oi + dst * ELEM, rd(srci + idx * ELEM));
                        }

                        dst++;
                    }
                }

                cb_push_back(cb_next_even_r, num_tiles);
                cb_push_back(cb_next_even_i, num_tiles);
                cb_push_back(cb_next_odd_r,  num_tiles);
                cb_push_back(cb_next_odd_i,  num_tiles);
            }
        }
    }
}