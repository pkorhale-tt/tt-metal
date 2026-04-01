// writer_fft_f32_mc.cpp  — MULTICORE writer  [OPTIMISED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS / FIXES vs. previous version
// ══════════════════════════════════════════════════════════════════════
//
//  OPT-7  Add a per-row loop for multi-row cores.
//  ─────────────────────────────────────────────────────────────────────
//  When rows_this > 1 the CBs contain rows_this independent FFT rows'
//  worth of data packed contiguously.  The previous version's shuffle
//  formula only processed half_row elements regardless of rows_this,
//  silently ignoring rows 1 … rows_this-1.
//
//  Fix: outer loop over rows [0, rows_this).  Within each row iteration,
//  all CB pointers are offset by row × half_row × ELEM (elements) or
//  row × tiles_per_row × tile_bytes (tiles).  The shuffle formula itself
//  is unchanged — it operates on row-local indices with core_elem_base=0.
//
//  OPT-8  Batch DRAM write for all tiles of all rows in one barrier.
//  ─────────────────────────────────────────────────────────────────────
//  Previously, the is_last path wrote tiles in a single loop over
//  local_tiles (= rows_this × tiles_per_row) — this was already batched.
//  The row loop below preserves that: all noc_async_write_tile calls are
//  issued before the single noc_async_write_barrier.
//
// ── Arg layout ────────────────────────────────────────────────────────
//   0   out0_r_addr
//   1   out0_i_addr
//   2   out1_r_addr
//   3   out1_i_addr
//   4   local_tiles       rows_this × tiles_per_row
//   5   num_stages        log2(N_row)
//   6   local_half        half_row (one row's half — row-local)
//   7   half_N            half_row (== local_half for row-decomp)
//   8   num_cores         1  (row-decomp: each core owns full rows)
//   9   core_id           0
//  10   log2_cores        0
//  11   tile_offset       first global tile index for this core
//  12   core_elem_base    0 (row-local addressing)
//  13   rows_this         number of rows this core owns    ← NEW
//  14   tiles_per_row     tiles per single FFT row          ← NEW

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr    = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr    = get_arg_val<uint32_t>(3);
    const uint32_t local_tiles    = get_arg_val<uint32_t>(4);
    const uint32_t num_stages     = get_arg_val<uint32_t>(5);
    const uint32_t local_half     = get_arg_val<uint32_t>(6); // half_row (one row)
    const uint32_t half_N         = get_arg_val<uint32_t>(7); // == local_half
    const uint32_t num_cores      = get_arg_val<uint32_t>(8);
    const uint32_t core_id        = get_arg_val<uint32_t>(9);
    const uint32_t log2_cores     = get_arg_val<uint32_t>(10);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(11);
    const uint32_t core_elem_base = get_arg_val<uint32_t>(12);
    const uint32_t rows_this      = get_arg_val<uint32_t>(13); // NEW
    const uint32_t tiles_per_row  = get_arg_val<uint32_t>(14); // NEW

    // Suppress unused-variable warnings for single-core-row params
    (void)num_cores; (void)core_id; (void)log2_cores;

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

    if (local_tiles == 0) return;

    constexpr uint32_t ELEM = sizeof(float);

    auto rd = [](uint32_t addr) -> float {
        uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
        float v; __builtin_memcpy(&v, &raw, 4); return v;
    };
    auto wr = [](uint32_t addr, float v) {
        uint32_t raw; __builtin_memcpy(&raw, &v, 4);
        *reinterpret_cast<volatile uint32_t*>(addr) = raw;
    };

    // Safe local-index helper (same as before, per-row offset applied by caller).
    auto safe_sub = [](uint32_t a, uint32_t b) -> uint32_t {
        ASSERT(a >= b);
        return (a >= b) ? a - b : 0u;
    };

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);

        // Wait for all rows' output tiles.
        cb_wait_front(cb_out0_r, local_tiles);
        cb_wait_front(cb_out0_i, local_tiles);
        cb_wait_front(cb_out1_r, local_tiles);
        cb_wait_front(cb_out1_i, local_tiles);

        const uint32_t src0r_base = get_read_ptr(cb_out0_r);
        const uint32_t src0i_base = get_read_ptr(cb_out0_i);
        const uint32_t src1r_base = get_read_ptr(cb_out1_r);
        const uint32_t src1i_base = get_read_ptr(cb_out1_i);

        if (is_last) {
            // ── OPT-8: issue all DRAM writes for all rows, then one barrier ──
            for (uint32_t row = 0; row < rows_this; row++) {
                const uint32_t row_tile_off  = row * tiles_per_row;
                const uint32_t row_elem_off  = row * local_half; // in ELEMs
                // row_elem_off in bytes = row * local_half * ELEM
                // (local_half elements per row, each element sizeof(float) bytes)
                const uint32_t row_byte_off  = row_elem_off * ELEM;

                // For tile-granularity writes the tile offset into the CB
                // is row_tile_off × tile_bytes.  However noc_async_write_tile
                // takes a CB address from the CB read-ptr + a byte offset.
                // We supply the base address explicitly.
                const uint32_t rsrc0r = src0r_base + row_tile_off * tile_bytes;
                const uint32_t rsrc0i = src0i_base + row_tile_off * tile_bytes;
                const uint32_t rsrc1r = src1r_base + row_tile_off * tile_bytes;
                const uint32_t rsrc1i = src1i_base + row_tile_off * tile_bytes;

                uint32_t gt_base = tile_offset + row_tile_off;
                for (uint32_t t = 0; t < tiles_per_row; t++) {
                    uint32_t gt = gt_base + t;
                    noc_async_write_tile(gt, out0_r_gen, rsrc0r + t * tile_bytes);
                    noc_async_write_tile(gt, out0_i_gen, rsrc0i + t * tile_bytes);
                    noc_async_write_tile(gt, out1_r_gen, rsrc1r + t * tile_bytes);
                    noc_async_write_tile(gt, out1_i_gen, rsrc1i + t * tile_bytes);
                }
            }
            noc_async_write_barrier(); // one barrier for everything

            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

        } else {
            // ── SHUFFLE ─────────────────────────────────────────────────
            // Reserve next-stage even/odd CBs for all rows.
            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er_base = get_write_ptr(cb_even_r);
            const uint32_t dst_ei_base = get_write_ptr(cb_even_i);
            const uint32_t dst_or_base = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi_base = get_write_ptr(cb_odd_i);

            // Butterfly stage parameters (same for every row).
            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = (half_m2 <= local_half)
                                     ? local_half / half_m2 : 0u;
            const uint32_t log2m   = stage + 1;
            const uint32_t m_mask  = m - 1u;

            // OPT-7: outer loop over rows.
            for (uint32_t row = 0; row < rows_this; row++) {
                // Byte offset into the CB for this row's elements.
                const uint32_t row_off = row * local_half * ELEM;

                const uint32_t src0r = src0r_base + row_off;
                const uint32_t src0i = src0i_base + row_off;
                const uint32_t src1r = src1r_base + row_off;
                const uint32_t src1i = src1i_base + row_off;

                const uint32_t dst_er = dst_er_base + row_off;
                const uint32_t dst_ei = dst_ei_base + row_off;
                const uint32_t dst_or = dst_or_base + row_off;
                const uint32_t dst_oi = dst_oi_base + row_off;

                uint32_t dst = 0;

                if (G2 > 0) {
                    for (uint32_t g2 = 0; g2 < G2; g2++) {
                        const uint32_t local_base_e = g2 * m2;
                        const uint32_t local_base_o = local_base_e + half_m2;

                        for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                            // new_even[dst]
                            {
                                uint32_t f      = core_elem_base + local_base_e + j2;
                                uint32_t g_old  = f >> log2m;
                                uint32_t offset = f & m_mask;
                                uint32_t global_idx = (offset < half_m)
                                    ? g_old * half_m + offset
                                    : g_old * half_m + (offset - half_m);
                                uint32_t local_idx = safe_sub(global_idx, core_elem_base);
                                const uint32_t rsrcr = (offset < half_m) ? src0r : src1r;
                                const uint32_t rsrci = (offset < half_m) ? src0i : src1i;
                                wr(dst_er + dst * ELEM, rd(rsrcr + local_idx * ELEM));
                                wr(dst_ei + dst * ELEM, rd(rsrci + local_idx * ELEM));
                            }
                            // new_odd[dst]
                            {
                                uint32_t f      = core_elem_base + local_base_o + j2;
                                uint32_t g_old  = f >> log2m;
                                uint32_t offset = f & m_mask;
                                uint32_t global_idx = (offset < half_m)
                                    ? g_old * half_m + offset
                                    : g_old * half_m + (offset - half_m);
                                uint32_t local_idx = safe_sub(global_idx, core_elem_base);
                                const uint32_t rsrcr = (offset < half_m) ? src0r : src1r;
                                const uint32_t rsrci = (offset < half_m) ? src0i : src1i;
                                wr(dst_or + dst * ELEM, rd(rsrcr + local_idx * ELEM));
                                wr(dst_oi + dst * ELEM, rd(rsrci + local_idx * ELEM));
                            }
                            dst++;
                        }
                    }
                } else {
                    // G2 == 0: this stage's butterfly groups span multiple cores.
                    // Bit-reversal guarantees out0 → even, out1 → odd directly.
                    for (uint32_t lp = 0; lp < local_half; lp++) {
                        wr(dst_er + lp * ELEM, rd(src0r + lp * ELEM));
                        wr(dst_ei + lp * ELEM, rd(src0i + lp * ELEM));
                        wr(dst_or + lp * ELEM, rd(src1r + lp * ELEM));
                        wr(dst_oi + lp * ELEM, rd(src1i + lp * ELEM));
                    }
                }
            } // end row loop

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