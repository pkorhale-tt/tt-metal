// reader_fft_f32_mc.cpp — MULTICORE reader (FIXED v3)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  CHANGES vs v2
// ══════════════════════════════════════════════════════════════════════
//
//  BUG FIX (CB index mismatch — stage-0 even/odd)
//  ─────────────────────────────────────────────────────────────────────
//  Previous: reader pushed stage-0 even/odd data into CB 6-9
//            (cb_even_r=6, cb_even_i=7, cb_odd_r=8, cb_odd_i=9)
//
//  Problem:  the compute kernel reads stage-0 inputs from CB 0-3
//            (cb_stage0_even_r=0, cb_stage0_even_i=1,
//             cb_stage0_odd_r=2,  cb_stage0_odd_i=3)
//            so compute would stall forever at stage 0 waiting on CB 0-3
//            while reader was filling CB 6-9 instead.
//
//  Fix:      reader now pushes stage-0 even/odd into CB 0-3, matching
//            the compute kernel's cb_stage0_* indices.
//
//  No other logic changes.  The twiddle path (CB 4,5) and compact path
//  (CB 10,11) were already correct and are unchanged.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  even_r_addr
//  [1]  even_i_addr
//  [2]  odd_r_addr
//  [3]  odd_i_addr
//  [4]  compact_r_addr
//  [5]  compact_i_addr
//  [6]  tiles_per_row
//  [7]  tile_offset
//  [8]  num_stages
//  [9]  half_N
//  [10] local_half (ABI compatibility, unused in scatter loop)
//  [11] rows_per_core
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr = get_arg_val<uint32_t>(4);
    const uint32_t compact_i_addr = get_arg_val<uint32_t>(5);
    const uint32_t tiles_per_row  = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t half_N         = get_arg_val<uint32_t>(9);
    // arg[10] local_half — ABI compat, not used in scatter loop
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(11);

    // ── CB indices ────────────────────────────────────────────────────
    //
    // FIX: stage-0 even/odd must land in CB 0-3 so the compute kernel
    //      (cb_stage0_even_r=0 … cb_stage0_odd_i=3) can consume them.
    //      Previously these were 6-9, causing compute to stall on stage 0.
    //
    constexpr uint32_t cb_even_r    = 0;   // FIX: was 6
    constexpr uint32_t cb_even_i    = 1;   // FIX: was 7
    constexpr uint32_t cb_odd_r     = 2;   // FIX: was 8
    constexpr uint32_t cb_odd_i     = 3;   // FIX: was 9
    constexpr uint32_t cb_tw_r      = 4;   // unchanged
    constexpr uint32_t cb_tw_i      = 5;   // unchanged
    constexpr uint32_t cb_compact_r = 10;  // unchanged
    constexpr uint32_t cb_compact_i = 11;  // unchanged

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    const uint32_t compact_bytes = half_N * sizeof(float);   // kept for reference

    constexpr uint32_t ELEM      = sizeof(float);
    // Total float elements per twiddle CB push = all slots in tiles_per_row tiles.
    const uint32_t elems_per_row = (tile_bytes / ELEM) * tiles_per_row;

    if (tiles_per_row == 0 || num_stages == 0 || rows_per_core == 0) return;

    // ── Address generators ────────────────────────────────────────────
    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_r_gen  = {
        .bank_base_address = odd_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_i_gen  = {
        .bank_base_address = odd_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_r_gen  = {
        .bank_base_address = compact_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_i_gen  = {
        .bank_base_address = compact_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // ── Load compact twiddle table once (shared across all rows) ─────
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
    noc_async_read_barrier();
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── Outer row loop ────────────────────────────────────────────────
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        const uint32_t row_elem_base = row_tile_base * (tile_bytes / ELEM);

        // ── Stage 0: load even/odd from DRAM into CB 0-3 ─────────────
        //
        // Issue all four NOC reads before the barrier so they run in
        // parallel on the NoC fabric.
        //
        cb_reserve_back(cb_even_r, tiles_per_row);
        cb_reserve_back(cb_even_i, tiles_per_row);
        cb_reserve_back(cb_odd_r,  tiles_per_row);
        cb_reserve_back(cb_odd_i,  tiles_per_row);

        for (uint32_t t = 0; t < tiles_per_row; t++) {
            const uint32_t gt = row_tile_base + t;
            noc_async_read_tile(gt, even_r_gen,
                get_write_ptr(cb_even_r) + t * tile_bytes);
            noc_async_read_tile(gt, even_i_gen,
                get_write_ptr(cb_even_i) + t * tile_bytes);
            noc_async_read_tile(gt, odd_r_gen,
                get_write_ptr(cb_odd_r)  + t * tile_bytes);
            noc_async_read_tile(gt, odd_i_gen,
                get_write_ptr(cb_odd_i)  + t * tile_bytes);
        }
        noc_async_read_barrier();

        cb_push_back(cb_even_r, tiles_per_row);
        cb_push_back(cb_even_i, tiles_per_row);
        cb_push_back(cb_odd_r,  tiles_per_row);
        cb_push_back(cb_odd_i,  tiles_per_row);

        // ── All stages: scatter twiddle factors into CB 4,5 ──────────
        //
        // cb_tw_r/i have depth=tiles_per_row.  The cb_reserve_back at
        // stage N+1 blocks until compute drains stage N's twiddle tiles,
        // creating a natural one-stage-ahead limit on the reader.
        //
        // elems_per_row fills ALL float slots in the tiles_per_row tiles
        // so no element is left uninitialised (BUG 4 fix retained).

        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const uint32_t half_m      = 1u << stage;
            const uint32_t N_over_m    = half_N >> stage;
            const uint32_t half_m_mask = half_m - 1u;

            cb_reserve_back(cb_tw_r, tiles_per_row);
            cb_reserve_back(cb_tw_i, tiles_per_row);
            const uint32_t dst_r = get_write_ptr(cb_tw_r);
            const uint32_t dst_i = get_write_ptr(cb_tw_i);

            for (uint32_t lp = 0; lp < elems_per_row; lp++) {
                const uint32_t p   = row_elem_base + lp;
                const uint32_t idx = (p & half_m_mask) * N_over_m;
                wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
            }

            cb_push_back(cb_tw_r, tiles_per_row);
            cb_push_back(cb_tw_i, tiles_per_row);
        }
        // Reader's job for this row is done.
        // Compute+writer pipeline drains the remaining CBs independently.
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}