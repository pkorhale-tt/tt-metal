// reader_fft_f32_mc.cpp — MULTICORE reader (FIXED v4)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  KEY INVARIANTS THIS KERNEL MUST SATISFY
// ══════════════════════════════════════════════════════════════════════
//
//  1. Stage-0 even/odd → CB 0-3  (compute reads them only at stage 0)
//  2. Twiddles         → CB 4-5  (all stages; depth = tiles_per_row)
//  3. Compact twiddle table lives in CB 10-11 for the lifetime of the
//     kernel — popped only at the very end.
//
//  PIPELINE FLOW (per row):
//  ────────────────────────
//  reader pushes CB 0-3  (stage-0 even/odd)
//  reader pushes CB 4-5  (stage-0 twiddles)    ← compute starts stage 0
//  [writer drains CB 16-19 and shuffles into CB 6-9]
//  reader pushes CB 4-5  (stage-1 twiddles)    ← compute starts stage 1
//  ...
//
//  The cb_reserve_back(cb_tw_r/i, tiles_per_row) at the top of each
//  stage iteration naturally rate-limits the reader to one stage ahead:
//  it blocks until compute has drained the previous twiddle batch.
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
//  [10] local_half  (ABI padding, unused)
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
    // arg[10] — ABI padding, not used
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(11);

    // ── CB indices ────────────────────────────────────────────────────
    constexpr uint32_t cb_even_r    = 0;   // stage-0 even real
    constexpr uint32_t cb_even_i    = 1;   // stage-0 even imag
    constexpr uint32_t cb_odd_r     = 2;   // stage-0 odd  real
    constexpr uint32_t cb_odd_i     = 3;   // stage-0 odd  imag
    constexpr uint32_t cb_tw_r      = 4;   // twiddle real  (all stages)
    constexpr uint32_t cb_tw_i      = 5;   // twiddle imag  (all stages)
    constexpr uint32_t cb_compact_r = 10;  // compact twiddle table real
    constexpr uint32_t cb_compact_i = 11;  // compact twiddle table imag

    if (tiles_per_row == 0 || num_stages == 0 || rows_per_core == 0) return;

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    constexpr uint32_t ELEM      = sizeof(float);

    // Total float elements that fill tiles_per_row tiles.
    const uint32_t elems_per_row = (tile_bytes / ELEM) * tiles_per_row;

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

    // ── Load compact twiddle table once (shared across all rows/stages) ─
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
    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        const uint32_t row_elem_base = row_tile_base * (tile_bytes / ELEM);

        // ── Stage 0: DMA even/odd inputs from DRAM into CB 0-3 ───────
        //
        // Issue all four reads before the barrier so NoC can pipeline them.
        //
        cb_reserve_back(cb_even_r, tiles_per_row);
        cb_reserve_back(cb_even_i, tiles_per_row);
        cb_reserve_back(cb_odd_r,  tiles_per_row);
        cb_reserve_back(cb_odd_i,  tiles_per_row);

        for (uint32_t t = 0; t < tiles_per_row; ++t) {
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

        // ── All stages: scatter twiddle factors into CB 4-5 ──────────
        //
        // cb_reserve_back blocks until compute has drained the previous
        // stage's twiddle tiles — this is the natural backpressure that
        // keeps the reader at most one stage ahead of compute.
        //
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const uint32_t half_m      = 1u << stage;
            const uint32_t N_over_m    = half_N >> stage;
            const uint32_t half_m_mask = half_m - 1u;

            cb_reserve_back(cb_tw_r, tiles_per_row);
            cb_reserve_back(cb_tw_i, tiles_per_row);
            const uint32_t dst_r = get_write_ptr(cb_tw_r);
            const uint32_t dst_i = get_write_ptr(cb_tw_i);

            // Scatter: element p maps to compact twiddle index
            // (p & half_m_mask) * N_over_m
            for (uint32_t lp = 0; lp < elems_per_row; ++lp) {
                const uint32_t p   = row_elem_base + lp;
                const uint32_t idx = (p & half_m_mask) * N_over_m;
                wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
            }

            cb_push_back(cb_tw_r, tiles_per_row);
            cb_push_back(cb_tw_i, tiles_per_row);
        }
        // Reader is done for this row.  Compute and writer drain the
        // remaining CBs (out0/out1 and the inter-stage shuffle) on their own.
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}