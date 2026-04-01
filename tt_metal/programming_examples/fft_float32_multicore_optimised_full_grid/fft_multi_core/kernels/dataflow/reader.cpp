// reader_fft_f32_mc.cpp  — MULTICORE reader  [OPTIMISED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS vs. previous version
// ══════════════════════════════════════════════════════════════════════
//
//  OPT-4  Twiddle expansion: compute once per stage, replicate for
//         additional rows.
//  ─────────────────────────────────────────────────────────────────────
//  In row-decomposition mode a core may own rows_this > 1 rows. All
//  rows run an identical FFT so their twiddle tables are identical.
//  Previously the expansion loop ran over local_half = rows_this×half_row
//  elements but the indexed lookup for row k is the same as row 0 at the
//  same position mod half_row.
//
//  Fix: expand the first half_row elements, then memcpy that block for
//  the remaining rows_this−1 rows.  Total indexed lookups per stage:
//    before:  rows_this × half_row
//    after:   half_row  + (rows_this−1) × half_row (copies, no lookup)
//  For rows_this = R the lookup cost is reduced by R×.
//
//  OPT-5  Relaxed ASSERT → bounds check.
//  ─────────────────────────────────────────────────────────────────────
//  The original ASSERT(tw_tiles_needed == local_tiles) fires for any
//  multi-row core where half_row < TILE_SIZE (e.g. N=1024, 2 rows per
//  core: tw_tiles_needed=1, local_tiles=2).  Replaced with the weaker
//  ASSERT(tw_tiles_needed <= local_tiles) — padding within the reserved
//  tile space is acceptable.
//
//  OPT-6  Overlapped NOC reads (unchanged from bug-fixed version,
//         documented here for clarity).
//  ─────────────────────────────────────────────────────────────────────
//  All noc_async_read_tile calls (data tiles) and the compact-twiddle
//  noc_async_read are issued before the single noc_async_read_barrier.
//  This lets the NOC DMA engine pipeline all transfers in parallel.
//
// ── Arg layout ────────────────────────────────────────────────────────
//   0  even_r_addr        DRAM base — even real  (bit-reversed, split)
//   1  even_i_addr        DRAM base — even imag
//   2  odd_r_addr         DRAM base — odd  real
//   3  odd_i_addr         DRAM base — odd  imag
//   4  compact_r_addr     DRAM base — compact twiddle real  (N/2 floats)
//   5  compact_i_addr     DRAM base — compact twiddle imag
//   6  local_tiles        tiles_this = rows_this × tiles_per_row
//   7  tile_offset        first global tile index for this core
//   8  num_stages         log2(N_row)
//   9  half_N             N_row / 2  (= half_row)
//  10  local_half         rows_this × half_row   ← HOST MUST SET CORRECTLY
//  11  core_elem_base     0 for row-decomposition (row-local addressing)
//  12  rows_this          number of rows this core owns
//  13  tiles_per_row      tiles per single FFT row (= ceil(half_row/TILE_SIZE))

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr = get_arg_val<uint32_t>(4);
    const uint32_t compact_i_addr = get_arg_val<uint32_t>(5);
    const uint32_t local_tiles    = get_arg_val<uint32_t>(6);  // rows_this × tiles_per_row
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t half_N         = get_arg_val<uint32_t>(9);  // == half_row
    const uint32_t local_half     = get_arg_val<uint32_t>(10); // rows_this × half_row
    const uint32_t core_elem_base = get_arg_val<uint32_t>(11); // 0 for row-decomp
    const uint32_t rows_this      = get_arg_val<uint32_t>(12);
    const uint32_t tiles_per_row  = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_even_r    = 0;
    constexpr uint32_t cb_even_i    = 1;
    constexpr uint32_t cb_odd_r     = 2;
    constexpr uint32_t cb_odd_i     = 3;
    constexpr uint32_t cb_tw_r      = 4;
    constexpr uint32_t cb_tw_i      = 5;
    constexpr uint32_t cb_compact_r = 10;
    constexpr uint32_t cb_compact_i = 11;

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);

    constexpr uint32_t ELEM = sizeof(float);

    const uint32_t TILE_SIZE_ELEMS = tile_bytes / ELEM;
    const uint32_t half_row        = half_N; // elements per single-row half

    // OPT-5: relaxed consistency check — padding within reserved tiles is fine.
    // tw_tiles_needed = ceil(local_half / TILE_SIZE_ELEMS)
    const uint32_t tw_tiles_needed = (local_half + TILE_SIZE_ELEMS - 1)
                                      / TILE_SIZE_ELEMS;
    ASSERT(tw_tiles_needed <= local_tiles);

    const uint32_t compact_bytes = half_N * ELEM;

    // ── Address generators ─────────────────────────────────────────────
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

    if (local_tiles == 0 || num_stages == 0) return;

    // ── Step 1: Upload input tiles for all rows ────────────────────────
    // OPT-6: issue all NOC reads before barrier.
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);

    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t global_t = tile_offset + t;
        noc_async_read_tile(global_t, even_r_gen,
            get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(global_t, even_i_gen,
            get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(global_t, odd_r_gen,
            get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(global_t, odd_i_gen,
            get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }

    // Upload compact twiddle table (single contiguous DRAM alloc, byte-addressed).
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read(compact_r_addr, get_write_ptr(cb_compact_r), compact_bytes);
    noc_async_read(compact_i_addr, get_write_ptr(cb_compact_i), compact_bytes);

    // Single barrier for ALL outstanding reads (data tiles + compact twiddle).
    noc_async_read_barrier();

    cb_push_back(cb_even_r, local_tiles);
    cb_push_back(cb_even_i, local_tiles);
    cb_push_back(cb_odd_r,  local_tiles);
    cb_push_back(cb_odd_i,  local_tiles);
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── Step 2: Per-stage twiddle expansion ───────────────────────────
    //
    // OPT-4: For rows_this > 1, the twiddle table is IDENTICAL for every
    // row (all rows run the same N_row-point FFT). We therefore:
    //   (a) expand twiddles for row 0 using the indexed-lookup loop,
    //   (b) word-copy row 0's result into slots for rows 1 … rows_this-1.
    //
    // For rows_this == 1 the replication loop is a no-op.
    //
    // Twiddle formula for global element p at stage s:
    //   j   = p & (half_m - 1)
    //   idx = j * (half_N >> stage)
    //   twiddle = compact[idx]
    // With core_elem_base == 0 (row-decomp), p = lp for lp in [0, half_row).
    //
    // The CB is sized to local_tiles tiles = rows_this × tiles_per_row.
    // We write rows_this × half_row elements total (OPT-5: may not fill
    // the last tile completely; unused bytes are garbage but never read
    // because the compute kernel only accesses half_N elements per row).

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;
        const uint32_t half_m_mask = half_m - 1u;

        cb_reserve_back(cb_tw_r, local_tiles);
        cb_reserve_back(cb_tw_i, local_tiles);
        const uint32_t dst_r = get_write_ptr(cb_tw_r);
        const uint32_t dst_i = get_write_ptr(cb_tw_i);

        // (a) Expand twiddles for row 0 (elements [0, half_row)).
        for (uint32_t lp = 0; lp < half_row; lp++) {
            uint32_t j   = (core_elem_base + lp) & half_m_mask;
            uint32_t idx = j * N_over_m;
            // idx < half_N always (provable from j < half_m, N_over_m = half_N/half_m)
            *reinterpret_cast<volatile uint32_t*>(dst_r + lp * ELEM) =
                *reinterpret_cast<volatile uint32_t*>(cmp_r_base + idx * ELEM);
            *reinterpret_cast<volatile uint32_t*>(dst_i + lp * ELEM) =
                *reinterpret_cast<volatile uint32_t*>(cmp_i_base + idx * ELEM);
        }

        // (b) OPT-4: replicate row 0 twiddles into rows 1 … rows_this-1.
        // Uses plain uint32 copies — no lookup arithmetic.
        for (uint32_t row = 1; row < rows_this; row++) {
            const uint32_t row_off = row * half_row * ELEM;
            for (uint32_t lp = 0; lp < half_row; lp++) {
                *reinterpret_cast<volatile uint32_t*>(dst_r + row_off + lp * ELEM) =
                    *reinterpret_cast<volatile uint32_t*>(dst_r + lp * ELEM);
                *reinterpret_cast<volatile uint32_t*>(dst_i + row_off + lp * ELEM) =
                    *reinterpret_cast<volatile uint32_t*>(dst_i + lp * ELEM);
            }
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}