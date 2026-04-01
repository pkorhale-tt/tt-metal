// reader_fft_1d_64core.cpp - FIXED
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Responsibilities:
//   1. Load bit-reversed even/odd input tiles from DRAM once.
//   2. Load the compact twiddle table (half_N floats) from DRAM once.
//   3. For each FFT stage, expand twiddle factors into per-element
//      cb_tw_r / cb_tw_i tiles and push them to the compute kernel.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Runtime args ─────────────────────────────────────────────
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr = get_arg_val<uint32_t>(4);
    const uint32_t compact_i_addr = get_arg_val<uint32_t>(5);
    const uint32_t local_tiles    = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t half_N         = get_arg_val<uint32_t>(9);
    const uint32_t local_half     = get_arg_val<uint32_t>(10);
    const uint32_t core_elem_base = get_arg_val<uint32_t>(11);
    // args 12-15 (core_id, num_cores, log2_cores, local_stages) reserved
    // for future cross-core twiddle selection

    // ── CB indices (must match host and compute kernel) ──────────
    constexpr uint32_t cb_even_r    = 0;
    constexpr uint32_t cb_even_i    = 1;
    constexpr uint32_t cb_odd_r     = 2;
    constexpr uint32_t cb_odd_i     = 3;
    constexpr uint32_t cb_tw_r      = 4;
    constexpr uint32_t cb_tw_i      = 5;
    constexpr uint32_t cb_compact_r = 10;
    constexpr uint32_t cb_compact_i = 11;

    const uint32_t    tile_bytes  = get_tile_size(cb_even_r);
    const DataFormat  data_format = get_dataformat(cb_even_r);
    constexpr uint32_t ELEM       = sizeof(float);
    constexpr uint32_t TILE_SIZE  = 1024;   // 32×32

    // ── DRAM address generators ──────────────────────────────────
    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_r_gen = {
        .bank_base_address = odd_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> odd_i_gen = {
        .bank_base_address = odd_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    // ═════════════════════════════════════════════════════════════
    // Phase 1: Load Initial (bit-reversed) Input Tiles from DRAM
    // ═════════════════════════════════════════════════════════════
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);

    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t gt = tile_offset + t;
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

    cb_push_back(cb_even_r, local_tiles);
    cb_push_back(cb_even_i, local_tiles);
    cb_push_back(cb_odd_r,  local_tiles);
    cb_push_back(cb_odd_i,  local_tiles);

    // ═════════════════════════════════════════════════════════════
    // Phase 2: Load Compact Twiddle Table (half_N floats, raw DMA)
    // ═════════════════════════════════════════════════════════════
    const uint32_t compact_bytes = half_N * ELEM;

    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);

    noc_async_read(compact_r_addr, get_write_ptr(cb_compact_r), compact_bytes);
    noc_async_read(compact_i_addr, get_write_ptr(cb_compact_i), compact_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);

    volatile uint32_t* cmp_r_base =
        reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_compact_r));
    volatile uint32_t* cmp_i_base =
        reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_compact_i));

    // ═════════════════════════════════════════════════════════════
    // Phase 3: Per-Stage Twiddle Expansion
    //
    // For DIT radix-2, stage s (0-indexed):
    //   m        = 2^(s+1)   butterfly group size
    //   half_m   = 2^s       elements in each half-group
    //   N_over_m = (N/2) >> s  stride in the compact twiddle table
    //
    // The twiddle index for global element i is:
    //   j   = i mod half_m          (position within half-group)
    //   idx = j * N_over_m          (index into W_N[] compact table)
    // ═════════════════════════════════════════════════════════════
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;
        const uint32_t half_m_mask = half_m - 1u;

        cb_reserve_back(cb_tw_r, local_tiles);
        cb_reserve_back(cb_tw_i, local_tiles);

        volatile uint32_t* dst_r =
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_tw_r));
        volatile uint32_t* dst_i =
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_tw_i));

        // Expand twiddles for the local_half active elements
        for (uint32_t lp = 0; lp < local_half; lp++) {
            uint32_t global_elem = core_elem_base + lp;
            uint32_t j           = global_elem & half_m_mask;
            uint32_t idx         = j * N_over_m;
            if (idx >= half_N) idx = 0;   // bounds safety

            dst_r[lp] = cmp_r_base[idx];
            dst_i[lp] = cmp_i_base[idx];
        }

        // Zero-pad the rest of the tile(s) to avoid reading stale data
        for (uint32_t lp = local_half; lp < local_tiles * TILE_SIZE; lp++) {
            dst_r[lp] = 0u;
            dst_i[lp] = 0u;
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }

    // Release compact twiddle scratch space
    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}