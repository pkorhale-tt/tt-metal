// reader_fft_f32_mc.cpp  — MULTICORE reader
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Each core reads its own slice of the input from DRAM.
// Slice: core_id * local_tiles .. (core_id+1) * local_tiles  (tile-indexed).
//
// Compact twiddle table strategy (same as single-core v2):
//   Host uploads N/2 twiddle entries once.
//   This reader uploads the full compact table to every core's L1.
//   Per-stage expansion is done locally — no extra DRAM per stage.
//
// IMPORTANT — inter-core stages:
//   For the early FFT stages (stage < log2(num_cores)) butterfly pairs
//   span across cores.  The writer on each core handles the cross-core
//   NOC transfer; the reader only needs to push its local compact table
//   and the twiddle expansion for every stage.
//
// Args:
//   0  even_r_addr        DRAM base — even real  (bit-reversed, split)
//   1  even_i_addr        DRAM base — even imag
//   2  odd_r_addr         DRAM base — odd  real
//   3  odd_i_addr         DRAM base — odd  imag
//   4  compact_r_addr     DRAM base — compact twiddle real  (N/2 floats)
//   5  compact_i_addr     DRAM base — compact twiddle imag
//   6  local_tiles        number of tiles this core owns
//   7  tile_offset        first global tile index for this core
//   8  num_stages         log2N
//   9  half_N             N/2 (global)
//  10  local_half         N / (2 * num_cores) — elements in this core's slice

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
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

    // Compact twiddle: page_size = N/2 floats
    const uint32_t compact_bytes = half_N * sizeof(float);

    // ── Address generators for per-core tile slice ────────────────────
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

    const InterleavedAddrGenFast<true> cmp_r_gen = {
        .bank_base_address = compact_r_addr,
        .page_size = compact_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_i_gen = {
        .bank_base_address = compact_i_addr,
        .page_size = compact_bytes, .data_format = data_format };

    if (local_tiles == 0 || num_stages == 0) return;

    constexpr uint32_t ELEM = sizeof(float);

    // ── Step 1: Upload this core's input slice + compact twiddle table ─
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);

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
    // Every core gets a full copy of the compact twiddle table (N/2 entries).
    // This is cheap: compact_bytes = N/2 * 4 = 2KB for N=1024.
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
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

    // ── Step 2: Per-stage twiddle expansion (local L1, no DRAM) ─────────
    //
    // The twiddle for global element index p (0..half_N-1) at stage s is:
    //   j   = p & (half_m - 1)       half_m = 1 << stage
    //   idx = j * (half_N >> stage)
    //   twiddle = compact[idx]
    //
    // Each core only writes twiddles for its own local slice:
    //   global_p = core_element_base + local_p
    //   where core_element_base = tile_offset * TILE_SIZE (for 1 tile/core)
    //
    // This means every core expands only local_half twiddle values per stage,
    // not the full N/2 — linear scaling with num_cores.

    const uint32_t core_elem_base = tile_offset * (tile_bytes / ELEM);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;   // = N >> (stage+1)
        const uint32_t half_m_mask = half_m - 1u;

        cb_reserve_back(cb_tw_r, local_tiles);
        cb_reserve_back(cb_tw_i, local_tiles);
        const uint32_t dst_r = get_write_ptr(cb_tw_r);
        const uint32_t dst_i = get_write_ptr(cb_tw_i);

        // Expand twiddle for local elements only
        for (uint32_t lp = 0; lp < local_half; lp++) {
            uint32_t p   = core_elem_base + lp;   // global butterfly index
            uint32_t j   = p & half_m_mask;
            uint32_t idx = j * N_over_m;

            uint32_t raw_r = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_r_base + idx * ELEM);
            uint32_t raw_i = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_i_base + idx * ELEM);
            *reinterpret_cast<volatile uint32_t*>(dst_r + lp * ELEM) = raw_r;
            *reinterpret_cast<volatile uint32_t*>(dst_i + lp * ELEM) = raw_i;
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);

        // CB depth=1 ensures back-pressure: next cb_reserve_back will stall
        // until compute + writer have consumed this stage's twiddles.
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}