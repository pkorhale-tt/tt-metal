// reader_fft_f32_mc.cpp — MULTICORE reader (FIXED: no ThCon in dataflow)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// REMOVED: All TT_SETDMAREG / TT_LOADIND / TT_STOREIND / p_ind / LOWER_HALFWORD
//          / UPPER_HALFWORD / LO_16 / HI_16 / copy128 / store4_via_bounce.
//
//   These are ThCon (Tensor Controller) intrinsics that only exist in the
//   TRISC compute kernel translation unit. BRISC (this file) compiles with
//   a different include path that does not expose llk_defs.h or the ThCon
//   register file. Using them here causes "not declared in this scope" errors.
//
//   The correct approach for dataflow kernels is to use NOC DMA for DRAM
//   transfers (noc_async_read_tile / noc_async_write_tile) and plain scalar
//   reads/writes (volatile uint32_t*) for L1 copies. These are the only
//   memory operations available to BRISC/NCRISC.
//
// WHAT IS KEPT:
//   - Outer rows_per_core loop (row 0..rows_per_core-1)
//   - NOC async reads for stage-0 input tiles
//   - Compact twiddle table loaded once before the row loop
//   - Per-stage twiddle expansion via scalar scatter-read into CB
//   - All CB push/pop/wait/reserve protocol

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
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(11);

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
    const uint32_t compact_bytes = half_N * sizeof(float);

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
        .page_size = compact_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_i_gen  = {
        .bank_base_address = compact_i_addr,
        .page_size = compact_bytes, .data_format = data_format };

    if (local_tiles == 0 || num_stages == 0 || rows_per_core == 0) return;

    constexpr uint32_t ELEM = sizeof(float);

    // Scalar L1 read/write — the only memory ops available in BRISC/NCRISC.
    // No ThCon, no LLK intrinsics — those are compute-core only.
    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // Load compact twiddle table once — same for every row.
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

    // Outer loop: process each row independently.
    // tile_offset is the base for this core; each row advances by local_tiles.
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * local_tiles;
        const uint32_t row_elem_base = row_tile_base * (tile_bytes / ELEM);

        // Load stage-0 input tiles for this row from DRAM.
        cb_reserve_back(cb_even_r, local_tiles);
        cb_reserve_back(cb_even_i, local_tiles);
        cb_reserve_back(cb_odd_r,  local_tiles);
        cb_reserve_back(cb_odd_i,  local_tiles);

        for (uint32_t t = 0; t < local_tiles; t++) {
            uint32_t gt = row_tile_base + t;
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

        // Per-stage twiddle expansion for this row.
        // Scalar scatter-read from compact table into CB.
        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const uint32_t half_m      = 1u << stage;
            const uint32_t N_over_m    = half_N >> stage;
            const uint32_t half_m_mask = half_m - 1u;

            cb_reserve_back(cb_tw_r, local_tiles);
            cb_reserve_back(cb_tw_i, local_tiles);
            const uint32_t dst_r = get_write_ptr(cb_tw_r);
            const uint32_t dst_i = get_write_ptr(cb_tw_i);

            for (uint32_t lp = 0; lp < local_half; lp++) {
                const uint32_t p   = row_elem_base + lp;
                const uint32_t idx = (p & half_m_mask) * N_over_m;
                wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
            }

            cb_push_back(cb_tw_r, local_tiles);
            cb_push_back(cb_tw_i, local_tiles);
        }
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}