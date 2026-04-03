// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FFT READER (IN DATA MOVER) KERNEL — Paper-Aligned Implementation
// Ref: "Exploring Fast Fourier Transforms on the Tenstorrent Wormhole"
//      Brown, Davies, Le Clair (arXiv:2506.15437v1)
//
// Responsibilities (Fig. 3 of paper):
//   Step 0:  Read input from external DRAM into cb_data0/1 (even/odd split)
//   Step 1+: Data already in SRAM (fed back from writer's single-copy stage)
//            so only load twiddle factors from SRAM each step.
//
// Optimizations implemented:
//   [1] Chunked loading — fills CBs in tile-sized chunks so compute can start
//       while more data is still being loaded (pipeline overlap).
//   [2] Batched NOC reads — single noc_async_read_barrier() per chunk rather
//       than one barrier per tile.
//   [3] Pre-computed twiddles — twiddle factors written to DRAM once at init
//       by the host; reader loads them each step via fast NOC DMA.
//   [4] Separate NOC paths — reader uses RISCV_0/NOC_0, writer uses RISCV_1/
//       NOC_1, so both can transfer simultaneously without contention.

#include <cstdint>
#include "dataflow_api.h"

void MAIN {
    // ── Runtime arguments (set per-core by host in fft_multi_core.cpp) ─────
    const uint32_t src_data0_r_addr  = get_arg_val<uint32_t>(0);  // even real
    const uint32_t src_data0_i_addr  = get_arg_val<uint32_t>(1);  // even imag
    const uint32_t src_data1_r_addr  = get_arg_val<uint32_t>(2);  // odd  real
    const uint32_t src_data1_i_addr  = get_arg_val<uint32_t>(3);  // odd  imag
    const uint32_t src_twiddle_r_addr= get_arg_val<uint32_t>(4);  // twiddle real
    const uint32_t src_twiddle_i_addr= get_arg_val<uint32_t>(5);  // twiddle imag
    const uint32_t tiles_per_row     = get_arg_val<uint32_t>(6);  // N/2 / TILE_SIZE
    const uint32_t tile_offset       = get_arg_val<uint32_t>(7);  // core's DRAM offset
    const uint32_t num_steps         = get_arg_val<uint32_t>(8);  // log2(N)
    const uint32_t rows_per_core     = get_arg_val<uint32_t>(9);

    // ── CB indices ─────────────────────────────────────────────────────────
    constexpr uint32_t cb_data0_r = 0;
    constexpr uint32_t cb_data0_i = 1;
    constexpr uint32_t cb_data1_r = 2;
    constexpr uint32_t cb_data1_i = 3;
    constexpr uint32_t cb_tw_r    = 4;
    constexpr uint32_t cb_tw_i    = 5;

    const uint32_t tile_bytes    = get_tile_size(cb_data0_r);
    const DataFormat data_format = get_dataformat(cb_data0_r);

    // ── Address generators — fast interleaved DRAM access ─────────────────
    // InterleavedAddrGenFast<true> uses DRAM interleaving for bank-level
    // parallelism across the 24 on-card GDDR6 banks.
    const InterleavedAddrGenFast<true> data0_r_gen = {
        .bank_base_address = src_data0_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> data0_i_gen = {
        .bank_base_address = src_data0_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> data1_r_gen = {
        .bank_base_address = src_data1_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> data1_i_gen = {
        .bank_base_address = src_data1_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = src_twiddle_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = src_twiddle_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    // ── Per-row processing ─────────────────────────────────────────────────
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        // ══════════════════════════════════════════════════════════════════
        // STEP 0: Load initial data from external DRAM (Fig. 3: "First step")
        //
        // Paper: "the in data mover core will read the input data for that step.
        //         This is read either from external on-card DDR for the first step
        //         or from local SRAM for subsequent ones."
        //
        // Optimization [1] Chunked: we push one tile at a time so compute
        // can begin on chunk 0 while we're still loading chunk 1, etc.
        // ══════════════════════════════════════════════════════════════════
        for (uint32_t t = 0; t < tiles_per_row; t++) {
            const uint32_t tile_id = row_tile_base + t;

            // Reserve one page in each CB before issuing NOC reads
            cb_reserve_back(cb_data0_r, 1);
            cb_reserve_back(cb_data0_i, 1);
            cb_reserve_back(cb_data1_r, 1);
            cb_reserve_back(cb_data1_i, 1);

            uint32_t d0r_ptr = get_write_ptr(cb_data0_r);
            uint32_t d0i_ptr = get_write_ptr(cb_data0_i);
            uint32_t d1r_ptr = get_write_ptr(cb_data1_r);
            uint32_t d1i_ptr = get_write_ptr(cb_data1_i);

            // Optimization [2]: Issue all 4 reads before a single barrier
            noc_async_read_tile(tile_id, data0_r_gen, d0r_ptr);
            noc_async_read_tile(tile_id, data0_i_gen, d0i_ptr);
            noc_async_read_tile(tile_id, data1_r_gen, d1r_ptr);
            noc_async_read_tile(tile_id, data1_i_gen, d1i_ptr);

            noc_async_read_barrier();  // One barrier per chunk, not per tile

            cb_push_back(cb_data0_r, 1);
            cb_push_back(cb_data0_i, 1);
            cb_push_back(cb_data1_r, 1);
            cb_push_back(cb_data1_i, 1);
        }

        // ══════════════════════════════════════════════════════════════════
        // ALL STEPS: Load pre-computed twiddle factors for this step/row
        //
        // Optimization [3]: Twiddle tiles are pre-computed by the host once
        // and stored in DRAM (fft_multi_core.cpp: precompute_all_twiddle_tiles).
        // This avoids expensive on-device sin/cos evaluation per step.
        //
        // The writer kernel's single-copy optimization (Fig. 5) means that
        // for steps 1+, data0/1 CBs are already filled from SRAM by the writer,
        // so the reader only needs to supply twiddles here.
        // ══════════════════════════════════════════════════════════════════
        for (uint32_t step = 0; step < num_steps; step++) {
            // Twiddle layout in DRAM:
            //   tile_id = (step * rows_per_core * num_cores + global_row) * tiles_per_row + t
            // Here tile_offset already encodes the core's base.
            const uint32_t tw_base = (step * rows_per_core + row) * tiles_per_row;

            for (uint32_t t = 0; t < tiles_per_row; t++) {
                const uint32_t tw_tile_id = tw_base + t;

                cb_reserve_back(cb_tw_r, 1);
                cb_reserve_back(cb_tw_i, 1);

                uint32_t tw_r_ptr = get_write_ptr(cb_tw_r);
                uint32_t tw_i_ptr = get_write_ptr(cb_tw_i);

                noc_async_read_tile(tw_tile_id, tw_r_gen, tw_r_ptr);
                noc_async_read_tile(tw_tile_id, tw_i_gen, tw_i_ptr);

                noc_async_read_barrier();

                cb_push_back(cb_tw_r, 1);
                cb_push_back(cb_tw_i, 1);
            }
        }
    }
}