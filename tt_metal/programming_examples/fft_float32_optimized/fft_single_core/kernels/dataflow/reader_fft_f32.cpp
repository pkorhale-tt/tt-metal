// reader_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  READER RESPONSIBILITIES
// ═══════════════════════════════════════════════════════════════════
//
//  Phase 1 — Input data (stage 0 only):
//    Read even_r, even_i, odd_r, odd_i from DRAM into CBs c_0..c_3.
//    All 4 reads batched under ONE barrier per tile.
//    Double-buffered: tile i+1 fetched while compute processes tile i.
//
//  Phase 2 — Twiddle streaming (every stage):
//    For each stage s in 0..num_stages-1:
//      Read twiddle slice [s * tiles_per_stage .. (s+1) * tiles_per_stage]
//      into cb_tw_r (c_4) and cb_tw_i (c_5).
//      Double-buffered per tile within each stage.
//
//  NOC optimization:
//    All reads for a tile batched before single barrier.
//    Previous version: 6 barriers per tile.
//    This version:     1 barrier per tile.
//
//  CB push order matches compute cb_wait_front order:
//    tw_r → tw_i → odd_r → odd_i → even_r → even_i
//    (compute waits on odd+twiddle first, then even)
//
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// ── helpers ─────────────────────────────────────────────────────────

// Issue one tile read into a CB slot (no barrier — caller batches)
FORCE_INLINE void issue_read(
    uint32_t tile_idx,
    const InterleavedAddrGenFast<true>& gen,
    uint32_t cb)
{
    cb_reserve_back(cb, 1);
    noc_async_read_tile(tile_idx, gen, get_write_ptr(cb));
    // NO barrier here — caller fires all reads then one barrier
}

// Push after barrier confirms data is in L1
FORCE_INLINE void push(uint32_t cb) {
    cb_push_back(cb, 1);
}

void kernel_main() {

    // ── Runtime args ───────────────────────────────────────────────
    const uint32_t even_r_addr     = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr     = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr      = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr      = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr       = get_arg_val<uint32_t>(4);  // all-stages twiddle real
    const uint32_t tw_i_addr       = get_arg_val<uint32_t>(5);  // all-stages twiddle imag
    const uint32_t num_tiles       = get_arg_val<uint32_t>(6);  // tiles per stage
    const uint32_t num_stages      = get_arg_val<uint32_t>(7);

    // ── CB indices ─────────────────────────────────────────────────
    // Push order matches compute wait order:
    //   tw_r → tw_i → odd_r → odd_i → even_r → even_i
    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;

    // ── Tile geometry ──────────────────────────────────────────────
    const uint32_t tile_bytes   = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);

    // ── Address generators ─────────────────────────────────────────
    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> odd_r_gen = {
        .bank_base_address = odd_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> odd_i_gen = {
        .bank_base_address = odd_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };

    if (num_tiles == 0) return;

    // ══════════════════════════════════════════════════════════════
    //  PHASE 1: Stream input data for stage 0
    //
    //  Double-buffer pattern:
    //    Pre-fetch tile 0 (no prior tile to overlap with).
    //    For tiles 1..N-1:
    //      Issue reads for tile i     ← NOC busy
    //      (compute is processing tile i-1 simultaneously)
    //      barrier (wait for tile i)
    //      push tile i
    //
    //  All 4 CBs (even_r/i, odd_r/i) read in ONE barrier per tile.
    //  Twiddle for stage 0 is interleaved here too.
    // ══════════════════════════════════════════════════════════════

    // Pre-fetch tile 0: all 6 CBs together, one barrier
    {
        issue_read(0, tw_r_gen,   cb_tw_r);
        issue_read(0, tw_i_gen,   cb_tw_i);
        issue_read(0, odd_r_gen,  cb_odd_r);
        issue_read(0, odd_i_gen,  cb_odd_i);
        issue_read(0, even_r_gen, cb_even_r);
        issue_read(0, even_i_gen, cb_even_i);
        noc_async_read_barrier();               // 1 barrier covers all 6
        push(cb_tw_r);
        push(cb_tw_i);
        push(cb_odd_r);
        push(cb_odd_i);
        push(cb_even_r);
        push(cb_even_i);
    }

    // Tiles 1..num_tiles-1: double-buffer pipeline
    for (uint32_t t = 1; t < num_tiles; t++) {
        // Issue reads for tile t — fires while compute processes tile t-1
        issue_read(t, tw_r_gen,   cb_tw_r);
        issue_read(t, tw_i_gen,   cb_tw_i);
        issue_read(t, odd_r_gen,  cb_odd_r);
        issue_read(t, odd_i_gen,  cb_odd_i);
        issue_read(t, even_r_gen, cb_even_r);
        issue_read(t, even_i_gen, cb_even_i);

        // Wait for tile t to arrive — tile t-1 is already in compute
        noc_async_read_barrier();

        push(cb_tw_r);
        push(cb_tw_i);
        push(cb_odd_r);
        push(cb_odd_i);
        push(cb_even_r);
        push(cb_even_i);
    }

    // ══════════════════════════════════════════════════════════════
    //  PHASE 2: Stream twiddle factors for stages 1..num_stages-1
    //
    //  The twiddle DRAM buffer layout (set up by host):
    //    [stage 0 twiddles: num_tiles tiles] [stage 1 twiddles: ...]
    //    ... [stage N-1 twiddles: ...]
    //
    //  Stage 0 twiddles were already streamed in Phase 1.
    //  For stages 1..N-1, only twiddles needed (data is in L1 ping/pong).
    //
    //  Double-buffer per tile within each stage.
    // ══════════════════════════════════════════════════════════════

    for (uint32_t stage = 1; stage < num_stages; stage++) {
        const uint32_t stage_tile_offset = stage * num_tiles;

        // Pre-fetch first tile of this stage's twiddles
        {
            const uint32_t global_tile = stage_tile_offset + 0;
            issue_read(global_tile, tw_r_gen, cb_tw_r);
            issue_read(global_tile, tw_i_gen, cb_tw_i);
            noc_async_read_barrier();
            push(cb_tw_r);
            push(cb_tw_i);
        }

        // Double-buffer remaining tiles
        for (uint32_t t = 1; t < num_tiles; t++) {
            const uint32_t global_tile = stage_tile_offset + t;

            // Issue tile t reads — fires while compute processes tile t-1
            issue_read(global_tile, tw_r_gen, cb_tw_r);
            issue_read(global_tile, tw_i_gen, cb_tw_i);

            noc_async_read_barrier();

            push(cb_tw_r);
            push(cb_tw_i);
        }
    }
}