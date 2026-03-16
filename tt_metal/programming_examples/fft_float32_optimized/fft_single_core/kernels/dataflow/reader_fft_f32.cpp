// reader_fft_f32.cpp  — OPTIMISED
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//   1. Phase 1: reserve all num_tiles slots upfront, issue all 6*num_tiles
//      NOC reads without any barrier inside the loop, then ONE barrier for
//      all of them, then push all num_tiles tiles.  NOC stays busy the whole
//      time instead of stalling after every tile.
//   2. Phase 2: same batch pattern for twiddle-only reads.  Also fixed the
//      write-pointer bug: each tile's read destination is now offset by
//      t * tile_bytes so tiles land in distinct CB slots instead of all
//      overwriting slot 0.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tw_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(6);
    const uint32_t num_stages  = get_arg_val<uint32_t>(7);

    // CB indices
    constexpr uint32_t cb_tw_r   = 4;
    constexpr uint32_t cb_tw_i   = 5;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;

    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);

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

    if (num_tiles == 0 || num_stages == 0) {
        return;
    }

    // ── Phase 1: Stage 0 — even, odd, and twiddle tiles ──────────────────
    // Reserve all slots upfront so the CB is ready for the full burst.
    cb_reserve_back(cb_tw_r,   num_tiles);
    cb_reserve_back(cb_tw_i,   num_tiles);
    cb_reserve_back(cb_odd_r,  num_tiles);
    cb_reserve_back(cb_odd_i,  num_tiles);
    cb_reserve_back(cb_even_r, num_tiles);
    cb_reserve_back(cb_even_i, num_tiles);

    // Issue all reads without any barrier between them — NOC stays busy.
    // Each tile lands at write_ptr + t*tile_bytes (distinct CB slot).
    for (uint32_t t = 0; t < num_tiles; t++) {
        noc_async_read_tile(t, tw_r_gen,   get_write_ptr(cb_tw_r)   + t * tile_bytes);
        noc_async_read_tile(t, tw_i_gen,   get_write_ptr(cb_tw_i)   + t * tile_bytes);
        noc_async_read_tile(t, odd_r_gen,  get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(t, odd_i_gen,  get_write_ptr(cb_odd_i)  + t * tile_bytes);
        noc_async_read_tile(t, even_r_gen, get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(t, even_i_gen, get_write_ptr(cb_even_i) + t * tile_bytes);
    }

    // One barrier waits for all 6*num_tiles in-flight reads to finish.
    noc_async_read_barrier();

    // Signal compute that a full stage worth of tiles is ready.
    cb_push_back(cb_tw_r,   num_tiles);
    cb_push_back(cb_tw_i,   num_tiles);
    cb_push_back(cb_odd_r,  num_tiles);
    cb_push_back(cb_odd_i,  num_tiles);
    cb_push_back(cb_even_r, num_tiles);
    cb_push_back(cb_even_i, num_tiles);

    // ── Phase 2: Stages 1..num_stages-1 — twiddle tiles only ─────────────
    // Even/odd data for stages 1+ comes from the ping-pong CBs written by
    // the compute kernel — the reader only needs to supply fresh twiddles.
    for (uint32_t stage = 1; stage < num_stages; stage++) {
        // Reserve a full stage of twiddle slots upfront.
        cb_reserve_back(cb_tw_r, num_tiles);
        cb_reserve_back(cb_tw_i, num_tiles);

        // Issue all reads for this stage in one burst.
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t global_tile = stage * num_tiles + t;
            // FIX (bug): offset by t*tile_bytes so each tile lands in its
            // own CB slot.  Original code always used get_write_ptr() with
            // no offset, so every tile overwrote slot 0.
            noc_async_read_tile(global_tile, tw_r_gen,
                                get_write_ptr(cb_tw_r) + t * tile_bytes);
            noc_async_read_tile(global_tile, tw_i_gen,
                                get_write_ptr(cb_tw_i) + t * tile_bytes);
        }

        // One barrier for the whole stage.
        noc_async_read_barrier();

        cb_push_back(cb_tw_r, num_tiles);
        cb_push_back(cb_tw_i, num_tiles);
    }
}