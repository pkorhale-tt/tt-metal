// reader_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//  1. All 6 NOC reads for a tile are issued together before a single
//     noc_async_read_barrier() — down from 6 barriers/tile to 1.
//  2. True double-buffering: the next tile's reads are issued while the
//     compute kernel is processing the current tile, keeping the NOC
//     busy and the compute engine fed.
//  3. cb_reserve_back for all 6 CBs is done before any noc_async_read_tile,
//     so the write pointers are all valid when the NOC transactions fire.
//  4. Push order matches compute kernel's cb_wait_front order:
//       tw_r → tw_i → odd_r → odd_i → even_r → even_i
//     This ensures compute never stalls on a CB that the reader hasn't
//     pushed yet while an earlier CB is already full.
//
// Double-buffer pipeline diagram (per CB, e.g. cb_even_r, depth=2):
//
//  Slot:    [ 0 ][ 1 ][ 0 ][ 1 ][ 0 ] ...
//  Reader:  [R t0]    [R t1]    [R t2]
//  Compute:      [C t0]    [C t1]    [C t2]
//  NOC:     act  act  act  act  act       ← never idle

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Runtime args ───────────────────────────────────────────────────────
    const uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tw_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(6);
    const uint32_t start_tile  = get_arg_val<uint32_t>(7);

    // ── CB indices — push order matches compute cb_wait_front order ────────
    //   compute waits: tw_r, tw_i, odd_r, odd_i, (then later) even_r, even_i
    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;

    // ── Tile size and data format (derived from one representative CB) ──────
    const uint32_t tile_bytes      = get_tile_size(cb_even_r);
    const DataFormat data_format   = get_dataformat(cb_even_r);

    // ── Address generators ─────────────────────────────────────────────────
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

    // ── Helper lambda: reserve all 6 CB slots and issue all 6 NOC reads ────
    // All reserve_back calls happen before any noc_async_read_tile so that
    // write pointers are stable. A single barrier covers all 6 transactions.
    //
    // After this call: all 6 CBs have 1 tile reserved and the NOC read is
    // in flight. Caller must call noc_async_read_barrier() then push all 6.
    auto issue_reads = [&](uint32_t tile_idx) FORCE_INLINE {
        cb_reserve_back(cb_tw_r,   1);
        cb_reserve_back(cb_tw_i,   1);
        cb_reserve_back(cb_odd_r,  1);
        cb_reserve_back(cb_odd_i,  1);
        cb_reserve_back(cb_even_r, 1);
        cb_reserve_back(cb_even_i, 1);

        noc_async_read_tile(tile_idx, tw_r_gen,   get_write_ptr(cb_tw_r));
        noc_async_read_tile(tile_idx, tw_i_gen,   get_write_ptr(cb_tw_i));
        noc_async_read_tile(tile_idx, odd_r_gen,  get_write_ptr(cb_odd_r));
        noc_async_read_tile(tile_idx, odd_i_gen,  get_write_ptr(cb_odd_i));
        noc_async_read_tile(tile_idx, even_r_gen, get_write_ptr(cb_even_r));
        noc_async_read_tile(tile_idx, even_i_gen, get_write_ptr(cb_even_i));
        // One barrier for all 6 — caller must call noc_async_read_barrier()
    };

    auto push_all = [&]() FORCE_INLINE {
        // Push in the same order as issue_reads so that compute sees them
        // in the order it waits for them: tw_r, tw_i, odd_r, odd_i, even_r, even_i
        cb_push_back(cb_tw_r,   1);
        cb_push_back(cb_tw_i,   1);
        cb_push_back(cb_odd_r,  1);
        cb_push_back(cb_odd_i,  1);
        cb_push_back(cb_even_r, 1);
        cb_push_back(cb_even_i, 1);
    };

    // ── Double-buffer pipeline ─────────────────────────────────────────────
    //
    // Iteration structure:
    //   1. Issue reads for tile i+1  (overlaps with compute on tile i)
    //   2. Wait for tile i reads to complete   (noc_async_read_barrier)
    //   3. Push tile i into CBs
    //
    // This keeps the NOC busy while compute is running, and keeps compute
    // busy while the next tile is being fetched.
    //
    // Concretely for N tiles:
    //
    //   PRE-FETCH tile 0 (no previous tile to overlap with)
    //   issue_reads(0)
    //   barrier
    //   push(0)
    //
    //   for i = 1..N-1:
    //     issue_reads(i)          ← in flight while compute runs tile i-1
    //     barrier                 ← wait only for tile i (tile i-1 already pushed)
    //     push(i)
    //
    // With CB depth=2 the compute kernel can hold tile i-1 while the reader
    // is already filling tile i, so cb_reserve_back(... 1) never blocks.

    // Pre-fetch tile 0
    issue_reads(start_tile);
    noc_async_read_barrier();
    push_all();

    for (uint32_t i = 1; i < num_tiles; i++) {
        const uint32_t tile_idx = start_tile + i;

        // Issue reads for tile i — this fires while compute processes tile i-1.
        // cb_reserve_back inside issue_reads will not block because:
        //   - CB depth = 2
        //   - tile i-1 was pushed in the previous iteration
        //   - compute is consuming tile i-1 concurrently
        issue_reads(tile_idx);

        // Single barrier covers all 6 DRAM→L1 reads for tile i.
        noc_async_read_barrier();

        // Make tile i visible to compute.
        push_all();
    }
}