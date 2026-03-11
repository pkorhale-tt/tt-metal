// writer_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//  1. All 4 NOC writes for a tile are issued together before a single
//     noc_async_write_barrier() — down from 4 barriers/tile to 1.
//  2. True double-buffering on drain side: cb_wait_front for all 4 output
//     CBs is done together, then all 4 writes issued, then one barrier,
//     then all 4 pops. This keeps the NOC write path saturated.
//  3. Ordering: wait all → write all → barrier → pop all
//     This is the write-side mirror of the reader's
//     reserve all → read all → barrier → push all pattern.
//
// Double-buffer pipeline diagram (per output CB, depth=2):
//
//  Compute: [prod t0][prod t1][prod t2] ...
//  Slot:    [  0  ][  1  ][  0  ][  1  ]
//  Writer:       [W t0]      [W t1]      [W t2]
//  NOC:          act         act         act   ← no idle stalls

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Runtime args ───────────────────────────────────────────────────────
    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);
    const uint32_t start_tile  = get_arg_val<uint32_t>(5);

    // ── CB indices ─────────────────────────────────────────────────────────
    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    // ── Tile size and data format ──────────────────────────────────────────
    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    // ── Address generators ─────────────────────────────────────────────────
    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format
    };

    if (num_tiles == 0) return;

    // ── Main loop: wait-all → write-all → barrier → pop-all ───────────────
    //
    // Why this order?
    //
    //   wait-all:    Ensures all 4 tiles are ready before touching the NOC.
    //                With CB depth=2 compute can be producing tile i+1 while
    //                we are writing tile i — no stall.
    //
    //   write-all:   Issue all 4 noc_async_write_tile calls back-to-back.
    //                The NOC can pipeline / coalesce these more efficiently
    //                than interleaved barrier-per-write.
    //
    //   barrier:     One noc_async_write_barrier() covers all 4 writes.
    //                Previously there were 4 barriers per tile iteration.
    //
    //   pop-all:     Free all 4 CB slots together after the writes are done.
    //                This gives compute the maximum window to fill the next
    //                pair of slots.
    //
    // With output CB depth=2, the pipeline looks like:
    //
    //   Compute: [push t0_r][push t0_i][push t0_r1][push t0_i1]
    //                                  [push t1_r] [push t1_i] ...
    //   Writer:                        [wait t0]
    //                                  [write all 4 t0] → barrier → pop
    //                                              [wait t1] ...
    //
    // The writer's barrier overlaps with compute pushing t1.

    for (uint32_t i = 0; i < num_tiles; i++) {
        const uint32_t tile_idx = start_tile + i;

        // ── Wait for all 4 output tiles to be produced by compute ──────────
        cb_wait_front(cb_out0_r, 1);
        cb_wait_front(cb_out0_i, 1);
        cb_wait_front(cb_out1_r, 1);
        cb_wait_front(cb_out1_i, 1);

        // ── Issue all 4 NOC writes before any barrier ──────────────────────
        noc_async_write_tile(tile_idx, out0_r_gen, get_read_ptr(cb_out0_r));
        noc_async_write_tile(tile_idx, out0_i_gen, get_read_ptr(cb_out0_i));
        noc_async_write_tile(tile_idx, out1_r_gen, get_read_ptr(cb_out1_r));
        noc_async_write_tile(tile_idx, out1_i_gen, get_read_ptr(cb_out1_i));

        // ── Single barrier for all 4 L1→DRAM writes ───────────────────────
        noc_async_write_barrier();

        // ── Free all 4 CB slots — compute can now refill them ──────────────
        cb_pop_front(cb_out0_r, 1);
        cb_pop_front(cb_out0_i, 1);
        cb_pop_front(cb_out1_r, 1);
        cb_pop_front(cb_out1_i, 1);
    }
}