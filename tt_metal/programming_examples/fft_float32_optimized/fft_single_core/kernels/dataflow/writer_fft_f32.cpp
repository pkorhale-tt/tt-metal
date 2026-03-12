// writer_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  WRITER RESPONSIBILITIES
// ═══════════════════════════════════════════════════════════════════
//
//  Drains ONLY the final stage output CBs (c_16..c_19) to DRAM.
//  All intermediate stages are handled by ping-pong L1 buffers —
//  writer is NOT involved between stages.
//
//  Pattern per tile:
//    wait all 4 output CBs     ← compute must have pushed all 4
//    write all 4 to DRAM       ← 4 noc_async_write_tile in one shot
//    single barrier             ← 1 barrier covers all 4 writes
//    pop all 4                  ← free slots for compute's next tile
//
//  NOC optimization:
//    1 barrier per tile → 4× better NOC utilization vs naive.
//
//  Double-buffer benefit:
//    Output CBs depth=2.
//    Compute pushes tile i+1 into slot 1 while writer waits for
//    NOC to drain tile i. No stall between compute and writer.
//
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {

    // ── Runtime args ───────────────────────────────────────────────
    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);

    // ── CB indices ─────────────────────────────────────────────────
    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    // ── Tile geometry ──────────────────────────────────────────────
    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    // ── Address generators ─────────────────────────────────────────
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

    // ══════════════════════════════════════════════════════════════
    //  DRAIN LOOP
    //
    //  wait_all → write_all → barrier → pop_all
    //
    //  With depth=2 output CBs:
    //    compute pushes tile i+1 into slot 1 while
    //    writer barrier waits for tile i's NOC write to complete.
    //    Zero stall between compute and writer.
    // ══════════════════════════════════════════════════════════════

    for (uint32_t t = 0; t < num_tiles; t++) {

        cb_wait_front(cb_out0_r, 1);
        cb_wait_front(cb_out0_i, 1);
        cb_wait_front(cb_out1_r, 1);
        cb_wait_front(cb_out1_i, 1);

        noc_async_write_tile(t, out0_r_gen, get_read_ptr(cb_out0_r));
        noc_async_write_tile(t, out0_i_gen, get_read_ptr(cb_out0_i));
        noc_async_write_tile(t, out1_r_gen, get_read_ptr(cb_out1_r));
        noc_async_write_tile(t, out1_i_gen, get_read_ptr(cb_out1_i));

        noc_async_write_barrier();

        cb_pop_front(cb_out0_r, 1);
        cb_pop_front(cb_out0_i, 1);
        cb_pop_front(cb_out1_r, 1);
        cb_pop_front(cb_out1_i, 1);
    }
}