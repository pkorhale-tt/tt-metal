// writer_fft_f32.cpp  — OPTIMISED
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//   1. cb_wait_front for all four output CBs now waits for num_tiles tiles
//      in one call outside the loop, instead of waiting 1 tile at a time
//      inside the loop.  All four CBs are produced together by compute so
//      there is no point serialising the waits.
//   2. All noc_async_write_tile calls are issued inside the loop without
//      any barrier between them — the NOC handles them concurrently.
//   3. ONE noc_async_write_barrier() after the loop waits for every
//      in-flight write to complete before the CB slots are freed.
//   4. cb_pop_front drains all num_tiles tiles in one call instead of 1-by-1.
//   5. Pointer stride: each tile's source address is offset by t*tile_bytes
//      so every tile reads from its own CB slot.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);

    // CB indices
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

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

    if (num_tiles == 0) {
        return;
    }

    // ── Wait for ALL output tiles to be ready before writing anything ─────
    // Compute produces all four CBs together, so once out0_r has num_tiles
    // tiles, the other three do too.  Waiting in bulk avoids the overhead
    // of num_tiles individual synchronisation checks.
    cb_wait_front(cb_out0_r, num_tiles);
    cb_wait_front(cb_out0_i, num_tiles);
    cb_wait_front(cb_out1_r, num_tiles);
    cb_wait_front(cb_out1_i, num_tiles);

    // ── Issue all DRAM writes in one burst ────────────────────────────────
    // No barrier inside the loop — the NOC queues these concurrently and
    // the single barrier below waits for all of them to complete.
    for (uint32_t t = 0; t < num_tiles; t++) {
        noc_async_write_tile(t, out0_r_gen,
                             get_read_ptr(cb_out0_r) + t * tile_bytes);
        noc_async_write_tile(t, out0_i_gen,
                             get_read_ptr(cb_out0_i) + t * tile_bytes);
        noc_async_write_tile(t, out1_r_gen,
                             get_read_ptr(cb_out1_r) + t * tile_bytes);
        noc_async_write_tile(t, out1_i_gen,
                             get_read_ptr(cb_out1_i) + t * tile_bytes);
    }

    // ONE barrier for the entire write burst.
    noc_async_write_barrier();

    // ── Free all CB slots in one call ─────────────────────────────────────
    cb_pop_front(cb_out0_r, num_tiles);
    cb_pop_front(cb_out0_i, num_tiles);
    cb_pop_front(cb_out1_r, num_tiles);
    cb_pop_front(cb_out1_i, num_tiles);
}