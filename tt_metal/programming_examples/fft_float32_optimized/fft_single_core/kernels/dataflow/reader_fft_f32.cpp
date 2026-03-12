// reader_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  READER — rewritten to use raw noc_async_read
//
//  Root cause of previous hang:
//    InterleavedAddrGenFast computes a NOC address from a tile index
//    using an internal banking formula. If the buffer was allocated
//    as a single contiguous DRAM buffer (page_size == buffer_size),
//    the banking formula maps tile_idx > 0 to a wrong address, and
//    noc_async_read_barrier() waits forever for a transfer that
//    either never arrives or writes to the wrong L1 location.
//
//  Fix: use raw noc_async_read(src_noc_addr, dst_l1_addr, size).
//    src_noc_addr = buffer_base + tile_idx * TILE_BYTES
//    dst_l1_addr  = get_write_ptr(cb)
//    noc_async_read does a flat address read — no banking, no formula.
//    This is correct for interleaved DRAM buffers where each "page"
//    is exactly one tile (TILE_BYTES = 4096 for Float32 32x32).
//
//  NOC address from DRAM buffer base address:
//    The buffer base address returned by buf->address() is a NOC
//    address that can be used directly with noc_async_read.
//    We compute: src = base + tile_idx * TILE_BYTES.
//
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_BYTES = 32 * 32 * sizeof(float);  // 4096

void kernel_main() {

    // ── Runtime args ───────────────────────────────────────────────
    const uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tw_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(6);   // tiles_per_stage
    const uint32_t num_stages  = get_arg_val<uint32_t>(7);

    // ── CB indices ─────────────────────────────────────────────────
    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;

    if (num_tiles == 0 || num_stages == 0) return;

    // ══════════════════════════════════════════════════════════════
    //  PHASE 1: stage 0 input data + stage 0 twiddles
    //
    //  For each tile t in 0..num_tiles-1:
    //    Read tw_r[t], tw_i[t], odd_r[t], odd_i[t], even_r[t], even_i[t]
    //    All 6 reads batched under one barrier.
    // ══════════════════════════════════════════════════════════════
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t byte_off = t * TILE_BYTES;

        // Reserve all 6 slots before firing NOC reads
        cb_reserve_back(cb_tw_r,   1);
        cb_reserve_back(cb_tw_i,   1);
        cb_reserve_back(cb_odd_r,  1);
        cb_reserve_back(cb_odd_i,  1);
        cb_reserve_back(cb_even_r, 1);
        cb_reserve_back(cb_even_i, 1);

        // Fire all 6 NOC reads — no barrier between them
        noc_async_read(tw_r_addr   + byte_off, get_write_ptr(cb_tw_r),   TILE_BYTES);
        noc_async_read(tw_i_addr   + byte_off, get_write_ptr(cb_tw_i),   TILE_BYTES);
        noc_async_read(odd_r_addr  + byte_off, get_write_ptr(cb_odd_r),  TILE_BYTES);
        noc_async_read(odd_i_addr  + byte_off, get_write_ptr(cb_odd_i),  TILE_BYTES);
        noc_async_read(even_r_addr + byte_off, get_write_ptr(cb_even_r), TILE_BYTES);
        noc_async_read(even_i_addr + byte_off, get_write_ptr(cb_even_i), TILE_BYTES);

        // One barrier covers all 6 transfers
        noc_async_read_barrier();

        // Signal compute that tiles are ready
        cb_push_back(cb_tw_r,   1);
        cb_push_back(cb_tw_i,   1);
        cb_push_back(cb_odd_r,  1);
        cb_push_back(cb_odd_i,  1);
        cb_push_back(cb_even_r, 1);
        cb_push_back(cb_even_i, 1);
    }

    // ══════════════════════════════════════════════════════════════
    //  PHASE 2: twiddles for stages 1..num_stages-1
    //
    //  Twiddle DRAM layout (set by host, stage-major):
    //    [stage 0: num_tiles tiles][stage 1: num_tiles tiles]...
    //
    //  Stage 0 was handled in Phase 1.
    //  For stages 1..N-1, only twiddles needed (data in L1 ping/pong).
    // ══════════════════════════════════════════════════════════════
    for (uint32_t stage = 1; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t global_tile = stage * num_tiles + t;
            const uint32_t byte_off    = global_tile * TILE_BYTES;

            cb_reserve_back(cb_tw_r, 1);
            cb_reserve_back(cb_tw_i, 1);

            noc_async_read(tw_r_addr + byte_off, get_write_ptr(cb_tw_r), TILE_BYTES);
            noc_async_read(tw_i_addr + byte_off, get_write_ptr(cb_tw_i), TILE_BYTES);

            noc_async_read_barrier();

            cb_push_back(cb_tw_r, 1);
            cb_push_back(cb_tw_i, 1);
        }
    }
}