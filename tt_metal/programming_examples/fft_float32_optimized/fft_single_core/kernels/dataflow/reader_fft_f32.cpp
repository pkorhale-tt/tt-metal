// reader_fft_f32.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  READER — uses get_noc_addr(addr) for DRAM reads
//
//  get_noc_addr(dram_addr) correctly encodes the DRAM bank address
//  into a NOC address. This is the correct low-level API when you
//  want flat sequential reads from a contiguous DRAM buffer without
//  the banking formula of InterleavedAddrGenFast.
//
//  noc_async_read(noc_addr, l1_ptr, size) transfers `size` bytes
//  from DRAM noc_addr into L1 at l1_ptr.
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_BYTES = 32 * 32 * sizeof(float);  // 4096

void kernel_main() {

    const uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tw_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tw_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(6);
    const uint32_t num_stages  = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;

    if (num_tiles == 0 || num_stages == 0) return;

    // ── Phase 1: stage 0 input data + stage 0 twiddles ────────────
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t off = t * TILE_BYTES;

        cb_reserve_back(cb_tw_r,   1);
        cb_reserve_back(cb_tw_i,   1);
        cb_reserve_back(cb_odd_r,  1);
        cb_reserve_back(cb_odd_i,  1);
        cb_reserve_back(cb_even_r, 1);
        cb_reserve_back(cb_even_i, 1);

        noc_async_read(get_noc_addr(tw_r_addr   + off), get_write_ptr(cb_tw_r),   TILE_BYTES);
        noc_async_read(get_noc_addr(tw_i_addr   + off), get_write_ptr(cb_tw_i),   TILE_BYTES);
        noc_async_read(get_noc_addr(odd_r_addr  + off), get_write_ptr(cb_odd_r),  TILE_BYTES);
        noc_async_read(get_noc_addr(odd_i_addr  + off), get_write_ptr(cb_odd_i),  TILE_BYTES);
        noc_async_read(get_noc_addr(even_r_addr + off), get_write_ptr(cb_even_r), TILE_BYTES);
        noc_async_read(get_noc_addr(even_i_addr + off), get_write_ptr(cb_even_i), TILE_BYTES);

        noc_async_read_barrier();

        cb_push_back(cb_tw_r,   1);
        cb_push_back(cb_tw_i,   1);
        cb_push_back(cb_odd_r,  1);
        cb_push_back(cb_odd_i,  1);
        cb_push_back(cb_even_r, 1);
        cb_push_back(cb_even_i, 1);
    }

    // ── Phase 2: twiddles for stages 1..num_stages-1 ──────────────
    for (uint32_t stage = 1; stage < num_stages; stage++) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t off = (stage * num_tiles + t) * TILE_BYTES;

            cb_reserve_back(cb_tw_r, 1);
            cb_reserve_back(cb_tw_i, 1);

            noc_async_read(get_noc_addr(tw_r_addr + off), get_write_ptr(cb_tw_r), TILE_BYTES);
            noc_async_read(get_noc_addr(tw_i_addr + off), get_write_ptr(cb_tw_i), TILE_BYTES);

            noc_async_read_barrier();

            cb_push_back(cb_tw_r, 1);
            cb_push_back(cb_tw_i, 1);
        }
    }
}