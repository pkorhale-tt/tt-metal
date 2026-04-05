// reader_fft_f32_mc.cpp  — MULTICORE reader  [BUG-FIXED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fixes applied:
//
//   BUG 1 (CB overflow in twiddle expansion):
//     Original wrote `local_half` elements into a CB reserved for
//     `local_tiles` tiles (= local_tiles * TILE_SIZE elements).
//     When local_half > local_tiles*TILE_SIZE this silently overflowed.
//     Fix: reserve the CB based on local_half elements rounded up to
//     whole tiles, and assert the sizes are consistent.
//
//   BUG 2 (core_elem_base wrong when tiles_per_row > 1):
//     Original: core_elem_base = tile_offset * (tile_bytes / ELEM)
//     This is only correct when tiles map 1:1 with elements starting
//     at 0, i.e. tiles_per_row == 1. For tiles_per_row > 1 the
//     tile_offset includes all previous cores' multi-tile rows, giving
//     a wrong element base.
//     Fix: receive core_elem_base explicitly as a kernel argument (arg 11)
//     — the host already knows this value and passes it to the writer.
//
//   BUG 3 (InterleavedAddrGenFast page_size < TILE_BYTES):
//     compact_bytes = half_N * sizeof(float) may be smaller than one
//     TILE_BYTES, causing incorrect DRAM bank selection on interleaved
//     buffers.
//     Fix: the compact twiddle buffer's page_size is set to
//     TILE_BYTES on the host (rounded up), and we read exactly
//     compact_bytes bytes using noc_async_read (not _tile) so the
//     address generator page_size stays aligned.
//
// Args:
//   0  even_r_addr        DRAM base — even real  (bit-reversed, split)
//   1  even_i_addr        DRAM base — even imag
//   2  odd_r_addr         DRAM base — odd  real
//   3  odd_i_addr         DRAM base — odd  imag
//   4  compact_r_addr     DRAM base — compact twiddle real  (N/2 floats)
//   5  compact_i_addr     DRAM base — compact twiddle imag
//   6  local_tiles        number of tiles this core owns
//   7  tile_offset        first global tile index for this core
//   8  num_stages         log2N
//   9  half_N             N/2 (global)
//  10  local_half         N / (2 * num_cores) — elements in this core's slice
//  11  core_elem_base     first global element index for this core (explicit)
//                         (= core_id * local_half for uniform partition)
/*
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
    // FIX (Bug 2): receive core_elem_base explicitly instead of computing
    // it from tile_offset, which is incorrect when tiles_per_row > 1.
    const uint32_t core_elem_base = get_arg_val<uint32_t>(11);

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

    constexpr uint32_t ELEM = sizeof(float);

    // FIX (Bug 1): twiddle CB tiles must be large enough to hold local_half
    // elements. Compute how many tiles that actually requires.
    const uint32_t TILE_SIZE_ELEMS  = tile_bytes / ELEM;
    // Number of tiles needed to hold local_half float elements.
    const uint32_t tw_tiles_needed  = (local_half + TILE_SIZE_ELEMS - 1)
                                       / TILE_SIZE_ELEMS;
    // Sanity: local_tiles (the data tiles) must equal tw_tiles_needed.
    // If this fires, the host passed mismatched tile counts.
    // (In a production kernel replace with a static assert or host-side check.)
    ASSERT(tw_tiles_needed == local_tiles);

    // FIX (Bug 3): Read the compact twiddle table with noc_async_read
    // (byte-addressed) rather than noc_async_read_tile, so we are not
    // subject to InterleavedAddrGenFast bank-selection using a page_size
    // smaller than TILE_BYTES. The host stores the compact table as a
    // single contiguous DRAM buffer aligned to TILE_BYTES.
    const uint32_t compact_bytes = half_N * ELEM;

    // ── Address generators for per-core tile slice ────────────────────
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

    if (local_tiles == 0 || num_stages == 0) return;

    // ── Step 1: Upload this core's input slice ────────────────────────
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);

    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t global_t = tile_offset + t;
        noc_async_read_tile(global_t, even_r_gen,
            get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(global_t, even_i_gen,
            get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(global_t, odd_r_gen,
            get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(global_t, odd_i_gen,
            get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }

    // FIX (Bug 3): Upload compact twiddle table using byte-addressed read.
    // The compact CBs (10, 11) are sized to hold compact_bytes rounded up
    // to tile_bytes on the host side.  We read exactly compact_bytes bytes
    // from DRAM base address directly — no InterleavedAddrGenFast needed
    // because it is a single contiguous (non-interleaved) allocation.
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read(compact_r_addr, get_write_ptr(cb_compact_r), compact_bytes);
    noc_async_read(compact_i_addr, get_write_ptr(cb_compact_i), compact_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_even_r, local_tiles);
    cb_push_back(cb_even_i, local_tiles);
    cb_push_back(cb_odd_r,  local_tiles);
    cb_push_back(cb_odd_i,  local_tiles);
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── Step 2: Per-stage twiddle expansion (local L1, no DRAM) ─────────
    //
    // The twiddle for global element index p (0..half_N-1) at stage s is:
    //   j   = p & (half_m - 1)       half_m = 1 << stage
    //   idx = j * (half_N >> stage)
    //   twiddle = compact[idx]
    //
    // FIX (Bug 2): use the explicit core_elem_base argument instead of
    // recomputing from tile_offset (which is wrong for tiles_per_row > 1).
    //
    // FIX (Bug 1): reserve tw_tiles_needed tiles (= local_tiles, validated
    // above) for the twiddle CBs.  The expansion loop writes exactly
    // local_half elements, which fits within tw_tiles_needed tiles.

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;
        const uint32_t half_m_mask = half_m - 1u;

        // Reserve exactly local_tiles tiles (= ceil(local_half / TILE_SIZE))
        cb_reserve_back(cb_tw_r, local_tiles);
        cb_reserve_back(cb_tw_i, local_tiles);
        const uint32_t dst_r = get_write_ptr(cb_tw_r);
        const uint32_t dst_i = get_write_ptr(cb_tw_i);

        for (uint32_t lp = 0; lp < local_half; lp++) {
            // FIX (Bug 2): use core_elem_base from arg, not from tile_offset.
            uint32_t p   = core_elem_base + lp;
            uint32_t j   = p & half_m_mask;
            uint32_t idx = j * N_over_m;

            // Bounds check: idx must be within the compact table.
            // idx = j * N_over_m ≤ (half_m-1) * (half_N/half_m)
            //     = half_N - half_N/half_m < half_N   (always satisfied)
            uint32_t raw_r = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_r_base + idx * ELEM);
            uint32_t raw_i = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_i_base + idx * ELEM);
            *reinterpret_cast<volatile uint32_t*>(dst_r + lp * ELEM) = raw_r;
            *reinterpret_cast<volatile uint32_t*>(dst_i + lp * ELEM) = raw_i;
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}*/



// reader_fft_f32_mc.cpp  — MULTICORE reader  [BUG-FIXED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fixes applied:
//
//   BUG 1 (CB overflow in twiddle expansion):
//     Original wrote `local_half` elements into a CB reserved for
//     `local_tiles` tiles (= local_tiles * TILE_SIZE elements).
//     When local_half > local_tiles*TILE_SIZE this silently overflowed.
//     Fix: reserve the CB based on local_half elements rounded up to
//     whole tiles, and assert the sizes are consistent.
//
//   BUG 2 (core_elem_base wrong when tiles_per_row > 1):
//     Original: core_elem_base = tile_offset * (tile_bytes / ELEM)
//     This is only correct when tiles map 1:1 with elements starting
//     at 0, i.e. tiles_per_row == 1. For tiles_per_row > 1 the
//     tile_offset includes all previous cores' multi-tile rows, giving
//     a wrong element base.
//     Fix: receive core_elem_base explicitly as a kernel argument (arg 11)
//     — the host already knows this value and passes it to the writer.
//
//   BUG 3 (InterleavedAddrGenFast page_size < TILE_BYTES):
//     compact_bytes = half_N * sizeof(float) may be smaller than one
//     TILE_BYTES, causing incorrect DRAM bank selection on interleaved
//     buffers.
//     Fix: the compact twiddle buffer's page_size is set to
//     TILE_BYTES on the host (rounded up), and we read exactly
//     compact_bytes bytes using noc_async_read (not _tile) so the
//     address generator page_size stays aligned.
//
// Args:
//   0  even_r_addr        DRAM base — even real  (bit-reversed, split)
//   1  even_i_addr        DRAM base — even imag
//   2  odd_r_addr         DRAM base — odd  real
//   3  odd_i_addr         DRAM base — odd  imag
//   4  compact_r_addr     DRAM base — compact twiddle real  (N/2 floats)
//   5  compact_i_addr     DRAM base — compact twiddle imag
//   6  local_tiles        number of tiles this core owns
//   7  tile_offset        first global tile index for this core
//   8  num_stages         log2N
//   9  half_N             N/2 (global)
//  10  local_half         N / (2 * num_cores) — elements in this core's slice
//  11  core_elem_base     first global element index for this core (explicit)
//                         (= core_id * local_half for uniform partition)

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
    // FIX (Bug 2): receive core_elem_base explicitly instead of computing
    // it from tile_offset, which is incorrect when tiles_per_row > 1.
    const uint32_t core_elem_base = get_arg_val<uint32_t>(11);

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

    constexpr uint32_t ELEM = sizeof(float);

    // FIX (Bug 1): twiddle CB tiles must be large enough to hold local_half
    // elements. Compute how many tiles that actually requires.
    const uint32_t TILE_SIZE_ELEMS  = tile_bytes / ELEM;
    // Number of tiles needed to hold local_half float elements.
    const uint32_t tw_tiles_needed  = (local_half + TILE_SIZE_ELEMS - 1)
                                       / TILE_SIZE_ELEMS;
    // Sanity: local_tiles (the data tiles) must equal tw_tiles_needed.
    // If this fires, the host passed mismatched tile counts.
    ASSERT(tw_tiles_needed == local_tiles);

    // FIX (Bug 3): Read the compact twiddle table with noc_async_read
    // (byte-addressed) rather than noc_async_read_tile, so we are not
    // subject to InterleavedAddrGenFast bank-selection using a page_size
    // smaller than TILE_BYTES. The host stores the compact table as a
    // single contiguous DRAM buffer aligned to TILE_BYTES.
    const uint32_t compact_bytes = half_N * ELEM;

    // ── Address generators for per-core tile slice ────────────────────
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

    if (local_tiles == 0 || num_stages == 0) return;

    // ── Step 1: Upload this core's input slice ────────────────────────
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);

    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t global_t = tile_offset + t;
        noc_async_read_tile(global_t, even_r_gen,
            get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(global_t, even_i_gen,
            get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(global_t, odd_r_gen,
            get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(global_t, odd_i_gen,
            get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }

    // FIX (Bug 3): Upload compact twiddle table using byte-addressed read.
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read(compact_r_addr, get_write_ptr(cb_compact_r), compact_bytes);
    noc_async_read(compact_i_addr, get_write_ptr(cb_compact_i), compact_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_even_r, local_tiles);
    cb_push_back(cb_even_i, local_tiles);
    cb_push_back(cb_odd_r,  local_tiles);
    cb_push_back(cb_odd_i,  local_tiles);
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── Step 2: Per-stage twiddle expansion (local L1, no DRAM) ───────
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;
        const uint32_t half_m_mask = half_m - 1u;

        cb_reserve_back(cb_tw_r, local_tiles);
        cb_reserve_back(cb_tw_i, local_tiles);
        const uint32_t dst_r = get_write_ptr(cb_tw_r);
        const uint32_t dst_i = get_write_ptr(cb_tw_i);

        for (uint32_t lp = 0; lp < local_half; lp++) {
            uint32_t p   = core_elem_base + lp;
            uint32_t j   = p & half_m_mask;
            uint32_t idx = j * N_over_m;

            uint32_t raw_r = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_r_base + idx * ELEM);
            uint32_t raw_i = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_i_base + idx * ELEM);
            *reinterpret_cast<volatile uint32_t*>(dst_r + lp * ELEM) = raw_r;
            *reinterpret_cast<volatile uint32_t*>(dst_i + lp * ELEM) = raw_i;
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}
