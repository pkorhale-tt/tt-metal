// reader_fft_f32.cpp  — OPTIMAL v2: compact twiddle table
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DRAM traffic (N=1024):
//   Upload: 4 input tiles (8 KB) + 1 compact twiddle tile (8 KB) = 16 KB
//   vs previous design: 8 KB + 80 KB = 88 KB
//   Saving: ~5.5× less DRAM traffic
//
// The compact twiddle table has N/2 entries:
//   compact[k] = (cos(sign*2π*k/N), sin(sign*2π*k/N))  k=0..N/2-1
//
// The reader uploads this once, then before each stage it expands
// the compact table into a per-element tile in L1 using the formula:
//   slot p: j = p & (half_m-1),  idx = j * (N >> (stage+1))
//   expanded[p] = compact[idx]
//
// This keeps the FPU tile-op butterfly (1024 elements per cycle) while
// cutting twiddle DRAM storage from log2N×N/2 to N/2 values.
//
// CB map:
//   0  cb_even_r   stage input even real  (stage 0: from DRAM; stages 1+: from writer)
//   1  cb_even_i   stage input even imag
//   2  cb_odd_r    stage input odd  real
//   3  cb_odd_i    stage input odd  imag
//   4  cb_tw_r     expanded twiddle real  (reader fills per stage)
//   5  cb_tw_i     expanded twiddle imag
//  10  cb_compact_r  compact twiddle real (uploaded once from DRAM, kept in L1)
//  11  cb_compact_i  compact twiddle imag

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr = get_arg_val<uint32_t>(4);  // N/2 twiddles
    const uint32_t compact_i_addr = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles      = get_arg_val<uint32_t>(6);  // tiles_per_stage
    const uint32_t num_stages     = get_arg_val<uint32_t>(7);  // log2N
    const uint32_t half_N         = get_arg_val<uint32_t>(8);  // N/2 elements

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

    // Compact twiddle tile size: N/2 floats, may be < one full tile
    const uint32_t compact_bytes = half_N * sizeof(float);

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

    // Compact twiddles: page_size = compact_bytes (N/2 floats)
    const InterleavedAddrGenFast<true> cmp_r_gen = {
        .bank_base_address = compact_r_addr,
        .page_size = compact_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_i_gen = {
        .bank_base_address = compact_i_addr,
        .page_size = compact_bytes, .data_format = data_format };

    if (num_tiles == 0 || num_stages == 0) return;

    constexpr uint32_t ELEM = sizeof(float);

    // ── Step 1: Upload stage-0 inputs and compact twiddle table ──────
    // All issued in one burst, one barrier.
    cb_reserve_back(cb_even_r, num_tiles);
    cb_reserve_back(cb_even_i, num_tiles);
    cb_reserve_back(cb_odd_r,  num_tiles);
    cb_reserve_back(cb_odd_i,  num_tiles);
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        noc_async_read_tile(t, even_r_gen, get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(t, even_i_gen, get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(t, odd_r_gen,  get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(t, odd_i_gen,  get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
    noc_async_read_barrier();

    cb_push_back(cb_even_r, num_tiles);
    cb_push_back(cb_even_i, num_tiles);
    cb_push_back(cb_odd_r,  num_tiles);
    cb_push_back(cb_odd_i,  num_tiles);
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    // Compact twiddle base pointers (stay valid for all stages)
    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── Step 2: Per-stage twiddle expansion ───────────────────────────
    // Before each stage, expand compact[j * (N >> (stage+1))] into the
    // per-element twiddle tile that the compute kernel expects.
    // This runs entirely in L1 — no DRAM access per stage.
    //
    // Expansion formula:
    //   half_m   = 1 << stage
    //   N_over_m = half_N >> stage      (= N / 2^(stage+1))
    //   half_m_mask = half_m - 1
    //   For slot p in 0..half_N-1:
    //     j   = p & half_m_mask
    //     idx = j * N_over_m
    //     expanded[p] = compact[idx]

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const uint32_t half_m      = 1u << stage;
        const uint32_t N_over_m    = half_N >> stage;   // = N >> (stage+1)
        const uint32_t half_m_mask = half_m - 1u;

        // Reserve one twiddle tile slot
        cb_reserve_back(cb_tw_r, num_tiles);
        cb_reserve_back(cb_tw_i, num_tiles);
        const uint32_t dst_r = get_write_ptr(cb_tw_r);
        const uint32_t dst_i = get_write_ptr(cb_tw_i);

        // Expand: direct RISC-V reads from compact CB, writes to twiddle CB
        for (uint32_t p = 0; p < half_N; p++) {
            uint32_t j   = p & half_m_mask;
            uint32_t idx = j * N_over_m;

            // Read compact[idx] via uint32 (avoids strict-aliasing)
            uint32_t raw_r = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_r_base + idx * ELEM);
            uint32_t raw_i = *reinterpret_cast<volatile uint32_t*>(
                                 cmp_i_base + idx * ELEM);
            // Write to expanded tile slot p
            *reinterpret_cast<volatile uint32_t*>(dst_r + p * ELEM) = raw_r;
            *reinterpret_cast<volatile uint32_t*>(dst_i + p * ELEM) = raw_i;
        }

        cb_push_back(cb_tw_r, num_tiles);
        cb_push_back(cb_tw_i, num_tiles);

        // Wait for compute + writer to finish this stage before overwriting
        // the twiddle slot (CB depth=1, so cb_reserve_back at next iteration
        // will block until compute pops it).
        // Also wait for writer to signal that next stage's even/odd is ready.
        // (For stage 0 the even/odd was pushed above; for stages 1+ the writer
        //  pushes after its shuffle. We don't need to wait here — the compute
        //  kernel's cb_wait_front handles synchronisation.)
    }

    // Compact twiddle CBs are no longer needed
    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}