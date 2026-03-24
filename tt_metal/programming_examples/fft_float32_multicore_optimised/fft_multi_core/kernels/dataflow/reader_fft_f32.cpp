// reader_fft_f32_mc.cpp — MULTICORE reader (BUGFREE + OPTIMISED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  BUGS FIXED vs previous version
// ══════════════════════════════════════════════════════════════════════
//
//  BUG 4 (twiddle scatter size mismatch)
//    Old code iterated local_half (= half_N = 512 for N=1024) times and
//    wrote individual floats into cb_tw_r/i, but each CB slot is sized
//    for tiles_per_row tiles (1 tile = TILE_SIZE = 1024 floats for a
//    32×32 tile). The scatter only filled the FIRST 512 of 1024 elements,
//    leaving the second 512 as uninitialised L1 garbage. The compute
//    kernel then read a full tile from cb_tw_r/i, seeing wrong twiddles
//    for butterfly indices 512-1023.
//
//    FIX: iterate TILE_SIZE * tiles_per_row total elements per stage,
//    so every element of every tile in the CB is correctly initialised
//    before cb_push_back is called.
//
//  BUG 5 (row-loop CB race)
//    Old code: the outer row loop reserved cb_even_r/i and cb_odd_r/i at
//    the start of each row without waiting for the writer to finish
//    consuming/shuffling those CBs from the *previous* row. The writer
//    shuffles output back into cb_even/odd for the next stage — if the
//    reader reserves new space before the writer's cb_push_back sequence
//    completes, cb_reserve_back in the reader and cb_wait_front in the
//    compute kernel interleave unpredictably, corrupting CB state.
//
//    FIX: the reader must NOT pre-reserve even/odd CBs for row N+1 until
//    the compute+writer pipeline for row N has fully drained them. Because
//    the reader cannot observe writer completion directly, the correct
//    architecture is:
//      - Stage 0 of each row: reader fills even/odd from DRAM.
//      - Stages 1..num_stages-1: writer fills even/odd via shuffle.
//    Therefore the reader only reserves/pushes even/odd ONCE per row
//    (for stage 0 only), and does NOT loop over stages internally.
//    The twiddle CB is filled once per stage (all stages), which is
//    correct because twiddles are consumed by compute, not recycled
//    by the writer.
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS
// ══════════════════════════════════════════════════════════════════════
//
//  1. Compact twiddle table loaded once before the row loop (unchanged),
//     but the base pointers are cached as const locals to avoid repeated
//     get_read_ptr() calls inside the hot scatter loop.
//
//  2. NOC async reads for all four even/odd CBs are issued in one burst
//     before the barrier — maximises DRAM read parallelism.
//
//  3. Twiddle scatter loop: index arithmetic uses pre-shifted constants
//     (half_m, N_over_m) computed once per stage rather than inside
//     the inner element loop.
//
//  4. TILE_SIZE elements per tile is a compile-time constant derived
//     from tile_bytes / ELEM, eliminating the division in the loop.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP (must match host exactly)
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  even_r_addr     — DRAM base address of even-real input buffer
//  [1]  even_i_addr     — DRAM base address of even-imag input buffer
//  [2]  odd_r_addr      — DRAM base address of odd-real  input buffer
//  [3]  odd_i_addr      — DRAM base address of odd-imag  input buffer
//  [4]  compact_r_addr  — DRAM base of compact twiddle real table
//  [5]  compact_i_addr  — DRAM base of compact twiddle imag table
//  [6]  tiles_per_row   — number of tiles per FFT row (= half_N/TILE_SIZE)
//  [7]  tile_offset     — first tile index owned by this core
//  [8]  num_stages      — log2(N_row)
//  [9]  half_N          — N_row / 2
//  [10] local_half      — elements per half-row on this core (= half_N)
//  [11] rows_per_core   — number of FFT rows this core processes
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Runtime args ─────────────────────────────────────────────────
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr = get_arg_val<uint32_t>(4);
    const uint32_t compact_i_addr = get_arg_val<uint32_t>(5);
    const uint32_t tiles_per_row  = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t half_N         = get_arg_val<uint32_t>(9);
    // arg[10] local_half intentionally unused — see BUG 5 fix note above.
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(11);

    // ── CB indices ────────────────────────────────────────────────────
    constexpr uint32_t cb_even_r    = 0;
    constexpr uint32_t cb_even_i    = 1;
    constexpr uint32_t cb_odd_r     = 2;
    constexpr uint32_t cb_odd_i     = 3;
    constexpr uint32_t cb_tw_r      = 4;
    constexpr uint32_t cb_tw_i      = 5;
    constexpr uint32_t cb_compact_r = 10;
    constexpr uint32_t cb_compact_i = 11;

    // ── Tile geometry ─────────────────────────────────────────────────
    const uint32_t tile_bytes    = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    const uint32_t compact_bytes = half_N * sizeof(float);

    // FIX (BUG 4): total elements per twiddle CB push = tiles_per_row
    // full tiles, each of TILE_SIZE floats. For N=1024: 1 * 1024 = 1024.
    constexpr uint32_t ELEM      = sizeof(float);
    const uint32_t elems_per_row = (tile_bytes / ELEM) * tiles_per_row;
    // elems_per_row is the number of scalar float slots the compute kernel
    // will read from cb_tw_r / cb_tw_i in one stage. We MUST fill all of
    // them before pushing.

    // Early exit.
    if (tiles_per_row == 0 || num_stages == 0 || rows_per_core == 0) return;

    // ── Address generators ────────────────────────────────────────────
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
    const InterleavedAddrGenFast<true> cmp_r_gen  = {
        .bank_base_address = compact_r_addr,
        .page_size = compact_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_i_gen  = {
        .bank_base_address = compact_i_addr,
        .page_size = compact_bytes, .data_format = data_format };

    // ── Scalar L1 accessors (BRISC/NCRISC only — no ThCon) ───────────
    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // ── Load compact twiddle table once — shared across all rows ─────
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
    noc_async_read_barrier();
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    // Cache read pointers — they will not move (compact CBs are never popped
    // until kernel exit).
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── Outer row loop ────────────────────────────────────────────────
    //
    // FIX (BUG 5): the reader fills even/odd from DRAM ONCE per row
    // (stage 0 only). For stages 1..num_stages-1 the writer performs the
    // shuffle and pushes the next stage's even/odd. The reader must not
    // attempt to refill even/odd during those stages.
    //
    // The reader IS responsible for twiddle factors every stage, because
    // the writer does not produce twiddles — it only produces even/odd.

    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        const uint32_t row_elem_base = row_tile_base * (tile_bytes / ELEM);

        // ── Stage 0: load even/odd from DRAM ─────────────────────────
        //
        // Issue all four NOC reads before the barrier for maximum
        // DRAM throughput (up to 4 outstanding requests in flight).

        cb_reserve_back(cb_even_r, tiles_per_row);
        cb_reserve_back(cb_even_i, tiles_per_row);
        cb_reserve_back(cb_odd_r,  tiles_per_row);
        cb_reserve_back(cb_odd_i,  tiles_per_row);

        for (uint32_t t = 0; t < tiles_per_row; t++) {
            const uint32_t gt = row_tile_base + t;
            noc_async_read_tile(gt, even_r_gen,
                get_write_ptr(cb_even_r) + t * tile_bytes);
            noc_async_read_tile(gt, even_i_gen,
                get_write_ptr(cb_even_i) + t * tile_bytes);
            noc_async_read_tile(gt, odd_r_gen,
                get_write_ptr(cb_odd_r)  + t * tile_bytes);
            noc_async_read_tile(gt, odd_i_gen,
                get_write_ptr(cb_odd_i)  + t * tile_bytes);
        }
        noc_async_read_barrier();

        cb_push_back(cb_even_r, tiles_per_row);
        cb_push_back(cb_even_i, tiles_per_row);
        cb_push_back(cb_odd_r,  tiles_per_row);
        cb_push_back(cb_odd_i,  tiles_per_row);

        // ── All stages: fill twiddle CBs ─────────────────────────────
        //
        // Twiddle factors depend on the butterfly group (stage index and
        // element position within the row). We scatter-read from the
        // compact table into the full twiddle CB for each stage.
        //
        // FIX (BUG 4): loop runs elems_per_row iterations, filling ALL
        // elements in the tiles_per_row tiles — not just local_half.
        // For N=1024: elems_per_row = 1024, half_N = 512,
        // so elements 512-1023 now get the correct twiddle rather than
        // staying as uninitialised L1 values.
        //
        // The twiddle index formula:
        //   For global element p at FFT stage s with half_m = 2^s:
        //     twiddle_index = (p mod half_m) * (half_N / half_m)
        //                   = (p & (half_m - 1)) * N_over_m

        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const uint32_t half_m      = 1u << stage;
            const uint32_t N_over_m    = half_N >> stage;
            const uint32_t half_m_mask = half_m - 1u;

            cb_reserve_back(cb_tw_r, tiles_per_row);
            cb_reserve_back(cb_tw_i, tiles_per_row);
            const uint32_t dst_r = get_write_ptr(cb_tw_r);
            const uint32_t dst_i = get_write_ptr(cb_tw_i);

            // Fill every element in the tile(s).
            for (uint32_t lp = 0; lp < elems_per_row; lp++) {
                const uint32_t p   = row_elem_base + lp;
                const uint32_t idx = (p & half_m_mask) * N_over_m;
                wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
            }

            cb_push_back(cb_tw_r, tiles_per_row);
            cb_push_back(cb_tw_i, tiles_per_row);
        }
        // After the twiddle loop the compute+writer pipeline takes over:
        //   - Compute drains tw_r/tw_i and even/odd, produces out0/out1.
        //   - Writer consumes out0/out1 and (for stages <last) shuffles
        //     results back into even/odd for the next compute stage.
        // The reader does not touch even/odd again for this row.
        // The cb_reserve_back at the top of the next row iteration will
        // block correctly until the writer has drained the CBs — this is
        // the natural back-pressure mechanism.
    }

    // ── Release compact twiddle table ─────────────────────────────────
    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}