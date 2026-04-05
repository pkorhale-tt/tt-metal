// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

// ── Correct semaphore protocol (see writer.cpp for full explanation) ───────────
//
// Reader uses noc_semaphore_set() (local L1 store) for BOTH reset and ack,
// matching the writer's local-store signals.  noc_semaphore_set_remote is NOT
// used — that would do an AT_INC which cannot be used to reset to 0.
//
// Reader does NOT initialise either flag. Writer owns initialisation before
// its step loop to avoid the race where reader's zero-write races with the
// writer's step-0 setup.
// ─────────────────────────────────────────────────────────────────────────────
/*
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dram_input_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t dram_input_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t n                 = get_arg_val<uint32_t>(2);
    const uint32_t num_steps         = get_arg_val<uint32_t>(3);
    const uint32_t num_chunks        = get_arg_val<uint32_t>(4);
    const uint32_t chunk_size        = get_arg_val<uint32_t>(5);
    const uint32_t sram_buf_r_addr   = get_arg_val<uint32_t>(6);
    const uint32_t sync_flag_addr    = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_data0_r   = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i   = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r   = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i   = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;

    const uint32_t row_bytes       = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

    const uint32_t sram_tw_r_addr = sram_buf_i_addr + row_bytes;
    const uint32_t sram_tw_i_addr = sram_tw_r_addr + num_steps * (n / 2u) * sizeof(float);

    // rdy_flag @ sync_flag_addr+0 : writer signals, reader polls
    volatile tt_l1_ptr uint32_t* rdy_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);

    // ack_flag @ sync_flag_addr+4 : reader signals, writer polls
    const uint32_t ack_flag_addr = sync_flag_addr + sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* ack_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_flag_addr);

    // No init here — writer initialises both flags before its step loop.

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m         = 1u << step;
        const uint32_t m              = half_m << 1u;
        const uint32_t tw_step_offset = step * (n / 2u);

        if (step == 0u) {
            // ── Step 0: load from DRAM and bit-reverse permute in SRAM ──────
            const uint64_t noc_r = get_noc_addr(dram_input_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_input_i_addr);
            noc_async_read(noc_r, sram_buf_r_addr, row_bytes);
            noc_async_read(noc_i, sram_buf_i_addr, row_bytes);
            noc_async_read_barrier();

            volatile tt_l1_ptr float* sr =
                reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
            volatile tt_l1_ptr float* si =
                reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

            for (uint32_t i = 0; i < n; ++i) {
                uint32_t j   = 0u;
                uint32_t tmp = i;
                for (uint32_t b = 0; b < num_steps; ++b) {
                    j   = (j << 1u) | (tmp & 1u);
                    tmp >>= 1u;
                }
                if (i < j) {
                    float tr = sr[i]; sr[i] = sr[j]; sr[j] = tr;
                    float ti = si[i]; si[i] = si[j]; si[j] = ti;
                }
            }
        } else {
            // ── Steps 1+: wait for writer's rdy signal ───────────────────────
            noc_semaphore_wait(rdy_flag, 1);
            noc_semaphore_set(rdy_flag, 0);   // reset — local store, safe same-Tensix
        }

        // ── Push butterfly pairs and twiddles into CBs ───────────────────────
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            // real data pair
            cb_reserve_back(cb_data0_r, 1);
            cb_reserve_back(cb_data1_r, 1);

            volatile tt_l1_ptr float* dst0_r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data0_r));
            volatile tt_l1_ptr float* dst1_r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data1_r));
            const volatile tt_l1_ptr float* src_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_buf_r_addr);

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;
                const uint32_t b        = a + half_m;
                dst0_r[p] = src_r[a];
                dst1_r[p] = src_r[b];
            }

            cb_push_back(cb_data0_r, 1);
            cb_push_back(cb_data1_r, 1);

            // imaginary data pair
            cb_reserve_back(cb_data0_i, 1);
            cb_reserve_back(cb_data1_i, 1);

            volatile tt_l1_ptr float* dst0_i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data0_i));
            volatile tt_l1_ptr float* dst1_i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data1_i));
            const volatile tt_l1_ptr float* src_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_buf_i_addr);

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;
                const uint32_t b        = a + half_m;
                dst0_i[p] = src_i[a];
                dst1_i[p] = src_i[b];
            }

            cb_push_back(cb_data0_i, 1);
            cb_push_back(cb_data1_i, 1);

            // twiddle factors
            cb_reserve_back(cb_twiddle_r, 1);
            cb_reserve_back(cb_twiddle_i, 1);

            volatile tt_l1_ptr float* tw_r_dst =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_twiddle_r));
            volatile tt_l1_ptr float* tw_i_dst =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_twiddle_i));
            const volatile tt_l1_ptr float* sram_tw_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_tw_r_addr) + tw_step_offset;
            const volatile tt_l1_ptr float* sram_tw_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_tw_i_addr) + tw_step_offset;

            for (uint32_t p = 0; p < chunk_size; ++p) {
                tw_r_dst[p] = sram_tw_r[pair_base + p];
                tw_i_dst[p] = sram_tw_i[pair_base + p];
            }

            cb_push_back(cb_twiddle_r, 1);
            cb_push_back(cb_twiddle_i, 1);
        }

        // Ack writer after all chunks are pushed, so it can proceed to drain
        // output CBs and scatter SRAM for the next step.
        // Step 0 needs no ack: the writer is not waiting for step 0.
        if (step > 0u) {
            noc_semaphore_set(ack_flag, 1);   // local store, safe same-Tensix
        }
    }
}
*/

//=====================

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