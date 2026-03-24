// writer_fft_f32.cpp — FIXED v4
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  ROOT CAUSE OF PREVIOUS DEADLOCK
// ══════════════════════════════════════════════════════════════════════
//
//  In the original code, for any stage s < num_stages-1 the writer did:
//
//    cb_wait_front(cb_out0_r, num_tiles)   ← blocks until compute pushes
//    cb_reserve_back(cb_next_even_r, ...)  ← blocks until compute DRAINS
//                                             CB 6-9 (it hasn't yet!)
//
//  Compute meanwhile was doing:
//    cb_wait_front(cb_next_even_r, 1)      ← blocks until writer pushes
//
//  Both sides blocked on each other → deadlock.
//
//  FIX: For intermediate stages the writer performs two passes:
//
//    Pass 1 — "Drain":  wait for all output tiles from compute, copy the
//             shuffle data into local L1 scratch, then immediately pop the
//             output CBs.  This unblocks compute for the next stage.
//
//    Pass 2 — "Fill":   reserve space in CB 6-9, copy the shuffled data
//             in, push.  Compute can now proceed with stage s+1.
//
//  Because Pass 1 pops CB 16-19 before Pass 2 reserves CB 6-9, compute
//  is never waiting on the writer while the writer is waiting on compute.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  out0_r_addr
//  [1]  out0_i_addr
//  [2]  out1_r_addr
//  [3]  out1_i_addr
//  [4]  num_tiles      (= tiles_per_row)
//  [5]  num_stages
//  [6]  half_N
//  [7]  (padding)
//  [8]  (padding)
//  [9]  (padding / core index)
//  [10] (padding)
//  [11] tile_offset
//  [12] (padding)
//  [13] rows_per_core
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr   = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles     = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t half_N        = get_arg_val<uint32_t>(6);
    // args [7-10]: padding
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);
    // arg [12]:    padding
    const uint32_t rows_per_core = get_arg_val<uint32_t>(13);

    // ── CB indices ────────────────────────────────────────────────────
    constexpr uint32_t cb_out0_r      = 16;
    constexpr uint32_t cb_out0_i      = 17;
    constexpr uint32_t cb_out1_r      = 18;
    constexpr uint32_t cb_out1_i      = 19;

    // Next-stage even/odd: writer fills these after shuffling, compute reads.
    constexpr uint32_t cb_next_even_r = 6;
    constexpr uint32_t cb_next_even_i = 7;
    constexpr uint32_t cb_next_odd_r  = 8;
    constexpr uint32_t cb_next_odd_i  = 9;

    if (num_tiles == 0 || num_stages == 0 || rows_per_core == 0) return;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);
    constexpr uint32_t ELEM      = sizeof(float);
    const uint32_t elems_per_tile_batch = num_tiles * (tile_bytes / ELEM);

    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // Temporary L1 scratch for the inter-stage shuffle.
    // Allocated once; reused every stage.  Size = 4 component arrays ×
    // elems_per_tile_batch floats.
    //
    // We use a flat uint32_t array.  The host kernel linker places L1
    // data in the local data segment, so a stack VLA is fine for small
    // sizes (tiles_per_row is typically 1-4).
    const uint32_t scratch_elems = elems_per_tile_batch;
    // Four scratch arrays laid out as local variables:
    //   scratch_er[0..scratch_elems-1]
    //   scratch_ei[0..scratch_elems-1]
    //   scratch_or[0..scratch_elems-1]
    //   scratch_oi[0..scratch_elems-1]
    //
    // Using __attribute__((aligned(4))) to be safe, but alignment should
    // already be 4 on RISC-V with uint32_t arrays.
    //
    // NOTE: If tiles_per_row is large (e.g. 8+) consider moving scratch
    //       to a dedicated L1 CB instead of the stack.
    uint32_t scratch_er[scratch_elems];
    uint32_t scratch_ei[scratch_elems];
    uint32_t scratch_or[scratch_elems];
    uint32_t scratch_oi[scratch_elems];

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * num_tiles;

        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const bool is_last = (stage == num_stages - 1);

            // ── Wait for compute to finish this stage ─────────────────
            cb_wait_front(cb_out0_r, num_tiles);
            cb_wait_front(cb_out0_i, num_tiles);
            cb_wait_front(cb_out1_r, num_tiles);
            cb_wait_front(cb_out1_i, num_tiles);

            if (is_last) {
                // ── Final stage: write results to DRAM ────────────────
                const uint32_t src0r = get_read_ptr(cb_out0_r);
                const uint32_t src0i = get_read_ptr(cb_out0_i);
                const uint32_t src1r = get_read_ptr(cb_out1_r);
                const uint32_t src1i = get_read_ptr(cb_out1_i);

                for (uint32_t t = 0; t < num_tiles; ++t) {
                    noc_async_write_tile(row_tile_base + t, out0_r_gen,
                        src0r + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out0_i_gen,
                        src0i + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_r_gen,
                        src1r + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_i_gen,
                        src1i + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);

            } else {
                // ── Intermediate stage: shuffle then feed next stage ──
                //
                // DEADLOCK-FREE PROTOCOL:
                //
                //   Pass 1: Read compute outputs, compute shuffle, store
                //           results in local scratch, then POP the output
                //           CBs immediately.  This frees compute to start
                //           working on stage s+1 twiddle tiles right away.
                //
                //   Pass 2: Reserve space in CB 6-9, copy from scratch,
                //           push.  Compute can now consume stage s+1.
                //
                // Because Pass 1 always happens before Pass 2, compute
                // is never blocked waiting for CB 6-9 while the writer
                // is blocked waiting for CB 16-19.

                // ── Pass 1: drain compute outputs, compute shuffle ─────
                const uint32_t m       = 1u << (stage + 1);
                const uint32_t half_m  = m >> 1;
                const uint32_t m2      = m << 1;
                const uint32_t half_m2 = m2 >> 1;
                const uint32_t G2      =
                    (half_N >= half_m2) ? (half_N / half_m2) : 0u;

                const uint32_t src0r = get_read_ptr(cb_out0_r);
                const uint32_t src0i = get_read_ptr(cb_out0_i);
                const uint32_t src1r = get_read_ptr(cb_out1_r);
                const uint32_t src1i = get_read_ptr(cb_out1_i);

                // Zero-initialise scratch so unused elements are 0.
                for (uint32_t lp = 0; lp < scratch_elems; ++lp) {
                    scratch_er[lp] = 0u;
                    scratch_ei[lp] = 0u;
                    scratch_or[lp] = 0u;
                    scratch_oi[lp] = 0u;
                }

                if (G2 != 0) {
                    const uint32_t log2m  = stage + 1;
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst = 0;

                    for (uint32_t g2 = 0; g2 < G2; ++g2) {
                        const uint32_t base_e = g2 * m2;
                        const uint32_t base_o = base_e + half_m2;

                        for (uint32_t j2 = 0; j2 < half_m2; ++j2) {
                            // Even output element
                            {
                                const uint32_t f      = base_e + j2;
                                const uint32_t g_old  = f >> log2m;
                                const uint32_t offset = f & m_mask;
                                uint32_t idx, srcr, srci;
                                if (offset < half_m) {
                                    idx  = g_old * half_m + offset;
                                    srcr = src0r; srci = src0i;
                                } else {
                                    idx  = g_old * half_m + (offset - half_m);
                                    srcr = src1r; srci = src1i;
                                }
                                scratch_er[dst] = rd32(srcr + idx * ELEM);
                                scratch_ei[dst] = rd32(srci + idx * ELEM);
                            }
                            // Odd output element
                            {
                                const uint32_t f      = base_o + j2;
                                const uint32_t g_old  = f >> log2m;
                                const uint32_t offset = f & m_mask;
                                uint32_t idx, srcr, srci;
                                if (offset < half_m) {
                                    idx  = g_old * half_m + offset;
                                    srcr = src0r; srci = src0i;
                                } else {
                                    idx  = g_old * half_m + (offset - half_m);
                                    srcr = src1r; srci = src1i;
                                }
                                scratch_or[dst] = rd32(srcr + idx * ELEM);
                                scratch_oi[dst] = rd32(srci + idx * ELEM);
                            }
                            ++dst;
                        }
                    }
                }

                // Pop compute outputs NOW — before reserving CB 6-9.
                // This is the key step that breaks the circular dependency.
                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);

                // ── Pass 2: push shuffled data into CB 6-9 ────────────
                //
                // cb_reserve_back may block if compute has not yet
                // consumed the previous push into CB 6-9, but that is
                // only possible at stage s >= 2 and only if compute is
                // behind — which is fine because compute is now free
                // to drain CB 6-9 (we already popped its inputs above).
                //
                cb_reserve_back(cb_next_even_r, num_tiles);
                cb_reserve_back(cb_next_even_i, num_tiles);
                cb_reserve_back(cb_next_odd_r,  num_tiles);
                cb_reserve_back(cb_next_odd_i,  num_tiles);

                const uint32_t dst_er = get_write_ptr(cb_next_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_next_even_i);
                const uint32_t dst_or = get_write_ptr(cb_next_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_next_odd_i);

                for (uint32_t lp = 0; lp < scratch_elems; ++lp) {
                    const uint32_t off = lp * ELEM;
                    wr32(dst_er + off, scratch_er[lp]);
                    wr32(dst_ei + off, scratch_ei[lp]);
                    wr32(dst_or + off, scratch_or[lp]);
                    wr32(dst_oi + off, scratch_oi[lp]);
                }

                cb_push_back(cb_next_even_r, num_tiles);
                cb_push_back(cb_next_even_i, num_tiles);
                cb_push_back(cb_next_odd_r,  num_tiles);
                cb_push_back(cb_next_odd_i,  num_tiles);
            }
        }
    }
}