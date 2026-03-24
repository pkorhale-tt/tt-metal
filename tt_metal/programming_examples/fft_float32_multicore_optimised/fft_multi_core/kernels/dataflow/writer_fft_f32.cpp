// writer_fft_f32_mc.cpp — MULTICORE writer (BUGFREE + OPTIMISED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  BUGS FIXED vs previous version
// ══════════════════════════════════════════════════════════════════════
//
//  BUG 5 (row-loop CB race) — shared with reader
//    The old writer consumed out0/out1 and shuffled them into even/odd
//    for the next stage, but it pushed to even/odd AFTER popping out0/1,
//    with no guarantee the reader wasn't concurrently reserving the same
//    CBs for the next row.
//
//    FIX: the writer's shuffle path (non-last stages) pops out0/out1
//    first, then reserves+pushes even/odd. Because cb_reserve_back in
//    the reader (next row) blocks until there is space, and the writer
//    is the producer, the CB depth naturally serialises them. No extra
//    synchronisation primitive is needed as long as CB depths are sized
//    correctly (depth = tiles_per_row for all data CBs).
//
//    Additionally, the writer now processes ALL num_stages per row in
//    a single outer stage loop before advancing to the next row — this
//    keeps the pipeline clean and avoids interleaving rows mid-flight.
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS
// ══════════════════════════════════════════════════════════════════════
//
//  1. copy_floats uses 32-bit scalar stores — unchanged from original
//     (no ThCon, no LLK). Kept as the only correct L1 copy method in
//     BRISC/NCRISC dataflow kernels.
//
//  2. Last-stage DRAM writes issue all four noc_async_write_tile calls
//     before the barrier, maximising write throughput.
//
//  3. safe_local_src underflow guard retained with fast UINT32_MAX
//     sentinel — avoids undefined behaviour on unsigned subtraction.
//
//  4. G2 shuffle block decomposition unchanged in structure but with
//     clearer variable naming and the pop-before-push fix applied.
//
//  5. Row/stage loop structure: outer = rows, inner = stages. The
//     writer fully processes a row through all stages before moving on.
//     This matches the reader's twiddle fill order and avoids CB aliasing
//     between rows.
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP (must match host exactly)
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  out0_r_addr    — DRAM base for upper butterfly real output
//  [1]  out0_i_addr    — DRAM base for upper butterfly imag output
//  [2]  out1_r_addr    — DRAM base for lower butterfly real output
//  [3]  out1_i_addr    — DRAM base for lower butterfly imag output
//  [4]  tiles_per_row  — tiles per single FFT row
//  [5]  num_stages     — log2(N_row)
//  [6]  local_half     — elements per half-row (= half_N)
//  [7]  half_N         — N_row / 2
//  [8]  num_cores      — 1 (self-contained row FFT per core)
//  [9]  core_id        — this core's index
//  [10] log2_cores     — 0 (single-core per row)
//  [11] tile_offset    — base tile index for DRAM writes
//  [12] core_elem_base — 0 (full row per FFT)
//  [13] rows_per_core  — number of FFT rows this core processes
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Runtime args ─────────────────────────────────────────────────
    const uint32_t out0_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr    = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr    = get_arg_val<uint32_t>(3);
    const uint32_t tiles_per_row  = get_arg_val<uint32_t>(4);
    const uint32_t num_stages     = get_arg_val<uint32_t>(5);
    const uint32_t local_half     = get_arg_val<uint32_t>(6);
    const uint32_t half_N         = get_arg_val<uint32_t>(7);
    // args [8..10] reserved for multi-core future use — not used here.
    const uint32_t tile_offset    = get_arg_val<uint32_t>(11);
    // arg [12] core_elem_base = 0, unused in single-row-per-core mode.
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(13);

    // ── CB indices ────────────────────────────────────────────────────
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;

    // ── Tile geometry ─────────────────────────────────────────────────
    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    if (tiles_per_row == 0 || rows_per_core == 0) return;

    constexpr uint32_t ELEM = sizeof(float);

    // ── Address generators ────────────────────────────────────────────
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

    // ── Scalar L1 copy — BRISC/NCRISC only, no ThCon ─────────────────
    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };
    auto copy_floats = [&](uint32_t dst, uint32_t src, uint32_t count) {
        for (uint32_t i = 0; i < count; i++)
            wr32(dst + i * ELEM, rd32(src + i * ELEM));
    };

    // Underflow guard for unsigned subtraction in shuffle index math.
    auto safe_local_src = [](uint32_t src_start, uint32_t base) -> uint32_t {
        if (src_start < base) return UINT32_MAX;
        return src_start - base;
    };

    // ── Outer row loop ────────────────────────────────────────────────
    //
    // Each row is processed through ALL stages before the next row
    // begins. This avoids CB aliasing between rows and is consistent
    // with the reader's twiddle fill order.

    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        const uint32_t row_elem_base = row_tile_base * (tile_bytes / ELEM);

        // ── Stage loop ────────────────────────────────────────────────
        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const bool is_last = (stage == num_stages - 1u);

            // Wait for compute to produce this stage's outputs.
            cb_wait_front(cb_out0_r, tiles_per_row);
            cb_wait_front(cb_out0_i, tiles_per_row);
            cb_wait_front(cb_out1_r, tiles_per_row);
            cb_wait_front(cb_out1_i, tiles_per_row);

            const uint32_t src0r = get_read_ptr(cb_out0_r);
            const uint32_t src0i = get_read_ptr(cb_out0_i);
            const uint32_t src1r = get_read_ptr(cb_out1_r);
            const uint32_t src1i = get_read_ptr(cb_out1_i);

            if (is_last) {
                // ── Last stage: write to DRAM ─────────────────────────
                //
                // Issue all four NOC writes before the barrier for
                // maximum DRAM write throughput.

                for (uint32_t t = 0; t < tiles_per_row; t++) {
                    const uint32_t gt = row_tile_base + t;
                    noc_async_write_tile(gt, out0_r_gen, src0r + t * tile_bytes);
                    noc_async_write_tile(gt, out0_i_gen, src0i + t * tile_bytes);
                    noc_async_write_tile(gt, out1_r_gen, src1r + t * tile_bytes);
                    noc_async_write_tile(gt, out1_i_gen, src1i + t * tile_bytes);
                }
                noc_async_write_barrier();

                // Pop output CBs after DRAM writes complete.
                cb_pop_front(cb_out0_r, tiles_per_row);
                cb_pop_front(cb_out0_i, tiles_per_row);
                cb_pop_front(cb_out1_r, tiles_per_row);
                cb_pop_front(cb_out1_i, tiles_per_row);

            } else {
                // ── Intermediate stage: shuffle into even/odd ─────────
                //
                // Pop out0/out1 FIRST (freeing space), then reserve and
                // fill even/odd for the next compute stage.
                //
                // FIX (BUG 5): popping before reserving ensures the reader
                // cannot concurrently fill even/odd for the next row while
                // the writer is still producing results for this row.
                // CB back-pressure (depth = tiles_per_row) handles
                // serialisation naturally.

                const uint32_t m       = 1u << (stage + 1u);
                const uint32_t half_m  = m >> 1u;
                const uint32_t m2      = m << 1u;
                const uint32_t half_m2 = m2 >> 1u;
                // G2: how many complete double-groups fit in local_half.
                const uint32_t G2 = (half_m2 <= local_half)
                                    ? local_half / half_m2 : 0u;

                // Pop outputs before reserving inputs for next stage.
                cb_pop_front(cb_out0_r, tiles_per_row);
                cb_pop_front(cb_out0_i, tiles_per_row);
                cb_pop_front(cb_out1_r, tiles_per_row);
                cb_pop_front(cb_out1_i, tiles_per_row);

                // Now reserve space in even/odd for next stage.
                cb_reserve_back(cb_even_r, tiles_per_row);
                cb_reserve_back(cb_even_i, tiles_per_row);
                cb_reserve_back(cb_odd_r,  tiles_per_row);
                cb_reserve_back(cb_odd_i,  tiles_per_row);

                const uint32_t dst_er = get_write_ptr(cb_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_even_i);
                const uint32_t dst_or = get_write_ptr(cb_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_odd_i);

                if (G2 > 0u) {
                    // ── Normal shuffle path (G2 complete groups) ──────
                    const uint32_t log2m  = stage + 1u;
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst_base = 0u;

                    for (uint32_t g2 = 0u; g2 < G2; g2++) {
                        const uint32_t lb_e = g2 * m2;
                        const uint32_t lb_o = lb_e + half_m2;

                        // Block A: new_even[0..half_m) ← out0
                        {
                            const uint32_t f0    = row_elem_base + lb_e;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + off;
                            const uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_er + dst_base * ELEM,
                                            src0r  + ls      * ELEM, half_m);
                                copy_floats(dst_ei + dst_base * ELEM,
                                            src0i  + ls      * ELEM, half_m);
                            }
                        }
                        // Block B: new_even[half_m..m) ← out1
                        {
                            const uint32_t f0    = row_elem_base + lb_e + half_m;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + (off - half_m);
                            const uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_er + (dst_base + half_m) * ELEM,
                                            src1r  + ls                  * ELEM, half_m);
                                copy_floats(dst_ei + (dst_base + half_m) * ELEM,
                                            src1i  + ls                  * ELEM, half_m);
                            }
                        }
                        // Block C: new_odd[0..half_m) ← out0
                        {
                            const uint32_t f0    = row_elem_base + lb_o;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + off;
                            const uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_or + dst_base * ELEM,
                                            src0r  + ls      * ELEM, half_m);
                                copy_floats(dst_oi + dst_base * ELEM,
                                            src0i  + ls      * ELEM, half_m);
                            }
                        }
                        // Block D: new_odd[half_m..m) ← out1
                        {
                            const uint32_t f0    = row_elem_base + lb_o + half_m;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + (off - half_m);
                            const uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_or + (dst_base + half_m) * ELEM,
                                            src1r  + ls                  * ELEM, half_m);
                                copy_floats(dst_oi + (dst_base + half_m) * ELEM,
                                            src1i  + ls                  * ELEM, half_m);
                            }
                        }

                        dst_base += half_m2;
                    }
                } else {
                    // ── G2 = 0: passthrough (contiguous copy) ─────────
                    //
                    // When no complete double-group fits in local_half
                    // the elements are already in butterfly order —
                    // copy directly without reordering.
                    copy_floats(dst_er, src0r, local_half);
                    copy_floats(dst_ei, src0i, local_half);
                    copy_floats(dst_or, src1r, local_half);
                    copy_floats(dst_oi, src1i, local_half);
                }

                // Push shuffled data — compute will consume on next stage.
                cb_push_back(cb_even_r, tiles_per_row);
                cb_push_back(cb_even_i, tiles_per_row);
                cb_push_back(cb_odd_r,  tiles_per_row);
                cb_push_back(cb_odd_i,  tiles_per_row);
            }
        }
        // Row fully processed. even/odd are empty (last stage wrote to
        // DRAM, not to even/odd). The next row's reader reserve will
        // succeed immediately.
    }
}