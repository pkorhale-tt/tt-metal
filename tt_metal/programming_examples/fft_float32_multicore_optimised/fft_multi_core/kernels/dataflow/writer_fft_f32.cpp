// writer_fft_f32_mc.cpp — MULTICORE writer (FIXED v2)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  BUGS FIXED vs previous version
// ══════════════════════════════════════════════════════════════════════
//
//  BUG W1 (shuffle index arithmetic uses global instead of local base)
//  ─────────────────────────────────────────────────────────────────────
//  Previous code computed src indices using `row_elem_base`, which for
//  row N is N * tiles_per_row * TILE_SIZE (a large global offset).
//  The safe_local_src guard then fired for every element on every row
//  except row 0, silently skipping all copy_floats calls and leaving
//  the even/odd CBs filled with uninitialised L1 values.  Stages 1..9
//  operated on garbage data for all rows except the first.
//
//  FIX: The shuffle operates on the LOCAL CB buffer which is always
//  indexed 0..local_half-1 regardless of which row is being processed.
//  Replace `row_elem_base` in the index formula with 0 (i.e. remove
//  it entirely).  The safe_local_src guard is retained as a defence
//  against genuine underflow on the first/last partial groups but is
//  now actually reachable only in legitimate edge cases.
//
//  BUG W2 (CB op / tile_regs invariant — shared with compute kernel)
//  ─────────────────────────────────────────────────────────────────────
//  Writer does not call tile_regs_* so this bug is in the compute
//  kernel (see fft_compute_f32.cpp).  Writer is clean in this regard.
//
//  BUG W3 (pop-before-reserve ordering on intermediate stages)
//  ─────────────────────────────────────────────────────────────────────
//  Retained from previous fix: out0/out1 are popped before reserving
//  even/odd, preventing a depth-1 CB from appearing to need depth 2.
//
// ══════════════════════════════════════════════════════════════════════
//  SHUFFLE CORRECTNESS DRY-RUN (stage=0, local indices, N=1024)
// ══════════════════════════════════════════════════════════════════════
//
//  m=2, half_m=1, m2=4, half_m2=2, G2=512/2=256, log2m=1, m_mask=1
//  local_elem_base = 0 (always)
//
//  g2=0:  lb_e=0, lb_o=2
//    Block A: f0=0, g_old=0, off=0, ss=0, ls=0 → copy out0[0..1) ✓
//    Block B: f0=1, g_old=0, off=1, ss=0+0=0,  ls=0 → copy out1[0..1) ✓
//    Block C: f0=2, g_old=1, off=0, ss=1, ls=1 → copy out0[1..2) ✓
//    Block D: f0=3, g_old=1, off=1, ss=1+0=1,  ls=1 → copy out1[1..2) ✓
//  g2=1:  lb_e=4, lb_o=6
//    Block A: f0=4, g_old=2, off=0, ss=2, ls=2 → copy out0[2..3) ✓
//    ... pattern continues, all 512 elements correctly placed ✓
//
//  Same dry-run for any row (local_elem_base=0 always): identical ✓
//
// ══════════════════════════════════════════════════════════════════════
//  ARGUMENT MAP (unchanged from previous version)
// ══════════════════════════════════════════════════════════════════════
//
//  [0]  out0_r_addr
//  [1]  out0_i_addr
//  [2]  out1_r_addr
//  [3]  out1_i_addr
//  [4]  tiles_per_row
//  [5]  num_stages
//  [6]  local_half
//  [7]  half_N
//  [8..10] reserved
//  [11] tile_offset
//  [12] core_elem_base (unused)
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
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t local_half    = get_arg_val<uint32_t>(6);
    const uint32_t half_N        = get_arg_val<uint32_t>(7);
    // args [8..10] reserved
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);
    // arg[12] core_elem_base unused
    const uint32_t rows_per_core = get_arg_val<uint32_t>(13);

    // ── CB indices ────────────────────────────────────────────────────
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;

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

    // ── Scalar L1 helpers ─────────────────────────────────────────────
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

    // Underflow guard — only fires on genuine partial-group edge cases.
    auto safe_local_src = [](uint32_t src_start, uint32_t base) -> uint32_t {
        if (src_start < base) return UINT32_MAX;
        return src_start - base;
    };

    // ── Outer row loop ────────────────────────────────────────────────
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const bool is_last = (stage == num_stages - 1u);

            cb_wait_front(cb_out0_r, tiles_per_row);
            cb_wait_front(cb_out0_i, tiles_per_row);
            cb_wait_front(cb_out1_r, tiles_per_row);
            cb_wait_front(cb_out1_i, tiles_per_row);

            const uint32_t src0r = get_read_ptr(cb_out0_r);
            const uint32_t src0i = get_read_ptr(cb_out0_i);
            const uint32_t src1r = get_read_ptr(cb_out1_r);
            const uint32_t src1i = get_read_ptr(cb_out1_i);

            if (is_last) {
                // ── Last stage: write butterfly outputs to DRAM ───────
                for (uint32_t t = 0; t < tiles_per_row; t++) {
                    const uint32_t gt = row_tile_base + t;
                    noc_async_write_tile(gt, out0_r_gen, src0r + t * tile_bytes);
                    noc_async_write_tile(gt, out0_i_gen, src0i + t * tile_bytes);
                    noc_async_write_tile(gt, out1_r_gen, src1r + t * tile_bytes);
                    noc_async_write_tile(gt, out1_i_gen, src1i + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_out0_r, tiles_per_row);
                cb_pop_front(cb_out0_i, tiles_per_row);
                cb_pop_front(cb_out1_r, tiles_per_row);
                cb_pop_front(cb_out1_i, tiles_per_row);

            } else {
                // ── Intermediate stage: shuffle out → even/odd ────────
                //
                // FIX W3: pop outputs BEFORE reserving even/odd so the
                // output CB slots are free and depth-1 CBs stay coherent.

                // Stage geometry for next butterfly level.
                const uint32_t m       = 1u << (stage + 1u);
                const uint32_t half_m  = m >> 1u;
                const uint32_t m2      = m << 1u;
                const uint32_t half_m2 = m2 >> 1u;
                const uint32_t G2      = (half_m2 <= local_half)
                                         ? local_half / half_m2 : 0u;

                // Pop outputs first.
                cb_pop_front(cb_out0_r, tiles_per_row);
                cb_pop_front(cb_out0_i, tiles_per_row);
                cb_pop_front(cb_out1_r, tiles_per_row);
                cb_pop_front(cb_out1_i, tiles_per_row);

                // Reserve and fill even/odd for next stage.
                cb_reserve_back(cb_even_r, tiles_per_row);
                cb_reserve_back(cb_even_i, tiles_per_row);
                cb_reserve_back(cb_odd_r,  tiles_per_row);
                cb_reserve_back(cb_odd_i,  tiles_per_row);

                const uint32_t dst_er = get_write_ptr(cb_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_even_i);
                const uint32_t dst_or = get_write_ptr(cb_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_odd_i);

                if (G2 > 0u) {
                    // ── Normal shuffle (G2 complete double-groups) ────
                    //
                    // FIX W1: index arithmetic uses local_elem_base=0,
                    // NOT row_elem_base.  The CB buffer is always a
                    // local 0..local_half-1 window regardless of row.
                    constexpr uint32_t local_elem_base = 0u;

                    const uint32_t log2m  = stage + 1u;
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst_base = 0u;

                    for (uint32_t g2 = 0u; g2 < G2; g2++) {
                        const uint32_t lb_e = g2 * m2;
                        const uint32_t lb_o = lb_e + half_m2;

                        // Block A: new_even[0..half_m) ← out0
                        {
                            const uint32_t f0    = local_elem_base + lb_e;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + off;
                            const uint32_t ls    = safe_local_src(ss, local_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_er + dst_base * ELEM,
                                            src0r  + ls      * ELEM, half_m);
                                copy_floats(dst_ei + dst_base * ELEM,
                                            src0i  + ls      * ELEM, half_m);
                            }
                        }
                        // Block B: new_even[half_m..m) ← out1
                        {
                            const uint32_t f0    = local_elem_base + lb_e + half_m;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            // off >= half_m guaranteed when f0 is in the
                            // second half of a butterfly group.
                            const uint32_t ss    = g_old * half_m + (off - half_m);
                            const uint32_t ls    = safe_local_src(ss, local_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_er + (dst_base + half_m) * ELEM,
                                            src1r  + ls                  * ELEM, half_m);
                                copy_floats(dst_ei + (dst_base + half_m) * ELEM,
                                            src1i  + ls                  * ELEM, half_m);
                            }
                        }
                        // Block C: new_odd[0..half_m) ← out0
                        {
                            const uint32_t f0    = local_elem_base + lb_o;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + off;
                            const uint32_t ls    = safe_local_src(ss, local_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_or + dst_base * ELEM,
                                            src0r  + ls      * ELEM, half_m);
                                copy_floats(dst_oi + dst_base * ELEM,
                                            src0i  + ls      * ELEM, half_m);
                            }
                        }
                        // Block D: new_odd[half_m..m) ← out1
                        {
                            const uint32_t f0    = local_elem_base + lb_o + half_m;
                            const uint32_t g_old = f0 >> log2m;
                            const uint32_t off   = f0 & m_mask;
                            const uint32_t ss    = g_old * half_m + (off - half_m);
                            const uint32_t ls    = safe_local_src(ss, local_elem_base);
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
                    // ── G2=0: passthrough (no reordering needed) ──────
                    copy_floats(dst_er, src0r, local_half);
                    copy_floats(dst_ei, src0i, local_half);
                    copy_floats(dst_or, src1r, local_half);
                    copy_floats(dst_oi, src1i, local_half);
                }

                cb_push_back(cb_even_r, tiles_per_row);
                cb_push_back(cb_even_i, tiles_per_row);
                cb_push_back(cb_odd_r,  tiles_per_row);
                cb_push_back(cb_odd_i,  tiles_per_row);
            }
        }
        // Row complete. even/odd are empty (last stage wrote to DRAM).
    }
}