// writer_fft_f32_mc.cpp — MULTICORE writer (FIXED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BUG FIX vs previous version:
//
//   Bug: rows_per_core argument was accepted but the stage loop was not
//   wrapped in an outer row loop. The writer processed local_tiles tiles
//   for num_stages stages once — corresponding to exactly one FFT row.
//   For rows_per_core=128, rows 1-127 were never written to DRAM, and
//   the compute kernel stalled after row 0 because output CBs were never
//   drained.
//
//   Fix: Added outer loop `for (row = 0; row < rows_per_core; row++)`.
//   Each iteration processes one complete FFT row (all stages).
//   tile_offset for DRAM writes advances by local_tiles each row.
//   The shuffle (inter-stage reorder) uses row_elem_base derived from
//   the row's tile offset for correct twiddle index computation.
//
//   Also fixed: safe_local_src() underflow guard on unsigned subtraction.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "llk_io.h"
#include "llk_defs.h"

void kernel_main() {
    const uint32_t out0_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr    = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr    = get_arg_val<uint32_t>(3);
    const uint32_t local_tiles    = get_arg_val<uint32_t>(4);
    const uint32_t num_stages     = get_arg_val<uint32_t>(5);
    const uint32_t local_half     = get_arg_val<uint32_t>(6);
    const uint32_t half_N         = get_arg_val<uint32_t>(7);
    const uint32_t num_cores      = get_arg_val<uint32_t>(8);
    const uint32_t core_id        = get_arg_val<uint32_t>(9);
    const uint32_t log2_cores     = get_arg_val<uint32_t>(10);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(11);  // base tile index for this core
    const uint32_t core_elem_base = get_arg_val<uint32_t>(12);  // base elem index for this core
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(13);  // FIX: was ignored before

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

    if (local_tiles == 0 || rows_per_core == 0) return;

    constexpr uint32_t ELEM    = sizeof(float);
    constexpr uint32_t ELEM128 = 4 * sizeof(float);

    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    auto copy128 = [](uint32_t dst, uint32_t src) {
        uint32_t base   = src >> 4, offset = src & 0xFu;
        TT_SETDMAREG(0, LOWER_HALFWORD(offset), 0, LO_16(0));
        TT_SETDMAREG(0, UPPER_HALFWORD(offset), 0, HI_16(0));
        TT_SETDMAREG(0, LOWER_HALFWORD(base),   0, LO_16(1));
        TT_SETDMAREG(0, UPPER_HALFWORD(base),   0, HI_16(1));
        TT_LOADIND(p_ind::LD_128bit, LO_16(0), p_ind::INC_NONE, 4, 1);
        uint32_t dbase = dst >> 4, doffset = dst & 0xFu;
        TT_SETDMAREG(0, LOWER_HALFWORD(doffset), 0, LO_16(2));
        TT_SETDMAREG(0, UPPER_HALFWORD(doffset), 0, HI_16(2));
        TT_SETDMAREG(0, LOWER_HALFWORD(dbase),   0, LO_16(3));
        TT_SETDMAREG(0, UPPER_HALFWORD(dbase),   0, HI_16(3));
        TT_STOREIND(p_ind::ST_128bit, LO_16(2), p_ind::INC_NONE, 4, 3);
    };

    auto copy_floats = [&](uint32_t dst, uint32_t src, uint32_t count) {
        while (count > 0 && (dst & 0xFu) != 0) {
            wr32(dst, rd32(src)); dst += ELEM; src += ELEM; count--;
        }
        while (count >= 4) {
            copy128(dst, src); dst += ELEM128; src += ELEM128; count -= 4;
        }
        while (count > 0) {
            wr32(dst, rd32(src)); dst += ELEM; src += ELEM; count--;
        }
    };

    // FIX: safe unsigned subtraction with underflow guard.
    auto safe_local_src = [](uint32_t src_start, uint32_t base) -> uint32_t {
        if (src_start < base) return UINT32_MAX; // partitioning invariant violated
        return src_start - base;
    };

    // ── FIX: outer loop over rows ─────────────────────────────────────────
    // Each iteration processes one complete FFT row through all stages.
    // row_tile_offset: DRAM tile base for this row's output.
    // row_elem_base:   element index base for this row's shuffle computation.

    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_offset = tile_offset + row * local_tiles;
        const uint32_t row_elem_base   = row_tile_offset * (tile_bytes / ELEM);

        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const bool is_last = (stage == num_stages - 1);

            cb_wait_front(cb_out0_r, local_tiles);
            cb_wait_front(cb_out0_i, local_tiles);
            cb_wait_front(cb_out1_r, local_tiles);
            cb_wait_front(cb_out1_i, local_tiles);

            const uint32_t src0r = get_read_ptr(cb_out0_r);
            const uint32_t src0i = get_read_ptr(cb_out0_i);
            const uint32_t src1r = get_read_ptr(cb_out1_r);
            const uint32_t src1i = get_read_ptr(cb_out1_i);

            if (is_last) {
                // DRAM write: all 4 arrays, tight loop for NOC pipeline depth
                for (uint32_t t = 0; t < local_tiles; t++) {
                    const uint32_t gt = row_tile_offset + t;
                    noc_async_write_tile(gt, out0_r_gen, src0r + t * tile_bytes);
                    noc_async_write_tile(gt, out0_i_gen, src0i + t * tile_bytes);
                    noc_async_write_tile(gt, out1_r_gen, src1r + t * tile_bytes);
                    noc_async_write_tile(gt, out1_i_gen, src1i + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_out0_r, local_tiles);
                cb_pop_front(cb_out0_i, local_tiles);
                cb_pop_front(cb_out1_r, local_tiles);
                cb_pop_front(cb_out1_i, local_tiles);

            } else {
                // Shuffle: reorder out0/out1 → even/odd for next stage
                const uint32_t m       = 1u << (stage + 1);
                const uint32_t half_m  = m >> 1;
                const uint32_t m2      = m << 1;
                const uint32_t half_m2 = m2 >> 1;
                const uint32_t G2      = (half_m2 <= local_half)
                                         ? local_half / half_m2 : 0u;

                cb_reserve_back(cb_even_r, local_tiles);
                cb_reserve_back(cb_even_i, local_tiles);
                cb_reserve_back(cb_odd_r,  local_tiles);
                cb_reserve_back(cb_odd_i,  local_tiles);

                const uint32_t dst_er = get_write_ptr(cb_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_even_i);
                const uint32_t dst_or = get_write_ptr(cb_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_odd_i);

                if (G2 > 0) {
                    const uint32_t log2m  = stage + 1;
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst = 0;

                    for (uint32_t g2 = 0; g2 < G2; g2++) {
                        const uint32_t lb_e = g2 * m2;
                        const uint32_t lb_o = lb_e + half_m2;

                        // Block A: new_even[0..half_m) ← out0
                        {
                            uint32_t f0  = row_elem_base + lb_e;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + off;
                            uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_er + dst*ELEM, src0r + ls*ELEM, half_m);
                                copy_floats(dst_ei + dst*ELEM, src0i + ls*ELEM, half_m);
                            }
                        }
                        // Block B: new_even[half_m..m) ← out1
                        {
                            uint32_t f0  = row_elem_base + lb_e + half_m;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + (off - half_m);
                            uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_er + (dst+half_m)*ELEM, src1r + ls*ELEM, half_m);
                                copy_floats(dst_ei + (dst+half_m)*ELEM, src1i + ls*ELEM, half_m);
                            }
                        }
                        // Block C: new_odd[0..half_m) ← out0
                        {
                            uint32_t f0  = row_elem_base + lb_o;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + off;
                            uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_or + dst*ELEM, src0r + ls*ELEM, half_m);
                                copy_floats(dst_oi + dst*ELEM, src0i + ls*ELEM, half_m);
                            }
                        }
                        // Block D: new_odd[half_m..m) ← out1
                        {
                            uint32_t f0  = row_elem_base + lb_o + half_m;
                            uint32_t g_old = f0 >> log2m;
                            uint32_t off   = f0 & m_mask;
                            uint32_t ss    = g_old * half_m + (off - half_m);
                            uint32_t ls    = safe_local_src(ss, row_elem_base);
                            if (ls != UINT32_MAX) {
                                copy_floats(dst_or + (dst+half_m)*ELEM, src1r + ls*ELEM, half_m);
                                copy_floats(dst_oi + (dst+half_m)*ELEM, src1i + ls*ELEM, half_m);
                            }
                        }

                        dst += half_m2;
                    }
                } else {
                    // G2=0: direct passthrough, fully contiguous
                    copy_floats(dst_er, src0r, local_half);
                    copy_floats(dst_ei, src0i, local_half);
                    copy_floats(dst_or, src1r, local_half);
                    copy_floats(dst_oi, src1i, local_half);
                }

                cb_pop_front(cb_out0_r, local_tiles);
                cb_pop_front(cb_out0_i, local_tiles);
                cb_pop_front(cb_out1_r, local_tiles);
                cb_pop_front(cb_out1_i, local_tiles);

                cb_push_back(cb_even_r, local_tiles);
                cb_push_back(cb_even_i, local_tiles);
                cb_push_back(cb_odd_r,  local_tiles);
                cb_push_back(cb_odd_i,  local_tiles);
            }
        }
        // End of row — all stages complete, DRAM write done for this row.
    }
}