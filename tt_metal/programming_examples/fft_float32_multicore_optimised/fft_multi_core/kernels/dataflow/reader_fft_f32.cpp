// reader_fft_f32_mc.cpp — MULTICORE reader (FIXED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BUG FIX vs previous version:
//
//   Bug: rows_per_core argument was accepted but never used.
//   The reader uploaded local_tiles tiles for stage-0 inputs once,
//   then looped over num_stages for twiddle expansion — but there was
//   no outer loop over rows. For rows_per_core=128 only row 0's data
//   was ever loaded. The compute kernel stalled waiting for CBs after
//   row 0, causing the writer to write garbage from uninitialized state.
//
//   Fix: Added outer loop `for (row = 0; row < rows_per_core; row++)`.
//   Each iteration:
//     1. Loads local_tiles tiles for this row's even_r/i, odd_r/i into CBs.
//     2. Loops over all stages expanding twiddles into cb_tw_r/cb_tw_i.
//   The compact twiddle table is loaded once before the outer loop
//   (it is identical for every row) and kept resident in cb_compact_r/i.
//   tile_offset advances by local_tiles each row so each row reads
//   the correct DRAM tile indices.
//
//   Also fixed: the bounce-buffer 128-bit store for strided twiddle reads.
//   The original claimed 128-bit writes but called wr32() in the bulk path.
//   Fixed version uses store4_via_bounce() which actually issues TT_STOREIND.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "llk_io.h"
#include "llk_defs.h"

void kernel_main() {
    const uint32_t even_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t compact_r_addr = get_arg_val<uint32_t>(4);
    const uint32_t compact_i_addr = get_arg_val<uint32_t>(5);
    const uint32_t local_tiles    = get_arg_val<uint32_t>(6);   // tiles per single FFT row
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);   // first tile index for this core
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t half_N         = get_arg_val<uint32_t>(9);
    const uint32_t local_half     = get_arg_val<uint32_t>(10);
    const uint32_t rows_per_core  = get_arg_val<uint32_t>(11);  // FIX: was ignored before

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
    const InterleavedAddrGenFast<true> cmp_r_gen = {
        .bank_base_address = compact_r_addr,
        .page_size = compact_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> cmp_i_gen = {
        .bank_base_address = compact_i_addr,
        .page_size = compact_bytes, .data_format = data_format };

    if (local_tiles == 0 || num_stages == 0 || rows_per_core == 0) return;

    constexpr uint32_t ELEM    = sizeof(float);
    constexpr uint32_t ELEM128 = 4 * sizeof(float);

    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    auto copy128 = [](uint32_t dst, uint32_t src) {
        uint32_t sbase = src >> 4, soff = src & 0xFu;
        TT_SETDMAREG(0, LOWER_HALFWORD(soff),  0, LO_16(0));
        TT_SETDMAREG(0, UPPER_HALFWORD(soff),  0, HI_16(0));
        TT_SETDMAREG(0, LOWER_HALFWORD(sbase), 0, LO_16(1));
        TT_SETDMAREG(0, UPPER_HALFWORD(sbase), 0, HI_16(1));
        TT_LOADIND(p_ind::LD_128bit, LO_16(0), p_ind::INC_NONE, 4, 1);
        uint32_t dbase = dst >> 4, doff = dst & 0xFu;
        TT_SETDMAREG(0, LOWER_HALFWORD(doff),  0, LO_16(2));
        TT_SETDMAREG(0, UPPER_HALFWORD(doff),  0, HI_16(2));
        TT_SETDMAREG(0, LOWER_HALFWORD(dbase), 0, LO_16(3));
        TT_SETDMAREG(0, UPPER_HALFWORD(dbase), 0, HI_16(3));
        TT_STOREIND(p_ind::ST_128bit, LO_16(2), p_ind::INC_NONE, 4, 3);
    };

    // Bounce-buffer 128-bit store for strided twiddle sources.
    // Reads 4 scalars at arbitrary addresses, stores contiguously via ThCon.
    alignas(16) uint32_t bounce_r[4];
    alignas(16) uint32_t bounce_i[4];
    const uint32_t bounce_r_addr = reinterpret_cast<uint32_t>(&bounce_r[0]);
    const uint32_t bounce_i_addr = reinterpret_cast<uint32_t>(&bounce_i[0]);

    auto store4_via_bounce = [&](uint32_t dst_r, uint32_t dst_i,
                                  uint32_t src_r_base, uint32_t src_i_base,
                                  uint32_t idx0, uint32_t idx1,
                                  uint32_t idx2, uint32_t idx3) {
        bounce_r[0] = rd32(src_r_base + idx0 * ELEM);
        bounce_r[1] = rd32(src_r_base + idx1 * ELEM);
        bounce_r[2] = rd32(src_r_base + idx2 * ELEM);
        bounce_r[3] = rd32(src_r_base + idx3 * ELEM);
        bounce_i[0] = rd32(src_i_base + idx0 * ELEM);
        bounce_i[1] = rd32(src_i_base + idx1 * ELEM);
        bounce_i[2] = rd32(src_i_base + idx2 * ELEM);
        bounce_i[3] = rd32(src_i_base + idx3 * ELEM);
        copy128(dst_r, bounce_r_addr);
        copy128(dst_i, bounce_i_addr);
    };

    // ── Load compact twiddle table ONCE — shared across all rows ─────────
    // The twiddle table depends only on N, not on the input data.
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
    noc_async_read_barrier();
    cb_push_back(cb_compact_r, 1);
    cb_push_back(cb_compact_i, 1);

    cb_wait_front(cb_compact_r, 1);
    cb_wait_front(cb_compact_i, 1);
    const uint32_t cmp_r_base = get_read_ptr(cb_compact_r);
    const uint32_t cmp_i_base = get_read_ptr(cb_compact_i);

    // ── FIX: outer loop over rows_per_core ───────────────────────────────
    // Each iteration processes one complete FFT row:
    //   1. Load this row's stage-0 input tiles from DRAM.
    //   2. Expand twiddles for each stage.
    // tile_offset advances by local_tiles each row so the correct DRAM
    // tile indices are read.

    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_offset  = tile_offset + row * local_tiles;
        const uint32_t row_elem_base    = row_tile_offset * (tile_bytes / ELEM);

        // Load stage-0 input tiles for this row
        cb_reserve_back(cb_even_r, local_tiles);
        cb_reserve_back(cb_even_i, local_tiles);
        cb_reserve_back(cb_odd_r,  local_tiles);
        cb_reserve_back(cb_odd_i,  local_tiles);

        for (uint32_t t = 0; t < local_tiles; t++) {
            uint32_t gt = row_tile_offset + t;
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

        cb_push_back(cb_even_r, local_tiles);
        cb_push_back(cb_even_i, local_tiles);
        cb_push_back(cb_odd_r,  local_tiles);
        cb_push_back(cb_odd_i,  local_tiles);

        // Per-stage twiddle expansion for this row
        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const uint32_t half_m      = 1u << stage;
            const uint32_t N_over_m    = half_N >> stage;
            const uint32_t half_m_mask = half_m - 1u;
            const bool contiguous_src  = (N_over_m == 1);

            cb_reserve_back(cb_tw_r, local_tiles);
            cb_reserve_back(cb_tw_i, local_tiles);
            const uint32_t dst_r = get_write_ptr(cb_tw_r);
            const uint32_t dst_i = get_write_ptr(cb_tw_i);

            if (contiguous_src) {
                // Last stage: compact source contiguous → full 128-bit read+write
                uint32_t src_r     = cmp_r_base;
                uint32_t src_i     = cmp_i_base;
                uint32_t dst_r_ptr = dst_r;
                uint32_t dst_i_ptr = dst_i;
                const uint32_t repeats = local_half / half_m;

                for (uint32_t rep = 0; rep < repeats; rep++) {
                    uint32_t count = half_m;
                    uint32_t sr = src_r, si = src_i;
                    while (count >= 4
                           && (sr & 0xFu) == 0
                           && (dst_r_ptr & 0xFu) == 0) {
                        copy128(dst_r_ptr, sr);
                        copy128(dst_i_ptr, si);
                        sr += ELEM128; si += ELEM128;
                        dst_r_ptr += ELEM128; dst_i_ptr += ELEM128;
                        count -= 4;
                    }
                    while (count > 0) {
                        wr32(dst_r_ptr, rd32(sr));
                        wr32(dst_i_ptr, rd32(si));
                        sr += ELEM; si += ELEM;
                        dst_r_ptr += ELEM; dst_i_ptr += ELEM;
                        count--;
                    }
                }
            } else {
                // General case: strided source, contiguous destination.
                // Scalar scatter-read + 128-bit bounce store.
                uint32_t lp = 0;

                // Scalar prologue to align destination
                while (lp < local_half && ((dst_r + lp * ELEM) & 0xFu)) {
                    const uint32_t p   = row_elem_base + lp;
                    const uint32_t idx = (p & half_m_mask) * N_over_m;
                    wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                    wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
                    lp++;
                }
                // Bulk: 4 scalar reads → bounce buffer → 1× 128-bit store
                while (lp + 4 <= local_half) {
                    const uint32_t p0   = row_elem_base + lp;
                    const uint32_t idx0 = (p0       & half_m_mask) * N_over_m;
                    const uint32_t idx1 = ((p0 + 1) & half_m_mask) * N_over_m;
                    const uint32_t idx2 = ((p0 + 2) & half_m_mask) * N_over_m;
                    const uint32_t idx3 = ((p0 + 3) & half_m_mask) * N_over_m;
                    store4_via_bounce(dst_r + lp * ELEM, dst_i + lp * ELEM,
                                      cmp_r_base, cmp_i_base,
                                      idx0, idx1, idx2, idx3);
                    lp += 4;
                }
                // Scalar epilogue
                while (lp < local_half) {
                    const uint32_t p   = row_elem_base + lp;
                    const uint32_t idx = (p & half_m_mask) * N_over_m;
                    wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                    wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
                    lp++;
                }
            }

            cb_push_back(cb_tw_r, local_tiles);
            cb_push_back(cb_tw_i, local_tiles);
        }
        // End of row — stage-0 inputs and twiddles for this row have been
        // pushed to CBs and will be consumed by compute + writer.
        // The compact twiddle CB stays live for the next row.
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}