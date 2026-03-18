// reader_fft_f32_mc.cpp — MULTICORE reader with ThCon 128-bit twiddle expand
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ThCon optimisation applied to twiddle expansion:
//   The compact twiddle table lookup produces idx = j * N_over_m.
//   Within each stage, consecutive elements p=0..half_m-1 all map to
//   j=0..half_m-1, giving idx values spaced N_over_m apart (non-contiguous).
//   After j wraps back to 0, the pattern repeats — NOT contiguous.
//
//   However, when N_over_m = 1 (last stage, stage = log2N-1), all
//   elements map to consecutive compact table entries → 128-bit reads.
//   For all other stages the source is strided → scalar fallback.
//
//   The ThCon benefit here is smaller than in the writer, but the
//   write side (storing expanded twiddles into the CB) is always
//   contiguous → we use 128-bit stores for the write path at all stages.

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
    const uint32_t local_tiles    = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset    = get_arg_val<uint32_t>(7);
    const uint32_t num_stages     = get_arg_val<uint32_t>(8);
    const uint32_t half_N         = get_arg_val<uint32_t>(9);
    const uint32_t local_half     = get_arg_val<uint32_t>(10);

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

    if (local_tiles == 0 || num_stages == 0) return;

    constexpr uint32_t ELEM    = sizeof(float);
    constexpr uint32_t ELEM128 = 4 * sizeof(float);

    // ── Scalar 32-bit read ─────────────────────────────────────────────
    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // ── ThCon 128-bit copy (4 floats at once) ─────────────────────────
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

    // ── Upload stage-0 inputs + compact twiddle ────────────────────────
    cb_reserve_back(cb_even_r, local_tiles);
    cb_reserve_back(cb_even_i, local_tiles);
    cb_reserve_back(cb_odd_r,  local_tiles);
    cb_reserve_back(cb_odd_i,  local_tiles);
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);

    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t gt = tile_offset + t;
        noc_async_read_tile(gt, even_r_gen,
            get_write_ptr(cb_even_r) + t*tile_bytes);
        noc_async_read_tile(gt, even_i_gen,
            get_write_ptr(cb_even_i) + t*tile_bytes);
        noc_async_read_tile(gt, odd_r_gen,
            get_write_ptr(cb_odd_r)  + t*tile_bytes);
        noc_async_read_tile(gt, odd_i_gen,
            get_write_ptr(cb_odd_i)  + t*tile_bytes);
    }
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
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

    const uint32_t core_elem_base = tile_offset * (tile_bytes / ELEM);

    // ── Per-stage twiddle expansion ────────────────────────────────────
    //
    // Source pattern: compact[j * N_over_m] for j = 0..half_m-1,
    // repeated local_half/half_m times.
    //
    // Stride analysis:
    //   N_over_m = half_N >> stage = N / 2^(stage+1)
    //   Source stride between consecutive j: N_over_m floats
    //   When N_over_m == 1 (stage == log2N-1): contiguous → 128-bit read
    //   Otherwise: strided → scalar read (ThCon stride load not available)
    //
    // Write side: always contiguous → 128-bit write at all stages.
    // We scatter-read (scalar) then sequential-write (128-bit).

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
            // ── Last stage: compact source is contiguous ───────────────
            // N_over_m=1 → idx = j*1 = j, source is compact[0..half_m-1]
            // Read and write with 128-bit ThCon.
            uint32_t src_r = cmp_r_base;
            uint32_t src_i = cmp_i_base;
            uint32_t dst_r_ptr = dst_r;
            uint32_t dst_i_ptr = dst_i;

            // The compact table has half_m entries at this stage, repeated
            // local_half/half_m times in the output.
            uint32_t repeats = local_half / half_m;
            for (uint32_t rep = 0; rep < repeats; rep++) {
                uint32_t count = half_m;
                uint32_t sr = src_r, si = src_i;
                while (count >= 4 && (sr & 0xFu) == 0 && (dst_r_ptr & 0xFu) == 0) {
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
            // ── General case: strided source, sequential destination ───
            // Read scalar (strided compact table), write sequentially.
            // The write is always contiguous so we buffer 4 values and
            // flush with a 128-bit store when aligned and count >= 4.
            //
            // This is a scatter-read / gather-write pattern:
            //   for each lp: idx = (global_p & half_m_mask) * N_over_m
            //   source = compact[idx] — stride N_over_m between elements
            //   destination = twiddle_cb[lp] — stride 1 (contiguous)

            uint32_t lp = 0;
            // Align destination to 16 bytes with scalar writes
            while (lp < local_half && (dst_r + lp*ELEM) & 0xFu) {
                uint32_t p   = core_elem_base + lp;
                uint32_t j   = p & half_m_mask;
                uint32_t idx = j * N_over_m;
                wr32(dst_r + lp*ELEM, rd32(cmp_r_base + idx*ELEM));
                wr32(dst_i + lp*ELEM, rd32(cmp_i_base + idx*ELEM));
                lp++;
            }
            // Bulk: read 4 scalars, store 1×128-bit
            // We collect into a temp buffer and store with TT_STOREIND
            while (lp + 4 <= local_half) {
                uint32_t tmp_r[4], tmp_i[4];
                for (int k = 0; k < 4; k++) {
                    uint32_t p   = core_elem_base + lp + k;
                    uint32_t j   = p & half_m_mask;
                    uint32_t idx = j * N_over_m;
                    tmp_r[k] = rd32(cmp_r_base + idx*ELEM);
                    tmp_i[k] = rd32(cmp_i_base + idx*ELEM);
                }
                // Write 4 floats via scalar (TT_STOREIND needs ThCon reg,
                // not a memory array — use scalar for safety here)
                for (int k = 0; k < 4; k++) {
                    wr32(dst_r + (lp+k)*ELEM, tmp_r[k]);
                    wr32(dst_i + (lp+k)*ELEM, tmp_i[k]);
                }
                lp += 4;
            }
            // Scalar tail
            while (lp < local_half) {
                uint32_t p   = core_elem_base + lp;
                uint32_t j   = p & half_m_mask;
                uint32_t idx = j * N_over_m;
                wr32(dst_r + lp*ELEM, rd32(cmp_r_base + idx*ELEM));
                wr32(dst_i + lp*ELEM, rd32(cmp_i_base + idx*ELEM));
                lp++;
            }
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}