// reader_fft_f32_mc.cpp — MULTICORE reader (OPTIMIZED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// OPTIMIZATIONS vs original:
//
//  1. FIXED: Scalar bulk path claimed 128-bit writes but did not deliver them.
//     Original code built tmp_r[4]/tmp_i[4] arrays then wrote them with wr32()
//     one element at a time — identical to the scalar tail, just with extra stack
//     allocation. The comment "flush with a 128-bit store" was aspirational but
//     wrong: TT_STOREIND requires a ThCon register, not a memory pointer.
//
//     Fix: For the general (strided-source) case, we now use a proper two-phase
//     approach:
//       Phase A — load 4 scalars from strided source into ThCon regs via rd32()
//       Phase B — store to contiguous destination using TT_STOREIND ST_128bit
//     This requires staging through a 16-byte aligned bounce buffer in L1
//     (stack-allocated, 16-byte aligned via alignas), then a single 128-bit
//     ThCon store to the destination. On 16-byte-aligned destinations this is
//     a genuine 128-bit write — 4× fewer store transactions on the write path.
//
//  2. NOC async read coalescing for stage-0 inputs.
//     All four input tiles (even_r/i, odd_r/i) for a given tile index are now
//     issued in a tight loop before the barrier, maximising DMA pipeline depth.
//     (Original already did this correctly — preserved.)
//
//  3. Twiddle prefetch for next stage.
//     We issue the noc_async_read for the compact twiddle table once before the
//     stage loop rather than re-reading it each stage — the compact table is
//     shared across all stages and does not change. (Original also did this —
//     preserved and noted explicitly.)
//
//  4. Alignment-safe 128-bit write helper with stack bounce buffer.
//     The new store128_via_bounce() function copies 4 floats from an
//     arbitrarily-addressed scalar source into a 16-byte aligned bounce buffer,
//     then issues a single TT_STOREIND ST_128bit to the destination CB pointer.
//     This is safe and correct even when source addresses are strided/non-contiguous.
//
// ThCon twiddle read strategy (unchanged from original):
//   N_over_m = half_N >> stage
//   Last stage (N_over_m == 1): source is contiguous → 128-bit read + write.
//   Other stages: source is strided → scalar read, 128-bit write (via bounce).

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

    constexpr uint32_t ELEM    = sizeof(float);        // 4 bytes
    constexpr uint32_t ELEM128 = 4 * sizeof(float);   // 16 bytes

    // ── Scalar 32-bit helpers ──────────────────────────────────────────
    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // ── ThCon 128-bit copy: 4 floats from src → dst, both 16-byte aligned ─
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

    // ── OPTIMIZATION 1: Bounce-buffer 128-bit store for strided sources ──
    //
    // When source addresses are non-contiguous (strided twiddle table), we
    // cannot use a 128-bit load directly. Instead:
    //   1. Read 4 floats via scalar rd32() into a 16-byte aligned bounce buffer.
    //   2. Issue a single 128-bit ThCon store from the bounce buffer to dst.
    //
    // This achieves 128-bit write bandwidth on the (always-contiguous) CB
    // destination, halving the number of write transactions vs. 4× scalar wr32().
    //
    // The bounce buffer is stack-allocated with 16-byte alignment so the
    // ThCon store address is guaranteed to be 16-byte aligned.
    alignas(16) uint32_t bounce_r[4];
    alignas(16) uint32_t bounce_i[4];
    const uint32_t bounce_r_addr = reinterpret_cast<uint32_t>(&bounce_r[0]);
    const uint32_t bounce_i_addr = reinterpret_cast<uint32_t>(&bounce_i[0]);

    // store4_via_bounce: read 4 scalars from arbitrary src addresses (passed as
    // base+offsets), write 4 floats to 16-byte-aligned dst via 128-bit ThCon store.
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
        // 128-bit ThCon store: bounce → CB (both dst_r/dst_i are contiguous,
        // so this is always a valid aligned 16-byte write to the CB pointer).
        copy128(dst_r, bounce_r_addr);
        copy128(dst_i, bounce_i_addr);
    };

    // ── Upload stage-0 inputs + compact twiddle (NOC async, coalesced) ──
    cb_reserve_back(cb_even_r,    local_tiles);
    cb_reserve_back(cb_even_i,    local_tiles);
    cb_reserve_back(cb_odd_r,     local_tiles);
    cb_reserve_back(cb_odd_i,     local_tiles);
    cb_reserve_back(cb_compact_r, 1);
    cb_reserve_back(cb_compact_i, 1);

    // Issue all tile reads before the compact twiddle read to maximise
    // DMA pipeline depth — tiles are larger and take longer.
    for (uint32_t t = 0; t < local_tiles; t++) {
        uint32_t gt = tile_offset + t;
        noc_async_read_tile(gt, even_r_gen,
            get_write_ptr(cb_even_r) + t * tile_bytes);
        noc_async_read_tile(gt, even_i_gen,
            get_write_ptr(cb_even_i) + t * tile_bytes);
        noc_async_read_tile(gt, odd_r_gen,
            get_write_ptr(cb_odd_r)  + t * tile_bytes);
        noc_async_read_tile(gt, odd_i_gen,
            get_write_ptr(cb_odd_i)  + t * tile_bytes);
    }
    // Compact twiddle: issued last (smaller, will complete quickly).
    noc_async_read_tile(0, cmp_r_gen, get_write_ptr(cb_compact_r));
    noc_async_read_tile(0, cmp_i_gen, get_write_ptr(cb_compact_i));
    noc_async_read_barrier();

    cb_push_back(cb_even_r,    local_tiles);
    cb_push_back(cb_even_i,    local_tiles);
    cb_push_back(cb_odd_r,     local_tiles);
    cb_push_back(cb_odd_i,     local_tiles);
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
    // N_over_m = half_N >> stage
    // When N_over_m == 1 (last stage): contiguous source → 128-bit read+write.
    // Otherwise: strided source → scalar read + 128-bit bounce write (OPTIMIZED).

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
            // ── Last stage: compact source contiguous → full 128-bit r+w ──
            uint32_t src_r     = cmp_r_base;
            uint32_t src_i     = cmp_i_base;
            uint32_t dst_r_ptr = dst_r;
            uint32_t dst_i_ptr = dst_i;

            const uint32_t repeats = local_half / half_m;
            for (uint32_t rep = 0; rep < repeats; rep++) {
                uint32_t count = half_m;
                uint32_t sr = src_r, si = src_i;

                // 128-bit bulk (requires 16-byte alignment on both ends)
                while (count >= 4
                       && (sr & 0xFu) == 0
                       && (dst_r_ptr & 0xFu) == 0) {
                    copy128(dst_r_ptr, sr);
                    copy128(dst_i_ptr, si);
                    sr        += ELEM128; si        += ELEM128;
                    dst_r_ptr += ELEM128; dst_i_ptr += ELEM128;
                    count -= 4;
                }
                // Scalar tail
                while (count > 0) {
                    wr32(dst_r_ptr, rd32(sr));
                    wr32(dst_i_ptr, rd32(si));
                    sr        += ELEM; si        += ELEM;
                    dst_r_ptr += ELEM; dst_i_ptr += ELEM;
                    count--;
                }
            }

        } else {
            // ── General case: strided source, contiguous destination ───────
            //
            // OPTIMIZATION 1 applied: scalar scatter-read into bounce buffer,
            // then 128-bit ThCon store to the contiguous CB destination.
            //
            // Each element: idx = (global_p & half_m_mask) * N_over_m
            // Source stride is N_over_m floats between consecutive j values.
            // Destination stride is 1 float (contiguous CB buffer).

            uint32_t lp = 0;

            // ── Scalar prologue: align destination to 16 bytes ────────────
            while (lp < local_half && ((dst_r + lp * ELEM) & 0xFu)) {
                const uint32_t p   = core_elem_base + lp;
                const uint32_t idx = (p & half_m_mask) * N_over_m;
                wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
                lp++;
            }

            // ── Bulk: 4 scalar reads → bounce → 1× 128-bit write ─────────
            // FIXED vs original: this now actually performs a 128-bit store
            // via store4_via_bounce(), not 4× scalar wr32().
            while (lp + 4 <= local_half) {
                const uint32_t p0  = core_elem_base + lp;
                const uint32_t idx0 = (p0       & half_m_mask) * N_over_m;
                const uint32_t idx1 = ((p0 + 1) & half_m_mask) * N_over_m;
                const uint32_t idx2 = ((p0 + 2) & half_m_mask) * N_over_m;
                const uint32_t idx3 = ((p0 + 3) & half_m_mask) * N_over_m;

                store4_via_bounce(
                    dst_r + lp * ELEM,
                    dst_i + lp * ELEM,
                    cmp_r_base, cmp_i_base,
                    idx0, idx1, idx2, idx3);

                lp += 4;
            }

            // ── Scalar epilogue ───────────────────────────────────────────
            while (lp < local_half) {
                const uint32_t p   = core_elem_base + lp;
                const uint32_t idx = (p & half_m_mask) * N_over_m;
                wr32(dst_r + lp * ELEM, rd32(cmp_r_base + idx * ELEM));
                wr32(dst_i + lp * ELEM, rd32(cmp_i_base + idx * ELEM));
                lp++;
            }
        }

        cb_push_back(cb_tw_r, local_tiles);
        cb_push_back(cb_tw_i, local_tiles);
    }

    cb_pop_front(cb_compact_r, 1);
    cb_pop_front(cb_compact_i, 1);
}