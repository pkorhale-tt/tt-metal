// writer_fft_f32_mc.cpp  — MULTICORE writer  [BUG-FIXED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fixes applied:
//
//   BUG 4 (G2==0 fallback — split loop):
//     Original split the copy into two loops over local_half/2 each.
//     Replaced with a single clean loop over local_half. Functionally
//     equivalent but removes the confusing split and makes the intent clear.
//
//   BUG 6 (uint32_t underflow in local_idx):
//     Original: uint32_t local_idx = global_src_idx - core_elem_base;
//     If global_src_idx < core_elem_base (due to any host-side bit-reversal
//     error) this wraps to a huge value and silently reads garbage.
//     Fix: assert global_src_idx >= core_elem_base before subtraction,
//     and clamp to 0 in release builds to prevent out-of-bounds memory reads.
//
// KEY INSIGHT (verified by working through N=8, 2 cores manually):
// ══════════════════════════════════════════════════════════════════
// The single-core shuffle formula works element-by-element. For each
// destination slot dst, it computes a source index from out0 or out1.
// In multicore, EVERY core runs the SAME formula — just over its own
// local slice [core_elem_base .. core_elem_base+local_half).
//
// There is NO cross-core data exchange needed at any stage, including
// cross-core stages (stage < log2_cores). The bit-reversal in
// prepare_stage0 on the host ensures that each core's butterfly inputs
// produce outputs that, when shuffled by the standard formula, land
// exactly in that core's local output range.
//
// NOTE (Bug 7 — intentional simplification):
//   In fft_multicore_2d.cpp the host passes num_cores=1 and log2_cores=0
//   to this writer for the row-decomposition case, because each core runs
//   a completely independent full-row FFT with core_elem_base=0. This is
//   correct for that use case. If this writer is ever reused for true
//   butterfly-partitioned multicore FFT, the host MUST pass the real
//   num_cores, log2_cores, and core_elem_base values.
//
// Args:
//   0-3   DRAM output addresses (out0_r/i, out1_r/i)
//   4     local_tiles
//   5     num_stages (log2N)
//   6     local_half
//   7     half_N
//   8     num_cores
//   9     core_id
//  10     log2_cores
//  11     tile_offset
//  12     core_elem_base

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr   = get_arg_val<uint32_t>(3);
    const uint32_t local_tiles   = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t local_half    = get_arg_val<uint32_t>(6);
    const uint32_t half_N        = get_arg_val<uint32_t>(7);
    const uint32_t num_cores     = get_arg_val<uint32_t>(8);
    const uint32_t core_id       = get_arg_val<uint32_t>(9);
    const uint32_t log2_cores    = get_arg_val<uint32_t>(10);
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);
    const uint32_t core_elem_base= get_arg_val<uint32_t>(12);

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

    if (local_tiles == 0) return;

    constexpr uint32_t ELEM = sizeof(float);

    auto rd = [](uint32_t addr) -> float {
        uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
        float v; __builtin_memcpy(&v, &raw, 4); return v;
    };
    auto wr = [](uint32_t addr, float v) {
        uint32_t raw; __builtin_memcpy(&raw, &v, 4);
        *reinterpret_cast<volatile uint32_t*>(addr) = raw;
    };

    // FIX (Bug 6): safe subtraction helper — asserts in debug, clamps in
    // release to prevent out-of-bounds memory access from uint32 underflow.
    // If this fires it means the host's bit-reversal is wrong.
    auto safe_local_idx = [&](uint32_t global_idx) -> uint32_t {
        ASSERT(global_idx >= core_elem_base);
        // In release builds, clamp rather than wrap to prevent wild reads.
        if (global_idx < core_elem_base) return 0u;
        return global_idx - core_elem_base;
    };

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
            // ── DRAM write ───────────────────────────────────────────
            for (uint32_t t = 0; t < local_tiles; t++) {
                uint32_t gt = tile_offset + t;
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
            // ── SHUFFLE ──────────────────────────────────────────────
            //
            // Same formula as single-core writer, with:
            //   G2 = local_half / half_m2   (our slice only)
            //   f  = core_elem_base + local_f  (global index)
            //   local_idx = global_src_idx - core_elem_base  (safe_local_idx)
            //
            // When half_m2 > local_half: G2=0 → fallback copies out0→even,
            // out1→odd directly (no shuffle needed at this scale).

            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = (half_m2 <= local_half)
                                     ? local_half / half_m2
                                     : 0u;

            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            const uint32_t log2m  = stage + 1;
            const uint32_t m_mask = m - 1u;

            uint32_t dst = 0;
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                const uint32_t local_base_e = g2 * m2;
                const uint32_t local_base_o = local_base_e + half_m2;

                for (uint32_t j2 = 0; j2 < half_m2; j2++) {

                    // new_even[dst]
                    {
                        uint32_t f      = core_elem_base + local_base_e + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        // FIX (Bug 6): use safe subtraction to detect host errors.
                        uint32_t local_idx = safe_local_idx(global_idx);
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_er + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_ei + dst*ELEM, rd(srci + local_idx*ELEM));
                    }

                    // new_odd[dst]
                    {
                        uint32_t f      = core_elem_base + local_base_o + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        // FIX (Bug 6): use safe subtraction.
                        uint32_t local_idx = safe_local_idx(global_idx);
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_or + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_oi + dst*ELEM, rd(srci + local_idx*ELEM));
                    }

                    dst++;
                }
            }

            // FIX (Bug 4): Replace the confusing split two-loop fallback
            // with a single clean loop over all local_half elements.
            // out0 → even, out1 → odd, element by element.
            // Correctness: the bit-reversed partition ensures out0[k]/out1[k]
            // are already the correct even/odd pair for the next stage when
            // the group spans multiple cores (G2==0 case).
            if (G2 == 0) {
                for (uint32_t lp = 0; lp < local_half; lp++) {
                    wr(dst_er + lp*ELEM, rd(src0r + lp*ELEM));
                    wr(dst_ei + lp*ELEM, rd(src0i + lp*ELEM));
                    wr(dst_or + lp*ELEM, rd(src1r + lp*ELEM));
                    wr(dst_oi + lp*ELEM, rd(src1i + lp*ELEM));
                }
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
}