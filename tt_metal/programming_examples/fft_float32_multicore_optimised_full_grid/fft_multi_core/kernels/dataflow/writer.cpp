// writer_fft_1d_64core.cpp - FIXED
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Responsibilities per stage:
//   LOCAL  stage (s < local_stages):
//     Shuffle compute outputs back into even/odd CBs for the next
//     butterfly stage. The shuffle re-orders elements within this
//     core's contiguous block.
//
//   CROSS-CORE stage (s >= local_stages, s < num_stages-1):
//     TODO: Real implementation requires NOC send/recv with the
//     butterfly-partner core.  Currently stubs as identity copy
//     (correct for single-core testing, wrong for multi-core FFT).
//
//   FINAL stage (s == num_stages-1):
//     Write the completed butterfly results directly to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Runtime args ─────────────────────────────────────────────
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
    const uint32_t tile_offset    = get_arg_val<uint32_t>(11);
    const uint32_t core_elem_base = get_arg_val<uint32_t>(12);
    const uint32_t local_stages   = get_arg_val<uint32_t>(13);

    // ── CB indices ───────────────────────────────────────────────
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;

    const uint32_t    tile_bytes  = get_tile_size(cb_out0_r);
    const DataFormat  data_format = get_dataformat(cb_out0_r);
    constexpr uint32_t ELEM       = sizeof(float);
    constexpr uint32_t TILE_SIZE  = 1024;

    // ── DRAM address generators (final-stage write) ──────────────
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

    // ── Typed L1 memory read/write helpers ───────────────────────
    auto rd = [](uint32_t addr) -> float {
        uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
        float v; __builtin_memcpy(&v, &raw, 4); return v;
    };
    auto wr = [](uint32_t addr, float v) {
        uint32_t raw; __builtin_memcpy(&raw, &v, 4);
        *reinterpret_cast<volatile uint32_t*>(addr) = raw;
    };

    // ── Per-stage loop ───────────────────────────────────────────
    for (uint32_t stage = 0; stage < num_stages; stage++) {

        const bool is_last       = (stage == num_stages - 1);
        const bool is_cross_core = (stage >= local_stages);

        cb_wait_front(cb_out0_r, local_tiles);
        cb_wait_front(cb_out0_i, local_tiles);
        cb_wait_front(cb_out1_r, local_tiles);
        cb_wait_front(cb_out1_i, local_tiles);

        const uint32_t src0r = get_read_ptr(cb_out0_r);
        const uint32_t src0i = get_read_ptr(cb_out0_i);
        const uint32_t src1r = get_read_ptr(cb_out1_r);
        const uint32_t src1i = get_read_ptr(cb_out1_i);

        if (is_last) {
            // ═════════════════════════════════════════════════════
            // FINAL STAGE: Write butterfly outputs to DRAM.
            // out0 holds the "upper" half, out1 the "lower" half.
            // ═════════════════════════════════════════════════════
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

        } else if (is_cross_core) {
            // ═════════════════════════════════════════════════════
            // CROSS-CORE STAGE (STUB)
            //
            // A full implementation requires:
            //   1. Identify partner core: partner = core_id XOR (1 << (stage - local_stages))
            //   2. Send this core's half-result to partner via NOC write.
            //   3. Receive partner's half-result via NOC read / semaphore sync.
            //   4. Re-pack into even/odd layout for the next stage.
            //
            // Until that is implemented, we do an identity copy (pass-
            // through) which produces wrong FFT results for multi-core
            // runs but won't cause a hang.
            // ═════════════════════════════════════════════════════
            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            for (uint32_t i = 0; i < local_half; i++) {
                wr(dst_er + i * ELEM, rd(src0r + i * ELEM));
                wr(dst_ei + i * ELEM, rd(src0i + i * ELEM));
                wr(dst_or + i * ELEM, rd(src1r + i * ELEM));
                wr(dst_oi + i * ELEM, rd(src1i + i * ELEM));
            }
            // Zero-pad the rest of the tile
            for (uint32_t i = local_half; i < local_tiles * TILE_SIZE; i++) {
                wr(dst_er + i * ELEM, 0.0f);
                wr(dst_ei + i * ELEM, 0.0f);
                wr(dst_or + i * ELEM, 0.0f);
                wr(dst_oi + i * ELEM, 0.0f);
            }

            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r,  local_tiles);
            cb_push_back(cb_odd_i,  local_tiles);

        } else {
            // ═════════════════════════════════════════════════════
            // LOCAL STAGE
            //
            // Butterfly group size at this stage: m = 2^(stage+1)
            // Each group contributes half_m outputs to out0 and
            // half_m outputs to out1.
            //
            // We need to re-map those outputs into the even/odd
            // layout expected by the NEXT stage:
            //   even[dst_even_idx] ← out0 or out1 element
            //   odd [dst_odd_idx]  ← out0 or out1 element
            // ═════════════════════════════════════════════════════
            const uint32_t m        = 1u << (stage + 1);
            const uint32_t half_m   = m >> 1;
            const uint32_t log2m    = stage + 1;
            const uint32_t m_mask   = m - 1u;
            const uint32_t num_groups = local_half / m;

            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            if (num_groups > 0) {
                for (uint32_t g = 0; g < num_groups; g++) {
                    for (uint32_t j = 0; j < half_m; j++) {

                        // ── even destination ──────────────────────
                        uint32_t dst_even_idx = g * m + j;
                        uint32_t global_pos   = core_elem_base + dst_even_idx;
                        uint32_t old_group    = global_pos >> log2m;
                        uint32_t offset       = global_pos & m_mask;
                        bool     from_out0    = (offset < half_m);
                        uint32_t src_idx      = old_group * half_m + offset;

                        if (src_idx >= core_elem_base &&
                            src_idx <  core_elem_base + local_half) {
                            uint32_t local_src = src_idx - core_elem_base;
                            uint32_t srcr = from_out0 ? src0r : src1r;
                            uint32_t srci = from_out0 ? src0i : src1i;
                            wr(dst_er + dst_even_idx * ELEM, rd(srcr + local_src * ELEM));
                            wr(dst_ei + dst_even_idx * ELEM, rd(srci + local_src * ELEM));
                        }

                        // ── odd destination ───────────────────────
                        uint32_t dst_odd_idx = g * half_m + j;
                        global_pos   = core_elem_base + (g * m + half_m + j);
                        old_group    = global_pos >> log2m;
                        offset       = global_pos & m_mask;
                        from_out0    = (offset < half_m);
                        src_idx      = old_group * half_m + offset;

                        if (src_idx >= core_elem_base &&
                            src_idx <  core_elem_base + local_half) {
                            uint32_t local_src = src_idx - core_elem_base;
                            uint32_t srcr = from_out0 ? src0r : src1r;
                            uint32_t srci = from_out0 ? src0i : src1i;
                            wr(dst_or + dst_odd_idx * ELEM, rd(srcr + local_src * ELEM));
                            wr(dst_oi + dst_odd_idx * ELEM, rd(srci + local_src * ELEM));
                        }
                    }
                }
            } else {
                // num_groups == 0: group spans multiple cores,
                // simple pass-through for now
                for (uint32_t lp = 0; lp < local_half; lp++) {
                    wr(dst_er + lp * ELEM, rd(src0r + lp * ELEM));
                    wr(dst_ei + lp * ELEM, rd(src0i + lp * ELEM));
                    wr(dst_or + lp * ELEM, rd(src1r + lp * ELEM));
                    wr(dst_oi + lp * ELEM, rd(src1i + lp * ELEM));
                }
            }

            // Zero-pad remainder of tile
            for (uint32_t lp = local_half; lp < local_tiles * TILE_SIZE; lp++) {
                wr(dst_er + lp * ELEM, 0.0f);
                wr(dst_ei + lp * ELEM, 0.0f);
                wr(dst_or + lp * ELEM, 0.0f);
                wr(dst_oi + lp * ELEM, 0.0f);
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