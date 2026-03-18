// writer_fft_f32_mc.cpp  — MULTICORE writer (corrected butterfly split)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// CROSS-CORE BUTTERFLY — CORRECT SPLIT
// ══════════════════════════════════════
// At stage s, the radix-2 DIT butterfly produces:
//   out0[k] = even[k] + W[k]*odd[k]   (butterfly sum)
//   out1[k] = even[k] - W[k]*odd[k]   (butterfly difference)
//
// For the NEXT stage, each butterfly needs one element from out0
// and one from out1 of the SAME index k. So out0[k] and out1[k]
// always travel together to the same destination core.
//
// Cross-core split rule (standard Cooley-Tukey):
//   Lower partner (core_id bit==0):
//     keeps   out0[0..local_half-1]  as new even
//     keeps   out1[0..local_half-1]  as new odd
//     sends   nothing — it already has what it needs
//     BUT receives out0/out1 from upper partner to complete its tile
//
// Wait — the correct model is simpler:
//   Each core after a cross-core butterfly stage gets a FULL tile of
//   out0 values (butterfly sums) OR out1 values (differences) depending
//   on whether it is the lower or upper partner in each group.
//
//   Lower partner keeps out0 (sums)   as its even+odd for next stage
//   Upper partner keeps out1 (diffs)  as its even+odd for next stage
//
// This is the standard in-place decimation-in-time FFT:
//   after butterfly, the "top" output goes to lower index, "bottom" to upper.
//
// IMPLEMENTATION:
//   Lower core: new_even = out0[0..half-1], new_odd = out0[half..local_half-1]
//   Upper core: new_even = out1[0..half-1], new_odd = out1[half..local_half-1]
//   No NOC exchange needed — each core already has its full out0 or out1.
//   The input split (prepare_stage0 / bit-reversal) ensures the correct
//   elements are on each core from the start.
//
// NOC IS ONLY NEEDED when the butterfly partners are on different cores
// AND the output of one core feeds the input of the other. In the
// standard Cooley-Tukey partitioning used here (contiguous block per core),
// after stage s < log2_cores the data is already correctly distributed —
// lower core takes out0, upper core takes out1, NO exchange required.
//
// This means the cross-core "exchange" is actually just a LOCAL selection:
//   is_lower → use out0 as next stage input
//   is_upper → use out1 as next stage input
//
// The NOC exchange we were doing was wrong — it was mixing outputs across
// cores when it shouldn't have been.

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

    // Cross-core args still passed (for future use / NOC twiddle sync)
    // but not used for data exchange in this corrected version.
    // Layout unchanged so host args don't need to change.

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

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last       = (stage == num_stages - 1);
        const bool is_cross_core = (stage < log2_cores);

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

        } else if (is_cross_core) {
            // ── CROSS-CORE STAGE: LOCAL SELECTION (no NOC exchange) ──
            //
            // Standard Cooley-Tukey DIT with contiguous core partitioning:
            // After the butterfly at stage s, the outputs are already on
            // the correct core — no data movement needed.
            //
            // The bit that determines lower/upper in the FFT butterfly group
            // at stage s is bit (log2_cores - 1 - s) of core_id.
            // Equivalently: group_bit = num_cores >> (s+1)
            //
            //   Lower partner (core_id & group_bit == 0):
            //     Takes out0 (butterfly sums) as next stage input.
            //     new_even = out0[0 .. half-1]
            //     new_odd  = out0[half .. local_half-1]
            //
            //   Upper partner (core_id & group_bit != 0):
            //     Takes out1 (butterfly differences) as next stage input.
            //     new_even = out1[0 .. half-1]
            //     new_odd  = out1[half .. local_half-1]
            //
            // Why this works: prepare_stage0 bit-reverses the input so that
            // elements destined for lower/upper outputs are already separated
            // into the correct core's slice. Each core runs its local butterfly
            // and then selects its output half — no cross-core communication.

            const uint32_t group_bit = (num_cores >> (stage + 1));
            const bool is_lower = ((core_id & group_bit) == 0);
            const uint32_t half = local_half / 2;

            // Select which output to use as next stage input
            const uint32_t use_r = is_lower ? src0r : src1r;
            const uint32_t use_i = is_lower ? src0i : src1i;

            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            // Copy selected output into next stage even/odd input
            for (uint32_t lp = 0; lp < half; lp++) {
                wr(dst_er + lp*ELEM, rd(use_r + lp*ELEM));
                wr(dst_ei + lp*ELEM, rd(use_i + lp*ELEM));
            }
            for (uint32_t lp = 0; lp < half; lp++) {
                wr(dst_or + lp*ELEM, rd(use_r + (half+lp)*ELEM));
                wr(dst_oi + lp*ELEM, rd(use_i + (half+lp)*ELEM));
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
            // ── LOCAL SHUFFLE (stage >= log2_cores) ──────────────────
            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = local_half / half_m2;

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
                        uint32_t f          = core_elem_base + local_base_e + j2;
                        uint32_t g_old      = f >> log2m;
                        uint32_t offset     = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx  = global_idx - core_elem_base;
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_er + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_ei + dst*ELEM, rd(srci + local_idx*ELEM));
                    }
                    // new_odd[dst]
                    {
                        uint32_t f          = core_elem_base + local_base_o + j2;
                        uint32_t g_old      = f >> log2m;
                        uint32_t offset     = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx  = global_idx - core_elem_base;
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_or + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_oi + dst*ELEM, rd(srci + local_idx*ELEM));
                    }
                    dst++;
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