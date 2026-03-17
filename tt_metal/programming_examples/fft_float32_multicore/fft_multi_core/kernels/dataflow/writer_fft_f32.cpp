// writer_fft_f32_mc.cpp  — MULTICORE writer
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// STAGE CLASSIFICATION
// ════════════════════
//  Local stages  (stage >= log2_cores):
//    Both butterfly partners on the SAME core.
//    L1-to-L1 shuffle, no NOC. Uses GLOBAL element index
//    (core_elem_base + local_p) for the shuffle formula.
//
//  Cross-core stages  (stage < log2_cores):
//    Partners on DIFFERENT cores. Each core sends half its
//    butterfly results to its partner via NOC unicast write,
//    keeps the other half locally.
//
// CRITICAL FIXES vs previous version
// ════════════════════════════════════
//  1. Local shuffle: stage offset corrected.
//     The single-core writer used raw `stage` in the m/half_m
//     formula because stage ran 0..log2N-1.  In multicore the
//     local-stage loop still iterates stage=log2_cores..log2N-1
//     so we pass `stage` directly — that is already the global
//     stage index, which is what the formula needs.  No change
//     needed here, but G2 must use local_half, not half_N.
//
//  2. Local shuffle: global element base added.
//     Each core owns elements [core_elem_base .. core_elem_base+local_half).
//     The shuffle index `f` in the formula refers to GLOBAL positions.
//     We add core_elem_base to f before computing g_old / offset.
//
//  3. NOC write: use noc_async_write, not multicast.
//     noc_async_write_multicast_loopback_src is for multicast
//     groups. For a simple core-to-core unicast use noc_async_write.
//
//  4. Cross-core sync: partner must signal us before we push_back
//     to compute. We use a semaphore: each writer increments the
//     partner's semaphore after its NOC writes land, then waits
//     on its own semaphore reaching 1 before calling cb_push_back.
//
// Args (positions must match host SetRuntimeArgs exactly):
//   0   out0_r_addr
//   1   out0_i_addr
//   2   out1_r_addr
//   3   out1_i_addr
//   4   local_tiles
//   5   num_stages      (log2N)
//   6   local_half      N/(2*num_cores)
//   7   half_N          N/2
//   8   num_cores
//   9   core_id
//  10   log2_cores
//  11   tile_offset     first global tile index for this core
//  12   core_elem_base  tile_offset * TILE_SIZE (first global element)
//  -- cross-core per-stage arrays (log2_cores entries each) --
//  13                   partner_noc_x[0..log2_cores-1]
//  13+log2_cores        partner_noc_y[0..log2_cores-1]
//  13+2*log2_cores      partner_cb_er[0..log2_cores-1]
//  13+3*log2_cores      partner_cb_ei[0..log2_cores-1]
//  13+4*log2_cores      partner_cb_or[0..log2_cores-1]
//  13+5*log2_cores      partner_cb_oi[0..log2_cores-1]
//  13+6*log2_cores      partner_sem_addr[0..log2_cores-1]
//  13+7*log2_cores      my_sem_addr[0..log2_cores-1]

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
    const uint32_t core_elem_base= get_arg_val<uint32_t>(12); // = tile_offset * TILE_SIZE

    const uint32_t cross_base    = 13;

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
            // ── CROSS-CORE SHUFFLE via NOC unicast ───────────────────
            //
            // At stage s, group size m = 2^(s+1).
            // Partner: core_id XOR (num_cores >> (s+1))
            //
            // Lower partner (bit==0) keeps out0[0..send_half-1] locally
            // and sends out0[send_half..local_half-1] to partner.
            // Upper partner (bit==1) keeps out0[send_half..local_half-1]
            // locally and sends out0[0..send_half-1] to partner.
            // Same logic for out1 → odd.

            const uint32_t partner_noc_x = get_arg_val<uint32_t>(
                cross_base + stage);
            const uint32_t partner_noc_y = get_arg_val<uint32_t>(
                cross_base + log2_cores + stage);
            const uint32_t p_dst_er = get_arg_val<uint32_t>(
                cross_base + 2*log2_cores + stage);
            const uint32_t p_dst_ei = get_arg_val<uint32_t>(
                cross_base + 3*log2_cores + stage);
            const uint32_t p_dst_or = get_arg_val<uint32_t>(
                cross_base + 4*log2_cores + stage);
            const uint32_t p_dst_oi = get_arg_val<uint32_t>(
                cross_base + 5*log2_cores + stage);
            const uint32_t p_sem_addr = get_arg_val<uint32_t>(
                cross_base + 6*log2_cores + stage);
            const uint32_t my_sem_l1  = get_arg_val<uint32_t>(
                cross_base + 7*log2_cores + stage);

            const uint32_t send_half = local_half / 2;
            const uint32_t group_bit = (num_cores >> (stage + 1));
            const bool is_lower = ((core_id & group_bit) == 0);

            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            // Reset semaphore to 0 before use
            volatile uint32_t* my_sem =
                reinterpret_cast<volatile uint32_t*>(my_sem_l1);
            *my_sem = 0;

            if (is_lower) {
                // Keep lower half locally
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    wr(dst_er + lp*ELEM, rd(src0r + lp*ELEM));
                    wr(dst_ei + lp*ELEM, rd(src0i + lp*ELEM));
                    wr(dst_or + lp*ELEM, rd(src1r + lp*ELEM));
                    wr(dst_oi + lp*ELEM, rd(src1i + lp*ELEM));
                }
                // Send upper half to partner's lower slot
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    noc_async_write(src0r + (send_half+lp)*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_er + lp*ELEM), ELEM);
                    noc_async_write(src0i + (send_half+lp)*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_ei + lp*ELEM), ELEM);
                    noc_async_write(src1r + (send_half+lp)*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_or + lp*ELEM), ELEM);
                    noc_async_write(src1i + (send_half+lp)*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_oi + lp*ELEM), ELEM);
                }
            } else {
                // Keep upper half locally
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    wr(dst_er + lp*ELEM, rd(src0r + (send_half+lp)*ELEM));
                    wr(dst_ei + lp*ELEM, rd(src0i + (send_half+lp)*ELEM));
                    wr(dst_or + lp*ELEM, rd(src1r + (send_half+lp)*ELEM));
                    wr(dst_oi + lp*ELEM, rd(src1i + (send_half+lp)*ELEM));
                }
                // Send lower half to partner's upper slot
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    noc_async_write(src0r + lp*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_er + (send_half+lp)*ELEM), ELEM);
                    noc_async_write(src0i + lp*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_ei + (send_half+lp)*ELEM), ELEM);
                    noc_async_write(src1r + lp*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_or + (send_half+lp)*ELEM), ELEM);
                    noc_async_write(src1i + lp*ELEM,
                        get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_oi + (send_half+lp)*ELEM), ELEM);
                }
            }

            // Wait for all NOC writes to land
            noc_async_write_barrier();

            // Signal partner: my data has arrived in your CB
            noc_semaphore_inc(
                get_noc_addr(partner_noc_x, partner_noc_y, p_sem_addr), 1);

            // Wait for partner to signal us: their data is in our CB
            noc_semaphore_wait(my_sem, 1);

            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r,  local_tiles);
            cb_push_back(cb_odd_i,  local_tiles);

        } else {
            // ── LOCAL SHUFFLE ────────────────────────────────────────
            //
            // Uses the same formula as single-core writer, but:
            //   - G2 = local_half / half_m2  (only our slice, not all N/2)
            //   - f uses GLOBAL index: f_global = core_elem_base + f_local
            //     so g_old and offset resolve correctly across all cores.

            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            // Number of next-stage groups whose both halves are on this core
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
                // base_e/base_o are LOCAL offsets within our slice.
                // Add core_elem_base to get the GLOBAL index for formula.
                const uint32_t local_base_e = g2 * m2;
                const uint32_t local_base_o = local_base_e + half_m2;

                for (uint32_t j2 = 0; j2 < half_m2; j2++) {

                    // new_even[dst]
                    {
                        // Global f: offset within global N/2 array
                        uint32_t f      = core_elem_base + local_base_e + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f &  m_mask;
                        // g_old and offset are global, but the data lives
                        // in our local src buffers at local index:
                        //   local_idx = g_old * half_m + offset - core_elem_base
                        // which simplifies to the same index arithmetic
                        // because core_elem_base is always a multiple of half_m
                        // at local stages (stage >= log2_cores guarantees this).
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx = global_idx - core_elem_base;
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_er + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_ei + dst*ELEM, rd(srci + local_idx*ELEM));
                    }

                    // new_odd[dst]
                    {
                        uint32_t f      = core_elem_base + local_base_o + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f &  m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx = global_idx - core_elem_base;
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