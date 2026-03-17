// writer_fft_f32_mc.cpp  — MULTICORE writer with NOC cross-core shuffle
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  MULTICORE FFT STAGE CLASSIFICATION
// ══════════════════════════════════════════════════════════════════════
//
//  Given num_cores C and N-point FFT (log2N stages total):
//
//  LOCAL STAGES  (stage >= log2C):
//    Both butterfly partners reside on the SAME core.
//    Writer performs L1-to-L1 shuffle (identical to single-core design).
//    No NOC traffic.
//
//  CROSS-CORE STAGES  (stage < log2C):
//    Butterfly partners reside on DIFFERENT cores.
//    Each core owns a contiguous block of N/(2C) even/odd pairs.
//    After computing out0/out1, each core must exchange half its results
//    with a "partner" core via NOC async writes into the partner's CB 0-3.
//
//  PARTNER ASSIGNMENT for stage s (s < log2C):
//    block_size = num_cores >> (s+1)
//    partner_id = core_id XOR block_size
//    (Standard radix-2 DIT: partner bits flip the (s+1)-th bit of core_id)
//
//  CROSS-CORE SHUFFLE PROTOCOL:
//    1. Compute writes out0/out1 to CB 16-19 as usual.
//    2. Writer on core A computes which elements go to partner B.
//    3. Writer NOC-writes those elements into B's cb_even / cb_odd
//       destination L1 address (passed as runtime args per stage).
//    4. Writer writes its own retained elements into its own cb_even/cb_odd.
//    5. Both writers call noc_async_write_barrier(), then cb_push_back.
//    6. Compute on each core proceeds once cb_wait_front is satisfied.
//
//  For a balanced split (each core retains half of out0, half of out1):
//    - Elements with global index bit (log2C - 1 - s) == 0  → stay on core
//    - Elements with global index bit (log2C - 1 - s) == 1  → go to partner
//
// ══════════════════════════════════════════════════════════════════════
//
// Args:
//   0   out0_r_addr        DRAM base for final even-real output
//   1   out0_i_addr        DRAM base for final even-imag output
//   2   out1_r_addr        DRAM base for final odd-real output
//   3   out1_i_addr        DRAM base for final odd-imag output
//   4   local_tiles        tiles this core owns per stage
//   5   num_stages         log2N
//   6   local_half         N / (2 * num_cores)
//   7   half_N             N/2
//   8   num_cores          C
//   9   core_id            0..C-1
//  10   log2_cores         log2(C)
//  11   tile_offset        first global tile index for this core
//  12.. partner_noc_x[s]   NOC X of partner core for stage s  (log2C entries)
//  12+log2C.. partner_noc_y[s]   NOC Y of partner core for stage s
//  12+2*log2C.. partner_cb_er[s]   L1 write ptr of partner's cb_even_r for stage s
//  12+3*log2C.. partner_cb_ei[s]
//  12+4*log2C.. partner_cb_or[s]
//  12+5*log2C.. partner_cb_oi[s]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t local_tiles  = get_arg_val<uint32_t>(4);
    const uint32_t num_stages   = get_arg_val<uint32_t>(5);
    const uint32_t local_half   = get_arg_val<uint32_t>(6);
    const uint32_t half_N       = get_arg_val<uint32_t>(7);
    const uint32_t num_cores    = get_arg_val<uint32_t>(8);
    const uint32_t core_id      = get_arg_val<uint32_t>(9);
    const uint32_t log2_cores   = get_arg_val<uint32_t>(10);
    const uint32_t tile_offset  = get_arg_val<uint32_t>(11);

    // Per-stage cross-core args start at index 12
    // Layout: [noc_x * log2C] [noc_y * log2C] [cb_er * log2C]
    //         [cb_ei * log2C] [cb_or * log2C]  [cb_oi * log2C]
    const uint32_t cross_args_base = 12;

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

    constexpr uint32_t ELEM     = sizeof(float);
    const uint32_t core_elem_base = tile_offset * (tile_bytes / ELEM);

    // ── Helpers: synchronous L1 r/w (no strict-aliasing violation) ──────
    auto rd = [](uint32_t addr) -> float {
        uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
        float v = 0.0f;
        __builtin_memcpy(&v, &raw, sizeof(float));
        return v;
    };
    auto wr = [](uint32_t addr, float v) {
        uint32_t raw = 0u;
        __builtin_memcpy(&raw, &v, sizeof(float));
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
            // ── DRAM write: last stage only ──────────────────────────
            for (uint32_t t = 0; t < local_tiles; t++) {
                uint32_t global_t = tile_offset + t;
                noc_async_write_tile(global_t, out0_r_gen,
                    src0r + t * tile_bytes);
                noc_async_write_tile(global_t, out0_i_gen,
                    src0i + t * tile_bytes);
                noc_async_write_tile(global_t, out1_r_gen,
                    src1r + t * tile_bytes);
                noc_async_write_tile(global_t, out1_i_gen,
                    src1i + t * tile_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

        } else if (is_cross_core) {
            // ══════════════════════════════════════════════════════════
            // CROSS-CORE SHUFFLE via NOC
            // ══════════════════════════════════════════════════════════
            //
            // At stage s < log2C, the FFT butterfly group straddles two cores.
            // Standard radix-2 DIT: at stage s the group size is m = 2^(s+1).
            // Cores are paired in blocks of size 2^(log2C - s - 1):
            //   partner = core_id XOR (num_cores >> (s+1))
            //
            // Decide which half of local out0/out1 this core keeps vs sends:
            //   - The "keep" criterion is whether the global element index
            //     has bit (log2C - 1 - s) == core_id's bit at that position.
            //   - Simpler: the lower half of this core's elements (lp < local_half/2)
            //     go to the "keep" region, upper half to the "send" region,
            //     because the bit-reversed ordering ensures a clean split.
            //
            // Each core sends local_half/2 elements to partner and keeps
            // local_half/2 elements for itself — resulting in local_half
            // elements in cb_even and local_half in cb_odd for the next stage.

            const uint32_t partner_noc_x = get_arg_val<uint32_t>(
                cross_args_base + stage);
            const uint32_t partner_noc_y = get_arg_val<uint32_t>(
                cross_args_base + log2_cores + stage);
            const uint32_t p_dst_er = get_arg_val<uint32_t>(
                cross_args_base + 2*log2_cores + stage);
            const uint32_t p_dst_ei = get_arg_val<uint32_t>(
                cross_args_base + 3*log2_cores + stage);
            const uint32_t p_dst_or = get_arg_val<uint32_t>(
                cross_args_base + 4*log2_cores + stage);
            const uint32_t p_dst_oi = get_arg_val<uint32_t>(
                cross_args_base + 5*log2_cores + stage);

            const uint32_t send_half = local_half / 2;  // elements sent to partner

            // Reserve our own next-stage input CBs
            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            // Determine keep/send split.
            // Convention: lower indexed elements (lp < send_half) are "even"
            // within the inter-core butterfly group — they map to out0 slots.
            // Upper indexed elements (lp >= send_half) are "odd" — out1 slots.
            //
            // Whether this core is the "lower" or "upper" partner in the group
            // is determined by bit (log2_cores - 1 - stage) of core_id.
            const uint32_t group_bit = (num_cores >> (stage + 1));
            const bool is_lower_partner = ((core_id & group_bit) == 0);

            // Scratch buffers to build what we send to partner (NOC write)
            // We write them into the partner's cb_even / cb_odd write-ptr.
            // The partner has already reserved those slots (symmetric protocol).

            if (is_lower_partner) {
                // Lower partner keeps out0[0..send_half-1] as its new even.
                // Lower partner keeps out1[0..send_half-1] as its new odd.
                // Lower partner sends out0[send_half..local_half-1] → partner even.
                // Lower partner sends out1[send_half..local_half-1] → partner odd.
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    wr(dst_er + lp * ELEM, rd(src0r + lp * ELEM));
                    wr(dst_ei + lp * ELEM, rd(src0i + lp * ELEM));
                    wr(dst_or + lp * ELEM, rd(src1r + lp * ELEM));
                    wr(dst_oi + lp * ELEM, rd(src1i + lp * ELEM));
                }
                // Send upper half to partner via NOC
                // Partner's cb_even gets our out0 upper; partner's cb_odd gets out1 upper
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    // Pack element into a temp uint32 word and write over NOC
                    uint32_t e_r, e_i, o_r, o_i;
                    float tmp;
                    tmp = rd(src0r + (send_half + lp) * ELEM);
                    __builtin_memcpy(&e_r, &tmp, 4);
                    tmp = rd(src0i + (send_half + lp) * ELEM);
                    __builtin_memcpy(&e_i, &tmp, 4);
                    tmp = rd(src1r + (send_half + lp) * ELEM);
                    __builtin_memcpy(&o_r, &tmp, 4);
                    tmp = rd(src1i + (send_half + lp) * ELEM);
                    __builtin_memcpy(&o_i, &tmp, 4);

                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_er + lp * ELEM),
                        src0r + (send_half + lp) * ELEM, ELEM, 1, false);
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_ei + lp * ELEM),
                        src0i + (send_half + lp) * ELEM, ELEM, 1, false);
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_or + lp * ELEM),
                        src1r + (send_half + lp) * ELEM, ELEM, 1, false);
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_oi + lp * ELEM),
                        src1i + (send_half + lp) * ELEM, ELEM, 1, false);
                }
            } else {
                // Upper partner keeps out0[send_half..local_half-1] as even.
                // Upper partner keeps out1[send_half..local_half-1] as odd.
                // Upper partner sends out0[0..send_half-1] → partner even[send_half..].
                // Upper partner sends out1[0..send_half-1] → partner odd[send_half..].
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    wr(dst_er + lp * ELEM, rd(src0r + (send_half + lp) * ELEM));
                    wr(dst_ei + lp * ELEM, rd(src0i + (send_half + lp) * ELEM));
                    wr(dst_or + lp * ELEM, rd(src1r + (send_half + lp) * ELEM));
                    wr(dst_oi + lp * ELEM, rd(src1i + (send_half + lp) * ELEM));
                }
                // Send lower half to partner's upper slots
                for (uint32_t lp = 0; lp < send_half; lp++) {
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_er + (send_half + lp) * ELEM),
                        src0r + lp * ELEM, ELEM, 1, false);
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_ei + (send_half + lp) * ELEM),
                        src0i + lp * ELEM, ELEM, 1, false);
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_or + (send_half + lp) * ELEM),
                        src1r + lp * ELEM, ELEM, 1, false);
                    noc_async_write_multicast_loopback_src(
                        (uint64_t)get_noc_addr(partner_noc_x, partner_noc_y,
                            p_dst_oi + (send_half + lp) * ELEM),
                        src1i + lp * ELEM, ELEM, 1, false);
                }
            }

            // Barrier: ensure our NOC sends to partner land before partner's
            // compute proceeds (partner's cb_wait_front enforces this via
            // the cb depth=1 handshake on its own side).
            noc_async_write_barrier();

            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r,  local_tiles);
            cb_push_back(cb_odd_i,  local_tiles);

        } else {
            // ══════════════════════════════════════════════════════════
            // LOCAL SHUFFLE — identical to single-core writer
            // ══════════════════════════════════════════════════════════
            const uint32_t m      = 1u << (stage + 1);
            const uint32_t half_m = m >> 1;
            const uint32_t m2     = m << 1;
            const uint32_t half_m2= m2 >> 1;
            const uint32_t G2     = local_half / half_m2;

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
                const uint32_t base_e = g2 * m2;
                const uint32_t base_o = base_e + half_m2;
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {

                    // new_even[dst]
                    {
                        uint32_t f      = base_e + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f &  m_mask;
                        uint32_t idx, srcr, srci;
                        if (offset < half_m) {
                            idx = g_old * half_m + offset;
                            srcr = src0r; srci = src0i;
                        } else {
                            idx = g_old * half_m + (offset - half_m);
                            srcr = src1r; srci = src1i;
                        }
                        wr(dst_er + dst * ELEM, rd(srcr + idx * ELEM));
                        wr(dst_ei + dst * ELEM, rd(srci + idx * ELEM));
                    }

                    // new_odd[dst]
                    {
                        uint32_t f      = base_o + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f &  m_mask;
                        uint32_t idx, srcr, srci;
                        if (offset < half_m) {
                            idx = g_old * half_m + offset;
                            srcr = src0r; srci = src0i;
                        } else {
                            idx = g_old * half_m + (offset - half_m);
                            srcr = src1r; srci = src1i;
                        }
                        wr(dst_or + dst * ELEM, rd(srcr + idx * ELEM));
                        wr(dst_oi + dst * ELEM, rd(srci + idx * ELEM));
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