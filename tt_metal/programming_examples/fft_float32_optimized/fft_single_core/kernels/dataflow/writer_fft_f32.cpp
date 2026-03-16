// writer_fft_f32.cpp  — OPTIMAL: L1-to-L1 inter-stage shuffle
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// This writer does two distinct jobs, selected by stage:
//
// INTERMEDIATE STAGES (0 .. log2N-2): L1-to-L1 shuffle.
//   The compute kernel writes out0 and out1 into CB 16-19 (in L1).
//   The writer reads them and shuffles the values into CB 0-3 (also in L1)
//   so that the next stage's butterfly receives the correct even/odd pairs.
//   No DRAM is touched. Uses noc_async_write with L1 destination addresses.
//
// LAST STAGE (log2N-1): DRAM write.
//   The final butterfly outputs are written from CB 16-19 to DRAM output
//   buffers. This happens exactly once regardless of log2N.
//
// SHUFFLE FORMULA (verified for N=4..1024, FFT and IFFT):
//   After stage s: m = 1<<(s+1), half_m = m>>1
//   Next stage group size: m2 = m<<1, half_m2 = m2>>1, G2 = N//m2
//
//   For new_even[dst] (dst=0..half_N-1):
//     for g2 in [0,G2), for j2 in [0,half_m2):
//       f      = g2*m2 + j2
//       g_old  = f / m
//       offset = f % m
//       if offset < half_m: new_even[dst] = out0[g_old*half_m + offset]
//       else:               new_even[dst] = out1[g_old*half_m + offset - half_m]
//       dst++
//
//   For new_odd[dst]: identical but f = g2*m2 + half_m2 + j2

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(4);  // tiles_per_stage
    const uint32_t num_stages  = get_arg_val<uint32_t>(5);  // log2N
    const uint32_t half_N      = get_arg_val<uint32_t>(6);  // N/2 elements

    // Output CBs (compute writes here, RISCV_1 drains here)
    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    // Input CBs for next stage (writer shuffles into these, compute reads from them)
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

    if (num_tiles == 0) return;

    constexpr uint32_t ELEM = sizeof(float);  // 4 bytes per element

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);

        // Wait for compute to finish this stage's butterfly output
        cb_wait_front(cb_out0_r, num_tiles);
        cb_wait_front(cb_out0_i, num_tiles);
        cb_wait_front(cb_out1_r, num_tiles);
        cb_wait_front(cb_out1_i, num_tiles);

        if (is_last) {
            // ── DRAM write (once, last stage only) ───────────────
            for (uint32_t t = 0; t < num_tiles; t++) {
                noc_async_write_tile(t, out0_r_gen,
                    get_read_ptr(cb_out0_r) + t * tile_bytes);
                noc_async_write_tile(t, out0_i_gen,
                    get_read_ptr(cb_out0_i) + t * tile_bytes);
                noc_async_write_tile(t, out1_r_gen,
                    get_read_ptr(cb_out1_r) + t * tile_bytes);
                noc_async_write_tile(t, out1_i_gen,
                    get_read_ptr(cb_out1_i) + t * tile_bytes);
            }
            noc_async_write_barrier();

            cb_pop_front(cb_out0_r, num_tiles);
            cb_pop_front(cb_out0_i, num_tiles);
            cb_pop_front(cb_out1_r, num_tiles);
            cb_pop_front(cb_out1_i, num_tiles);

        } else {
            // ── L1-to-L1 shuffle (intermediate stages) ───────────
            //
            // Source:  CB 16-19 (out0_r/i and out1_r/i) — in L1
            // Destination: CB 0-3 (even_r/i and odd_r/i) — also in L1
            //
            // The shuffle regroups out0/out1 into the correct even/odd pair
            // layout needed by the NEXT stage's butterfly.

            const uint32_t m      = 1u << (stage + 1);
            const uint32_t half_m = m >> 1;
            const uint32_t m2     = m << 1;
            const uint32_t half_m2= m2 >> 1;
            const uint32_t G2     = half_N / half_m2;  // = N / m2

            // Base L1 read pointers (out0/out1 in CB 16-19)
            const uint32_t src0r = get_read_ptr(cb_out0_r);
            const uint32_t src0i = get_read_ptr(cb_out0_i);
            const uint32_t src1r = get_read_ptr(cb_out1_r);
            const uint32_t src1i = get_read_ptr(cb_out1_i);

            // Reserve destination slots in CB 0-3
            cb_reserve_back(cb_even_r, num_tiles);
            cb_reserve_back(cb_even_i, num_tiles);
            cb_reserve_back(cb_odd_r,  num_tiles);
            cb_reserve_back(cb_odd_i,  num_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            // ── Shuffle: direct RISC-V pointer writes (L1-to-L1) ────
            //
            // noc_async_write is NOT used here because:
            //  a) The NOC does not guarantee ordering between two separate
            //     write bursts to different destinations on the same core.
            //  b) Both the new_even and new_odd loops issue NOC writes
            //     concurrently; without a barrier between them the writes
            //     can land out of order and corrupt both destination CBs.
            //
            // Direct RISC-V dereference is synchronous, always ordered,
            // requires no barrier, and is faster for small data (< 1KB).

            // Helper: read one float from L1 address
            auto rd = [](uint32_t addr) -> float {
                float v;
                *reinterpret_cast<uint32_t*>(&v) =
                    *reinterpret_cast<volatile uint32_t*>(addr);
                return v;
            };
            // Helper: write one float to L1 address
            auto wr = [](uint32_t addr, float v) {
                *reinterpret_cast<volatile uint32_t*>(addr) =
                    *reinterpret_cast<uint32_t*>(&v);
            };

            // ── new_even ─────────────────────────────────────────
            uint32_t dst = 0;
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                    uint32_t f      = g2 * m2 + j2;
                    uint32_t g_old  = f / m;
                    uint32_t offset = f % m;
                    uint32_t idx;
                    uint32_t srcr, srci;
                    if (offset < half_m) {
                        idx  = g_old * half_m + offset;
                        srcr = src0r; srci = src0i;
                    } else {
                        idx  = g_old * half_m + (offset - half_m);
                        srcr = src1r; srci = src1i;
                    }
                    wr(dst_er + dst * ELEM, rd(srcr + idx * ELEM));
                    wr(dst_ei + dst * ELEM, rd(srci + idx * ELEM));
                    dst++;
                }
            }

            // ── new_odd ──────────────────────────────────────────
            dst = 0;
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                    uint32_t f      = g2 * m2 + half_m2 + j2;
                    uint32_t g_old  = f / m;
                    uint32_t offset = f % m;
                    uint32_t idx;
                    uint32_t srcr, srci;
                    if (offset < half_m) {
                        idx  = g_old * half_m + offset;
                        srcr = src0r; srci = src0i;
                    } else {
                        idx  = g_old * half_m + (offset - half_m);
                        srcr = src1r; srci = src1i;
                    }
                    wr(dst_or + dst * ELEM, rd(srcr + idx * ELEM));
                    wr(dst_oi + dst * ELEM, rd(srci + idx * ELEM));
                    dst++;
                }
            }
            // No barrier needed — direct RISC-V writes are synchronous.

            // Free compute's output slots
            cb_pop_front(cb_out0_r, num_tiles);
            cb_pop_front(cb_out0_i, num_tiles);
            cb_pop_front(cb_out1_r, num_tiles);
            cb_pop_front(cb_out1_i, num_tiles);

            // Signal compute that next stage's inputs are ready
            cb_push_back(cb_even_r, num_tiles);
            cb_push_back(cb_even_i, num_tiles);
            cb_push_back(cb_odd_r,  num_tiles);
            cb_push_back(cb_odd_i,  num_tiles);
        }
    }
}