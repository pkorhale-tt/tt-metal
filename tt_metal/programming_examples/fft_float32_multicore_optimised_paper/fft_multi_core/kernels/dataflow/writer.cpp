// // SPDX-FileCopyrightText: © 2025 (paper faithful port)
// // SPDX-License-Identifier: Apache-2.0

// // ── Correct semaphore protocol ────────────────────────────────────────────────
// //
// // KEY INSIGHT: noc_semaphore_set_remote() is a NOC ATOMIC INCREMENT (AT_INC),
// // NOT a store.  Passing value 0 is a no-op; value N increments by N.
// // So "noc_semaphore_set_remote(addr, 0)" for reset does NOTHING — the
// // previous fix was broken because it tried to reset to 0 via set_remote.
// //
// // noc_semaphore_set(ptr, val) is a plain LOCAL L1 store.
// // noc_semaphore_wait(ptr, target) polls *ptr until it equals target.
// //
// // RISCV_0 (reader) and RISCV_1 (writer) are on the SAME Tensix core and
// // share the same 1.3MB L1 SRAM.  A local store from either core is visible
// // to the other core once it exits the store buffer — which is guaranteed
// // by the time the other core's noc_semaphore_wait() load executes (the
// // wait loop is a polling load with an implicit acquire fence).
// //
// // Therefore for same-Tensix communication: use LOCAL stores (noc_semaphore_set)
// // for BOTH signal and reset.  Do NOT use noc_semaphore_set_remote for reset
// // because AT_INC(0) is a no-op and AT_INC on a stale NOC-cached value can
// // produce values > 1, causing the wait(==1) to spin forever.
// //
// // CORRECT PROTOCOL:
// //   Init (writer only, before step loop):
// //     noc_semaphore_set(rdy_flag, 0)
// //     noc_semaphore_set(ack_flag, 0)
// //
// //   Writer end-of-step N (not last):
// //     noc_semaphore_set(rdy_flag, 1)     // local store, visible to RISCV_0
// //     noc_semaphore_wait(ack_flag, 1)    // poll until reader acks
// //     noc_semaphore_set(ack_flag, 0)     // reset
// //
// //   Reader start-of-step N+1:
// //     noc_semaphore_wait(rdy_flag, 1)    // poll until writer signals
// //     noc_semaphore_set(rdy_flag, 0)     // reset
// //     < push chunks >
// //     noc_semaphore_set(ack_flag, 1)     // ack writer
// // ─────────────────────────────────────────────────────────────────────────────

// #include <cstdint>
// #include "api/dataflow/dataflow_api.h"

// void kernel_main() {
//     const uint32_t dram_output_r_addr = get_arg_val<uint32_t>(0);
//     const uint32_t dram_output_i_addr = get_arg_val<uint32_t>(1);
//     const uint32_t n                  = get_arg_val<uint32_t>(2);
//     const uint32_t num_steps          = get_arg_val<uint32_t>(3);
//     const uint32_t num_chunks         = get_arg_val<uint32_t>(4);
//     const uint32_t chunk_size         = get_arg_val<uint32_t>(5);
//     const uint32_t sram_buf_r_addr    = get_arg_val<uint32_t>(6);
//     const uint32_t sync_flag_addr     = get_arg_val<uint32_t>(7);

//     constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
//     constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
//     constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
//     constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

//     const uint32_t row_bytes       = n * sizeof(float);
//     const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

//     // rdy_flag @ sync_flag_addr+0 : writer signals → reader polls
//     volatile tt_l1_ptr uint32_t* rdy_flag =
//         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);

//     // ack_flag @ sync_flag_addr+4 : reader signals → writer polls
//     const uint32_t ack_flag_addr = sync_flag_addr + sizeof(uint32_t);
//     volatile tt_l1_ptr uint32_t* ack_flag =
//         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_flag_addr);

//     volatile tt_l1_ptr float* sram_r =
//         reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
//     volatile tt_l1_ptr float* sram_i =
//         reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

//     // Writer owns initialisation. Reader does NOT initialise — single owner
//     // avoids the race where reader's init zeros a flag the writer already set.
//     noc_semaphore_set(rdy_flag, 0);
//     noc_semaphore_set(ack_flag, 0);

//     for (uint32_t step = 0; step < num_steps; ++step) {
//         const uint32_t half_m   = 1u << step;
//         const uint32_t m        = half_m << 1u;
//         const bool is_last_step = (step + 1u == num_steps);

//         for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
//             const uint32_t pair_base = chunk * chunk_size;

//             // Push order matches compute.cpp: out0_r, out0_i, out1_r, out1_i.
//             cb_wait_front(cb_out0_r, 1);
//             cb_wait_front(cb_out0_i, 1);
//             cb_wait_front(cb_out1_r, 1);
//             cb_wait_front(cb_out1_i, 1);

//             const volatile tt_l1_ptr float* out0_r =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out0_r));
//             const volatile tt_l1_ptr float* out0_i =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out0_i));
//             const volatile tt_l1_ptr float* out1_r =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out1_r));
//             const volatile tt_l1_ptr float* out1_i =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out1_i));

//             // Scatter butterfly results back to natural order in SRAM.
//             for (uint32_t p = 0; p < chunk_size; ++p) {
//                 const uint32_t global_p = pair_base + p;
//                 const uint32_t group    = global_p / half_m;
//                 const uint32_t j        = global_p % half_m;
//                 const uint32_t a        = group * m + j;
//                 const uint32_t b        = a + half_m;
//                 sram_r[a] = out0_r[p];
//                 sram_i[a] = out0_i[p];
//                 sram_r[b] = out1_r[p];
//                 sram_i[b] = out1_i[p];
//             }

//             cb_pop_front(cb_out0_r, 1);
//             cb_pop_front(cb_out0_i, 1);
//             cb_pop_front(cb_out1_r, 1);
//             cb_pop_front(cb_out1_i, 1);
//         }

//         if (is_last_step) {
//             // DMA final SRAM results to DRAM.
//             const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
//             const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
//             noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
//             noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
//             noc_async_write_barrier();
//         } else {
//             // All SRAM scatter writes done. Signal reader via local store —
//             // same Tensix L1, no NOC needed, immediately visible to RISCV_0.
//             noc_semaphore_set(rdy_flag, 1);

//             // Wait for reader to ack after it has started pushing CBs.
//             noc_semaphore_wait(ack_flag, 1);
//             noc_semaphore_set(ack_flag, 0);   // reset for next step
//         }
//     }
// }

// writer_fft_f32_mc.cpp  — MULTICORE writer  [BUG-FIXED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fixes applied:
//
//   BUG 4 (G2==0 fallback — split loop):
//     Original split the copy into two loops over local_half/2 each.
//     Replaced with a single clean loop over local_half.
//
//   BUG 6 (uint32_t underflow in local_idx):
//     Original: uint32_t local_idx = global_src_idx - core_elem_base;
//     If global_src_idx < core_elem_base this wraps and reads garbage.
//     Fix: assert + clamp.
//
// NOTE (Bug 7 — intentional simplification):
//   For row decomposition, host can pass num_cores=1 and log2_cores=0.
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

    auto safe_local_idx = [&](uint32_t global_idx) -> uint32_t {
        ASSERT(global_idx >= core_elem_base);
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

                    {
                        uint32_t f      = core_elem_base + local_base_e + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx = safe_local_idx(global_idx);
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_er + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_ei + dst*ELEM, rd(srci + local_idx*ELEM));
                    }

                    {
                        uint32_t f      = core_elem_base + local_base_o + j2;
                        uint32_t g_old  = f >> log2m;
                        uint32_t offset = f & m_mask;
                        uint32_t global_idx = (offset < half_m)
                            ? g_old * half_m + offset
                            : g_old * half_m + (offset - half_m);
                        uint32_t local_idx = safe_local_idx(global_idx);
                        uint32_t srcr = (offset < half_m) ? src0r : src1r;
                        uint32_t srci = (offset < half_m) ? src0i : src1i;
                        wr(dst_or + dst*ELEM, rd(srcr + local_idx*ELEM));
                        wr(dst_oi + dst*ELEM, rd(srci + local_idx*ELEM));
                    }

                    dst++;
                }
            }

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