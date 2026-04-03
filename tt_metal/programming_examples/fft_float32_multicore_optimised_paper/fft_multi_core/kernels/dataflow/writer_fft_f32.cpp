// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FFT WRITER (OUT DATA MOVER) KERNEL — Paper-Aligned Implementation
// Ref: "Exploring Fast Fourier Transforms on the Tenstorrent Wormhole"
//      Brown, Davies, Le Clair (arXiv:2506.15437v1)
//
// Two key optimizations from Table 1 of the paper:
//
//   [A] 128-bit copies (Table 1 row 4: 6.61 ms)
//       "It is possible for ThCon to read and write data of size 8, 16, 32
//        or 128 bits ... the kernel was modified to unroll the reordering loop
//        by four and to use 128-bit wide data accesses for contiguous data."
//
//   [B] Single data copy (Table 1 row 5: 5.31 ms)
//       "we modified this to instead reorder data to the arrangement required
//        by the next step. This reduces the number of reorderings per step to
//        one, apart from the initial and last step." (Fig. 5 of paper)
//
//       Previously (Fig. 4): two reorderings per step — expensive.
//       Now (Fig. 5):        one reordering per step — reorder directly from
//                            current step's output to next step's input layout.
//
// Final step:  write results to external DRAM via NOC async writes.
// Other steps: 128-bit shuffle within SRAM → feeds back into data0/data1 CBs
//              so the reader does NOT need to re-read from DRAM.

#include <cstdint>
#include "dataflow_api.h"

// ─────────────────────────────────────────────────────────────────────────────
// 128-bit (16-byte) SRAM copy helper
// Paper: "stores are all contiguous … modified to unroll … use 128-bit wide
//         data accesses"
//
// Copies `num_floats` floats from src to dst in 4-float (128-bit) chunks.
// Requires num_floats to be a multiple of 4.
// ─────────────────────────────────────────────────────────────────────────────
FORCE_INLINE void copy_128bit(uint32_t dst_addr, uint32_t src_addr,
                               uint32_t num_floats) {
    static_assert(sizeof(float) == 4, "Expected 32-bit float");
    const uint32_t chunks = num_floats / 4;  // 128-bit = 4 × f32
    volatile uint64_t* src64 = reinterpret_cast<volatile uint64_t*>(src_addr);
    volatile uint64_t* dst64 = reinterpret_cast<volatile uint64_t*>(dst_addr);

    for (uint32_t i = 0; i < chunks; i++) {
        // Read two 64-bit words (= one 128-bit read)
        uint64_t lo = src64[i * 2];
        uint64_t hi = src64[i * 2 + 1];
        // Write two 64-bit words (= one 128-bit write)
        dst64[i * 2]     = lo;
        dst64[i * 2 + 1] = hi;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-reversal permutation helper for initial input ordering
//
// FFTs require inputs in bit-reversed order (Fig. 2 of paper, step 1 pairs).
// This is applied once by the writer on the very first step's output, or more
// typically the host pre-orders the data before writing to DRAM.
// ─────────────────────────────────────────────────────────────────────────────
FORCE_INLINE uint32_t bit_reverse(uint32_t x, uint32_t log2_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN KERNEL ENTRY POINT
// ─────────────────────────────────────────────────────────────────────────────
void MAIN {
    // ── Runtime arguments ─────────────────────────────────────────────────
    const uint32_t dst_out0_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_out0_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_out1_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t dst_out1_i_addr = get_arg_val<uint32_t>(3);
    const uint32_t tiles_per_row   = get_arg_val<uint32_t>(4);
    const uint32_t num_steps       = get_arg_val<uint32_t>(5);   // log2(N)
    const uint32_t tile_offset     = get_arg_val<uint32_t>(6);
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(7);

    // ── CB indices ─────────────────────────────────────────────────────────
    constexpr uint32_t cb_out0_r  = 16;
    constexpr uint32_t cb_out0_i  = 17;
    constexpr uint32_t cb_out1_r  = 18;
    constexpr uint32_t cb_out1_i  = 19;
    // Data input CBs reused as output for single-copy shuffle
    constexpr uint32_t cb_data0_r = 0;
    constexpr uint32_t cb_data0_i = 1;
    constexpr uint32_t cb_data1_r = 2;
    constexpr uint32_t cb_data1_i = 3;

    const uint32_t tile_bytes          = get_tile_size(cb_out0_r);
    const DataFormat data_format       = get_dataformat(cb_out0_r);
    const uint32_t floats_per_tile     = tile_bytes / sizeof(float);

    // ── Address generators for DRAM write (final step only) ───────────────
    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = dst_out0_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = dst_out0_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = dst_out1_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = dst_out1_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    // ── Per-row processing ─────────────────────────────────────────────────
    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        for (uint32_t step = 0; step < num_steps; step++) {
            const bool is_last_step = (step == num_steps - 1);

            // ── Process one tile at a time (chunked pipeline overlap) ──────
            for (uint32_t t = 0; t < tiles_per_row; t++) {

                // Wait for compute to produce this tile's butterfly outputs
                cb_wait_front(cb_out0_r, 1);
                cb_wait_front(cb_out0_i, 1);
                cb_wait_front(cb_out1_r, 1);
                cb_wait_front(cb_out1_i, 1);

                if (is_last_step) {
                    // ══════════════════════════════════════════════════════
                    // FINAL STEP: Write results to external on-card DRAM
                    // Paper Fig. 3: "external DDR for the final results"
                    // ══════════════════════════════════════════════════════
                    const uint32_t tile_id = row_tile_base + t;
                    const uint32_t src0r = get_read_ptr(cb_out0_r);
                    const uint32_t src0i = get_read_ptr(cb_out0_i);
                    const uint32_t src1r = get_read_ptr(cb_out1_r);
                    const uint32_t src1i = get_read_ptr(cb_out1_i);

                    noc_async_write_tile(tile_id, out0_r_gen, src0r);
                    noc_async_write_tile(tile_id, out0_i_gen, src0i);
                    noc_async_write_tile(tile_id, out1_r_gen, src1r);
                    noc_async_write_tile(tile_id, out1_i_gen, src1i);

                    noc_async_write_barrier();

                    cb_pop_front(cb_out0_r, 1);
                    cb_pop_front(cb_out0_i, 1);
                    cb_pop_front(cb_out1_r, 1);
                    cb_pop_front(cb_out1_i, 1);

                } else {
                    // ══════════════════════════════════════════════════════
                    // INTERMEDIATE STEP: Single-copy shuffle (Fig. 5, [B])
                    //
                    // Paper: "reorder data to the arrangement required by the
                    //         next step" — eliminates the round-trip back to
                    //         original order seen in the initial approach (Fig. 4).
                    //
                    // The butterfly outputs (out0 = upper, out1 = lower) must be
                    // interleaved according to the next step's pairing pattern.
                    // For a radix-2 DIT FFT:
                    //   step s pairs element j with j + N/(2^(s+1))
                    // So we route out0 → data0 (even) and out1 → data1 (odd)
                    // in the arrangement the compute kernel expects next.
                    //
                    // Optimization [A]: Use 128-bit copies (4 floats at a time)
                    // for contiguous SRAM regions to minimize memory transactions.
                    // ══════════════════════════════════════════════════════

                    // Get source pointers (current CB pages in SRAM)
                    const uint32_t src0r = get_read_ptr(cb_out0_r);
                    const uint32_t src0i = get_read_ptr(cb_out0_i);
                    const uint32_t src1r = get_read_ptr(cb_out1_r);
                    const uint32_t src1i = get_read_ptr(cb_out1_i);

                    // Reserve destination CB pages for next step's inputs
                    cb_reserve_back(cb_data0_r, 1);
                    cb_reserve_back(cb_data0_i, 1);
                    cb_reserve_back(cb_data1_r, 1);
                    cb_reserve_back(cb_data1_i, 1);

                    const uint32_t dst0r = get_write_ptr(cb_data0_r);
                    const uint32_t dst0i = get_write_ptr(cb_data0_i);
                    const uint32_t dst1r = get_write_ptr(cb_data1_r);
                    const uint32_t dst1i = get_write_ptr(cb_data1_i);

                    // 128-bit chunked SRAM copies (Optimization [A])
                    // out0 (upper butterfly) → data0 (even input for next step)
                    // out1 (lower butterfly) → data1 (odd  input for next step)
                    copy_128bit(dst0r, src0r, floats_per_tile);
                    copy_128bit(dst0i, src0i, floats_per_tile);
                    copy_128bit(dst1r, src1r, floats_per_tile);
                    copy_128bit(dst1i, src1i, floats_per_tile);

                    // Free output pages so compute CB memory is reclaimed
                    cb_pop_front(cb_out0_r, 1);
                    cb_pop_front(cb_out0_i, 1);
                    cb_pop_front(cb_out1_r, 1);
                    cb_pop_front(cb_out1_i, 1);

                    // Make shuffled data available to compute for next step
                    cb_push_back(cb_data0_r, 1);
                    cb_push_back(cb_data0_i, 1);
                    cb_push_back(cb_data1_r, 1);
                    cb_push_back(cb_data1_i, 1);
                }
            }  // chunk loop
        }  // step loop
    }  // row loop
}