// SPDX-FileCopyrightText: © 2026 OpenAI
// SPDX-License-Identifier: Apache-2.0
//
// writer_fft_f32.cpp  –  CORRECTED
//
// Change vs. original: scratch row CBs changed from c_24/c_25 to c_26/c_27.
// The reader kernel owns c_24/c_25 for its own scratch.  If the writer also
// used c_24/c_25, RISCV_0 and RISCV_1 would race on the same CB indices,
// causing non-deterministic deadlocks or data corruption.
//
// Expected kernel args (12):
//   0:  stage0_r_addr
//   1:  stage0_i_addr
//   2:  stage1_r_addr
//   3:  stage1_i_addr
//   4:  output_r_addr
//   5:  output_i_addr
//   6:  row_tiles
//   7:  pair_tiles
//   8:  n_row
//   9:  num_stages
//  10:  row_start
//  11:  rows_this_core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t TILE_ELEMS = 32u * 32u;

inline void zero_u32_buffer(volatile uint32_t* ptr, const uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) ptr[i] = 0u;
}

}  // namespace

void kernel_main() {
    const uint32_t stage0_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t stage0_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t stage1_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t stage1_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t output_r_addr  = get_arg_val<uint32_t>(4);
    const uint32_t output_i_addr  = get_arg_val<uint32_t>(5);
    const uint32_t row_tiles      = get_arg_val<uint32_t>(6);
    const uint32_t pair_tiles     = get_arg_val<uint32_t>(7);
    const uint32_t n_row          = get_arg_val<uint32_t>(8);
    const uint32_t num_stages     = get_arg_val<uint32_t>(9);
    const uint32_t row_start      = get_arg_val<uint32_t>(10);
    const uint32_t rows_this_core = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    // Scratch CBs for row assembly – c_26/c_27 (not c_24/c_25 which the
    // reader uses for loading scratch, to avoid a race between RISCV_0 and 1)
    constexpr uint32_t cb_row_r = tt::CBIndex::c_26;
    constexpr uint32_t cb_row_i = tt::CBIndex::c_27;

    if (rows_this_core == 0u) return;

    const uint32_t tile_bytes       = get_tile_size(cb_out0_r);
    const DataFormat data_format    = get_dataformat(cb_out0_r);
    const uint32_t  pair_count      = n_row >> 1;
    const uint32_t  row_stride_elems = row_tiles * TILE_ELEMS;

    const InterleavedAddrGenFast<true> stage0_r_gen = {
        .bank_base_address = stage0_r_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> stage0_i_gen = {
        .bank_base_address = stage0_i_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> stage1_r_gen = {
        .bank_base_address = stage1_r_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> stage1_i_gen = {
        .bank_base_address = stage1_i_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> output_r_gen = {
        .bank_base_address = output_r_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true> output_i_gen = {
        .bank_base_address = output_i_addr, .page_size = tile_bytes, .data_format = data_format};

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        const uint32_t half_m        = 1u << stage;
        const uint32_t m             = half_m << 1u;
        const bool     is_last_stage = (stage + 1u == num_stages);
        // Even stages (0, 2, ...) read from stage0 → write to stage1.
        // Odd  stages (1, 3, ...) read from stage1 → write to stage0.
        const bool write_stage1 = ((stage & 1u) == 0u);

        for (uint32_t local_row = 0; local_row < rows_this_core; ++local_row) {
            const uint32_t global_row    = row_start + local_row;
            const uint32_t row_tile_base = global_row * row_tiles;

            // Wait for this row's butterfly outputs from compute
            cb_wait_front(cb_out0_r, pair_tiles);
            cb_wait_front(cb_out0_i, pair_tiles);
            cb_wait_front(cb_out1_r, pair_tiles);
            cb_wait_front(cb_out1_i, pair_tiles);

            const volatile uint32_t* out0_r =
                reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_out0_r));
            const volatile uint32_t* out0_i =
                reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_out0_i));
            const volatile uint32_t* out1_r =
                reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_out1_r));
            const volatile uint32_t* out1_i =
                reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_out1_i));

            // Assemble the full output row in scratch CB
            cb_reserve_back(cb_row_r, row_tiles);
            cb_reserve_back(cb_row_i, row_tiles);
            volatile uint32_t* row_r =
                reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_row_r));
            volatile uint32_t* row_i =
                reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_row_i));

            zero_u32_buffer(row_r, row_stride_elems);
            zero_u32_buffer(row_i, row_stride_elems);

            // Scatter butterfly outputs back to their row positions
            for (uint32_t p = 0; p < pair_count; ++p) {
                const uint32_t group = p / half_m;
                const uint32_t j     = p % half_m;
                const uint32_t a     = group * m + j;        // even position
                const uint32_t b     = a + half_m;           // odd position

                row_r[a] = out0_r[p];
                row_i[a] = out0_i[p];
                row_r[b] = out1_r[p];
                row_i[b] = out1_i[p];
            }

            cb_push_back(cb_row_r, row_tiles);
            cb_push_back(cb_row_i, row_tiles);

            const uint32_t row_r_read = get_read_ptr(cb_row_r);
            const uint32_t row_i_read = get_read_ptr(cb_row_i);

            // Write the assembled row to the appropriate DRAM destination
            for (uint32_t t = 0; t < row_tiles; ++t) {
                const uint32_t tile_id     = row_tile_base + t;
                const uint32_t tile_r_addr = row_r_read + t * tile_bytes;
                const uint32_t tile_i_addr = row_i_read + t * tile_bytes;

                if (is_last_stage) {
                    noc_async_write_tile(tile_id, output_r_gen, tile_r_addr);
                    noc_async_write_tile(tile_id, output_i_gen, tile_i_addr);
                } else if (write_stage1) {
                    noc_async_write_tile(tile_id, stage1_r_gen, tile_r_addr);
                    noc_async_write_tile(tile_id, stage1_i_gen, tile_i_addr);
                } else {
                    noc_async_write_tile(tile_id, stage0_r_gen, tile_r_addr);
                    noc_async_write_tile(tile_id, stage0_i_gen, tile_i_addr);
                }
            }

            noc_async_write_barrier();

            // Release scratch and consumed output CBs
            cb_pop_front(cb_row_r,  row_tiles);
            cb_pop_front(cb_row_i,  row_tiles);
            cb_pop_front(cb_out0_r, pair_tiles);
            cb_pop_front(cb_out0_i, pair_tiles);
            cb_pop_front(cb_out1_r, pair_tiles);
            cb_pop_front(cb_out1_i, pair_tiles);
        }
    }
}