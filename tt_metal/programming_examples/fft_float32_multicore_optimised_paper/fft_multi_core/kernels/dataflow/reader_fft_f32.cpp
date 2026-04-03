// SPDX-FileCopyrightText: © 2026 OpenAI
// SPDX-License-Identifier: Apache-2.0
//
// Paper-style FFT reader kernel for Wormhole / TT-Metalium.
// Safe implementation of the paper's basic design:
//   - load one row from a stage source buffer,
//   - reorder into stage-specific even/odd butterfly pairs,
//   - bulk-load precomputed twiddle tiles.

#include <cstdint>
#include "dataflow_api.h"

namespace {

constexpr uint32_t TILE_ELEMS = 32u * 32u;

inline uint32_t ceil_div_u32(const uint32_t a, const uint32_t b) {
    return (a + b - 1u) / b;
}

inline void zero_u32_buffer(volatile uint32_t* ptr, const uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        ptr[i] = 0u;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t stage0_r_addr    = get_arg_val<uint32_t>(0);
    const uint32_t stage0_i_addr    = get_arg_val<uint32_t>(1);
    const uint32_t stage1_r_addr    = get_arg_val<uint32_t>(2);
    const uint32_t stage1_i_addr    = get_arg_val<uint32_t>(3);
    const uint32_t twiddle_r_addr   = get_arg_val<uint32_t>(4);
    const uint32_t twiddle_i_addr   = get_arg_val<uint32_t>(5);
    const uint32_t row_tiles        = get_arg_val<uint32_t>(6);
    const uint32_t pair_tiles       = get_arg_val<uint32_t>(7);
    const uint32_t n_row            = get_arg_val<uint32_t>(8);
    const uint32_t num_stages       = get_arg_val<uint32_t>(9);
    const uint32_t total_rows       = get_arg_val<uint32_t>(10);
    const uint32_t row_start        = get_arg_val<uint32_t>(11);
    const uint32_t rows_this_core   = get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;
    constexpr uint32_t cb_odd_r  = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i  = tt::CBIndex::c_3;
    constexpr uint32_t cb_tw_r   = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i   = tt::CBIndex::c_5;

    constexpr uint32_t cb_row_r  = tt::CBIndex::c_24;
    constexpr uint32_t cb_row_i  = tt::CBIndex::c_25;

    if (rows_this_core == 0u) {
        return;
    }

    const uint32_t tile_bytes = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    const uint32_t pair_count = n_row >> 1;
    const uint32_t row_stride_elems = row_tiles * TILE_ELEMS;
    const uint32_t pair_stride_elems = pair_tiles * TILE_ELEMS;

    const InterleavedAddrGenFast<true> stage0_r_gen = {
        .bank_base_address = stage0_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format};
    const InterleavedAddrGenFast<true> stage0_i_gen = {
        .bank_base_address = stage0_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format};
    const InterleavedAddrGenFast<true> stage1_r_gen = {
        .bank_base_address = stage1_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format};
    const InterleavedAddrGenFast<true> stage1_i_gen = {
        .bank_base_address = stage1_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format};
    const InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = twiddle_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format};
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = twiddle_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format};

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        const uint32_t half_m = 1u << stage;
        const uint32_t m = half_m << 1u;

        for (uint32_t local_row = 0; local_row < rows_this_core; ++local_row) {
            const uint32_t global_row = row_start + local_row;
            const uint32_t row_tile_base = global_row * row_tiles;
            const uint32_t tw_tile_base = (stage * total_rows + global_row) * pair_tiles;
            const bool use_stage0 = ((stage & 1u) == 0u);

            // Load source row into scratch CBs.
            cb_reserve_back(cb_row_r, row_tiles);
            cb_reserve_back(cb_row_i, row_tiles);
            uint32_t row_r_write = get_write_ptr(cb_row_r);
            uint32_t row_i_write = get_write_ptr(cb_row_i);

            for (uint32_t t = 0; t < row_tiles; ++t) {
                const uint32_t tile_id = row_tile_base + t;
                if (use_stage0) {
                    noc_async_read_tile(tile_id, stage0_r_gen, row_r_write + t * tile_bytes);
                    noc_async_read_tile(tile_id, stage0_i_gen, row_i_write + t * tile_bytes);
                } else {
                    noc_async_read_tile(tile_id, stage1_r_gen, row_r_write + t * tile_bytes);
                    noc_async_read_tile(tile_id, stage1_i_gen, row_i_write + t * tile_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_row_r, row_tiles);
            cb_push_back(cb_row_i, row_tiles);

            const volatile uint32_t* row_r = reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_row_r));
            const volatile uint32_t* row_i = reinterpret_cast<volatile uint32_t*>(get_read_ptr(cb_row_i));

            // Build stage-specific even / odd pair arrays.
            cb_reserve_back(cb_even_r, pair_tiles);
            cb_reserve_back(cb_even_i, pair_tiles);
            cb_reserve_back(cb_odd_r, pair_tiles);
            cb_reserve_back(cb_odd_i, pair_tiles);

            volatile uint32_t* even_r = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_even_r));
            volatile uint32_t* even_i = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_even_i));
            volatile uint32_t* odd_r  = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_odd_r));
            volatile uint32_t* odd_i  = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_odd_i));

            zero_u32_buffer(even_r, pair_stride_elems);
            zero_u32_buffer(even_i, pair_stride_elems);
            zero_u32_buffer(odd_r,  pair_stride_elems);
            zero_u32_buffer(odd_i,  pair_stride_elems);

            for (uint32_t p = 0; p < pair_count; ++p) {
                const uint32_t group = p / half_m;
                const uint32_t j = p % half_m;
                const uint32_t a = group * m + j;
                const uint32_t b = a + half_m;

                even_r[p] = row_r[a];
                even_i[p] = row_i[a];
                odd_r[p]  = row_r[b];
                odd_i[p]  = row_i[b];
            }

            cb_push_back(cb_even_r, pair_tiles);
            cb_push_back(cb_even_i, pair_tiles);
            cb_push_back(cb_odd_r,  pair_tiles);
            cb_push_back(cb_odd_i,  pair_tiles);

            // Load precomputed twiddle tiles aligned to pair order.
            cb_reserve_back(cb_tw_r, pair_tiles);
            cb_reserve_back(cb_tw_i, pair_tiles);
            uint32_t tw_r_write = get_write_ptr(cb_tw_r);
            uint32_t tw_i_write = get_write_ptr(cb_tw_i);

            for (uint32_t t = 0; t < pair_tiles; ++t) {
                const uint32_t tw_tile_id = tw_tile_base + t;
                noc_async_read_tile(tw_tile_id, tw_r_gen, tw_r_write + t * tile_bytes);
                noc_async_read_tile(tw_tile_id, tw_i_gen, tw_i_write + t * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_tw_r, pair_tiles);
            cb_push_back(cb_tw_i, pair_tiles);

            cb_pop_front(cb_row_r, row_tiles);
            cb_pop_front(cb_row_i, row_tiles);
        }
    }
}
