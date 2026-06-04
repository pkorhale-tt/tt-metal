// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// complex_mul_reader.cpp — BRISC0 / reader for the elementwise complex
// multiply used by Bluestein chirp pre/post mul.
//
// Streams (a_R, a_I, b_R, b_I) tile pairs to the compute engine.
// `a` is consumed sequentially (tile_idx = first_tile + k);
// `b` broadcasts: tile_idx = (first_tile + k) % num_b_tiles. This lets the
// same length-M chirp tile sequence be reused across every input row.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "complex_mul_common.h"

void kernel_main() {
    const uint32_t a_r_addr      = get_arg_val<uint32_t>(0);
    const uint32_t a_i_addr      = get_arg_val<uint32_t>(1);
    const uint32_t b_r_addr      = get_arg_val<uint32_t>(2);
    const uint32_t b_i_addr      = get_arg_val<uint32_t>(3);
    const uint32_t first_tile    = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles     = get_arg_val<uint32_t>(5);
    const uint32_t num_b_tiles   = get_arg_val<uint32_t>(6);

    const DataFormat df = get_dataformat(CB_A_R);
    const uint32_t   ts = get_tile_size(CB_A_R);

    InterleavedAddrGenFast<true> a_r_gen = {
        .bank_base_address = a_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> a_i_gen = {
        .bank_base_address = a_i_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> b_r_gen = {
        .bank_base_address = b_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> b_i_gen = {
        .bank_base_address = b_i_addr, .page_size = ts, .data_format = df};

    for (uint32_t k = 0; k < num_tiles; ++k) {
        const uint32_t a_idx = first_tile + k;
        const uint32_t b_idx = (num_b_tiles > 0u) ? (a_idx % num_b_tiles) : 0u;

        cb_reserve_back(CB_A_R, 1);
        cb_reserve_back(CB_A_I, 1);
        cb_reserve_back(CB_B_R, 1);
        cb_reserve_back(CB_B_I, 1);

        noc_async_read_tile(a_idx, a_r_gen, get_write_ptr(CB_A_R));
        noc_async_read_tile(a_idx, a_i_gen, get_write_ptr(CB_A_I));
        noc_async_read_tile(b_idx, b_r_gen, get_write_ptr(CB_B_R));
        noc_async_read_tile(b_idx, b_i_gen, get_write_ptr(CB_B_I));
        noc_async_read_barrier();

        cb_push_back(CB_A_R, 1);
        cb_push_back(CB_A_I, 1);
        cb_push_back(CB_B_R, 1);
        cb_push_back(CB_B_I, 1);
    }
}
