// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_r_addr  = get_arg_val<uint32_t>(0); // LHS Real
    uint32_t src0_i_addr  = get_arg_val<uint32_t>(1); // LHS Imaginary
    uint32_t src1_r_addr  = get_arg_val<uint32_t>(2); // RHS Real
    uint32_t src1_i_addr  = get_arg_val<uint32_t>(3); // RHS Imaginary
    uint32_t twiddles_r_addr = get_arg_val<uint32_t>(4); // Twiddle Real
    uint32_t twiddles_i_addr = get_arg_val<uint32_t>(5); // Twiddle Imaginary
    uint32_t num_tiles    = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in0_i = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in1_r = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_in1_i = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_twiddle_i = tt::CBIndex::c_5;

    // Single tile sizes (assuming float16_b which is 2 bytes per element, 32x32 tile = 2048 bytes)
    uint32_t tile_bytes = get_tile_size(cb_id_in0_r);
    uint32_t twiddle_bytes = get_tile_size(cb_id_twiddle_r);

    const InterleavedAddrGenFast<true> s0_r = { .bank_base_address = src0_r_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> s0_i = { .bank_base_address = src0_i_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> s1_r = { .bank_base_address = src1_r_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> s1_i = { .bank_base_address = src1_i_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> tw_r = { .bank_base_address = twiddles_r_addr, .page_size = twiddle_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> tw_i = { .bank_base_address = twiddles_i_addr, .page_size = twiddle_bytes, .data_format = DataFormat::Float16_b };

    for (uint32_t i = 0; i < num_tiles; i++) {
        // LHS
        cb_reserve_back(cb_id_in0_r, 1);
        cb_reserve_back(cb_id_in0_i, 1);
        uint32_t l1_write_addr_in0_r = get_write_ptr(cb_id_in0_r);
        uint32_t l1_write_addr_in0_i = get_write_ptr(cb_id_in0_i);
        noc_async_read_tile(i, s0_r, l1_write_addr_in0_r);
        noc_async_read_tile(i, s0_i, l1_write_addr_in0_i);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0_r, 1);
        cb_push_back(cb_id_in0_i, 1);

        // RHS
        cb_reserve_back(cb_id_in1_r, 1);
        cb_reserve_back(cb_id_in1_i, 1);
        uint32_t l1_write_addr_in1_r = get_write_ptr(cb_id_in1_r);
        uint32_t l1_write_addr_in1_i = get_write_ptr(cb_id_in1_i);
        noc_async_read_tile(i, s1_r, l1_write_addr_in1_r);
        noc_async_read_tile(i, s1_i, l1_write_addr_in1_i);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1_r, 1);
        cb_push_back(cb_id_in1_i, 1);

        // Twiddles
        cb_reserve_back(cb_id_twiddle_r, 1);
        cb_reserve_back(cb_id_twiddle_i, 1);
        uint32_t l1_write_addr_tw_r = get_write_ptr(cb_id_twiddle_r);
        uint32_t l1_write_addr_tw_i = get_write_ptr(cb_id_twiddle_i);
        noc_async_read_tile(i, tw_r, l1_write_addr_tw_r);
        noc_async_read_tile(i, tw_i, l1_write_addr_tw_i);
        noc_async_read_barrier();
        cb_push_back(cb_id_twiddle_r, 1);
        cb_push_back(cb_id_twiddle_i, 1);
    }
}
