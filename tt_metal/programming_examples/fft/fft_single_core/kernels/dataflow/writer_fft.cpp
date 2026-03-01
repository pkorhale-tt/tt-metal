// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst0_r_addr  = get_arg_val<uint32_t>(0); // LHS Final output Real
    uint32_t dst0_i_addr  = get_arg_val<uint32_t>(1); // LHS Final output Imaginary
    uint32_t dst1_r_addr  = get_arg_val<uint32_t>(2); // RHS Final output Real
    uint32_t dst1_i_addr  = get_arg_val<uint32_t>(3); // RHS Final output Imaginary
    uint32_t num_tiles    = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_id_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_id_out1_i = tt::CBIndex::c_19;

    uint32_t tile_bytes = get_tile_size(cb_id_out0_r);

    const InterleavedAddrGenFast<true> d0_r = { .bank_base_address = dst0_r_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> d0_i = { .bank_base_address = dst0_i_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> d1_r = { .bank_base_address = dst1_r_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };
    const InterleavedAddrGenFast<true> d1_i = { .bank_base_address = dst1_i_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b };

    for (uint32_t i = 0; i < num_tiles; i++) {
        // output 0 (LHS)
        cb_wait_front(cb_id_out0_r, 1);
        cb_wait_front(cb_id_out0_i, 1);
        uint32_t l1_read_addr_out0_r = get_read_ptr(cb_id_out0_r);
        uint32_t l1_read_addr_out0_i = get_read_ptr(cb_id_out0_i);
        noc_async_write_tile(i, d0_r, l1_read_addr_out0_r);
        noc_async_write_tile(i, d0_i, l1_read_addr_out0_i);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0_r, 1);
        cb_pop_front(cb_id_out0_i, 1);

        // output 1 (RHS)
        cb_wait_front(cb_id_out1_r, 1);
        cb_wait_front(cb_id_out1_i, 1);
        uint32_t l1_read_addr_out1_r = get_read_ptr(cb_id_out1_r);
        uint32_t l1_read_addr_out1_i = get_read_ptr(cb_id_out1_i);
        noc_async_write_tile(i, d1_r, l1_read_addr_out1_r);
        noc_async_write_tile(i, d1_i, l1_read_addr_out1_i);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out1_r, 1);
        cb_pop_front(cb_id_out1_i, 1);
    }
}
