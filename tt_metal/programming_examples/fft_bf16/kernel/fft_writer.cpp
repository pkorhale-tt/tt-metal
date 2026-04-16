// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_writer.cpp (bf16) — BRISC1 / writer
//
// Identical to the fp32 version: the DataFormat / tile size are picked up
// from the CB metadata via get_dataformat / get_tile_size, so this file
// works unchanged for either precision.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

void kernel_main() {
    const uint32_t out_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_core    = get_arg_val<uint32_t>(2);

    const DataFormat df = get_dataformat(CB_STATE_R);
    const uint32_t   ts = get_tile_size(CB_STATE_R);

    InterleavedAddrGenFast<true> out_r_gen = {
        .bank_base_address = out_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> out_i_gen = {
        .bank_base_address = out_i_addr, .page_size = ts, .data_format = df};

    cb_wait_front(CB_SYNC, 1);
    cb_wait_front(CB_STATE_R, 1);
    cb_wait_front(CB_STATE_I, 1);

    noc_async_write_tile(my_core, out_r_gen, get_read_ptr(CB_STATE_R));
    noc_async_write_tile(my_core, out_i_gen, get_read_ptr(CB_STATE_I));
    noc_async_write_barrier();
}
