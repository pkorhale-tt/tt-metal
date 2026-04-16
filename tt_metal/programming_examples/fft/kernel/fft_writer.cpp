// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_writer.cpp — BRISC1 / writer
//
// Waits for CB_SYNC from the reader (which indicates the final FFT state is
// sitting in CB_STATE_{R,I}), then writes those two tiles to DRAM.
//
// The writer sits on NOC 1 so the DRAM output write runs concurrently with
// any remaining NOC 0 traffic from the reader; in the single-core FFT there
// is no such traffic, but we keep the conventional reader/writer split so the
// design extends cleanly to the multi-core case where NOC butterflies will
// live on this kernel.

#include <cstdint>
#include "dataflow_api.h"
#include "fft_common.h"

void kernel_main() {
    const uint32_t out_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_i_addr = get_arg_val<uint32_t>(1);

    const DataFormat df = get_dataformat(CB_STATE_R);
    const uint32_t   ts = get_tile_size(CB_STATE_R);

    InterleavedAddrGenFast<true> out_r_gen = {
        .bank_base_address = out_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> out_i_gen = {
        .bank_base_address = out_i_addr, .page_size = ts, .data_format = df};

    cb_wait_front(CB_SYNC, 1);
    cb_wait_front(CB_STATE_R, 1);
    cb_wait_front(CB_STATE_I, 1);

    noc_async_write_tile(0, out_r_gen, get_read_ptr(CB_STATE_R));
    noc_async_write_tile(0, out_i_gen, get_read_ptr(CB_STATE_I));
    noc_async_write_barrier();
}
