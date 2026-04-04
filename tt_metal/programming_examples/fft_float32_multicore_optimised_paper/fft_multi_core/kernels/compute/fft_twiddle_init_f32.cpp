// SPDX-FileCopyrightText: © 2025
// SPDX-License-Identifier: Apache-2.0
//
// twiddle_init_f32.cpp  –  copies twiddle factors from DRAM into core SRAM
//
// Args:
//   0  dram_tw_r_addr  – DRAM source for twiddle real
//   1  dram_tw_i_addr  – DRAM source for twiddle imag
//   2  sram_tw_r_addr  – SRAM destination for twiddle real
//   3  sram_tw_i_addr  – SRAM destination for twiddle imag
//   4  bytes           – number of bytes to copy for each component

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t dram_tw_r = get_arg_val<uint32_t>(0);
    const uint32_t dram_tw_i = get_arg_val<uint32_t>(1);
    const uint32_t sram_tw_r = get_arg_val<uint32_t>(2);
    const uint32_t sram_tw_i = get_arg_val<uint32_t>(3);
    const uint32_t bytes     = get_arg_val<uint32_t>(4);

    uint64_t noc_addr_r = get_noc_addr(dram_tw_r);
    noc_async_read(noc_addr_r, sram_tw_r, bytes);

    uint64_t noc_addr_i = get_noc_addr(dram_tw_i);
    noc_async_read(noc_addr_i, sram_tw_i, bytes);

    noc_async_read_barrier();
}
