// SPDX-FileCopyrightText: © 2025
// SPDX-License-Identifier: Apache-2.0
//
// twiddle_init_f32.cpp
//
// Copies precomputed twiddle tables from DRAM into per-core SRAM.
//
// Runtime args:
//   0 : dram_tw_r_addr  - DRAM base address of twiddle real table
//   1 : dram_tw_i_addr  - DRAM base address of twiddle imag table
//   2 : sram_tw_r_addr  - SRAM destination address for twiddle real table
//   3 : sram_tw_i_addr  - SRAM destination address for twiddle imag table
//   4 : bytes           - number of bytes to copy for each table

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dramTwRAddr = get_arg_val<uint32_t>(0);
    const uint32_t dramTwIAddr = get_arg_val<uint32_t>(1);
    const uint32_t sramTwRAddr = get_arg_val<uint32_t>(2);
    const uint32_t sramTwIAddr = get_arg_val<uint32_t>(3);
    const uint32_t bytes       = get_arg_val<uint32_t>(4);

    const uint64_t nocTwRAddr = get_noc_addr(dramTwRAddr);
    const uint64_t nocTwIAddr = get_noc_addr(dramTwIAddr);

    noc_async_read(nocTwRAddr, sramTwRAddr, bytes);
    noc_async_read(nocTwIAddr, sramTwIAddr, bytes);
    noc_async_read_barrier();
}
