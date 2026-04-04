// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dram_output_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t dram_output_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t n                  = get_arg_val<uint32_t>(2);
    const uint32_t num_steps          = get_arg_val<uint32_t>(3);
    const uint32_t num_chunks         = get_arg_val<uint32_t>(4);
    const uint32_t chunk_size         = get_arg_val<uint32_t>(5);
    const uint32_t sram_buf_r_addr    = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    // NEW: writer -> reader step barrier token
    constexpr uint32_t cb_step_sync = tt::CBIndex::c_24;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    const uint32_t sram_buf_bytes  = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + sram_buf_bytes;

    const InterleavedAddrGenFast<true> dram_r_gen = {
        .bank_base_address = dram_output_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};
    const InterleavedAddrGenFast<true> dram_i_gen = {
        .bank_base_address = dram_output_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m       = 1u << step;
        const uint32_t m            = half_m << 1u;
        const bool is_last_step     = (step + 1u == num_steps);

        volatile float* sram_r =
            reinterpret_cast<volatile float*>(sram_buf_r_addr);
        volatile float* sram_i =
            reinterpret_cast<volatile float*>(sram_buf_i_addr);

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            cb_wait_front(cb_out0_r, 1);
            cb_wait_front(cb_out0_i, 1);
            cb_wait_front(cb_out1_r, 1);
            cb_wait_front(cb_out1_i, 1);

            const volatile float* out0_r =
                reinterpret_cast<const volatile float*>(get_read_ptr(cb_out0_r));
            const volatile float* out0_i =
                reinterpret_cast<const volatile float*>(get_read_ptr(cb_out0_i));
            const volatile float* out1_r =
                reinterpret_cast<const volatile float*>(get_read_ptr(cb_out1_r));
            const volatile float* out1_i =
                reinterpret_cast<const volatile float*>(get_read_ptr(cb_out1_i));

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;
                const uint32_t b        = a + half_m;

                sram_r[a] = out0_r[p];
                sram_i[a] = out0_i[p];
                sram_r[b] = out1_r[p];
                sram_i[b] = out1_i[p];
            }

            if (is_last_step && (chunk + 1u == num_chunks)) {
                uint32_t sram_r_read = sram_buf_r_addr;
                uint32_t sram_i_read = sram_buf_i_addr;

                noc_async_write_tile(0u, dram_r_gen, sram_r_read);
                noc_async_write_tile(0u, dram_i_gen, sram_i_read);
                noc_async_write_barrier();
            }

            cb_pop_front(cb_out0_r, 1);
            cb_pop_front(cb_out0_i, 1);
            cb_pop_front(cb_out1_r, 1);
            cb_pop_front(cb_out1_i, 1);

            // NEW: after the last chunk of this step, signal reader
            if (chunk + 1u == num_chunks) {
                cb_reserve_back(cb_step_sync, 1);
                cb_push_back(cb_step_sync, 1);
            }
        }
    }
}