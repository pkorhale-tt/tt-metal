// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TILE_ELEMS = 32u * 32u;
}

void kernel_main() {
    const uint32_t dram_input_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t dram_input_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t n                 = get_arg_val<uint32_t>(2);
    const uint32_t num_steps         = get_arg_val<uint32_t>(3);
    const uint32_t num_chunks        = get_arg_val<uint32_t>(4);
    const uint32_t chunk_size        = get_arg_val<uint32_t>(5);
    const uint32_t sram_buf_r_addr   = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_data0_r   = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i   = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r   = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i   = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;

    // NEW: writer -> reader step barrier token
    constexpr uint32_t cb_step_sync = tt::CBIndex::c_24;

    const uint32_t tile_bytes    = get_tile_size(cb_data0_r);
    const DataFormat data_format = get_dataformat(cb_data0_r);

    const uint32_t sram_buf_bytes  = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + sram_buf_bytes;

    const InterleavedAddrGenFast<true> dram_r_gen = {
        .bank_base_address = dram_input_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};
    const InterleavedAddrGenFast<true> dram_i_gen = {
        .bank_base_address = dram_input_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};

    const uint32_t sram_tw_r_addr = sram_buf_i_addr + sram_buf_bytes;
    const uint32_t sram_tw_i_addr = sram_tw_r_addr + num_steps * (n / 2) * sizeof(float);

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m = 1u << step;
        const uint32_t m      = half_m << 1u;
        const bool is_first_step = (step == 0u);
        const uint32_t tw_step_offset = step * (n / 2u);

        // NEW: wait until writer finished previous step
        if (!is_first_step) {
            cb_wait_front(cb_step_sync, 1);
            cb_pop_front(cb_step_sync, 1);
        }

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            cb_reserve_back(cb_data0_r, 1);
            cb_reserve_back(cb_data0_i, 1);
            cb_reserve_back(cb_data1_r, 1);
            cb_reserve_back(cb_data1_i, 1);

            volatile float* dst0_r = reinterpret_cast<volatile float*>(get_write_ptr(cb_data0_r));
            volatile float* dst0_i = reinterpret_cast<volatile float*>(get_write_ptr(cb_data0_i));
            volatile float* dst1_r = reinterpret_cast<volatile float*>(get_write_ptr(cb_data1_r));
            volatile float* dst1_i = reinterpret_cast<volatile float*>(get_write_ptr(cb_data1_i));

            if (is_first_step) {
                constexpr uint32_t temp_buf_addr = 0x80000;
                volatile float* temp_r = reinterpret_cast<volatile float*>(temp_buf_addr);
                volatile float* temp_i = reinterpret_cast<volatile float*>(temp_buf_addr + tile_bytes);

                noc_async_read_tile(0u, dram_r_gen, temp_buf_addr);
                noc_async_read_tile(0u, dram_i_gen, temp_buf_addr + tile_bytes);
                noc_async_read_barrier();

                for (uint32_t p = 0; p < chunk_size; ++p) {
                    const uint32_t global_p = pair_base + p;
                    const uint32_t group    = global_p / half_m;
                    const uint32_t j        = global_p % half_m;
                    const uint32_t a        = group * m + j;
                    const uint32_t b        = a + half_m;

                    dst0_r[p] = temp_r[a];
                    dst0_i[p] = temp_i[a];
                    dst1_r[p] = temp_r[b];
                    dst1_i[p] = temp_i[b];
                }
            } else {
                const volatile float* src_r =
                    reinterpret_cast<const volatile float*>(sram_buf_r_addr);
                const volatile float* src_i =
                    reinterpret_cast<const volatile float*>(sram_buf_i_addr);

                for (uint32_t p = 0; p < chunk_size; ++p) {
                    const uint32_t global_p = pair_base + p;
                    const uint32_t group    = global_p / half_m;
                    const uint32_t j        = global_p % half_m;
                    const uint32_t a        = group * m + j;
                    const uint32_t b        = a + half_m;

                    dst0_r[p] = src_r[a];
                    dst0_i[p] = src_i[a];
                    dst1_r[p] = src_r[b];
                    dst1_i[p] = src_i[b];
                }
            }

            cb_push_back(cb_data0_r, 1);
            cb_push_back(cb_data0_i, 1);
            cb_push_back(cb_data1_r, 1);
            cb_push_back(cb_data1_i, 1);

            cb_reserve_back(cb_twiddle_r, 1);
            cb_reserve_back(cb_twiddle_i, 1);

            volatile float* tw_r_dst = reinterpret_cast<volatile float*>(get_write_ptr(cb_twiddle_r));
            volatile float* tw_i_dst = reinterpret_cast<volatile float*>(get_write_ptr(cb_twiddle_i));

            const volatile float* sram_tw_r =
                reinterpret_cast<const volatile float*>(sram_tw_r_addr) + tw_step_offset;
            const volatile float* sram_tw_i =
                reinterpret_cast<const volatile float*>(sram_tw_i_addr) + tw_step_offset;

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                tw_r_dst[p] = sram_tw_r[global_p];
                tw_i_dst[p] = sram_tw_i[global_p];
            }

            cb_push_back(cb_twiddle_r, 1);
            cb_push_back(cb_twiddle_i, 1);
        }
    }
}