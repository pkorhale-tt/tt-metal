// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dram_input_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t dram_input_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t n                 = get_arg_val<uint32_t>(2);
    const uint32_t num_steps         = get_arg_val<uint32_t>(3);
    const uint32_t num_chunks        = get_arg_val<uint32_t>(4);
    const uint32_t chunk_size        = get_arg_val<uint32_t>(5);
    const uint32_t sram_buf_r_addr   = get_arg_val<uint32_t>(6);
    const uint32_t sync_flag_addr    = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_data0_r   = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i   = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r   = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i   = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;

    const uint32_t row_bytes       = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

    const uint32_t sram_tw_r_addr = sram_buf_i_addr + row_bytes;
    const uint32_t sram_tw_i_addr = sram_tw_r_addr + num_steps * (n / 2u) * sizeof(float);

    volatile tt_l1_ptr uint32_t* sync_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);
    *sync_flag = 0u;

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m         = 1u << step;
        const uint32_t m              = half_m << 1u;
        const uint32_t tw_step_offset = step * (n / 2u);

        if (step == 0u) {
            const uint64_t noc_r = get_noc_addr(dram_input_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_input_i_addr);
            noc_async_read(noc_r, sram_buf_r_addr, row_bytes);
            noc_async_read(noc_i, sram_buf_i_addr, row_bytes);
            noc_async_read_barrier();

            volatile tt_l1_ptr float* sr =
                reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
            volatile tt_l1_ptr float* si =
                reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

            for (uint32_t i = 0; i < n; ++i) {
                uint32_t j = 0u;
                uint32_t tmp = i;
                for (uint32_t b = 0; b < num_steps; ++b) {
                    j = (j << 1u) | (tmp & 1u);
                    tmp >>= 1u;
                }
                if (i < j) {
                    float tr = sr[i]; sr[i] = sr[j]; sr[j] = tr;
                    float ti = si[i]; si[i] = si[j]; si[j] = ti;
                }
            }
        } else {
            while (*sync_flag == 0u) { }
            *sync_flag = 0u;
        }

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            cb_reserve_back(cb_data0_r, 1);
            cb_reserve_back(cb_data1_r, 1);

            volatile tt_l1_ptr float* dst0_r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data0_r));
            volatile tt_l1_ptr float* dst1_r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data1_r));
            const volatile tt_l1_ptr float* src_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_buf_r_addr);

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;
                const uint32_t b        = a + half_m;
                dst0_r[p] = src_r[a];
                dst1_r[p] = src_r[b];
            }

            cb_push_back(cb_data0_r, 1);
            cb_push_back(cb_data1_r, 1);

            cb_reserve_back(cb_data0_i, 1);
            cb_reserve_back(cb_data1_i, 1);

            volatile tt_l1_ptr float* dst0_i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data0_i));
            volatile tt_l1_ptr float* dst1_i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_data1_i));
            const volatile tt_l1_ptr float* src_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_buf_i_addr);

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;
                const uint32_t b        = a + half_m;
                dst0_i[p] = src_i[a];
                dst1_i[p] = src_i[b];
            }

            cb_push_back(cb_data0_i, 1);
            cb_push_back(cb_data1_i, 1);

            cb_reserve_back(cb_twiddle_r, 1);
            cb_reserve_back(cb_twiddle_i, 1);

            volatile tt_l1_ptr float* tw_r_dst =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_twiddle_r));
            volatile tt_l1_ptr float* tw_i_dst =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_twiddle_i));
            const volatile tt_l1_ptr float* sram_tw_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_tw_r_addr) + tw_step_offset;
            const volatile tt_l1_ptr float* sram_tw_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_tw_i_addr) + tw_step_offset;

            for (uint32_t p = 0; p < chunk_size; ++p) {
                tw_r_dst[p] = sram_tw_r[pair_base + p];
                tw_i_dst[p] = sram_tw_i[pair_base + p];
            }

            cb_push_back(cb_twiddle_r, 1);
            cb_push_back(cb_twiddle_i, 1);
        }
    }
}