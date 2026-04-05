// SPDX-FileCopyrightText: © 2025
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#if __has_include("compute_kernel_api/cb_api.h")
#include "compute_kernel_api/cb_api.h"
#elif __has_include("compute_kernel_api/common.h")
#include "compute_kernel_api/common.h"
#else
#include "compute_kernel_api.h"
#endif

void kernel_main() {
    const uint32_t num_steps = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t chunk_size = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_data0_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    for (uint32_t step = 0; step < num_steps; ++step) {
        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            cb_wait_front(cb_data0_r, 1);
            cb_wait_front(cb_data0_i, 1);
            cb_wait_front(cb_data1_r, 1);
            cb_wait_front(cb_data1_i, 1);
            cb_wait_front(cb_twiddle_r, 1);
            cb_wait_front(cb_twiddle_i, 1);

            const volatile tt_l1_ptr float* in0r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_data0_r));
            const volatile tt_l1_ptr float* in0i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_data0_i));
            const volatile tt_l1_ptr float* in1r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_data1_r));
            const volatile tt_l1_ptr float* in1i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_data1_i));
            const volatile tt_l1_ptr float* twr =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_twiddle_r));
            const volatile tt_l1_ptr float* twi =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_twiddle_i));

            cb_reserve_back(cb_out0_r, 1);
            cb_reserve_back(cb_out0_i, 1);
            cb_reserve_back(cb_out1_r, 1);
            cb_reserve_back(cb_out1_i, 1);

            volatile tt_l1_ptr float* out0r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_out0_r));
            volatile tt_l1_ptr float* out0i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_out0_i));
            volatile tt_l1_ptr float* out1r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_out1_r));
            volatile tt_l1_ptr float* out1i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_out1_i));

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const float a_r = in0r[p];
                const float a_i = in0i[p];
                const float b_r = in1r[p];
                const float b_i = in1i[p];
                const float w_r = twr[p];
                const float w_i = twi[p];

                const float t_r = b_r * w_r - b_i * w_i;
                const float t_i = b_r * w_i + b_i * w_r;

                out0r[p] = a_r + t_r;
                out0i[p] = a_i + t_i;
                out1r[p] = a_r - t_r;
                out1i[p] = a_i - t_i;
            }

            cb_push_back(cb_out0_r, 1);
            cb_push_back(cb_out0_i, 1);
            cb_push_back(cb_out1_r, 1);
            cb_push_back(cb_out1_i, 1);

            cb_pop_front(cb_data0_r, 1);
            cb_pop_front(cb_data0_i, 1);
            cb_pop_front(cb_data1_r, 1);
            cb_pop_front(cb_data1_i, 1);
            cb_pop_front(cb_twiddle_r, 1);
            cb_pop_front(cb_twiddle_i, 1);
        }
    }
}