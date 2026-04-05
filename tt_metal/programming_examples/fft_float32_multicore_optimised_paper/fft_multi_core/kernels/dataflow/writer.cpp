// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

inline uint32_t linearToNfacesIndex(uint32_t linearIdx) {
    const uint32_t row = linearIdx / 32;
    const uint32_t col = linearIdx % 32;

    const uint32_t faceRow = row / 16;
    const uint32_t faceCol = col / 16;
    const uint32_t face = faceRow * 2 + faceCol;

    const uint32_t inFaceRow = row % 16;
    const uint32_t inFaceCol = col % 16;

    return face * 256 + inFaceRow * 16 + inFaceCol;
}

inline float readLogicalValueFromTile(
    const volatile tt_l1_ptr float* tileBase,
    uint32_t logicalIdx) {
    return tileBase[linearToNfacesIndex(logicalIdx)];
}

void kernel_main() {
    const uint32_t dram_output_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t dram_output_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t n                  = get_arg_val<uint32_t>(2);
    const uint32_t num_steps          = get_arg_val<uint32_t>(3);
    const uint32_t num_chunks         = get_arg_val<uint32_t>(4);
    const uint32_t chunk_size         = get_arg_val<uint32_t>(5);
    const uint32_t sram_buf_r_addr    = get_arg_val<uint32_t>(6);
    const uint32_t sync_flag_addr     = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    const uint32_t row_bytes       = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

    volatile tt_l1_ptr uint32_t* sync_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);

    volatile tt_l1_ptr float* sram_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
    volatile tt_l1_ptr float* sram_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m   = 1u << step;
        const uint32_t m        = half_m << 1u;
        const bool is_last_step = (step + 1u == num_steps);

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            cb_wait_front(cb_out0_r, 1);
            cb_wait_front(cb_out0_i, 1);
            cb_wait_front(cb_out1_r, 1);
            cb_wait_front(cb_out1_i, 1);

            const volatile tt_l1_ptr float* out0_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out0_r));
            const volatile tt_l1_ptr float* out0_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out0_i));
            const volatile tt_l1_ptr float* out1_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out1_r));
            const volatile tt_l1_ptr float* out1_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out1_i));

            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;
                const uint32_t b        = a + half_m;

                sram_r[a] = readLogicalValueFromTile(out0_r, p);
                sram_i[a] = readLogicalValueFromTile(out0_i, p);
                sram_r[b] = readLogicalValueFromTile(out1_r, p);
                sram_i[b] = readLogicalValueFromTile(out1_i, p);
            }

            cb_pop_front(cb_out0_r, 1);
            cb_pop_front(cb_out0_i, 1);
            cb_pop_front(cb_out1_r, 1);
            cb_pop_front(cb_out1_i, 1);
        }

        if (is_last_step) {
            // DEBUG SENTINEL:
            // Force very obvious output values before final DRAM write.
            sram_r[0] = 999.0f;
            sram_i[0] = -777.0f;

            const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
            noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
            noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
            noc_async_write_barrier();
        } else {
            *sync_flag = 1u;
        }
    }
}