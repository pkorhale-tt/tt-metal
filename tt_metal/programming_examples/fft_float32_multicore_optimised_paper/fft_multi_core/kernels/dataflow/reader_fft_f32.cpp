// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for FFT.
//
// Runtime args:
//   0 : dram_input_r_addr
//   1 : dram_input_i_addr
//   2 : n
//   3 : num_steps
//   4 : num_chunks
//   5 : chunk_size
//   6 : sram_buf_r_addr
//   7 : sync_flag_addr   ← NEW: L1 address of the single uint32 handshake flag
//                          shared with the writer on RISCV_1.
//                          Writer sets it to 1 after scattering each step's
//                          results; reader clears it to 0 before moving on.

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
    const uint32_t sync_flag_addr    = get_arg_val<uint32_t>(7);  // NEW

    constexpr uint32_t cb_data0_r   = tt::CBIndex::c_0;
    constexpr uint32_t cb_data0_i   = tt::CBIndex::c_1;
    constexpr uint32_t cb_data1_r   = tt::CBIndex::c_2;
    constexpr uint32_t cb_data1_i   = tt::CBIndex::c_3;
    constexpr uint32_t cb_twiddle_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_twiddle_i = tt::CBIndex::c_5;

    const uint32_t tile_bytes    = get_tile_size(cb_data0_r);
    const DataFormat data_format = get_dataformat(cb_data0_r);

    const uint32_t sram_buf_bytes  = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + sram_buf_bytes;

    // Twiddle tables sit above the two data ping-pong buffers
    const uint32_t sram_tw_r_addr = sram_buf_i_addr + sram_buf_bytes;
    const uint32_t sram_tw_i_addr = sram_tw_r_addr + num_steps * (n / 2u) * sizeof(float);

    const uint32_t row_tiles = (n * sizeof(float) + tile_bytes - 1) / tile_bytes;

    const InterleavedAddrGenFast<true> dram_r_gen = {
        .bank_base_address = dram_input_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};
    const InterleavedAddrGenFast<true> dram_i_gen = {
        .bank_base_address = dram_input_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};

    // Initialise the sync flag to 0 so the writer can set it to 1 when ready.
    // Both RISCV_0 and RISCV_1 start simultaneously; only RISCV_0 (us) writes
    // the initial value, so there is no write-after-write hazard here.
    volatile tt_l1_ptr uint32_t* sync_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);
    *sync_flag = 0u;

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m         = 1u << step;
        const uint32_t m              = half_m << 1u;
        const uint32_t tw_step_offset = step * (n / 2u);

        if (step == 0u) {
            // Load the entire row from DRAM into SRAM on the first step only.
            for (uint32_t t = 0; t < row_tiles; ++t) {
                noc_async_read_tile(t, dram_r_gen, sram_buf_r_addr + t * tile_bytes);
                noc_async_read_tile(t, dram_i_gen, sram_buf_i_addr + t * tile_bytes);
            }
            noc_async_read_barrier();
        } else {
            // For steps 1+, wait until the writer has finished scattering the
            // previous step's butterfly outputs back into sram_buf.
            // Without this barrier the reader can race ahead and pick up stale
            // (or partially-written) values, producing garbage / inf results.
            while (*sync_flag == 0u) { /* spin */ }
            *sync_flag = 0u;  // reset for the next step
        }

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            // -- data0 and data1 real --
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

            // -- data0 and data1 imaginary --
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

            // -- twiddle factors --
            cb_reserve_back(cb_twiddle_r, 1);
            cb_reserve_back(cb_twiddle_i, 1);

            volatile tt_l1_ptr float* tw_r_dst =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_twiddle_r));
            volatile tt_l1_ptr float* tw_i_dst =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_twiddle_i));

            const volatile tt_l1_ptr float* sram_tw_r =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_tw_r_addr)
                + tw_step_offset;
            const volatile tt_l1_ptr float* sram_tw_i =
                reinterpret_cast<const volatile tt_l1_ptr float*>(sram_tw_i_addr)
                + tw_step_offset;

            for (uint32_t p = 0; p < chunk_size; ++p) {
                tw_r_dst[p] = sram_tw_r[pair_base + p];
                tw_i_dst[p] = sram_tw_i[pair_base + p];
            }

            cb_push_back(cb_twiddle_r, 1);
            cb_push_back(cb_twiddle_i, 1);
        }
    }
}