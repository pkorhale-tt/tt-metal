// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0
//
// writer_fft_f32.cpp  –  EXACT match to paper Section 4 / Figure 3
//
// Paper design (Fig. 3):
//   "The output of these calculations is stored in real and imaginary CBs
//    which are consumed by the out data mover core which then reorders the
//    data into the original order and will either store this in SRAM, ready
//    to be consumed by the next step, or external DDR for the final results."
//
// For each chunk the writer:
//   1. Waits for out0_{r,i} and out1_{r,i} pages from compute.
//   2. Scatters the butterfly outputs back to their original-order indices
//      in the SRAM ping buffer.
//   3. On the final step, also writes the assembled result row to DRAM.
//   4. Pops the output CBs so they can be reused by compute.
//
// No scratch row CB is needed here because the SRAM buffer IS the
// scratch area (paper design: entire domain lives in local SRAM).
//
// CB indices (must match compute kernel and host):
//   16  cb_out0_r   result data0, real
//   17  cb_out0_i   result data0, imaginary
//   18  cb_out1_r   result data1, real
//   19  cb_out1_i   result data1, imaginary
//
// Kernel args (7):
//   0  dram_output_r_addr – DRAM address for final real output
//   1  dram_output_i_addr – DRAM address for final imaginary output
//   2  n                  – FFT size
//   3  num_steps          – log2(n)
//   4  num_chunks         – number of chunks per step
//   5  chunk_size         – element pairs per chunk
//   6  sram_buf_r_addr    – SRAM ping buffer base address (real)
//      (imaginary follows immediately at +n*sizeof(float))

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
        const bool     is_last_step = (step + 1u == num_steps);

        // Pointers to the SRAM ping buffer for this step's results.
        volatile float* sram_r =
            reinterpret_cast<volatile float*>(sram_buf_r_addr);
        volatile float* sram_i =
            reinterpret_cast<volatile float*>(sram_buf_i_addr);

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            // ------------------------------------------------------------------
            // Wait for this chunk's butterfly outputs from compute.
            // Paper Listing 1.2: outputs are in cb_out0 (data0) and
            // cb_out1 (data1) CBs.
            // ------------------------------------------------------------------
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

            // ------------------------------------------------------------------
            // Scatter results back to original-order positions in SRAM.
            // Paper: "reorders the data into the original order and will
            // either store this in SRAM, ready to be consumed by the next
            // step, or external DDR for the final results."
            // ------------------------------------------------------------------
            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;       // data0 (even) position
                const uint32_t b        = a + half_m;          // data1 (odd)  position

                sram_r[a] = out0_r[p];
                sram_i[a] = out0_i[p];
                sram_r[b] = out1_r[p];
                sram_i[b] = out1_i[p];
            }

            // ------------------------------------------------------------------
            // Final step: write assembled results to DRAM output buffer.
            // Paper Fig. 3: "external DDR for the final results."
            // We write the entire domain after the last chunk completes.
            // ------------------------------------------------------------------
            if (is_last_step && (chunk + 1u == num_chunks)) {
                // Write real component tile to DRAM.
                uint32_t sram_r_read = sram_buf_r_addr;
                noc_async_write_tile(0u, dram_r_gen, sram_r_read);

                // Write imaginary component tile to DRAM.
                uint32_t sram_i_read = sram_buf_i_addr;
                noc_async_write_tile(0u, dram_i_gen, sram_i_read);

                noc_async_write_barrier();
            }

            // Release output CBs so compute can reuse them.
            cb_pop_front(cb_out0_r, 1);
            cb_pop_front(cb_out0_i, 1);
            cb_pop_front(cb_out1_r, 1);
            cb_pop_front(cb_out1_i, 1);
        }
    }
}