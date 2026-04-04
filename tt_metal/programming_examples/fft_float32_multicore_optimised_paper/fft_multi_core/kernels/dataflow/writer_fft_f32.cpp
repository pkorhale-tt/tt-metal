// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for FFT.
//
// Runtime args:
//   0 : dram_output_r_addr
//   1 : dram_output_i_addr
//   2 : n
//   3 : num_steps
//   4 : num_chunks
//   5 : chunk_size
//   6 : sram_buf_r_addr
//   7 : sync_flag_addr   ← NEW: same L1 word that the reader spins on.
//                          After scattering each step's results the writer
//                          sets this to 1 so the reader can proceed to the
//                          next step.  The reader resets it to 0.
//
// After the final butterfly stage the writer applies a bit-reversal
// permutation in SRAM before writing results back to DRAM.
// The Cooley-Tukey DIF butterfly loop naturally produces output in
// bit-reversed index order; without this step every bin lands at the
// wrong position, making all outputs appear wrong when compared against
// a natural-order CPU FFT reference.

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
    const uint32_t sync_flag_addr     = get_arg_val<uint32_t>(7);  // NEW

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    const uint32_t sram_buf_bytes  = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + sram_buf_bytes;

    const uint32_t row_tiles = (n * sizeof(float) + tile_bytes - 1) / tile_bytes;

    const InterleavedAddrGenFast<true> dram_r_gen = {
        .bank_base_address = dram_output_r_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};
    const InterleavedAddrGenFast<true> dram_i_gen = {
        .bank_base_address = dram_output_i_addr,
        .page_size         = tile_bytes,
        .data_format       = data_format};

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

                sram_r[a] = out0_r[p];
                sram_i[a] = out0_i[p];
                sram_r[b] = out1_r[p];
                sram_i[b] = out1_i[p];
            }

            cb_pop_front(cb_out0_r, 1);
            cb_pop_front(cb_out0_i, 1);
            cb_pop_front(cb_out1_r, 1);
            cb_pop_front(cb_out1_i, 1);
        }

        if (is_last_step) {
            // ── Bit-reversal permutation ──────────────────────────────────────
            // The DIF butterfly loop above produces results in bit-reversed
            // index order.  Swap each element with its bit-reversed counterpart
            // (in-place, only when i < bit_rev(i) to avoid double-swapping).
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t j   = 0u;
                uint32_t tmp = i;
                for (uint32_t b = 0; b < num_steps; ++b) {
                    j   = (j << 1u) | (tmp & 1u);
                    tmp >>= 1u;
                }
                if (i < j) {
                    float tr  = sram_r[i]; sram_r[i] = sram_r[j]; sram_r[j] = tr;
                    float ti  = sram_i[i]; sram_i[i] = sram_i[j]; sram_i[j] = ti;
                }
            }

            // ── Write final results to DRAM ───────────────────────────────────
            for (uint32_t t = 0; t < row_tiles; ++t) {
                noc_async_write_tile(t, dram_r_gen, sram_buf_r_addr + t * tile_bytes);
                noc_async_write_tile(t, dram_i_gen, sram_buf_i_addr + t * tile_bytes);
            }
            noc_async_write_barrier();
        } else {
            // ── Signal reader that SRAM is ready for the next step ────────────
            // The reader on RISCV_0 is spinning on this flag.  Setting it to 1
            // tells it the scatter for this step is complete and it is safe to
            // read sram_buf for step+1.
            *sync_flag = 1u;
        }
    }
}