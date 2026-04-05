// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

// See reader.cpp for full explanation of the two-semaphore NOC handshake.
//
// Writer side of the protocol (end of each non-last step):
//   1. All SRAM scatter writes for step N are done (scalar stores in loop above).
//   2. noc_semaphore_set_remote(rdy_noc_addr, 1)
//      → NOC write to rdy_flag, visible to RISCV_0 (reader).
//   3. noc_semaphore_wait(ack_flag, 1)
//      → spin until reader confirms it has started pushing CBs for step N+1.
//   4. noc_semaphore_set(ack_flag, 0)  → reset ack for next step.
//
// Output CB push order: out0_r, out0_i, out1_r, out1_i
//   Matches compute kernel push order so the writer's cb_wait_front(out0_r)
//   is never blocked by a full cb_out1_r.

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
    const uint32_t sync_flag_addr     = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    const uint32_t row_bytes       = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

    // rdy_flag: we (writer) set this to signal reader that SRAM is ready
    const uint32_t rdy_flag_addr = sync_flag_addr;
    volatile tt_l1_ptr uint32_t* rdy_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rdy_flag_addr);
    const uint64_t rdy_noc_addr = get_noc_addr(rdy_flag_addr);

    // ack_flag: reader sets this to signal us that it has seen the data
    const uint32_t ack_flag_addr = sync_flag_addr + sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* ack_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_flag_addr);

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

            // Wait for compute to push outputs in order: out0_r, out0_i, out1_r, out1_i
            // This order matches the compute kernel push order — no CB-fill deadlock.
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

            // Scatter butterfly results back to natural (original) order in SRAM
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
            // DMA final results from SRAM to DRAM
            const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
            noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
            noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
            noc_async_write_barrier();
        } else {
            // Signal reader that all SRAM writes for this step are done.
            // noc_semaphore_set_remote goes through the NOC router so it is
            // visible to RISCV_0 (reader) — unlike a plain scalar L1 store.
            noc_semaphore_set_remote(rdy_noc_addr, 1);

            // Wait for reader to acknowledge it has seen the signal and started
            // pushing input CBs for the next step, so we know ack_flag will be
            // reset before we check it again next iteration.
            noc_semaphore_wait(ack_flag, 1);
            noc_semaphore_set(ack_flag, 0);
        }
    }
}