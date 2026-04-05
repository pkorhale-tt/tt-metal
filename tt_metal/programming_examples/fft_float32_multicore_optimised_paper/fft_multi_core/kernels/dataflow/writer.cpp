// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

// ── Semaphore fix summary ──────────────────────────────────────────────────
//
// Root cause of deadlock:
//
// 1. noc_semaphore_set_remote performs a NOC ATOMIC INCREMENT, not a plain
//    store of 1.  After the reader resets rdy_flag to 0 with a local
//    noc_semaphore_set() call, the NOC router's internal counter for that
//    address may still lag.  A subsequent set_remote from the writer then
//    increments a stale value and the flag can accumulate past 1, causing
//    noc_semaphore_wait(rdy_flag, 1) in the reader to see 2 and spin
//    forever (it waits for == 1, not >= 1 on some SDK versions), OR the
//    reader resets to 0 before the writer has finished its own atomic, so
//    the flag under-counts.
//
// 2. The writer initialized neither rdy_flag nor ack_flag.  On a freshly
//    dispatched program, L1 content at SYNC_FLAG_ADDR is leftover from the
//    twiddle-init program.  If that byte happened to be non-zero the writer's
//    noc_semaphore_wait(ack_flag, 1) at the end of step 0 returns immediately
//    even though the reader never sent the ack, corrupting the handshake from
//    step 1 onward.
//
// Fix applied here:
//   • Writer initialises BOTH flags to 0 via noc_semaphore_set_remote to its
//     own NOC address — this goes through the NOC router and is therefore
//     coherent with every subsequent NOC semaphore operation on those words.
//   • All resets (rdy_flag after writer signals, ack_flag after writer
//     receives) also use noc_semaphore_set_remote(local_noc_addr, 0) instead
//     of the local noc_semaphore_set().  This keeps every write to the two
//     words in the NOC domain and avoids the core-local / NOC-coherence gap.
//
// The reader mirrors this change (see reader.cpp).
//
// Protocol (unchanged):
//   Writer end-of-step N (not last):
//     noc_semaphore_set_remote(rdy_noc_addr, 1)    signal reader
//     noc_semaphore_wait(ack_flag, 1)              wait for reader ack
//     noc_semaphore_set_remote(ack_noc_local, 0)   NOC-coherent reset
//
//   Reader start-of-step N+1:
//     noc_semaphore_wait(rdy_flag, 1)              wait for writer signal
//     noc_semaphore_set_remote(rdy_noc_local, 0)   NOC-coherent reset
//     < push chunks >
//     noc_semaphore_set_remote(ack_noc_addr, 1)    ack writer
// ─────────────────────────────────────────────────────────────────────────────

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

    // rdy_flag @ sync_flag_addr+0 : writer → reader  ("SRAM data committed")
    const uint32_t rdy_flag_addr = sync_flag_addr;
    volatile tt_l1_ptr uint32_t* rdy_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rdy_flag_addr);

    // ack_flag @ sync_flag_addr+4 : reader → writer  ("reader done, moving on")
    const uint32_t ack_flag_addr = sync_flag_addr + sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* ack_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_flag_addr);

    // NOC addresses for this core's own L1 semaphore words.
    // Using get_noc_addr() (single-arg) gives the local Tensix NOC XY + offset.
    // All writes go through the NOC router → fully coherent with noc_semaphore_wait.
    const uint64_t rdy_noc_local = get_noc_addr(rdy_flag_addr);
    const uint64_t ack_noc_local = get_noc_addr(ack_flag_addr);

    // NOC address of rdy_flag as seen by the reader (same Tensix core → same addr).
    const uint64_t rdy_noc_addr = rdy_noc_local;

    volatile tt_l1_ptr float* sram_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
    volatile tt_l1_ptr float* sram_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

    // ── Initialise both semaphore words to 0 via NOC ──────────────────────
    // Must use set_remote (NOC path) so the init is coherent with every
    // subsequent noc_semaphore_wait / noc_semaphore_set_remote on these words.
    noc_semaphore_set_remote(rdy_noc_local, 0);
    noc_semaphore_set_remote(ack_noc_local, 0);
    // Barrier ensures the two zeroing writes have landed before we proceed.
    noc_async_write_barrier();

    for (uint32_t step = 0; step < num_steps; ++step) {
        const uint32_t half_m   = 1u << step;
        const uint32_t m        = half_m << 1u;
        const bool is_last_step = (step + 1u == num_steps);

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const uint32_t pair_base = chunk * chunk_size;

            // Wait for compute outputs in push order: out0_r, out0_i, out1_r, out1_i.
            // This matches compute.cpp push order → no CB-fill deadlock.
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

            // Scatter butterfly results back to natural order in SRAM.
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
            // DMA final SRAM results → DRAM output buffers.
            const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
            noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
            noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
            noc_async_write_barrier();
        } else {
            // All SRAM scatter writes are done.  Signal the reader that SRAM
            // data for step N is committed and it may read pairs for step N+1.
            noc_semaphore_set_remote(rdy_noc_addr, 1);

            // Wait for reader to acknowledge that it has started consuming the
            // SRAM data (i.e. has begun pushing input CBs for step N+1).
            noc_semaphore_wait(ack_flag, 1);

            // Reset ack_flag to 0 via NOC so the reset is coherent with the
            // reader's next noc_semaphore_set_remote(ack_noc_addr, 1).
            noc_semaphore_set_remote(ack_noc_local, 0);
            noc_async_write_barrier();  // ensure reset lands before next iteration
        }
    }
}