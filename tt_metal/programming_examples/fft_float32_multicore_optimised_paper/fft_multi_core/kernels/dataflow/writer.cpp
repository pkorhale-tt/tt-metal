// // SPDX-FileCopyrightText: © 2025 (paper faithful port)
// // SPDX-License-Identifier: Apache-2.0

// // ── Correct semaphore protocol ────────────────────────────────────────────────
// //
// // KEY INSIGHT: noc_semaphore_set_remote() is a NOC ATOMIC INCREMENT (AT_INC),
// // NOT a store.  Passing value 0 is a no-op; value N increments by N.
// // So "noc_semaphore_set_remote(addr, 0)" for reset does NOTHING — the
// // previous fix was broken because it tried to reset to 0 via set_remote.
// //
// // noc_semaphore_set(ptr, val) is a plain LOCAL L1 store.
// // noc_semaphore_wait(ptr, target) polls *ptr until it equals target.
// //
// // RISCV_0 (reader) and RISCV_1 (writer) are on the SAME Tensix core and
// // share the same 1.3MB L1 SRAM.  A local store from either core is visible
// // to the other core once it exits the store buffer — which is guaranteed
// // by the time the other core's noc_semaphore_wait() load executes (the
// // wait loop is a polling load with an implicit acquire fence).
// //
// // Therefore for same-Tensix communication: use LOCAL stores (noc_semaphore_set)
// // for BOTH signal and reset.  Do NOT use noc_semaphore_set_remote for reset
// // because AT_INC(0) is a no-op and AT_INC on a stale NOC-cached value can
// // produce values > 1, causing the wait(==1) to spin forever.
// //
// // CORRECT PROTOCOL:
// //   Init (writer only, before step loop):
// //     noc_semaphore_set(rdy_flag, 0)
// //     noc_semaphore_set(ack_flag, 0)
// //
// //   Writer end-of-step N (not last):
// //     noc_semaphore_set(rdy_flag, 1)     // local store, visible to RISCV_0
// //     noc_semaphore_wait(ack_flag, 1)    // poll until reader acks
// //     noc_semaphore_set(ack_flag, 0)     // reset
// //
// //   Reader start-of-step N+1:
// //     noc_semaphore_wait(rdy_flag, 1)    // poll until writer signals
// //     noc_semaphore_set(rdy_flag, 0)     // reset
// //     < push chunks >
// //     noc_semaphore_set(ack_flag, 1)     // ack writer
// // ─────────────────────────────────────────────────────────────────────────────

// #include <cstdint>
// #include "api/dataflow/dataflow_api.h"

// void kernel_main() {
//     const uint32_t dram_output_r_addr = get_arg_val<uint32_t>(0);
//     const uint32_t dram_output_i_addr = get_arg_val<uint32_t>(1);
//     const uint32_t n                  = get_arg_val<uint32_t>(2);
//     const uint32_t num_steps          = get_arg_val<uint32_t>(3);
//     const uint32_t num_chunks         = get_arg_val<uint32_t>(4);
//     const uint32_t chunk_size         = get_arg_val<uint32_t>(5);
//     const uint32_t sram_buf_r_addr    = get_arg_val<uint32_t>(6);
//     const uint32_t sync_flag_addr     = get_arg_val<uint32_t>(7);

//     constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
//     constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
//     constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
//     constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

//     const uint32_t row_bytes       = n * sizeof(float);
//     const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

//     // rdy_flag @ sync_flag_addr+0 : writer signals → reader polls
//     volatile tt_l1_ptr uint32_t* rdy_flag =
//         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);

//     // ack_flag @ sync_flag_addr+4 : reader signals → writer polls
//     const uint32_t ack_flag_addr = sync_flag_addr + sizeof(uint32_t);
//     volatile tt_l1_ptr uint32_t* ack_flag =
//         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_flag_addr);

//     volatile tt_l1_ptr float* sram_r =
//         reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
//     volatile tt_l1_ptr float* sram_i =
//         reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

//     // Writer owns initialisation. Reader does NOT initialise — single owner
//     // avoids the race where reader's init zeros a flag the writer already set.
//     noc_semaphore_set(rdy_flag, 0);
//     noc_semaphore_set(ack_flag, 0);

//     for (uint32_t step = 0; step < num_steps; ++step) {
//         const uint32_t half_m   = 1u << step;
//         const uint32_t m        = half_m << 1u;
//         const bool is_last_step = (step + 1u == num_steps);

//         for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
//             const uint32_t pair_base = chunk * chunk_size;

//             // Push order matches compute.cpp: out0_r, out0_i, out1_r, out1_i.
//             cb_wait_front(cb_out0_r, 1);
//             cb_wait_front(cb_out0_i, 1);
//             cb_wait_front(cb_out1_r, 1);
//             cb_wait_front(cb_out1_i, 1);

//             const volatile tt_l1_ptr float* out0_r =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out0_r));
//             const volatile tt_l1_ptr float* out0_i =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out0_i));
//             const volatile tt_l1_ptr float* out1_r =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out1_r));
//             const volatile tt_l1_ptr float* out1_i =
//                 reinterpret_cast<const volatile tt_l1_ptr float*>(get_read_ptr(cb_out1_i));

//             // Scatter butterfly results back to natural order in SRAM.
//             for (uint32_t p = 0; p < chunk_size; ++p) {
//                 const uint32_t global_p = pair_base + p;
//                 const uint32_t group    = global_p / half_m;
//                 const uint32_t j        = global_p % half_m;
//                 const uint32_t a        = group * m + j;
//                 const uint32_t b        = a + half_m;
//                 sram_r[a] = out0_r[p];
//                 sram_i[a] = out0_i[p];
//                 sram_r[b] = out1_r[p];
//                 sram_i[b] = out1_i[p];
//             }

//             cb_pop_front(cb_out0_r, 1);
//             cb_pop_front(cb_out0_i, 1);
//             cb_pop_front(cb_out1_r, 1);
//             cb_pop_front(cb_out1_i, 1);
//         }

//         if (is_last_step) {
//             // DMA final SRAM results to DRAM.
//             const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
//             const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
//             noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
//             noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
//             noc_async_write_barrier();
//         } else {
//             // All SRAM scatter writes done. Signal reader via local store —
//             // same Tensix L1, no NOC needed, immediately visible to RISCV_0.
//             noc_semaphore_set(rdy_flag, 1);

//             // Wait for reader to ack after it has started pushing CBs.
//             noc_semaphore_wait(ack_flag, 1);
//             noc_semaphore_set(ack_flag, 0);   // reset for next step
//         }
//     }
// }


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
    const uint32_t sync_flag_addr     = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out0_r = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i = tt::CBIndex::c_19;

    const uint32_t row_bytes       = n * sizeof(float);
    const uint32_t sram_buf_i_addr = sram_buf_r_addr + row_bytes;

    // rdy_flag @ sync_flag_addr+0 : writer signals -> reader polls
    volatile tt_l1_ptr uint32_t* rdy_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_flag_addr);

    // ack_flag @ sync_flag_addr+4 : reader signals -> writer polls
    const uint32_t ack_flag_addr = sync_flag_addr + sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* ack_flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_flag_addr);

    volatile tt_l1_ptr float* sram_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_r_addr);
    volatile tt_l1_ptr float* sram_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(sram_buf_i_addr);

    // Writer owns initialisation.
    noc_semaphore_set(rdy_flag, 0);
    noc_semaphore_set(ack_flag, 0);

    // Debug selector:
    // 0 -> dump raw words from out0 tile page
    // 1 -> dump raw words from out1 tile page
    constexpr uint32_t debugDumpWhich = 0;

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

            // DEBUG PROOF MODE:
            // For n=8 on the last step, dump the first 8 raw float words
            // from one packed output tile page directly into SRAM bins 0..7.
            // This proves what the writer is actually reading.
            if (n == 8 && is_last_step && chunk == 0) {
                const volatile tt_l1_ptr float* dump_r =
                    (debugDumpWhich == 0) ? out0_r : out1_r;
                const volatile tt_l1_ptr float* dump_i =
                    (debugDumpWhich == 0) ? out0_i : out1_i;

                for (uint32_t k = 0; k < 8; ++k) {
                    sram_r[k] = dump_r[k];
                    sram_i[k] = dump_i[k];
                }

                cb_pop_front(cb_out0_r, 1);
                cb_pop_front(cb_out0_i, 1);
                cb_pop_front(cb_out1_r, 1);
                cb_pop_front(cb_out1_i, 1);
                continue;
            }

            // Normal path
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
            const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
            noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
            noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
            noc_async_write_barrier();
        } else {
            noc_semaphore_set(rdy_flag, 1);
            noc_semaphore_wait(ack_flag, 1);
            noc_semaphore_set(ack_flag, 0);
        }
    }
}