// SPDX-FileCopyrightText: © 2025 (paper faithful port)
// SPDX-License-Identifier: Apache-2.0

// ── FIXES vs original ────────────────────────────────────────────────────────
// 1. RISC-V weak memory model: added `__asm__ volatile("fence w,w" ::: "memory")`
//    before every `*sync_flag = 1u` write.  Without this fence the compiler
//    or the hardware can reorder the scalar SRAM stores (sram_r[a]/sram_i[b])
//    after the flag write, so the reader on RISCV_0 could see the flag go
//    high, clear it, start pushing CB tiles, and then observe partially
//    written SRAM data — producing wrong results or a deadlock on the next
//    step boundary.
//
// 2. The sync_flag is shared between RISCV_0 (reader) and RISCV_1 (writer).
//    The reader resets it to 0 right before spinning.  There is a race if the
//    writer sets the flag to 1 for step N while the reader has not yet reached
//    its spin-wait for step N+1, meaning the reader would then clear the flag
//    itself and spin forever.  The fence + the existing reader-side
//    `*sync_flag = 0` just before the spin closes the window: the writer's
//    flag write is now ordered after all SRAM data writes, so by the time the
//    reader sees the flag it can safely proceed.
//
// 3. Reordered the cb_wait_front calls at the top of each chunk to match the
//    push order in the fixed compute kernel (out0_r, out0_i, out1_r, out1_i).
//    Waiting on out0_r first while compute pushes out1_r first (old order)
//    could stall the writer and fill cb_out1_r with depth-2 tiles, blocking
//    compute from pushing out0_r, which is the exact CB-depth deadlock path.
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

    // CB indices must match host registration and compute kernel
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

            // ── FIX 3: wait in the same order compute pushes (out0 first) ──
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

            // Scatter butterfly outputs back into natural (original) order
            for (uint32_t p = 0; p < chunk_size; ++p) {
                const uint32_t global_p = pair_base + p;
                const uint32_t group    = global_p / half_m;
                const uint32_t j        = global_p % half_m;
                const uint32_t a        = group * m + j;       // upper element
                const uint32_t b        = a + half_m;          // lower element

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
            // Final step: DMA results from SRAM to DRAM output buffers
            const uint64_t noc_r = get_noc_addr(dram_output_r_addr);
            const uint64_t noc_i = get_noc_addr(dram_output_i_addr);
            noc_async_write(sram_buf_r_addr, noc_r, row_bytes);
            noc_async_write(sram_buf_i_addr, noc_i, row_bytes);
            noc_async_write_barrier();
        } else {
            // ── FIX 1 & 2: memory fence before signalling the reader ───────
            // Guarantees all scalar sram_r[]/sram_i[] stores committed above
            // are globally visible to RISCV_0 before the flag store.
            // Without this, the reader can observe the flag=1, clear it to 0,
            // and start reading SRAM data that hasn't landed yet.
            __asm__ volatile("fence w,w" ::: "memory");
            *sync_flag = 1u;
        }
    }
}