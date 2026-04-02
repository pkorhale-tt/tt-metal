// =============================================================================
// kernels/fft_large_writer.cpp
// Tensix DATA-MOVEMENT-1 kernel — Tier 3 Large FFT
//
// After phase-1 row FFTs are complete, this kernel performs the ALL-to-ALL
// NOC transpose: it takes each element [row r, col c] from this core's output
// and NOC-writes it to the core that owns column c (which becomes row c after
// the transpose).
//
// After phase-2 column FFTs are done, it writes final results to DRAM.
//
// Core coordinate layout (assumed 8-wide row-major grid):
//   core_id = row * 8 + col  (matches host-side assignment)
//   Col-owning core for column c = c / rows_per_core
//
// Runtime args:
//   [0] = dst_buf_addr
//   [1] = batch_idx
//   [2] = row_start
//   [3] = rows_per_core
//   [4] = R
//   [5] = S
// =============================================================================

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr      = get_arg_val<uint32_t>(0);
    uint32_t batch_idx     = get_arg_val<uint32_t>(1);
    uint32_t row_start     = get_arg_val<uint32_t>(2);
    uint32_t rows_per_core = get_arg_val<uint32_t>(3);
    uint32_t R             = get_arg_val<uint32_t>(4);
    uint32_t S             = get_arg_val<uint32_t>(5);

    const uint32_t N           = R * S;
    const uint32_t cores_in_grp = R / rows_per_core;

    // -------------------------------------------------------------------
    // Phase-1 output is in CB3 (CB_OUT). Wait for compute to push it.
    // -------------------------------------------------------------------
    cb_wait_front(3, 1);
    const volatile float* phase1_out =
        reinterpret_cast<const volatile float*>(get_read_ptr(3));

    // -------------------------------------------------------------------
    // NOC transpose: for each of my rows, scatter each column element to
    // the core that owns that column.
    //
    // After transpose: col c is owned by core  c / rows_per_core
    // That core's CB0 L1 base address is known from the core's NOC coords.
    // Within that core's CB0 buffer, our row r occupies slot:
    //   offset = row_start * R * 2 * sizeof(float)  + r_local * R * 2 * sizeof(float)
    //   (i.e. we're filling column row_start + r_local of the transposed matrix)
    // -------------------------------------------------------------------

    // Scatter phase-1 row data to the cores that own the corresponding columns.
    //
    // Layout in each receiver's CB0 (an R × rows_per_core block, row-major):
    //   element [abs_row][c_local] is at byte offset:
    //     abs_row * rows_per_core * 2 * sizeof(float)
    //   where c_local = c - target_core_id * rows_per_core (0-based within the
    //   slice), and we transfer all rows_per_core complex floats in one write.
    //   Because each write targets a *different* core for each c value, abs_row
    //   alone uniquely identifies the slot within that target core's buffer.
    for (uint32_t r_local = 0; r_local < rows_per_core; ++r_local) {
        uint32_t abs_row = row_start + r_local;

        for (uint32_t c = 0; c < S; c += rows_per_core) {
            uint32_t target_core_id = c / rows_per_core;

            uint32_t packed_noc = get_arg_val<uint32_t>(6 + target_core_id);
            uint32_t target_col = packed_noc & 0xFFFF;
            uint32_t target_row = packed_noc >> 16;

            uint64_t noc_dst = get_noc_addr(target_col, target_row, get_write_ptr(0));

            // Row slot for abs_row in the receiver's R × rows_per_core buffer.
            uint32_t dst_l1_offset = abs_row * rows_per_core * 2 * sizeof(float);

            // Source: columns [c .. c+rows_per_core) of row r_local
            const volatile float* src_ptr = phase1_out + r_local * S * 2 + c * 2;

            noc_async_write(
                reinterpret_cast<uintptr_t>(src_ptr),
                noc_dst + dst_l1_offset,
                rows_per_core * 2 * sizeof(float)
            );
        }
    }

    // Flush all NOC writes before incrementing any semaphore.
    // Each target core's reader spins on its semaphore reaching cores_in_grp,
    // counting one increment per sender core (not per row).  We fire all
    // semaphore increments after a single barrier, so every write is visible
    // before the receiver is unblocked.
    noc_async_write_barrier();

    // Increment semaphore on every target core exactly once — the reader waits
    // for cores_in_grp increments total, one from each sender core.
    for (uint32_t target_core_id = 0; target_core_id < cores_in_grp; ++target_core_id) {
        uint32_t packed_noc = get_arg_val<uint32_t>(6 + target_core_id);
        uint32_t tc = packed_noc & 0xFFFF;
        uint32_t tr = packed_noc >> 16;
        uint64_t sem_noc = get_noc_addr(tc, tr, get_semaphore(0));
        noc_semaphore_inc(sem_noc, 1);
    }

    cb_pop_front(3, 1);   // release phase-1 output CB

    // -------------------------------------------------------------------
    // Wait for phase-2 output from compute (second push to CB3 = CB_OUT)
    // -------------------------------------------------------------------
    cb_wait_front(3, 1);
    const volatile float* final_out =
        reinterpret_cast<const volatile float*>(get_read_ptr(3));

    // Write final results to DRAM
    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size         = 8
    };

    // Output layout matches input: [batch_idx * N + row_start * R + ...]
    // After the 2D FFT, our core holds rows [row_start .. row_start+rows_per_core)
    // of the R×S output matrix (now S×R after col FFTs, reinterpreted as N).
    uint32_t base_page = batch_idx * N + row_start * R;
    uint64_t dst_noc   = get_noc_addr(base_page, dst_gen);

    noc_async_write(
        reinterpret_cast<uintptr_t>(final_out),
        dst_noc,
        rows_per_core * R * 2 * sizeof(float)
    );
    noc_async_write_barrier();

    cb_pop_front(3, 1);
}