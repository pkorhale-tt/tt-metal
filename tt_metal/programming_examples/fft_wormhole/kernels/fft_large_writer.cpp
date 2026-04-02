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

    for (uint32_t r_local = 0; r_local < rows_per_core; ++r_local) {
        uint32_t abs_row = row_start + r_local;

        for (uint32_t c = 0; c < S; c += rows_per_core) {
            // Determine target core
            uint32_t target_core_id = c / rows_per_core;
            uint32_t target_col     = target_core_id % 8;
            uint32_t target_row     = target_core_id / 8;

            // Get L1 address of CB0 on the target core
            // (Metalium provides get_noc_addr_from_bank_id for L1 targets)
            uint32_t dst_l1_offset =
                abs_row * 2 * sizeof(float) +     // column r maps to row r in transpose
                c * rows_per_core * sizeof(float); // position within the column block

            // NOC address for target core's L1
            uint64_t noc_dst = get_noc_addr(target_col, target_row,
                                            get_write_ptr_of_core(0, target_col, target_row));

            // Source pointer into our phase-1 output CB
            const volatile float* src_ptr =
                phase1_out + r_local * S * 2 + c * 2;

            // Write rows_per_core column elements to target core
            noc_async_write(
                reinterpret_cast<uintptr_t>(src_ptr),
                noc_dst + dst_l1_offset,
                rows_per_core * 2 * sizeof(float)  // rows_per_core complex samples
            );
        }
    }

    // Flush all NOC writes before signalling
    noc_async_write_barrier();

    // Signal each target core's semaphore 0 that we've written our slice
    for (uint32_t target = 0; target < cores_in_grp; ++target) {
        uint32_t tc = target % 8, tr = target / 8;
        uint64_t sem_noc = get_noc_addr(tc, tr, get_semaphore_addr(0, tc, tr));
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
    uint32_t base_page = (batch_idx * N + row_start * R) * 2;
    uint64_t dst_noc   = get_noc_addr(base_page, dst_gen);

    noc_async_write(
        reinterpret_cast<uintptr_t>(final_out),
        dst_noc,
        rows_per_core * R * 2 * sizeof(float)
    );
    noc_async_write_barrier();

    cb_pop_front(3, 1);
}