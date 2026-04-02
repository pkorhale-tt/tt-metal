// =============================================================================
// kernels/fft_large_reader1.cpp
// Tensix DATA-MOVEMENT-0 kernel — Tier 3 Large FFT
//
// Two jobs:
//   1. Load initial row slice + twiddle tables from DRAM → L1 (Phase 1 setup)
//   2. Receive transposed column data from other cores via NOC (Phase 2 setup)
//
// The NOC transpose works as follows:
//   After phase-1 row FFTs, core X owns rows [row_start … row_start+rows_per_core).
//   Each row r has S frequency-domain values; value at column c belongs to the
//   core that owns column c after the R×S matrix transpose.
//   DM1 (writer) on core X sends row r's value at column c to the core owning c.
//   DM0 (this kernel, reader) on core Y receives data sent TO it and assembles
//   the column data for phase-2.
//
// Runtime args:
//   [0] = src_buf_addr    — DRAM base
//   [1] = tw_buf_addr     — DRAM twiddle base
//   [2] = batch_idx       — which FFT in the batch
//   [3] = row_start       — first row this core owns
//   [4] = rows_per_core
//   [5] = R               — row FFT length
//   [6] = S               — col FFT length
//   [7] = inverse
// =============================================================================

#include "dataflow_kernel_api.h"
#include "debug/dprint.h"      // optional, remove for production

void kernel_main() {
    uint32_t src_addr     = get_arg_val<uint32_t>(0);
    uint32_t tw_addr      = get_arg_val<uint32_t>(1);
    uint32_t batch_idx    = get_arg_val<uint32_t>(2);
    uint32_t row_start    = get_arg_val<uint32_t>(3);
    uint32_t rows_per_core = get_arg_val<uint32_t>(4);
    uint32_t R            = get_arg_val<uint32_t>(5);
    uint32_t S            = get_arg_val<uint32_t>(6);

    const uint32_t N          = R * S;
    const uint32_t row_bytes  = S * 2 * sizeof(float);     // one row in bytes
    const uint32_t col_bytes  = R * 2 * sizeof(float);     // one col in bytes
    const uint32_t chunk_bytes = rows_per_core * row_bytes; // this core's data

    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size         = 8
    };
    const InterleavedAddrGen<true> tw_gen = {
        .bank_base_address = tw_addr,
        .page_size         = 8
    };

    // -----------------------------------------------------------------------
    // 1. Load twiddle table for R-point FFT into CB_TW_R (CB1)
    // -----------------------------------------------------------------------
    cb_reserve_back(1, 1);
    uint64_t tw_r_noc = get_noc_addr(0, tw_gen);   // twiddle[0..R-1]
    noc_async_read(tw_r_noc, get_write_ptr(1), R * 2 * sizeof(float));
    noc_async_read_barrier();
    cb_push_back(1, 1);

    // -----------------------------------------------------------------------
    // 2. Load twiddle table for S-point FFT into CB_TW_S (CB2)
    //    These are stored right after the R twiddles in tw_buf
    // -----------------------------------------------------------------------
    cb_reserve_back(2, 1);
    uint32_t tw_s_page  = R * 2;    // S twiddles start at page R*2
    uint64_t tw_s_noc   = get_noc_addr(tw_s_page, tw_gen);
    noc_async_read(tw_s_noc, get_write_ptr(2), S * 2 * sizeof(float));
    noc_async_read_barrier();
    cb_push_back(2, 1);

    // -----------------------------------------------------------------------
    // 3. Load this core's row slice from DRAM into CB_DATA (CB0)
    //    Flat layout: data[batch_idx * N + row_start * S + col]
    //    Page index = (batch_idx * N + row_start * S) * 2  (×2 for complex)
    // -----------------------------------------------------------------------
    cb_reserve_back(0, 1);
    uint32_t base_page  = (batch_idx * N + row_start * S) * 2;
    uint64_t src_noc    = get_noc_addr(base_page, src_gen);
    noc_async_read(src_noc, get_write_ptr(0), chunk_bytes);
    noc_async_read_barrier();
    cb_push_back(0, 1);

    // -----------------------------------------------------------------------
    // 4. Receive transposed column data (Phase 2 setup)
    //    The writer kernel on every peer core will NOC-write column data
    //    addressed directly to our L1.  We pre-reserve the CB so the address
    //    is stable, then wait for all writes to arrive using a semaphore.
    //
    //    Protocol:
    //      • Every core in the group (including us) will write rows_per_core
    //        column slices to this core.  Total expected writes = CORES_PER_FFT.
    //      • A semaphore at a fixed L1 address counts arriving writes.
    //      • We spin until semaphore == CORES_PER_FFT.
    // -----------------------------------------------------------------------
    cb_reserve_back(0, 1);          // reserve the slot for transposed data
    // (writer kernels on remote cores will noc_async_write directly to
    //  get_write_ptr(0) on this core — the address is communicated via
    //  compile-time layout: each core's CB0 L1 address is deterministic
    //  given the core coordinate and the CB config established by the host)

    // Semaphore-based barrier: host pre-initialises semaphore 0 to 0.
    // Each peer writer increments it after finishing its write to us.
    volatile uint32_t* sem = reinterpret_cast<volatile uint32_t*>(
        get_semaphore(0));    // semaphore 0 on this core

    // Wait until all CORES_PER_FFT peers have written their slice
    // (compile-time constant injected; use a local variable here)
    uint32_t cores_per_fft = R / rows_per_core;   // == CORES_PER_FFT
    while (*sem < cores_per_fft) {
        // Spin — NOC writes are in-flight; memory is coherent on Wormhole
        // because the NOC write path goes through L1 directly.
        asm volatile("" ::: "memory");
    }
    // Reset semaphore for reuse
    *sem = 0;

    cb_push_back(0, 1);   // signal compute: transposed data is ready
}