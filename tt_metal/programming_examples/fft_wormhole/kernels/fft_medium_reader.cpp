// =============================================================================
// kernels/fft_medium_reader.cpp
// Tensix DATA-MOVEMENT-0 kernel — Tier 2 Medium FFT  (4K < size ≤ 32K)
//
// Each core handles ceil(batch / active_cores) complete FFTs sequentially.
// One full FFT fits in L1 SRAM (verified by host before dispatch).
//
// CB layout (set up by host):
//   CB0 — input data   (1 page = size * 2 * sizeof(float))
//   CB1 — scratch/ping-pong (same size, used by compute)
//   CB2 — twiddle table (1 page = size * 2 * sizeof(float))
//   CB3 — output data  (1 page = size * 2 * sizeof(float))
//
// Runtime args:
//   [0] = src_buf_addr   — DRAM base of input buffer
//   [1] = tw_buf_addr    — DRAM base of twiddle table
//   [2] = fft_offset     — index of first FFT this core owns
//   [3] = my_batch       — number of FFTs this core processes
//   [4] = size           — N, complex points per FFT
// =============================================================================

#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t src_addr   = get_arg_val<uint32_t>(0);
    uint32_t tw_addr    = get_arg_val<uint32_t>(1);
    uint32_t fft_offset = get_arg_val<uint32_t>(2);
    uint32_t my_batch   = get_arg_val<uint32_t>(3);
    uint32_t size       = get_arg_val<uint32_t>(4);

    const uint32_t fft_bytes    = size * 2 * sizeof(float);
    const uint32_t twiddle_bytes = fft_bytes;

    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size         = 8   // 1 complex float = 8 bytes
    };
    const InterleavedAddrGen<true> tw_gen = {
        .bank_base_address = tw_addr,
        .page_size         = 8
    };

    // -----------------------------------------------------------------------
    // 1. Load twiddle table once into CB2 — stays resident for all my_batch FFTs
    // -----------------------------------------------------------------------
    cb_reserve_back(2, 1);
    uint64_t tw_noc = get_noc_addr(0, tw_gen);
    noc_async_read(tw_noc, get_write_ptr(2), twiddle_bytes);
    noc_async_read_barrier();
    cb_push_back(2, 1);

    // -----------------------------------------------------------------------
    // 2. Stream each FFT input one at a time into CB0
    //    Compute processes it while we pre-fetch the next (natural pipeline).
    // -----------------------------------------------------------------------
    for (uint32_t i = 0; i < my_batch; ++i) {
        cb_reserve_back(0, 1);

        uint32_t abs_fft    = fft_offset + i;
        uint32_t page_start = abs_fft * size * 2;   // complex page index
        uint64_t noc_addr   = get_noc_addr(page_start, src_gen);

        noc_async_read(noc_addr, get_write_ptr(0), fft_bytes);
        noc_async_read_barrier();

        cb_push_back(0, 1);
    }
}