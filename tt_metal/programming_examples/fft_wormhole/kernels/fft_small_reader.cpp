// =============================================================================
// kernels/fft_small_reader.cpp
// Tensix DATA-MOVEMENT-0 kernel — Tier 1 Small FFT
//
// Responsibilities:
//   1. Load the twiddle table for this FFT size from DRAM → CB1 (once)
//   2. Load each FFT's input data from DRAM → CB0  (my_batch times)
//
// Runtime args (set by host per-core):
//   [0] = src_buf_addr    — DRAM base address of input buffer
//   [1] = tw_buf_addr     — DRAM base address of twiddle table
//   [2] = fft_offset      — index of first FFT this core owns (in [0, batch))
//   [3] = my_batch        — number of FFTs this core processes
//   [4] = size            — N, number of complex points per FFT
// =============================================================================

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ---- Runtime args -------------------------------------------------------
    uint32_t src_addr   = get_arg_val<uint32_t>(0);
    uint32_t tw_addr    = get_arg_val<uint32_t>(1);
    uint32_t fft_offset = get_arg_val<uint32_t>(2);
    uint32_t my_batch   = get_arg_val<uint32_t>(3);
    uint32_t size       = get_arg_val<uint32_t>(4);

    const uint32_t fft_bytes    = size * 2 * sizeof(float);  // N complex floats
    const uint32_t twiddle_bytes = fft_bytes;                 // same layout

    // NOC address helpers — src is an interleaved DRAM buffer
    // We use noc_async_read which takes (src_noc_addr, dst_l1_addr, bytes)
    const InterleavedAddrGen<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size         = 8   // 1 complex float = 8 bytes page granularity
    };
    const InterleavedAddrGen<true> tw_gen = {
        .bank_base_address = tw_addr,
        .page_size         = 8
    };

    // ---- 1. Load twiddle table (once) into CB1 ----------------------------
    cb_reserve_back(1, 1);
    uint32_t tw_l1 = get_write_ptr(1);

    // Read the entire twiddle table in one contiguous NOC transfer
    uint64_t tw_noc_addr = get_noc_addr(0, tw_gen);  // page 0 = start of table
    noc_async_read(tw_noc_addr, tw_l1, twiddle_bytes);
    noc_async_read_barrier();     // wait for transfer to complete

    cb_push_back(1, 1);           // signal compute: twiddles ready

    // ---- 2. Stream each FFT's input data ----------------------------------
    for (uint32_t i = 0; i < my_batch; ++i) {
        cb_reserve_back(0, 1);
        uint32_t in_l1 = get_write_ptr(0);

        // Absolute FFT index in the batch
        uint32_t abs_fft = fft_offset + i;

        // Byte offset in the flat DRAM buffer: abs_fft * fft_bytes
        // Page index for InterleavedAddrGen: abs_fft * size * 2  pages
        //   (each page = 8 bytes = 1 complex sample)
        uint32_t page_start = abs_fft * size;  // page index of first sample
        uint64_t noc_addr   = get_noc_addr(page_start, src_gen);

        noc_async_read(noc_addr, in_l1, fft_bytes);
        noc_async_read_barrier();

        cb_push_back(0, 1);   // signal compute: input page ready
    }
}