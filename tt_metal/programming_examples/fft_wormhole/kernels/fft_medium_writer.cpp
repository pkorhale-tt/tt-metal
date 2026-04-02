// =============================================================================
// kernels/fft_medium_writer.cpp
// Tensix DATA-MOVEMENT-1 kernel — Tier 2 Medium FFT  (4K < size ≤ 32K)
//
// Waits for compute to push completed FFT pages into CB3, then writes
// each page back to the correct location in the DRAM output buffer.
//
// Runtime args:
//   [0] = dst_buf_addr   — DRAM base of output buffer
//   [1] = fft_offset     — index of first FFT this core owns
//   [2] = my_batch       — number of FFTs this core writes back
//   [3] = size           — N, complex points per FFT
// =============================================================================

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t fft_offset = get_arg_val<uint32_t>(1);
    uint32_t my_batch   = get_arg_val<uint32_t>(2);
    uint32_t size       = get_arg_val<uint32_t>(3);

    const uint32_t fft_bytes = size * 2 * sizeof(float);

    const InterleavedAddrGen<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size         = 8   // 1 complex float = 8 bytes
    };

    for (uint32_t i = 0; i < my_batch; ++i) {
        // Wait for one completed FFT from compute
        cb_wait_front(3, 1);
        uint32_t out_l1 = get_read_ptr(3);

        uint32_t abs_fft    = fft_offset + i;
        uint32_t page_start = abs_fft * size;
        uint64_t noc_addr   = get_noc_addr(page_start, dst_gen);

        noc_async_write(out_l1, noc_addr, fft_bytes);
        noc_async_write_barrier();

        cb_pop_front(3, 1);
    }
}