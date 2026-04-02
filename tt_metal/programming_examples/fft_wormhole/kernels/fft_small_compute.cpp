// =============================================================================
// kernels/fft_small_compute.cpp
// Tensix COMPUTE kernel — Tier 1 Small FFT
//
// Runs on: RISCV_2/3/4  (math core)
// Each core computes `my_batch` independent FFTs of length N = 2^log2n.
// All data stays in L1 SRAM — zero DRAM traffic during butterfly computation.
//
// Algorithm: iterative Cooley–Tukey radix-2 DIT (Decimation In Time)
//   1. Bit-reverse permutation of input (in-place)
//   2. log2(N) butterfly stages
//   Each butterfly:  X[k]   = E + W·O
//                    X[k+N/2] = E - W·O
//   where E=even, O=odd sub-array element, W=twiddle factor
//
// Compile args (injected by host via compile_args vector):
//   [0] = log2n
//   [1] = inverse  (0 = forward FFT, 1 = IFFT)
// =============================================================================

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"

// Compile-time constants injected from host
constexpr uint32_t LOG2N   = get_compile_time_arg_val(0);
constexpr uint32_t INVERSE = get_compile_time_arg_val(1);
constexpr uint32_t N       = 1u << LOG2N;

// Runtime args (set per-core by host SetRuntimeArgs)
//   [0] = my_batch
//   [1] = size   (== N, redundant but explicit)
//   [2] = log2n  (== LOG2N, redundant)
//   [3] = inverse

// Circular buffer indices
//   CB0 = input  (my_batch pages, each N complex floats)
//   CB1 = twiddle table (1 page, N complex floats)
//   CB2 = output (my_batch pages)
constexpr uint32_t CB_IN  = 0;
constexpr uint32_t CB_TW  = 1;
constexpr uint32_t CB_OUT = 2;

// ---------------------------------------------------------------------------
// Bit-reverse permutation — standard in-place swap
// Operates on raw L1 pointer: data[i] = {re, im} packed as 2 floats
// ---------------------------------------------------------------------------
static inline void bit_reverse(volatile float* __restrict__ data, uint32_t n) {
    uint32_t j = 0;
    for (uint32_t i = 1; i < n; ++i) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            // swap complex elements at positions i and j
            float re = data[2*i],   im = data[2*i+1];
            data[2*i]   = data[2*j];   data[2*i+1] = data[2*j+1];
            data[2*j]   = re;          data[2*j+1] = im;
        }
    }
}

// ---------------------------------------------------------------------------
// Single radix-2 DIT FFT butterfly stage
//   stage_len = current sub-FFT size (starts at 2, doubles each stage)
//   twiddles  = precomputed W factors for this stage length
// ---------------------------------------------------------------------------
static inline void butterfly_stage(
    volatile float* __restrict__ data,
    const volatile float* __restrict__ tw,
    uint32_t n,
    uint32_t stage_len)
{
    uint32_t half = stage_len >> 1;
    uint32_t tw_stride = n / stage_len;   // step through twiddle table

    for (uint32_t k = 0; k < n; k += stage_len) {
        for (uint32_t m = 0; m < half; ++m) {
            uint32_t e_idx = (k + m) * 2;
            uint32_t o_idx = (k + m + half) * 2;
            uint32_t w_idx = (m * tw_stride) * 2;

            // Twiddle multiply:  T = W * O  (complex multiply)
            float W_re = tw[w_idx],     W_im = tw[w_idx+1];
            float O_re = data[o_idx],   O_im = data[o_idx+1];
            float E_re = data[e_idx],   E_im = data[e_idx+1];

            // For inverse FFT, conjugate the twiddle (W_im = -W_im)
            if constexpr (INVERSE) W_im = -W_im;

            float T_re = W_re * O_re - W_im * O_im;
            float T_im = W_re * O_im + W_im * O_re;

            // Butterfly output
            data[e_idx]   = E_re + T_re;
            data[e_idx+1] = E_im + T_im;
            data[o_idx]   = E_re - T_re;
            data[o_idx+1] = E_im - T_im;
        }
    }
}

#include "api/compute/cb_api.h"

// Polyfills for dataflow-like raw pointer access inside the compute kernel
inline uint32_t get_read_ptr(uint32_t cb_id) {
    return get_tile_address(cb_id, 0);
}

inline uint32_t get_write_ptr(uint32_t cb_id) {
    uint32_t address = 0;
#if COMPILE_FOR_TRISC == 0
    address = get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
    mailbox_write(ckernel::ThreadId::MathThreadId, address);
    mailbox_write(ckernel::ThreadId::PackThreadId, address);
#elif COMPILE_FOR_TRISC == 1
    address = mailbox_read(ckernel::ThreadId::UnpackThreadId);
#elif COMPILE_FOR_TRISC == 2
    address = mailbox_read(ckernel::ThreadId::UnpackThreadId);
#endif
    return address;
}

// ---------------------------------------------------------------------------
// KERNEL ENTRY POINT
// ---------------------------------------------------------------------------
void kernel_main() {
    // Runtime args
    uint32_t my_batch = get_arg_val<uint32_t>(0);
    // size and log2n are known at compile time (LOG2N, N)

    // Acquire twiddle table — it stays in L1 for the entire kernel lifetime
    cb_wait_front(CB_TW, 1);
    const volatile float* tw_ptr =
        reinterpret_cast<const volatile float*>(get_read_ptr(CB_TW));

    // Process each FFT in this core's batch slice
    for (uint32_t fft_i = 0; fft_i < my_batch; ++fft_i) {

        // Wait for reader to deliver one page of input data
        cb_wait_front(CB_IN, 1);
        volatile float* data =
            reinterpret_cast<volatile float*>(get_read_ptr(CB_IN));

        // ---- Step 1: bit-reverse permutation --------------------------------
#if COMPILE_FOR_TRISC == 1
        bit_reverse(data, N);

        // ---- Step 2: butterfly stages  (log2N stages total) ----------------
        uint32_t stage_len = 2;
        for (uint32_t s = 0; s < LOG2N; ++s) {
            butterfly_stage(data, tw_ptr, N, stage_len);
            stage_len <<= 1;
        }

        // signal UNPACK and PACK that math is done
        mailbox_write(ckernel::ThreadId::UnpackThreadId, fft_i + 1);
        mailbox_write(ckernel::ThreadId::PackThreadId, fft_i + 1);
#else
        while(mailbox_read(ckernel::ThreadId::MathThreadId) != fft_i + 1) { /* spin */ }
#endif

        // data now contains the FFT result in L1

        // ---- Step 3: reserve output CB and copy ----------------------------
        cb_reserve_back(CB_OUT, 1);
        volatile float* out_ptr =
            reinterpret_cast<volatile float*>(get_write_ptr(CB_OUT));

        for (uint32_t i = 0; i < N * 2; ++i) {
            out_ptr[i] = data[i];
        }

        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_IN,  1);
    }

    // Release twiddle table
    cb_pop_front(CB_TW, 1);
}