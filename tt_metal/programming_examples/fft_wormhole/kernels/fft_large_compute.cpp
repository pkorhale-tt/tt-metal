// =============================================================================
// kernels/fft_large_compute.cpp
// Tensix COMPUTE kernel — Tier 3 Large FFT
//
// Implements 2D Cooley–Tukey FFT decomposition across multiple cores.
//
// Algorithm overview (N = R × S, R=2^log2R, S=2^log2S):
//
//   Each core owns [rows_per_core] rows of the N-point input.
//
//   PHASE 1 — Row FFTs:
//     Each core independently computes S-point FFTs on its row slice.
//     Pure SRAM operation, no NOC.
//
//   TRANSPOSE via NOC:
//     DM1 (writer kernel) multicasts each row to the core that owns the
//     corresponding column after transpose.  The transpose is handled in
//     the reader kernel (fft_large_reader1) on the receiving side.
//
//   PHASE 2 — Column FFTs + twiddle multiplication:
//     Each core receives transposed data and computes R-point FFTs on
//     its column slice, applying the mixed-radix twiddle factors.
//
//   The host signals phase transition via a semaphore written over NOC.
//
// Compile args:
//   [0] = log2R          — size of row FFT  (= log2(R))
//   [1] = log2S          — size of col FFT  (= log2(S))
//   [2] = inverse        — 0=FFT, 1=IFFT
//   [3] = cores_per_fft  — cores cooperating on this FFT
//   [4] = rows_per_core  — rows / cols each core is responsible for
// =============================================================================

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t LOG2R          = get_compile_time_arg_val(0);
constexpr uint32_t LOG2S          = get_compile_time_arg_val(1);
constexpr uint32_t INVERSE        = get_compile_time_arg_val(2);
constexpr uint32_t CORES_PER_FFT  = get_compile_time_arg_val(3);
constexpr uint32_t ROWS_PER_CORE  = get_compile_time_arg_val(4);

constexpr uint32_t R_LEN = 1u << LOG2R;   // row FFT length  (= sqrt(N))
constexpr uint32_t S_LEN = 1u << LOG2S;   // col FFT length  (= N/R)

// CB indices
constexpr uint32_t CB_DATA  = 0;   // input / working data (rows_per_core rows × S cols)
constexpr uint32_t CB_TW_R  = 1;   // twiddle table for R-point FFTs
constexpr uint32_t CB_TW_S  = 2;   // twiddle table for S-point FFTs
constexpr uint32_t CB_OUT   = 3;   // output rows

// Runtime args:
//   [0] = row_start      — first row this core owns (0-indexed in the R×S matrix)
//   [1] = rows_per_core  — (same as compile arg but explicit for clarity)
//   [2] = R
//   [3] = S
//   [4] = local_core_id  — position within the cores_per_fft group (0-based)
//   [5] = batch_idx      — which FFT in the batch this group handles

// -------------------------------------------------------------------------
// Radix-2 DIT FFT  (same logic as small kernel, but parametrised at runtime)
// -------------------------------------------------------------------------
#if COMPILE_FOR_TRISC == 1
static void bit_reverse_inplace(volatile float* data, uint32_t n) {
    uint32_t j = 0;
    for (uint32_t i = 1; i < n; ++i) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float re = data[2*i], im = data[2*i+1];
            data[2*i]   = data[2*j];   data[2*i+1] = data[2*j+1];
            data[2*j]   = re;          data[2*j+1] = im;
        }
    }
}

static void radix2_dit(
    volatile float* data,
    const volatile float* tw,
    uint32_t n,
    uint32_t log2n,
    bool inverse)
{
    bit_reverse_inplace(data, n);
    uint32_t stage_len = 2;
    for (uint32_t s = 0; s < log2n; ++s) {
        uint32_t half       = stage_len >> 1;
        uint32_t tw_stride  = n / stage_len;

        for (uint32_t k = 0; k < n; k += stage_len) {
            for (uint32_t m = 0; m < half; ++m) {
                uint32_t ei = (k+m)*2, oi = (k+m+half)*2, wi = (m*tw_stride)*2;

                float Wre = tw[wi], Wim = inverse ? -tw[wi+1] : tw[wi+1];
                float Ore = data[oi], Oim = data[oi+1];
                float Ere = data[ei], Eim = data[ei+1];

                float Tre = Wre*Ore - Wim*Oim;
                float Tim = Wre*Oim + Wim*Ore;

                data[ei]   = Ere + Tre;   data[ei+1] = Eim + Tim;
                data[oi]   = Ere - Tre;   data[oi+1] = Eim - Tim;
            }
        }
        stage_len <<= 1;
    }
}

// -------------------------------------------------------------------------
// Mixed-radix twiddle multiply  W_N^(r·s_col)  applied to transposed data.
// W_N^{r*s} = exp(-2πi·r·s / N)  where N = R*S.
//
// Rather than calling __builtin_cos/__builtin_sin for every element (slow
// on Tensix RISC-V), we derive W_step = exp(-2πi·s_col/N) from the
// pre-loaded twiddle tables and accumulate by sequential complex multiply:
//   W^0 = (1, 0)
//   W^r = W^{r-1} * W_step
//
// W_step itself equals tw_s[s_col * R / N] only when s_col is a multiple
// of (N/S) = R.  For the general case we construct it from tw_r and tw_s:
//   exp(-2πi·s_col/N) = exp(-2πi·s_col/(R*S))
// which is NOT directly available as a single entry in either table.
// We therefore keep one cos/sin call — but only ONCE per column, not once
// per element, reducing the cost from O(R) transcendentals to O(1).
// -------------------------------------------------------------------------
static void apply_mixed_twiddles(
    volatile float* data,           // R-point column buffer after transpose
    uint32_t s_col,                 // column index in [0, S_LEN)
    uint32_t R_val,
    uint32_t N)
{
    // One transcendental pair per column (not per element).
    double angle = -2.0 * 3.14159265358979323846 * static_cast<double>(s_col)
                   / static_cast<double>(N);
    float wstep_re = static_cast<float>(__builtin_cos(angle));
    float wstep_im = static_cast<float>(__builtin_sin(angle));

    // Accumulator starts at W^0 = (1, 0)
    float wacc_re = 1.0f, wacc_im = 0.0f;

    for (uint32_t r = 0; r < R_val; ++r) {
        float re = data[2*r], im = data[2*r+1];
        data[2*r]   = re * wacc_re - im * wacc_im;
        data[2*r+1] = re * wacc_im + im * wacc_re;

        // Advance accumulator: wacc *= wstep  (complex multiply)
        float new_re = wacc_re * wstep_re - wacc_im * wstep_im;
        float new_im = wacc_re * wstep_im + wacc_im * wstep_re;
        wacc_re = new_re;
        wacc_im = new_im;
    }
}
#endif

#include "api/compute/cb_api.h"

inline uint32_t get_read_ptr(uint32_t cb_id) {
    return get_tile_address(cb_id, 0);
}


// =========================================================================
// KERNEL ENTRY POINT
//
// TRISC thread responsibilities:
//   TRISC 0 (UNPACK) — exits immediately; all DRAM→L1 is done by the
//                       DATA-MOVEMENT-0 reader kernel.
//   TRISC 1 (MATH)   — no CB calls.  Waits for PACK to signal readiness
//                       via MathThreadId mailbox, runs FFT math, signals
//                       PACK via PackThreadId.
//   TRISC 2 (PACK)   — owns ALL cb_wait_front / cb_reserve_back /
//                       cb_push_back / cb_pop_front calls.  Coordinates
//                       MATH via mailbox and handles data copies.
// =========================================================================
void kernel_main() {
#if COMPILE_FOR_TRISC == 1
    // Clear stale mailbox values from any prior kernel run.
    mailbox_write(ckernel::ThreadId::MathThreadId, 0);

    // -----------------------------------------------------------------------
    // MATH thread
    // -----------------------------------------------------------------------
    uint32_t row_start     = get_arg_val<uint32_t>(0);
    uint32_t rows_per_core = get_arg_val<uint32_t>(1);

    // PACK holds the twiddle CBs open; read their pointers here after PACK
    // has signalled phase-1-ready (MathThreadId >= 1).
    const volatile float* tw_r = nullptr;
    const volatile float* tw_s = nullptr;

    // --- Phase 1: row FFTs ---
    // Wait for PACK to confirm twiddles + data are in L1.
    while (mailbox_read(ckernel::ThreadId::MathThreadId) < 1) {
        asm volatile("" ::: "memory");
    }
    tw_s = reinterpret_cast<const volatile float*>(get_read_ptr(CB_TW_S));
    volatile float* data = reinterpret_cast<volatile float*>(get_read_ptr(CB_DATA));

    for (uint32_t r = 0; r < rows_per_core; ++r) {
        volatile float* row_ptr = data + r * S_LEN * 2;
        radix2_dit(row_ptr, tw_s, S_LEN, LOG2S, (bool)INVERSE);
    }
    // Tell PACK phase-1 math is done.
    mailbox_write(ckernel::ThreadId::PackThreadId, 1);

    // --- Phase 2: column FFTs ---
    // Wait for PACK to push transposed data into CB_DATA and signal us.
    while (mailbox_read(ckernel::ThreadId::MathThreadId) < 2) {
        asm volatile("" ::: "memory");
    }
    tw_r     = reinterpret_cast<const volatile float*>(get_read_ptr(CB_TW_R));
    // Phase-2 output goes directly to the CB_OUT write region.  PACK has
    // reserved it and passes its address via PackThreadId mailbox.
    volatile float* col_data   = reinterpret_cast<volatile float*>(get_read_ptr(CB_DATA));
    volatile float* col_out_ptr = reinterpret_cast<volatile float*>(
        static_cast<uintptr_t>(mailbox_read(ckernel::ThreadId::PackThreadId)));

    // Un-interleave: received layout is [R][rows_per_core]; output is [rows_per_core][R]
    for (uint32_t r = 0; r < R_LEN; ++r) {
        for (uint32_t c = 0; c < rows_per_core; ++c) {
            uint32_t src_idx = (r * rows_per_core + c) * 2;
            uint32_t dst_idx = (c * R_LEN + r) * 2;
            col_out_ptr[dst_idx]     = col_data[src_idx];
            col_out_ptr[dst_idx + 1] = col_data[src_idx + 1];
        }
    }

    const uint32_t N = R_LEN * S_LEN;
    for (uint32_t c = 0; c < rows_per_core; ++c) {
        volatile float* col_ptr = col_out_ptr + c * R_LEN * 2;
        uint32_t s_col = row_start + c;
        apply_mixed_twiddles(col_ptr, s_col, R_LEN, N);
        radix2_dit(col_ptr, tw_r, R_LEN, LOG2R, (bool)INVERSE);
    }
    // Signal PACK that phase-2 is done.
    mailbox_write(ckernel::ThreadId::PackThreadId, 2);

#elif COMPILE_FOR_TRISC == 2
    // -----------------------------------------------------------------------
    // PACK thread — owns all CB operations.
    // -----------------------------------------------------------------------
    mailbox_write(ckernel::ThreadId::PackThreadId, 0);

    uint32_t rows_per_core = get_arg_val<uint32_t>(1);

    // Wait for reader to deliver twiddles and initial row data.
    cb_wait_front(CB_TW_R, 1);
    cb_wait_front(CB_TW_S, 1);
    cb_wait_front(CB_DATA, 1);

    // Tell MATH: twiddles + data are ready (phase-1 start).
    mailbox_write(ckernel::ThreadId::MathThreadId, 1);

    // Wait for MATH to finish phase-1 row FFTs.
    while (mailbox_read(ckernel::ThreadId::PackThreadId) < 1) {
        asm volatile("" ::: "memory");
    }

    // Copy phase-1 results to CB_OUT for the writer to scatter via NOC.
    cb_reserve_back(CB_OUT, 1);
    volatile float* out_ptr =
        reinterpret_cast<volatile float*>(
            get_local_cb_interface(CB_OUT).fifo_wr_ptr << 4);
    const volatile float* data =
        reinterpret_cast<const volatile float*>(get_read_ptr(CB_DATA));

    for (uint32_t i = 0; i < rows_per_core * S_LEN * 2; ++i) {
        out_ptr[i] = data[i];
    }
    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_DATA, 1);   // release phase-1 input

    // Wait for the transposed column data that the NOC scatter delivers.
    cb_wait_front(CB_DATA, 1);

    // Reserve phase-2 output slot and pass its L1 address to MATH via mailbox.
    cb_reserve_back(CB_OUT, 1);
    uint32_t col_out_addr =
        get_local_cb_interface(CB_OUT).fifo_wr_ptr << 4;

    // Tell MATH: transposed data is in CB_DATA, output slot is at col_out_addr.
    // We encode the address in PackThreadId so MATH can derive the write pointer.
    // First bump MathThreadId to 2 to release MATH's phase-2 wait, then
    // write the address into PackThreadId (MATH reads it after unblocking).
    mailbox_write(ckernel::ThreadId::PackThreadId, col_out_addr);
    mailbox_write(ckernel::ThreadId::MathThreadId, 2);

    // Wait for MATH to finish phase-2 column FFTs.
    while (mailbox_read(ckernel::ThreadId::PackThreadId) < 2) {
        asm volatile("" ::: "memory");
    }

    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_DATA, 1);

    // Release twiddle tables.
    cb_pop_front(CB_TW_R, 1);
    cb_pop_front(CB_TW_S, 1);

#endif
    // TRISC 0 (UNPACK): nothing to do.
}