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
// Mixed-radix twiddle multiply  W_n^(r·s) applied to transposed data
// W_n^{r*s} = exp(-2πi·r·s / N)  where N = R*S
// -------------------------------------------------------------------------
static void apply_mixed_twiddles(
    volatile float* data,   // column data after transpose  [R points]
    uint32_t s_col,         // column index in the S dimension
    uint32_t R_val,
    uint32_t N)
{
    for (uint32_t r = 0; r < R_val; ++r) {
        double angle = -2.0 * 3.14159265358979323846 * (double)r * (double)s_col / (double)N;
        float wre = static_cast<float>(__builtin_cos(angle));
        float wim = static_cast<float>(__builtin_sin(angle));

        float re = data[2*r], im = data[2*r+1];
        data[2*r]   = re*wre - im*wim;
        data[2*r+1] = re*wim + im*wre;
    }
}

#include "api/compute/cb_api.h"

inline uint32_t get_read_ptr(uint32_t cb_id) {
    return get_tile_address(cb_id, 0);
}
inline uint32_t get_write_ptr(uint32_t cb_id) {
    uint32_t address = 0;
    PACK({
        address = get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
        mailbox_write(ckernel::ThreadId::MathThreadId, address);
        mailbox_write(ckernel::ThreadId::UnpackThreadId, address);
    })
    MATH(address = mailbox_read(ckernel::ThreadId::PackThreadId);)
    UNPACK(address = mailbox_read(ckernel::ThreadId::PackThreadId);)
    return address;
}

// =========================================================================
// KERNEL ENTRY POINT
// =========================================================================
void kernel_main() {
    uint32_t row_start     = get_arg_val<uint32_t>(0);
    uint32_t rows_per_core = get_arg_val<uint32_t>(1);
    // R, S from compile-time constants

    // Wait for reader to deliver: data block + both twiddle tables
    cb_wait_front(CB_TW_R, 1);
    cb_wait_front(CB_TW_S, 1);
    cb_wait_front(CB_DATA, 1);

    volatile float* data   = reinterpret_cast<volatile float*>(get_read_ptr(CB_DATA));
    const volatile float* tw_r = reinterpret_cast<const volatile float*>(get_read_ptr(CB_TW_R));
    const volatile float* tw_s = reinterpret_cast<const volatile float*>(get_read_ptr(CB_TW_S));

    // -----------------------------------------------------------------------
    // PHASE 1: Row FFTs  (S-point FFT on each of my rows_per_core rows)
    // Each row is stored contiguously: data[row * S_LEN * 2 ... (row+1)*S_LEN*2 - 1]
    // -----------------------------------------------------------------------
    for (uint32_t r = 0; r < rows_per_core; ++r) {
        volatile float* row_ptr = data + r * S_LEN * 2;
        radix2_dit(row_ptr, tw_s, S_LEN, LOG2S, (bool)INVERSE);
    }

    // Signal DM1 (writer) that phase-1 rows are ready for NOC transpose
    // Writer will multicast each row to the appropriate target core.
    // The CB push acts as the producer signal.
    cb_reserve_back(CB_OUT, 1);
    volatile float* out_ptr = reinterpret_cast<volatile float*>(get_write_ptr(CB_OUT));

    // Copy phase-1 results to output CB for writer to transmit
    for (uint32_t i = 0; i < rows_per_core * S_LEN * 2; ++i) {
        out_ptr[i] = data[i];
    }
    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_DATA, 1);

    // -----------------------------------------------------------------------
    // Wait for transposed data to arrive from other cores via NOC
    // (Reader kernel fft_large_reader1 handles the receive side and places
    //  transposed column data back into CB_DATA)
    // -----------------------------------------------------------------------
    cb_wait_front(CB_DATA, 1);
    volatile float* col_data = reinterpret_cast<volatile float*>(get_read_ptr(CB_DATA));

    // -----------------------------------------------------------------------
    // PHASE 2: Apply mixed-radix twiddles, then R-point column FFTs
    // After transpose, col_data holds [rows_per_core] columns of R_LEN points each.
    // -----------------------------------------------------------------------
    uint32_t N = R_LEN * S_LEN;
    for (uint32_t c = 0; c < rows_per_core; ++c) {
        volatile float* col_ptr = col_data + c * R_LEN * 2;
        uint32_t s_col = row_start + c;   // actual column index in S_LEN dimension

        apply_mixed_twiddles(col_ptr, s_col, R_LEN, N);
        radix2_dit(col_ptr, tw_r, R_LEN, LOG2R, (bool)INVERSE);
    }

    // Push final output
    cb_reserve_back(CB_OUT, 1);
    out_ptr = reinterpret_cast<volatile float*>(get_write_ptr(CB_OUT));
    for (uint32_t i = 0; i < rows_per_core * R_LEN * 2; ++i) {
        out_ptr[i] = col_data[i];
    }
    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_DATA, 1);

    // Release twiddle tables
    cb_pop_front(CB_TW_R, 1);
    cb_pop_front(CB_TW_S, 1);
}