#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t LOG2N   = get_compile_time_arg_val(0);
constexpr uint32_t INVERSE = get_compile_time_arg_val(1);
constexpr uint32_t N       = 1u << LOG2N;

constexpr uint32_t CB_IN  = 0;
constexpr uint32_t CB_TW  = 1;
constexpr uint32_t CB_OUT = 2;

inline uint32_t cb_front_ptr(uint32_t cb_id) {
    return get_local_cb_interface(cb_id).fifo_rd_ptr << 4;
}

inline uint32_t cb_back_ptr(uint32_t cb_id) {
    return get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
}

static inline void bit_reverse(volatile float* __restrict__ data, uint32_t n) {
    uint32_t j = 0;
    for (uint32_t i = 1; i < n; ++i) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float re = data[2*i],   im = data[2*i+1];
            data[2*i]   = data[2*j];   data[2*i+1] = data[2*j+1];
            data[2*j]   = re;          data[2*j+1] = im;
        }
    }
}

static inline void butterfly_stage(
    volatile float* __restrict__ data,
    const volatile float* __restrict__ tw,
    uint32_t n,
    uint32_t stage_len)
{
    uint32_t half = stage_len >> 1;
    uint32_t tw_stride = n / stage_len;

    for (uint32_t k = 0; k < n; k += stage_len) {
        for (uint32_t m = 0; m < half; ++m) {
            uint32_t e_idx = (k + m) * 2;
            uint32_t o_idx = (k + m + half) * 2;
            uint32_t w_idx = (m * tw_stride) * 2;

            float W_re = tw[w_idx];
            float W_im = tw[w_idx + 1];
            float O_re = data[o_idx];
            float O_im = data[o_idx + 1];
            float E_re = data[e_idx];
            float E_im = data[e_idx + 1];

            if constexpr (INVERSE) W_im = -W_im;

            float T_re = W_re * O_re - W_im * O_im;
            float T_im = W_re * O_im + W_im * O_re;

            data[e_idx]     = E_re + T_re;
            data[e_idx + 1] = E_im + T_im;
            data[o_idx]     = E_re - T_re;
            data[o_idx + 1] = E_im - T_im;
        }
    }
}

void kernel_main() {
#if COMPILE_FOR_TRISC == 1
    mailbox_write(ckernel::ThreadId::MathThreadId, 0);
#elif COMPILE_FOR_TRISC == 2
    mailbox_write(ckernel::ThreadId::PackThreadId, 0);
#endif

#if COMPILE_FOR_TRISC == 1
    uint32_t my_batch = get_arg_val<uint32_t>(0);

    for (uint32_t fft_i = 0; fft_i < my_batch; ++fft_i) {
        while (mailbox_read(ckernel::ThreadId::MathThreadId) < fft_i + 1) {
            asm volatile("" ::: "memory");
        }

        const volatile float* tw_ptr =
            reinterpret_cast<const volatile float*>(cb_front_ptr(CB_TW));
        volatile float* data =
            reinterpret_cast<volatile float*>(cb_front_ptr(CB_IN));

        bit_reverse(data, N);

        uint32_t stage_len = 2;
        for (uint32_t s = 0; s < LOG2N; ++s) {
            butterfly_stage(data, tw_ptr, N, stage_len);
            stage_len <<= 1;
        }

        mailbox_write(ckernel::ThreadId::PackThreadId, fft_i + 1);
    }

#elif COMPILE_FOR_TRISC == 2
    uint32_t my_batch = get_arg_val<uint32_t>(0);

    cb_wait_front(CB_TW, 1);

    for (uint32_t fft_i = 0; fft_i < my_batch; ++fft_i) {
        cb_wait_front(CB_IN, 1);
        mailbox_write(ckernel::ThreadId::MathThreadId, fft_i + 1);

        cb_reserve_back(CB_OUT, 1);

        while (mailbox_read(ckernel::ThreadId::PackThreadId) < fft_i + 1) {
            asm volatile("" ::: "memory");
        }

        volatile float* out_ptr =
            reinterpret_cast<volatile float*>(cb_back_ptr(CB_OUT));
        const volatile float* data =
            reinterpret_cast<const volatile float*>(cb_front_ptr(CB_IN));

        for (uint32_t i = 0; i < N * 2; ++i) {
            out_ptr[i] = data[i];
        }

        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_IN, 1);
    }

    cb_pop_front(CB_TW, 1);
#endif
}