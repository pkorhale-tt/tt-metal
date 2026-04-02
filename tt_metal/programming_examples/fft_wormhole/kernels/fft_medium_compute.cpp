#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"

constexpr uint32_t LOG2N   = get_compile_time_arg_val(0);
constexpr uint32_t INVERSE = get_compile_time_arg_val(1);

constexpr uint32_t CB_IN  = 0;
constexpr uint32_t CB_SCR = 1;
constexpr uint32_t CB_TW  = 2;
constexpr uint32_t CB_OUT = 3;

inline uint32_t cb_front_ptr(uint32_t cb_id) {
    return get_local_cb_interface(cb_id).fifo_rd_ptr << 4;
}

inline uint32_t cb_back_ptr(uint32_t cb_id) {
    return get_local_cb_interface(cb_id).fifo_wr_ptr << 4;
}

static inline void bit_reverse(volatile float* data, uint32_t n) {
    uint32_t j = 0;
    for (uint32_t i = 1; i < n; ++i) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float re = data[2*i], im = data[2*i+1];
            data[2*i] = data[2*j]; data[2*i+1] = data[2*j+1];
            data[2*j] = re;        data[2*j+1] = im;
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
    bit_reverse(data, n);

    uint32_t stage_len = 2;
    for (uint32_t s = 0; s < log2n; ++s) {
        uint32_t half      = stage_len >> 1;
        uint32_t tw_stride = n / stage_len;

        for (uint32_t k = 0; k < n; k += stage_len) {
            for (uint32_t m = 0; m < half; ++m) {
                uint32_t ei = (k + m) * 2;
                uint32_t oi = (k + m + half) * 2;
                uint32_t wi = (m * tw_stride) * 2;

                float Wre = tw[wi];
                float Wim = inverse ? -tw[wi + 1] : tw[wi + 1];
                float Ore = data[oi];
                float Oim = data[oi + 1];
                float Ere = data[ei];
                float Eim = data[ei + 1];

                float Tre = Wre * Ore - Wim * Oim;
                float Tim = Wre * Oim + Wim * Ore;

                data[ei]     = Ere + Tre;
                data[ei + 1] = Eim + Tim;
                data[oi]     = Ere - Tre;
                data[oi + 1] = Eim - Tim;
            }
        }
        stage_len <<= 1;
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
    uint32_t size     = get_arg_val<uint32_t>(1);
    uint32_t log2n    = get_arg_val<uint32_t>(2);
    uint32_t inv      = get_arg_val<uint32_t>(3);

    for (uint32_t i = 0; i < my_batch; ++i) {
        while (mailbox_read(ckernel::ThreadId::MathThreadId) < i + 1) {
            asm volatile("" ::: "memory");
        }

        const volatile float* tw =
            reinterpret_cast<const volatile float*>(cb_front_ptr(CB_TW));
        volatile float* data =
            reinterpret_cast<volatile float*>(cb_front_ptr(CB_IN));

        radix2_dit(data, tw, size, log2n, static_cast<bool>(inv));
        mailbox_write(ckernel::ThreadId::PackThreadId, i + 1);
    }

#elif COMPILE_FOR_TRISC == 2
    uint32_t my_batch = get_arg_val<uint32_t>(0);
    uint32_t size     = get_arg_val<uint32_t>(1);

    cb_wait_front(CB_TW, 1);

    for (uint32_t i = 0; i < my_batch; ++i) {
        cb_wait_front(CB_IN, 1);
        mailbox_write(ckernel::ThreadId::MathThreadId, i + 1);

        cb_reserve_back(CB_OUT, 1);

        while (mailbox_read(ckernel::ThreadId::PackThreadId) < i + 1) {
            asm volatile("" ::: "memory");
        }

        volatile float* out =
            reinterpret_cast<volatile float*>(cb_back_ptr(CB_OUT));
        const volatile float* data =
            reinterpret_cast<const volatile float*>(cb_front_ptr(CB_IN));

        for (uint32_t j = 0; j < size * 2; ++j) out[j] = data[j];

        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_IN, 1);
    }

    cb_pop_front(CB_TW, 1);
#endif
}