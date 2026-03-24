#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// ---------------------------------------------------------------------------
// Helper: multiply two tiles from CBs, write result into cb_out.
// Neither input CB is popped here — caller is responsible for pops.
// ---------------------------------------------------------------------------
inline void mulIntoCb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_init(cb_a);
    copy_tile(cb_a, 0, 0);
    copy_tile_init(cb_b);
    copy_tile(cb_b, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

inline void addIntoCb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_init(cb_a);
    copy_tile(cb_a, 0, 0);
    copy_tile_init(cb_b);
    copy_tile(cb_b, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

inline void subIntoCb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_init(cb_a);
    copy_tile(cb_a, 0, 0);
    copy_tile_init(cb_b);
    copy_tile(cb_b, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t rows_per_core   = get_arg_val<uint32_t>(2);

    // ── CB index map ────────────────────────────────────────────────────
    //
    // Stage-0 inputs come from the reader (CB 0-3).
    // Stage-1+ inputs come from the writer's inter-stage shuffle (CB 6-9).
    // Twiddle factors always come from reader on CB 4-5.
    // Butterfly outputs always go to writer on CB 16-19.
    // Scratch CBs 20-23 have depth=1 and are private to this kernel.
    //
    constexpr uint32_t cb_stage0_even_r = 0;
    constexpr uint32_t cb_stage0_even_i = 1;
    constexpr uint32_t cb_stage0_odd_r  = 2;
    constexpr uint32_t cb_stage0_odd_i  = 3;

    constexpr uint32_t cb_tw_r          = 4;
    constexpr uint32_t cb_tw_i          = 5;

    constexpr uint32_t cb_next_even_r   = 6;
    constexpr uint32_t cb_next_even_i   = 7;
    constexpr uint32_t cb_next_odd_r    = 8;
    constexpr uint32_t cb_next_odd_i    = 9;

    constexpr uint32_t cb_out0_r        = 16;
    constexpr uint32_t cb_out0_i        = 17;
    constexpr uint32_t cb_out1_r        = 18;
    constexpr uint32_t cb_out1_i        = 19;

    // Depth-1 scratch tiles — must be fully consumed before reuse.
    // CB 12-15 are used by the writer as L1 scratch memory (no push/pop).
    constexpr uint32_t cb_tmp0          = 20;
    constexpr uint32_t cb_tmp1          = 21;
    constexpr uint32_t cb_tmp2          = 22;
    constexpr uint32_t cb_tmp3          = 23;

    if (num_stages == 0 || tiles_per_stage == 0 || rows_per_core == 0) {
        return;
    }

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            // Pick the correct even/odd input CBs for this stage.
            const uint32_t cb_even_r = (stage == 0) ? cb_stage0_even_r : cb_next_even_r;
            const uint32_t cb_even_i = (stage == 0) ? cb_stage0_even_i : cb_next_even_i;
            const uint32_t cb_odd_r  = (stage == 0) ? cb_stage0_odd_r  : cb_next_odd_r;
            const uint32_t cb_odd_i  = (stage == 0) ? cb_stage0_odd_i  : cb_next_odd_i;

            for (uint32_t t = 0; t < tiles_per_stage; ++t) {
                // ----------------------------------------------------------
                // Wait for all inputs this tile needs.
                // ----------------------------------------------------------
                cb_wait_front(cb_even_r, 1);
                cb_wait_front(cb_even_i, 1);
                cb_wait_front(cb_odd_r,  1);
                cb_wait_front(cb_odd_i,  1);
                cb_wait_front(cb_tw_r,   1);
                cb_wait_front(cb_tw_i,   1);

                // ----------------------------------------------------------
                // Compute twiddle × odd  (complex multiply):
                //   tw_real = tw_r * odd_r - tw_i * odd_i   → cb_tmp2
                //   tw_imag = tw_r * odd_i + tw_i * odd_r   → cb_tmp3
                //
                // We use cb_tmp0 / cb_tmp1 as intermediate products and
                // pop them immediately so depth-1 scratch never stalls.
                // ----------------------------------------------------------

                // tw_r * odd_r → tmp0
                mulIntoCb(cb_tw_r, cb_odd_r, cb_tmp0);
                cb_wait_front(cb_tmp0, 1);

                // tw_i * odd_i → tmp1
                mulIntoCb(cb_tw_i, cb_odd_i, cb_tmp1);
                cb_wait_front(cb_tmp1, 1);

                // tmp2 = tmp0 - tmp1  (real part of twiddle*odd)
                subIntoCb(cb_tmp0, cb_tmp1, cb_tmp2);
                cb_pop_front(cb_tmp0, 1);
                cb_pop_front(cb_tmp1, 1);

                // tw_r * odd_i → tmp0
                mulIntoCb(cb_tw_r, cb_odd_i, cb_tmp0);
                cb_wait_front(cb_tmp0, 1);

                // tw_i * odd_r → tmp1
                mulIntoCb(cb_tw_i, cb_odd_r, cb_tmp1);
                cb_wait_front(cb_tmp1, 1);

                // tmp3 = tmp0 + tmp1  (imag part of twiddle*odd)
                addIntoCb(cb_tmp0, cb_tmp1, cb_tmp3);
                cb_pop_front(cb_tmp0, 1);
                cb_pop_front(cb_tmp1, 1);

                // Twiddle and odd inputs fully consumed — pop now.
                cb_pop_front(cb_tw_r,  1);
                cb_pop_front(cb_tw_i,  1);
                cb_pop_front(cb_odd_r, 1);
                cb_pop_front(cb_odd_i, 1);

                // ----------------------------------------------------------
                // Wait for the twiddle products, then compute butterfly:
                //   out0 = even + tw*odd
                //   out1 = even - tw*odd
                // ----------------------------------------------------------
                cb_wait_front(cb_tmp2, 1);
                cb_wait_front(cb_tmp3, 1);

                // out0_r = even_r + tmp2
                addIntoCb(cb_even_r, cb_tmp2, cb_out0_r);
                // out0_i = even_i + tmp3
                addIntoCb(cb_even_i, cb_tmp3, cb_out0_i);
                // out1_r = even_r - tmp2
                subIntoCb(cb_even_r, cb_tmp2, cb_out1_r);
                // out1_i = even_i - tmp3
                subIntoCb(cb_even_i, cb_tmp3, cb_out1_i);

                // Even inputs and scratch fully consumed — pop now.
                cb_pop_front(cb_even_r, 1);
                cb_pop_front(cb_even_i, 1);
                cb_pop_front(cb_tmp2,   1);
                cb_pop_front(cb_tmp3,   1);

                // cb_out0_r/i and cb_out1_r/i are pushed by the helpers
                // above and will be drained by the writer kernel.
            }
        }
    }
}