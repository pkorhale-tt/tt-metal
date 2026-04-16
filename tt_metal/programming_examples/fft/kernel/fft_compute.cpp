// fft_compute.cpp — multi-core FFT, correct two-output butterfly
// CB layout:
//   c_0/1 = even_r/i  (lower half of butterfly pairs)
//   c_2/3 = odd_r/i   (upper half of butterfly pairs)
//   c_4/5 = twiddle_r/i
//   c_6/7 = out0_r/i  (even + W*odd)
//   c_8/9 = out1_r/i  (even - W*odd)
//   c_10  = sync (NOC stage signal)
//   c_11-14 = scratch_r/i, tmp_r/i

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

constexpr uint32_t ADD=0, SUB=1, MUL=2, NEG=3;

// CBs
constexpr auto CB_EVEN_R  = tt::CBIndex::c_0;
constexpr auto CB_EVEN_I  = tt::CBIndex::c_1;
constexpr auto CB_ODD_R   = tt::CBIndex::c_2;
constexpr auto CB_ODD_I   = tt::CBIndex::c_3;
constexpr auto CB_TW_R    = tt::CBIndex::c_4;
constexpr auto CB_TW_I    = tt::CBIndex::c_5;
constexpr auto CB_OUT0_R  = tt::CBIndex::c_6;   // even + W*odd
constexpr auto CB_OUT0_I  = tt::CBIndex::c_7;
constexpr auto CB_OUT1_R  = tt::CBIndex::c_8;   // even - W*odd
constexpr auto CB_OUT1_I  = tt::CBIndex::c_9;
constexpr auto CB_SYNC    = tt::CBIndex::c_10;
constexpr auto CB_SCRATCH_R = tt::CBIndex::c_11; // incoming NOC data
constexpr auto CB_SCRATCH_I = tt::CBIndex::c_12;
constexpr auto CB_TMP_R   = tt::CBIndex::c_13;
constexpr auto CB_TMP_I   = tt::CBIndex::c_14;
constexpr auto CB_TW_ODD_R = tt::CBIndex::c_15; // W*odd result
constexpr auto CB_TW_ODD_I = tt::CBIndex::c_16;

constexpr uint32_t num_local_stages = get_compile_time_arg_val(0);
constexpr uint32_t num_noc_stages   = get_compile_time_arg_val(1);
constexpr uint32_t is_ifft          = get_compile_time_arg_val(2);

template <uint32_t OP, bool P1=false, bool P2=false>
FORCE_INLINE void mm_op(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (P1) cb_wait_front(a,1);
    if constexpr (P2) cb_wait_front(b,1);
    tile_regs_acquire();
    if constexpr (OP==ADD){add_tiles_init(a,b);add_tiles(a,b,0,0,0);}
    else if constexpr(OP==SUB){sub_tiles_init(a,b);sub_tiles(a,b,0,0,0);}
    else if constexpr(OP==MUL){mul_tiles_init(a,b);mul_tiles(a,b,0,0,0);}
    tile_regs_commit();
    if constexpr (P1) cb_pop_front(a,1);
    if constexpr (P2) cb_pop_front(b,1);
    cb_reserve_back(out,1); tile_regs_wait(); pack_tile(0,out);
    tile_regs_release(); cb_push_back(out,1);
}

template <bool P=false>
FORCE_INLINE void neg_op(uint32_t in, uint32_t out) {
    if constexpr (P) cb_wait_front(in,1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in); copy_tile(in,0,0);
    negative_tile_init(); negative_tile(0);
    tile_regs_commit();
    if constexpr (P) cb_pop_front(in,1);
    cb_reserve_back(out,1); tile_regs_wait(); pack_tile(0,out);
    tile_regs_release(); cb_push_back(out,1);
}

// Complex multiply: (a+bi)*(c+di) → out_r + out_i*j
FORCE_INLINE void cmul(uint32_t ar,uint32_t ai,uint32_t br,uint32_t bi,
                        uint32_t outr,uint32_t outi) {
    mm_op<MUL>(ar,br,CB_TMP_R); mm_op<MUL>(ai,bi,CB_TMP_I);
    mm_op<SUB,true,true>(CB_TMP_R,CB_TMP_I,outr);
    mm_op<MUL>(ar,bi,CB_TMP_R); mm_op<MUL>(ai,br,CB_TMP_I);
    mm_op<ADD,true,true>(CB_TMP_R,CB_TMP_I,outi);
}

// Full butterfly producing both outputs:
//   out0 = even + W*odd
//   out1 = even - W*odd
FORCE_INLINE void butterfly(
    uint32_t ev_r, uint32_t ev_i,
    uint32_t od_r, uint32_t od_i,
    uint32_t tw_r, uint32_t tw_i)
{
    // Step 1: W*odd → CB_TW_ODD_R/I
    cb_wait_front(od_r,1); cb_wait_front(od_i,1);
    cb_wait_front(tw_r,1); cb_wait_front(tw_i,1);
    cmul(od_r,od_i, tw_r,tw_i, CB_TW_ODD_R,CB_TW_ODD_I);
    cb_pop_front(od_r,1); cb_pop_front(od_i,1);
    cb_pop_front(tw_r,1); cb_pop_front(tw_i,1);

    // Step 2: even + W*odd → out0,  even - W*odd → out1
    cb_wait_front(ev_r,1); cb_wait_front(ev_i,1);
    cb_wait_front(CB_TW_ODD_R,1); cb_wait_front(CB_TW_ODD_I,1);

    mm_op<ADD>(ev_r, CB_TW_ODD_R, CB_OUT0_R);
    mm_op<ADD>(ev_i, CB_TW_ODD_I, CB_OUT0_I);
    mm_op<SUB>(ev_r, CB_TW_ODD_R, CB_OUT1_R);
    mm_op<SUB>(ev_i, CB_TW_ODD_I, CB_OUT1_I);

    cb_pop_front(ev_r,1); cb_pop_front(ev_i,1);
    cb_pop_front(CB_TW_ODD_R,1); cb_pop_front(CB_TW_ODD_I,1);
}

void kernel_main() {
    unary_op_init_common(CB_EVEN_R, CB_OUT0_R);
    copy_tile_to_dst_init_short(CB_EVEN_R);

    for (uint32_t s=0; s<num_local_stages; s++) {
        cb_wait_front(CB_EVEN_R,1); cb_wait_front(CB_EVEN_I,1);
        cb_wait_front(CB_ODD_R,1);  cb_wait_front(CB_ODD_I,1);

        if constexpr (is_ifft) {
            neg_op<true>(CB_TW_I, CB_TMP_R);
            cb_wait_front(CB_TMP_R,1);
            butterfly(CB_EVEN_R,CB_EVEN_I, CB_ODD_R,CB_ODD_I, CB_TW_R,CB_TMP_R);
            cb_pop_front(CB_TMP_R,1);
        } else {
            cb_wait_front(CB_TW_R,1); cb_wait_front(CB_TW_I,1);
            butterfly(CB_EVEN_R,CB_EVEN_I, CB_ODD_R,CB_ODD_I, CB_TW_R,CB_TW_I);
        }
    }

    // NOC stages: use CB_OUT0 as even, CB_SCRATCH as odd
    for (uint32_t s=0; s<num_noc_stages; s++) {
        cb_wait_front(CB_SYNC,1); cb_pop_front(CB_SYNC,1);

        cb_wait_front(CB_OUT0_R,1); cb_wait_front(CB_OUT0_I,1);
        cb_wait_front(CB_SCRATCH_R,1); cb_wait_front(CB_SCRATCH_I,1);

        if constexpr (is_ifft) {
            neg_op<true>(CB_TW_I, CB_TMP_R);
            cb_wait_front(CB_TMP_R,1);
            butterfly(CB_OUT0_R,CB_OUT0_I, CB_SCRATCH_R,CB_SCRATCH_I, CB_TW_R,CB_TMP_R);
            cb_pop_front(CB_TMP_R,1);
        } else {
            cb_wait_front(CB_TW_R,1); cb_wait_front(CB_TW_I,1);
            butterfly(CB_OUT0_R,CB_OUT0_I, CB_SCRATCH_R,CB_SCRATCH_I, CB_TW_R,CB_TW_I);
        }
    }
}