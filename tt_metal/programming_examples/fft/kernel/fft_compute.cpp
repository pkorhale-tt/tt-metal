// fft_compute.cpp — multi-core 1D FFT compute kernel
// Includes match existing fft_float32 kernels in this repo.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

constexpr uint32_t ADD = 0;
constexpr uint32_t SUB = 1;
constexpr uint32_t MUL = 2;
constexpr uint32_t NEG = 3;

// CB indices
constexpr auto CB_LHS_R     = tt::CBIndex::c_0;
constexpr auto CB_LHS_I     = tt::CBIndex::c_1;
constexpr auto CB_RHS_R     = tt::CBIndex::c_2;
constexpr auto CB_RHS_I     = tt::CBIndex::c_3;
constexpr auto CB_TWIDDLE_R = tt::CBIndex::c_4;
constexpr auto CB_TWIDDLE_I = tt::CBIndex::c_5;
constexpr auto CB_OUT_R     = tt::CBIndex::c_6;
constexpr auto CB_OUT_I     = tt::CBIndex::c_7;
constexpr auto CB_SCRATCH_R = tt::CBIndex::c_8;
constexpr auto CB_SCRATCH_I = tt::CBIndex::c_9;
constexpr auto CB_SYNC      = tt::CBIndex::c_10;
constexpr auto CB_TMP_R     = tt::CBIndex::c_11;
constexpr auto CB_TMP_I     = tt::CBIndex::c_12;
constexpr auto CB_WR_R      = tt::CBIndex::c_13;
constexpr auto CB_WR_I      = tt::CBIndex::c_14;

constexpr uint32_t num_local_stages = get_compile_time_arg_val(0);
constexpr uint32_t num_noc_stages   = get_compile_time_arg_val(1);
constexpr uint32_t is_ifft          = get_compile_time_arg_val(2);
constexpr uint32_t total_N          = get_compile_time_arg_val(3);

// Binary FPU op (matrix unit)
template <uint32_t OP, bool POP1=false, bool POP2=false>
FORCE_INLINE void mm_op(uint32_t a, uint32_t b, uint32_t out) {
    if constexpr (POP1) cb_wait_front(a, 1);
    if constexpr (POP2) cb_wait_front(b, 1);
    tile_regs_acquire();
    if constexpr (OP==ADD) { add_tiles_init(a,b); add_tiles(a,b,0,0,0); }
    else if constexpr (OP==SUB) { sub_tiles_init(a,b); sub_tiles(a,b,0,0,0); }
    else if constexpr (OP==MUL) { mul_tiles_init(a,b); mul_tiles(a,b,0,0,0); }
    tile_regs_commit();
    if constexpr (POP1) cb_pop_front(a,1);
    if constexpr (POP2) cb_pop_front(b,1);
    cb_reserve_back(out,1); tile_regs_wait(); pack_tile(0,out); tile_regs_release(); cb_push_back(out,1);
}

// Negate via SFPU
template <bool POP=false>
FORCE_INLINE void neg_op(uint32_t in, uint32_t out) {
    if constexpr (POP) cb_wait_front(in,1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in); copy_tile(in,0,0);
    negative_tile_init(); negative_tile(0);
    tile_regs_commit();
    if constexpr (POP) cb_pop_front(in,1);
    cb_reserve_back(out,1); tile_regs_wait(); pack_tile(0,out); tile_regs_release(); cb_push_back(out,1);
}

// Complex multiply: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
FORCE_INLINE void cmul(uint32_t ar, uint32_t ai, uint32_t br, uint32_t bi,
                        uint32_t outr, uint32_t outi) {
    mm_op<MUL>(ar,br,CB_TMP_R);  mm_op<MUL>(ai,bi,CB_TMP_I);
    mm_op<SUB,true,true>(CB_TMP_R,CB_TMP_I,outr);
    mm_op<MUL>(ar,bi,CB_TMP_R);  mm_op<MUL>(ai,br,CB_TMP_I);
    mm_op<ADD,true,true>(CB_TMP_R,CB_TMP_I,outi);
}

// Butterfly: out = even + W*odd  (result in CB_OUT_R/I)
FORCE_INLINE void butterfly(
    uint32_t er, uint32_t ei,   // even (LHS)
    uint32_t or_, uint32_t oi,  // odd  (RHS)
    uint32_t wr, uint32_t wi)   // twiddle
{
    cb_wait_front(or_,1); cb_wait_front(oi,1);
    cb_wait_front(wr,1);  cb_wait_front(wi,1);
    cmul(or_,oi,wr,wi, CB_WR_R,CB_WR_I);
    cb_pop_front(or_,1); cb_pop_front(oi,1);
    cb_pop_front(wr,1);  cb_pop_front(wi,1);

    cb_wait_front(er,1); cb_wait_front(ei,1);
    cb_wait_front(CB_WR_R,1); cb_wait_front(CB_WR_I,1);
    mm_op<ADD>(er,CB_WR_R, CB_OUT_R);
    mm_op<ADD>(ei,CB_WR_I, CB_OUT_I);
    cb_pop_front(er,1); cb_pop_front(ei,1);
    cb_pop_front(CB_WR_R,1); cb_pop_front(CB_WR_I,1);
}

void kernel_main() {
    unary_op_init_common(CB_LHS_R, CB_OUT_R);
    copy_tile_to_dst_init_short(CB_LHS_R);

    // Local stages
    for (uint32_t s=0; s<num_local_stages; s++) {
        cb_wait_front(CB_LHS_R,1); cb_wait_front(CB_LHS_I,1);
        cb_wait_front(CB_RHS_R,1); cb_wait_front(CB_RHS_I,1);
        cb_wait_front(CB_TWIDDLE_R,1); cb_wait_front(CB_TWIDDLE_I,1);

        if constexpr (is_ifft) {
            // Conjugate twiddle for IFFT
            neg_op<true>(CB_TWIDDLE_I, CB_TMP_R);
            cb_wait_front(CB_TMP_R,1);
            butterfly(CB_LHS_R,CB_LHS_I, CB_RHS_R,CB_RHS_I, CB_TWIDDLE_R,CB_TMP_R);
            cb_pop_front(CB_TMP_R,1);
        } else {
            butterfly(CB_LHS_R,CB_LHS_I, CB_RHS_R,CB_RHS_I, CB_TWIDDLE_R,CB_TWIDDLE_I);
        }
        cb_pop_front(CB_TWIDDLE_R,1);
        if constexpr (!is_ifft) cb_pop_front(CB_TWIDDLE_I,1);
    }

    // NOC stages: wait for CB_SYNC, use SCRATCH as RHS
    for (uint32_t s=0; s<num_noc_stages; s++) {
        cb_wait_front(CB_SYNC,1); cb_pop_front(CB_SYNC,1);

        cb_wait_front(CB_OUT_R,1);   cb_wait_front(CB_OUT_I,1);
        cb_wait_front(CB_SCRATCH_R,1); cb_wait_front(CB_SCRATCH_I,1);
        cb_wait_front(CB_TWIDDLE_R,1); cb_wait_front(CB_TWIDDLE_I,1);

        if constexpr (is_ifft) {
            neg_op<true>(CB_TWIDDLE_I, CB_TMP_R);
            cb_wait_front(CB_TMP_R,1);
            butterfly(CB_OUT_R,CB_OUT_I, CB_SCRATCH_R,CB_SCRATCH_I, CB_TWIDDLE_R,CB_TMP_R);
            cb_pop_front(CB_TMP_R,1);
        } else {
            butterfly(CB_OUT_R,CB_OUT_I, CB_SCRATCH_R,CB_SCRATCH_I, CB_TWIDDLE_R,CB_TWIDDLE_I);
        }
        cb_pop_front(CB_TWIDDLE_R,1);
        if constexpr (!is_ifft) cb_pop_front(CB_TWIDDLE_I,1);
    }
}