// fft_compute.cpp — Cooley-Tukey butterfly, all stages
// Pipeline per local stage:
//   Reader pushes: CB_RHS_R, CB_RHS_I, CB_TWIDDLE_R, CB_TWIDDLE_I
//   LHS comes from CB_LHS_R/I (initial load) or recycled from CB_OUT
//   Compute reads LHS+RHS+TW, writes CB_OUT
//   Writer pops CB_OUT, copies back to CB_LHS for next stage
// Pipeline per NOC stage:
//   Writer signals CB_SYNC after peer exchange
//   Compute reads CB_OUT(LHS) + CB_SCRATCH(RHS) + CB_TWIDDLE

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

constexpr uint32_t ADD = 0, SUB = 1, MUL = 2, NEG = 3;

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

// Complex multiply (a+bi)*(c+di) → (ac-bd) + (ad+bc)i
FORCE_INLINE void cmul(uint32_t ar,uint32_t ai,uint32_t br,uint32_t bi,
                        uint32_t outr,uint32_t outi) {
    mm_op<MUL>(ar,br,CB_TMP_R); mm_op<MUL>(ai,bi,CB_TMP_I);
    mm_op<SUB,true,true>(CB_TMP_R,CB_TMP_I,outr);
    mm_op<MUL>(ar,bi,CB_TMP_R); mm_op<MUL>(ai,br,CB_TMP_I);
    mm_op<ADD,true,true>(CB_TMP_R,CB_TMP_I,outi);
}

// Butterfly: out_r = even_r + (tw*odd).real
//            out_i = even_i + (tw*odd).imag
// Pops: odd, tw (even popped by caller after)
FORCE_INLINE void do_butterfly(
    uint32_t er,uint32_t ei,
    uint32_t or_,uint32_t oi,
    uint32_t wr,uint32_t wi)
{
    cb_wait_front(or_,1); cb_wait_front(oi,1);
    cb_wait_front(wr,1);  cb_wait_front(wi,1);
    cmul(or_,oi,wr,wi,CB_WR_R,CB_WR_I);
    cb_pop_front(or_,1); cb_pop_front(oi,1);
    cb_pop_front(wr,1);  cb_pop_front(wi,1);

    cb_wait_front(er,1); cb_wait_front(ei,1);
    cb_wait_front(CB_WR_R,1); cb_wait_front(CB_WR_I,1);
    mm_op<ADD>(er,CB_WR_R,CB_OUT_R);
    mm_op<ADD>(ei,CB_WR_I,CB_OUT_I);
    cb_pop_front(er,1); cb_pop_front(ei,1);
    cb_pop_front(CB_WR_R,1); cb_pop_front(CB_WR_I,1);
}

void kernel_main() {
    unary_op_init_common(CB_LHS_R, CB_OUT_R);
    copy_tile_to_dst_init_short(CB_LHS_R);

    // ── Local stages ──────────────────────────────────────────
    // Stage 0: LHS from initial reader load
    // Stage 1+: LHS recycled from previous OUT by writer
    for (uint32_t s=0; s<num_local_stages; s++) {
        // Wait for reader to push RHS and twiddles
        cb_wait_front(CB_RHS_R,1); cb_wait_front(CB_RHS_I,1);

        if constexpr (is_ifft) {
            neg_op<true>(CB_TWIDDLE_I, CB_TMP_R);
            cb_wait_front(CB_TMP_R,1);
            do_butterfly(CB_LHS_R,CB_LHS_I, CB_RHS_R,CB_RHS_I,
                         CB_TWIDDLE_R,CB_TMP_R);
            cb_pop_front(CB_TMP_R,1);
        } else {
            cb_wait_front(CB_TWIDDLE_R,1); cb_wait_front(CB_TWIDDLE_I,1);
            do_butterfly(CB_LHS_R,CB_LHS_I, CB_RHS_R,CB_RHS_I,
                         CB_TWIDDLE_R,CB_TWIDDLE_I);
        }
        // OUT_R/I now has result.
        // Writer will pop OUT and push back into LHS for next stage.
    }

    // ── NOC stages ────────────────────────────────────────────
    for (uint32_t s=0; s<num_noc_stages; s++) {
        // Wait for writer: peer data in scratch, OUT is valid LHS
        cb_wait_front(CB_SYNC,1); cb_pop_front(CB_SYNC,1);

        cb_wait_front(CB_SCRATCH_R,1); cb_wait_front(CB_SCRATCH_I,1);

        if constexpr (is_ifft) {
            neg_op<true>(CB_TWIDDLE_I, CB_TMP_R);
            cb_wait_front(CB_TMP_R,1);
            do_butterfly(CB_OUT_R,CB_OUT_I, CB_SCRATCH_R,CB_SCRATCH_I,
                         CB_TWIDDLE_R,CB_TMP_R);
            cb_pop_front(CB_TMP_R,1);
        } else {
            cb_wait_front(CB_TWIDDLE_R,1); cb_wait_front(CB_TWIDDLE_I,1);
            do_butterfly(CB_OUT_R,CB_OUT_I, CB_SCRATCH_R,CB_SCRATCH_I,
                         CB_TWIDDLE_R,CB_TWIDDLE_I);
        }
    }
}