#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "debug/dprint.h"

#define USE_SFPU 1

namespace NAMESPACE {

enum {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    NEG = 4,
};

void do_copy_tile(uint32_t, uint32_t);
void copy_tiles(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
int getLog(int);

template <int OPERATION, bool CB_OP_IN=false>
void unary_sfpu_op(uint32_t cb_in, uint32_t cb_tgt) {
    if constexpr(CB_OP_IN) cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    if (OPERATION == NEG) {
        negative_tile_init();
        negative_tile(0);
    }
    tile_regs_commit();
    if constexpr(CB_OP_IN) cb_pop_front(cb_in, 1);
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

template <int OPERATION, bool CB_OP_IN1=false, bool CB_OP_IN2=false>
void maths_sfpu_op(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt) {
    // Copy from input CBs into index 0 and 1 of DST regs
    if constexpr(CB_OP_IN1) cb_wait_front(cb_in_1, 1);
    if constexpr(CB_OP_IN2) cb_wait_front(cb_in_2, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in_1);
    copy_tile(cb_in_1, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_in_1, cb_in_2);
    copy_tile(cb_in_2, 0, 1);
    if (OPERATION == ADD) {
        add_binary_tile_init();
        add_binary_tile(0, 1);
    } else if (OPERATION == SUB) {
        sub_binary_tile_init();
        sub_binary_tile(0, 1);
    } else if (OPERATION == MUL) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1);
    } else if (OPERATION == DIV) {
        div_binary_tile_init();
        div_binary_tile(0, 1);
    }
    tile_regs_commit(); 
    if constexpr(CB_OP_IN1) cb_pop_front(cb_in_1, 1);
    if constexpr(CB_OP_IN2) cb_pop_front(cb_in_2, 1);
    // Result is in 0 index of DST regs, extract
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

template <int OPERATION, bool CB_OP_IN1=false, bool CB_OP_IN2=false>
void maths_mm_op(uint32_t cb_in_1, uint32_t cb_in_2, uint32_t cb_tgt) {
    if constexpr(CB_OP_IN1) cb_wait_front(cb_in_1, 1);
    if constexpr(CB_OP_IN2) cb_wait_front(cb_in_2, 1);
    tile_regs_acquire();
    if (OPERATION == ADD) {
        add_tiles_init(cb_in_1, cb_in_2);
        add_tiles(cb_in_1, cb_in_2, 0, 0, 0);
    } else if (OPERATION == SUB) {
        sub_tiles_init(cb_in_1, cb_in_2);
        sub_tiles(cb_in_1, cb_in_2, 0, 0, 0);
    } else if (OPERATION == MUL) {
        mul_tiles_init(cb_in_1, cb_in_2);
        mul_tiles(cb_in_1, cb_in_2, 0, 0, 0);
    }
    tile_regs_commit();
    if constexpr(CB_OP_IN1) cb_pop_front(cb_in_1, 1);
    if constexpr(CB_OP_IN2) cb_pop_front(cb_in_2, 1);
    cb_reserve_back(cb_tgt, 1);
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
    cb_push_back(cb_tgt, 1);
}

void MAIN {
    // Direction is 0 for forward FFT and 1 for backward FFT
    uint32_t direction = get_arg_val<uint32_t>(0);
    uint32_t domain_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_data0_r = tt::CBIndex::c_0;
    constexpr auto cb_data0_i = tt::CBIndex::c_1;
    constexpr auto cb_data1_r = tt::CBIndex::c_2;
    constexpr auto cb_data1_i = tt::CBIndex::c_3;
    constexpr auto cb_twiddle_r = tt::CBIndex::c_4;
    constexpr auto cb_twiddle_i = tt::CBIndex::c_5;
    constexpr auto cb_out_data0_r = tt::CBIndex::c_16;
    constexpr auto cb_out_data0_i = tt::CBIndex::c_17;
    constexpr auto cb_out_data1_r = tt::CBIndex::c_18;
    constexpr auto cb_out_data1_i = tt::CBIndex::c_19;

    constexpr auto cb_intermediate0 = tt::CBIndex::c_23;
    constexpr auto cb_intermediate1 = tt::CBIndex::c_24;
    constexpr auto cb_intermediate2 = tt::CBIndex::c_25;
    constexpr auto cb_f0 = tt::CBIndex::c_6;
    constexpr auto cb_f1 = tt::CBIndex::c_7;

    unary_op_init_common(cb_data1_r, cb_out_data1_r);    
    binary_op_init_common(cb_data1_r, cb_data1_i, cb_intermediate0);

    copy_tile_to_dst_init_short(cb_data1_r);

    int num_steps=getLog(domain_size);
    for (int step=0; step <= num_steps; step++) {
        cb_wait_front(cb_data1_r, 1);
        cb_wait_front(cb_data1_i, 1);

        cb_wait_front(cb_twiddle_r, 1);
        cb_wait_front(cb_twiddle_i, 1);

        // If this is a backwards FFT then we need to invert imaginary data on the 
        // first step and use this as input
        bool requires_imaginary_neg=(direction == 1 && step == 0);

        if (requires_imaginary_neg) {
            unary_sfpu_op<NEG>(cb_data1_i, cb_intermediate2);
            cb_wait_front(cb_intermediate2, 1);
        }

        // Calculate f0
#ifdef USE_SFPU
        maths_sfpu_op<MUL>(cb_data1_r, cb_twiddle_r, cb_intermediate0);
        maths_sfpu_op<MUL>(requires_imaginary_neg ? cb_intermediate2 : cb_data1_i, cb_twiddle_i, cb_intermediate1);
        maths_sfpu_op<SUB,true,true>(cb_intermediate0, cb_intermediate1, cb_f0);
#else
        maths_mm_op<MUL>(cb_data1_r, cb_twiddle_r, cb_intermediate0);
        maths_mm_op<MUL>(requires_imaginary_neg ? cb_intermediate2 : cb_data1_i, cb_twiddle_i, cb_intermediate1);
        maths_mm_op<SUB,true,true>(cb_intermediate0, cb_intermediate1, cb_f0);
#endif

        // Calculate f1      
#ifdef USE_SFPU
        maths_sfpu_op<MUL>(cb_data1_r, cb_twiddle_i, cb_intermediate0);
        maths_sfpu_op<MUL>(requires_imaginary_neg ? cb_intermediate2 : cb_data1_i, cb_twiddle_r, cb_intermediate1);
        maths_sfpu_op<ADD,true,true>(cb_intermediate0, cb_intermediate1, cb_f1);
#else
        maths_mm_op<MUL>(cb_data1_r, cb_twiddle_i, cb_intermediate0);
        maths_mm_op<MUL>(requires_imaginary_neg ? cb_intermediate2 : cb_data1_i, cb_twiddle_r, cb_intermediate1);
        maths_mm_op<ADD,true,true>(cb_intermediate0, cb_intermediate1, cb_f1);
#endif

        cb_pop_front(cb_twiddle_r, 1);
        cb_pop_front(cb_twiddle_i, 1);

        // Wait on data for data 0 CBs to be available as we are about to use these
        cb_wait_front(cb_data0_r, 1);
        cb_wait_front(cb_data0_i, 1);

        if (requires_imaginary_neg) {
            // Now invert the data 0 imaginary numbers if this is required
            cb_pop_front(cb_intermediate2, 1);            
            unary_sfpu_op<NEG>(cb_data0_i, cb_intermediate2);
            cb_wait_front(cb_intermediate2, 1);
        }

        cb_wait_front(cb_f0, 1);
        cb_wait_front(cb_f1, 1);

        // Calculate data_1 real
#ifdef USE_SFPU
        maths_sfpu_op<SUB>(cb_data0_r, cb_f0, cb_out_data1_r);
#else
        maths_mm_op<SUB>(cb_data0_r, cb_f0, cb_out_data1_r);
#endif
        // Calculate data_1 imaginary
#ifdef USE_SFPU
        maths_sfpu_op<SUB>(requires_imaginary_neg ? cb_intermediate2 : cb_data0_i, cb_f1, cb_out_data1_i);
#else
        maths_mm_op<SUB>(requires_imaginary_neg ? cb_intermediate2 : cb_data0_i, cb_f1, cb_out_data1_i);
#endif
        // Calculate data_0 real
#ifdef USE_SFPU
        maths_sfpu_op<ADD>(cb_data0_r, cb_f0, cb_out_data0_r);
#else
        maths_mm_op<ADD>(cb_data0_r, cb_f0, cb_out_data0_r);
#endif
        // Calculate data_0 imaginary
#ifdef USE_SFPU
        maths_sfpu_op<ADD>(requires_imaginary_neg ? cb_intermediate2 : cb_data0_i, cb_f1, cb_out_data0_i);
#else
        maths_mm_op<ADD>(requires_imaginary_neg ? cb_intermediate2 : cb_data0_i, cb_f1, cb_out_data0_i);
#endif        

        if (requires_imaginary_neg) cb_pop_front(cb_intermediate2, 1);

        cb_pop_front(cb_f0, 1);
        cb_pop_front(cb_f1, 1);
   
        cb_pop_front(cb_data0_r, 1);
        cb_pop_front(cb_data0_i, 1);
        cb_pop_front(cb_data1_r, 1);
        cb_pop_front(cb_data1_i, 1);
    }
}

void do_copy_tile(uint32_t cb_src, uint32_t cb_tgt) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_src);
    copy_tile(cb_src, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_tgt);
    tile_regs_release();
}

void copy_tiles(uint32_t cb_data1_r, uint32_t cb_data1_i, uint32_t cb_data0_r, uint32_t cb_data0_i, uint32_t cb_out_data1_r, uint32_t cb_out_data1_i, uint32_t cb_out_data0_r, uint32_t cb_out_data0_i) {
    do_copy_tile(cb_data1_r, cb_out_data1_r);
    do_copy_tile(cb_data1_i, cb_out_data1_i);
    do_copy_tile(cb_data0_r, cb_out_data0_r);
    do_copy_tile(cb_data0_i, cb_out_data0_i);
}

int getLog(int n) {
   int logn=0;
   n >>= 1;
   while ((n >>=1) > 0) {
      logn++;
   }
   return logn;
}

}  // namespace NAMESPACE
