#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"

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
        add_binary_tile(0, 1, 0);
    } else if (OPERATION == SUB) {
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
    } else if (OPERATION == MUL) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    } else if (OPERATION == DIV) {
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
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

template<int ADDR_OFFSET_REG_SRC, int BASE_ADDR_REG_SRC, int ADDR_OFFSET_REG_DEST, int BASE_ADDR_REG_DEST, int DATA_REG>
inline void copy_data(uint32_t from_addr, uint32_t to_addr, bool addr) {
    PACK(

    if (addr) {
        uint32_t base_address=from_addr / 16;
        uint32_t addr_offset=from_addr-(base_address*16);

        TT_SETDMAREG(0, LOWER_HALFWORD(addr_offset), 0, LO_16(ADDR_OFFSET_REG_SRC));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr_offset), 0, HI_16(ADDR_OFFSET_REG_SRC));
        TT_SETDMAREG(0, LOWER_HALFWORD(base_address), 0, LO_16(BASE_ADDR_REG_SRC));
        TT_SETDMAREG(0, UPPER_HALFWORD(base_address), 0, HI_16(BASE_ADDR_REG_SRC));
    }
    TT_LOADIND(p_ind::LD_32bit, LO_16(ADDR_OFFSET_REG_SRC), p_ind::INC_4B, DATA_REG, BASE_ADDR_REG_SRC);

    /*
    volatile uint32_t * reg=(uint32_t*) ((void*) 0xFFE00000);
    for (int i=0;i<64;i++) {
    //DPRINT << "Reg " << i<< " val " << reg[i] << ENDL();
    }*/
    
    if (addr) {
        uint32_t base_address=to_addr / 16;
        uint32_t addr_offset=to_addr-(base_address*16);

        TT_SETDMAREG(0, LOWER_HALFWORD(addr_offset), 0, LO_16(ADDR_OFFSET_REG_DEST));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr_offset), 0, HI_16(ADDR_OFFSET_REG_DEST));
        TT_SETDMAREG(0, LOWER_HALFWORD(base_address), 0, LO_16(BASE_ADDR_REG_DEST));
        TT_SETDMAREG(0, UPPER_HALFWORD(base_address), 0, HI_16(BASE_ADDR_REG_DEST));
    }
    TT_STOREIND(1, 0, p_ind::LD_32bit, LO_16(ADDR_OFFSET_REG_DEST), p_ind::INC_4B, DATA_REG, BASE_ADDR_REG_DEST);

    /*
    for (int i=0;i<64;i++) {
    DPRINT << "Reg " << i<< " val " << reg[i] << ENDL();
    }*/

    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    );
}

void copy_in(uint32_t step, uint32_t num_steps, uint32_t domain_size, uint32_t in_data_r, uint32_t in_data_i, uint32_t cb_data0_r, uint32_t cb_data0_i, uint32_t cb_data1_r, uint32_t cb_data1_i, uint32_t twiddle_data, uint32_t cb_twiddle_r, uint32_t cb_twiddle_i) {
    uint32_t num_spectra_in_step=step == 0 ? 1 : 2 << (step-1);
    uint32_t increment_next_point_in_step=2 << step;
    uint32_t matching_second_point=increment_next_point_in_step/2;
    uint32_t tgt_data_idx=0;
    for (uint32_t spectra=0; spectra < num_spectra_in_step; spectra++) {
        int twiddle_index=spectra << (num_steps-step);
        for (uint32_t point=0; point < domain_size; point+=increment_next_point_in_step) {
            uint32_t d0_data_index=spectra + point;
            uint32_t d1_data_index=spectra + point + matching_second_point;
            copy_data<5,6,7,8,9>(in_data_r+(d0_data_index*4), cb_data0_r+(tgt_data_idx*4), true);
            copy_data<5,6,7,8,9>(in_data_i+(d0_data_index*4), cb_data0_i+(tgt_data_idx*4), true);
            copy_data<5,6,7,8,9>(in_data_r+(d1_data_index*4), cb_data1_r+(tgt_data_idx*4), true);
            copy_data<5,6,7,8,9>(in_data_i+(d1_data_index*4), cb_data1_i+(tgt_data_idx*4), true);
                
            copy_data<5,6,7,8,9>(twiddle_data+((twiddle_index*2)*4), cb_twiddle_r+(tgt_data_idx*4), true);
            copy_data<5,6,7,8,9>(twiddle_data+(((twiddle_index*2)+1)*4), cb_twiddle_i+(tgt_data_idx*4), true);
            tgt_data_idx++;
        }
    }  
}

void copy_out(uint32_t step, uint32_t num_steps, uint32_t domain_size, uint32_t out_data_r, uint32_t out_data_i, uint32_t cb_data0_r, uint32_t cb_data0_i, uint32_t cb_data1_r, uint32_t cb_data1_i) {
    uint32_t num_spectra_in_step=step == 0 ? 1 : 2 << (step-1);
    uint32_t increment_next_point_in_step=2 << step;
    uint32_t matching_second_point=increment_next_point_in_step/2;
    uint32_t tgt_data_idx=0;
    for (uint32_t spectra=0; spectra < num_spectra_in_step; spectra++) {
        for (uint32_t point=0; point < domain_size; point+=increment_next_point_in_step) {
            uint32_t d0_data_index=spectra + point;
            uint32_t d1_data_index=spectra + point + matching_second_point;
            copy_data<5,6,7,8,9>(cb_data0_r+(tgt_data_idx*4), out_data_r+(d0_data_index*4), true);
            copy_data<5,6,7,8,9>(cb_data0_i+(tgt_data_idx*4), out_data_i+(d0_data_index*4), true);
            copy_data<5,6,7,8,9>(cb_data1_r+(tgt_data_idx*4), out_data_r+(d1_data_index*4), true);
            copy_data<5,6,7,8,9>(cb_data1_i+(tgt_data_idx*4), out_data_i+(d1_data_index*4), true);
            tgt_data_idx++;
        }
    }
}

void MAIN {
    // Direction is 0 for forward FFT and 1 for backward FFT
    uint32_t direction = get_arg_val<uint32_t>(0);
    uint32_t domain_size = get_arg_val<uint32_t>(1);
    uint32_t step_results_r_buffer_addr = get_arg_val<uint32_t>(2);
    uint32_t step_results_i_buffer_addr = get_arg_val<uint32_t>(3);

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

    constexpr auto cb_out_data_r = tt::CBIndex::c_8;
    constexpr auto cb_out_data_i = tt::CBIndex::c_9;

    constexpr auto cb_intermediate0 = tt::CBIndex::c_23;
    constexpr auto cb_intermediate1 = tt::CBIndex::c_24;
    constexpr auto cb_intermediate2 = tt::CBIndex::c_25;
    constexpr auto cb_f0 = tt::CBIndex::c_6;
    constexpr auto cb_f1 = tt::CBIndex::c_7;

    constexpr auto cb_ddr_data_r = tt::CBIndex::c_20;
    constexpr auto cb_ddr_data_i = tt::CBIndex::c_21;
    constexpr auto cb_ddr_twiddle_data = tt::CBIndex::c_22;

    cb_wait_front(cb_ddr_data_r, 1);
    cb_wait_front(cb_ddr_data_i, 1);
    cb_wait_front(cb_ddr_twiddle_data, 1);

    cb_reserve_back(cb_twiddle_r, 1);
    cb_reserve_back(cb_twiddle_i, 1);

    cb_reserve_back(cb_data0_r, 1);
    cb_reserve_back(cb_data0_i, 1);
    cb_reserve_back(cb_data1_r, 1);
    cb_reserve_back(cb_data1_i, 1);

    volatile uint *cb_ddr_data_r_addr, *cb_ddr_data_i_addr, *cb_data0_r_addr, *cb_data0_i_addr, *cb_data1_r_addr, *cb_data1_i_addr, *cb_ddr_twiddle_data_addr, *cb_twiddle_r_addr, *cb_twiddle_i_addr;
    cb_get_tile(cb_ddr_data_r, 0, &cb_ddr_data_r_addr);
    cb_get_tile(cb_ddr_data_i, 0, &cb_ddr_data_i_addr);
    cb_get_tile(cb_data0_r, 0, &cb_data0_r_addr);
    cb_get_tile(cb_data0_i, 0, &cb_data0_i_addr);
    cb_get_tile(cb_data1_r, 0, &cb_data1_r_addr);
    cb_get_tile(cb_data1_i, 0, &cb_data1_i_addr);
    cb_get_tile(cb_ddr_twiddle_data, 0, &cb_ddr_twiddle_data_addr);
    cb_get_tile(cb_twiddle_r, 0, &cb_twiddle_r_addr);
    cb_get_tile(cb_twiddle_i, 0, &cb_twiddle_i_addr);

    uint32_t idx_cb_ddr_data_r_addr = reinterpret_cast<uint32_t>(cb_ddr_data_r_addr);
    uint32_t idx_cb_ddr_data_i_addr = reinterpret_cast<uint32_t>(cb_ddr_data_i_addr);
    uint32_t idx_cb_data0_r_addr = reinterpret_cast<uint32_t>(cb_data0_r_addr);
    uint32_t idx_cb_data0_i_addr = reinterpret_cast<uint32_t>(cb_data0_i_addr);
    uint32_t idx_cb_data1_r_addr = reinterpret_cast<uint32_t>(cb_data1_r_addr);
    uint32_t idx_cb_data1_i_addr = reinterpret_cast<uint32_t>(cb_data1_i_addr);
    uint32_t idx_cb_ddr_twiddle_data_addr = reinterpret_cast<uint32_t>(cb_ddr_twiddle_data_addr);
    uint32_t idx_cb_twiddle_r_addr = reinterpret_cast<uint32_t>(cb_twiddle_r_addr);
    uint32_t idx_cb_twiddle_i_addr = reinterpret_cast<uint32_t>(cb_twiddle_i_addr);

    int num_steps=getLog(domain_size);
    for (int step=0; step <= num_steps; step++) {
    //PACK( DPRINT << "Go step " <<step<<ENDL(););
        if (step == 0) {
            // Data is sourced from DDR
            copy_in(step, num_steps, domain_size, idx_cb_ddr_data_r_addr, idx_cb_ddr_data_i_addr, idx_cb_data0_r_addr, 
                    idx_cb_data0_i_addr, idx_cb_data1_r_addr, idx_cb_data1_i_addr, 
                    idx_cb_ddr_twiddle_data_addr, idx_cb_twiddle_r_addr, idx_cb_twiddle_i_addr); 
            cb_pop_front(cb_ddr_data_r, 1);
            cb_pop_front(cb_ddr_data_i, 1);
        } else {
            // Data is sourced from internal buffer of previous step
            copy_in(step, num_steps, domain_size, step_results_r_buffer_addr, step_results_i_buffer_addr, idx_cb_data0_r_addr,
                    idx_cb_data0_i_addr, idx_cb_data1_r_addr, idx_cb_data1_i_addr,
                    idx_cb_ddr_twiddle_data_addr, idx_cb_twiddle_r_addr, idx_cb_twiddle_i_addr);
        }

        //PACK( DPRINT << "Data loaded " <<ENDL(););

        unary_op_init_common(cb_data1_r, cb_data1_r);
        binary_op_init_common(cb_data1_r, cb_data1_i, cb_intermediate0);
        copy_tile_to_dst_init_short(cb_data1_r);

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

        volatile uint *cb_out_data0_r_addr, cb_out_data0_i_addr, cb_out_data1_r_addr, cb_out_data1_i_addr;
        cb_get_tile(cb_out_data0_r, 0, &cb_out_data0_r_addr);
        cb_get_tile(cb_out_data0_i, 0, &cb_out_data0_i_addr);
        cb_get_tile(cb_out_data1_r, 0, &cb_out_data1_r_addr);
        cb_get_tile(cb_out_data1_i, 0, &cb_out_data1_i_addr);

        uint32_t idx_cb_out_data0_r_addr = reinterpret_cast<uint32_t>(cb_out_data0_r_addr);
        uint32_t idx_cb_out_data0_i_addr = reinterpret_cast<uint32_t>(cb_out_data0_i_addr);
        uint32_t idx_cb_out_data1_r_addr = reinterpret_cast<uint32_t>(cb_out_data1_r_addr);
        uint32_t idx_cb_out_data1_i_addr = reinterpret_cast<uint32_t>(cb_out_data1_i_addr);

        if (step == num_steps-1) {
            cb_reserve_back(cb_out_data_r, 1);
            cb_reserve_back(cb_out_data_i, 1);

            volatile uint *cb_out_data_r_addr, cb_out_data_i_addr;
            cb_get_tile(cb_out_data_r, 0, &cb_out_data_r_addr);
            cb_get_tile(cb_out_data_i, 0, &cb_out_data_i_addr);

            uint32_t idx_cb_out_data_r_addr = reinterpret_cast<uint32_t>(cb_out_data_r_addr);
            uint32_t idx_cb_out_data_i_addr = reinterpret_cast<uint32_t>(cb_out_data_i_addr);
            copy_out(step, num_steps, domain_size, idx_cb_out_data_r_addr, idx_cb_out_data_i_addr, idx_cb_out_data0_r_addr, idx_cb_out_data0_i_addr, idx_cb_out_data1_r_addr, idx_cb_out_data1_i_addr);
            cb_push_back(cb_out_data_r, 1);
            cb_push_back(cb_out_data_i, 1);
        } else {
            copy_out(step, num_steps, domain_size, step_results_r_buffer_addr, step_results_i_buffer_addr, idx_cb_out_data0_r_addr, idx_cb_out_data0_i_addr, idx_cb_out_data1_r_addr, idx_cb_out_data1_i_addr);
        }

        cb_pop_front(cb_out_data0_r, 1);
        cb_pop_front(cb_out_data0_i, 1);
        cb_pop_front(cb_out_data1_r, 1);
        cb_pop_front(cb_out_data1_i, 1);
    }
    cb_pop_front(cb_data0_r, 1);
    cb_pop_front(cb_data0_i, 1);
    cb_pop_front(cb_data1_r, 1);
    cb_pop_front(cb_data1_i, 1);
    cb_pop_front(cb_twiddle_r, 1);
    cb_pop_front(cb_twiddle_i, 1);
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
