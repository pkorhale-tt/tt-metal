// fft_reader.cpp — BRISC
// Loads even/odd split input, twiddles each stage.
// CB layout: c_0/1=even_r/i, c_2/3=odd_r/i, c_4/5=tw_r/i

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

void kernel_main() {
    uint32_t even_r_addr   = get_arg_val<uint32_t>(0);
    uint32_t even_i_addr   = get_arg_val<uint32_t>(1);
    uint32_t odd_r_addr    = get_arg_val<uint32_t>(2);
    uint32_t odd_i_addr    = get_arg_val<uint32_t>(3);
    uint32_t tw_r_addr     = get_arg_val<uint32_t>(4);
    uint32_t tw_i_addr     = get_arg_val<uint32_t>(5);
    uint32_t num_tiles     = get_arg_val<uint32_t>(6);  // tiles per channel
    uint32_t num_stages    = get_arg_val<uint32_t>(7);
    uint32_t my_id         = get_arg_val<uint32_t>(8);
    uint32_t num_local_stg = get_arg_val<uint32_t>(9);

    constexpr auto CB_EVEN_R = tt::CBIndex::c_0;
    constexpr auto CB_EVEN_I = tt::CBIndex::c_1;
    constexpr auto CB_ODD_R  = tt::CBIndex::c_2;
    constexpr auto CB_ODD_I  = tt::CBIndex::c_3;
    constexpr auto CB_TW_R   = tt::CBIndex::c_4;
    constexpr auto CB_TW_I   = tt::CBIndex::c_5;

    const DataFormat df  = get_dataformat(CB_EVEN_R);
    const uint32_t tsize = get_tile_size(CB_EVEN_R);

    InterleavedAddrGenFast<true> even_r_gen = {.bank_base_address=even_r_addr,.page_size=tsize,.data_format=df};
    InterleavedAddrGenFast<true> even_i_gen = {.bank_base_address=even_i_addr,.page_size=tsize,.data_format=df};
    InterleavedAddrGenFast<true> odd_r_gen  = {.bank_base_address=odd_r_addr, .page_size=tsize,.data_format=df};
    InterleavedAddrGenFast<true> odd_i_gen  = {.bank_base_address=odd_i_addr, .page_size=tsize,.data_format=df};
    InterleavedAddrGenFast<true> tw_r_gen   = {.bank_base_address=tw_r_addr,  .page_size=tsize,.data_format=df};
    InterleavedAddrGenFast<true> tw_i_gen   = {.bank_base_address=tw_i_addr,  .page_size=tsize,.data_format=df};

    for (uint32_t s=0; s<num_local_stg; s++) {
        // Twiddles first (matches compute cb_wait order)
        cb_reserve_back(CB_TW_R,1); cb_reserve_back(CB_TW_I,1);
        noc_async_read_tile(s * num_tiles + my_id, tw_r_gen, get_write_ptr(CB_TW_R));
        noc_async_read_tile(s * num_tiles + my_id, tw_i_gen, get_write_ptr(CB_TW_I));
        noc_async_read_barrier();
        cb_push_back(CB_TW_R,1); cb_push_back(CB_TW_I,1);

        // Even/odd input (stage 0: from DRAM; stage 1+: recycled by writer)
        if (s == 0) {
            cb_reserve_back(CB_EVEN_R,1); cb_reserve_back(CB_EVEN_I,1);
            cb_reserve_back(CB_ODD_R,1);  cb_reserve_back(CB_ODD_I,1);
            noc_async_read_tile(my_id, even_r_gen, get_write_ptr(CB_EVEN_R));
            noc_async_read_tile(my_id, even_i_gen, get_write_ptr(CB_EVEN_I));
            noc_async_read_tile(my_id, odd_r_gen,  get_write_ptr(CB_ODD_R));
            noc_async_read_tile(my_id, odd_i_gen,  get_write_ptr(CB_ODD_I));
            noc_async_read_barrier();
            cb_push_back(CB_EVEN_R,1); cb_push_back(CB_EVEN_I,1);
            cb_push_back(CB_ODD_R,1);  cb_push_back(CB_ODD_I,1);
        }
        // Stage 1+: writer recycles OUT0→EVEN, OUT1→ODD, so reader just pushes twiddles
    }

    // NOC stages: twiddles only
    for (uint32_t s=num_local_stg; s<num_stages; s++) {
        cb_reserve_back(CB_TW_R,1); cb_reserve_back(CB_TW_I,1);
        noc_async_read_tile(s * num_tiles + my_id, tw_r_gen, get_write_ptr(CB_TW_R));
        noc_async_read_tile(s * num_tiles + my_id, tw_i_gen, get_write_ptr(CB_TW_I));
        noc_async_read_barrier();
        cb_push_back(CB_TW_R,1); cb_push_back(CB_TW_I,1);
    }
}