// reader_fft_f32.cpp
#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Arguments
    uint32_t even_r_addr = get_arg_val<uint32_t>(0);
    uint32_t even_i_addr = get_arg_val<uint32_t>(1);
    uint32_t odd_r_addr = get_arg_val<uint32_t>(2);
    uint32_t odd_i_addr = get_arg_val<uint32_t>(3);
    uint32_t tw_r_addr = get_arg_val<uint32_t>(4);
    uint32_t tw_i_addr = get_arg_val<uint32_t>(5);
    uint32_t num_tiles = get_arg_val<uint32_t>(6);
    uint32_t start_tile = get_arg_val<uint32_t>(7);
    
    // CB indices
    constexpr uint32_t cb_even_r = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i = tt::CBIndex::c_1;
    constexpr uint32_t cb_odd_r = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i = tt::CBIndex::c_3;
    constexpr uint32_t cb_tw_r = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i = tt::CBIndex::c_5;
    
    uint32_t tile_bytes = get_tile_size(cb_even_r);
    const DataFormat data_format = get_dataformat(cb_even_r);
    
    const InterleavedAddrGenFast<true> even_r_gen = {
        .bank_base_address = even_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> even_i_gen = {
        .bank_base_address = even_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> odd_r_gen = {
        .bank_base_address = odd_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> odd_i_gen = {
        .bank_base_address = odd_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    
    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_idx = start_tile + i;
        
        // Read even (data0) real
        cb_reserve_back(cb_even_r, 1);
        uint32_t l1_even_r = get_write_ptr(cb_even_r);
        noc_async_read_tile(tile_idx, even_r_gen, l1_even_r);
        noc_async_read_barrier();
        cb_push_back(cb_even_r, 1);
        
        // Read even (data0) imaginary
        cb_reserve_back(cb_even_i, 1);
        uint32_t l1_even_i = get_write_ptr(cb_even_i);
        noc_async_read_tile(tile_idx, even_i_gen, l1_even_i);
        noc_async_read_barrier();
        cb_push_back(cb_even_i, 1);
        
        // Read odd (data1) real
        cb_reserve_back(cb_odd_r, 1);
        uint32_t l1_odd_r = get_write_ptr(cb_odd_r);
        noc_async_read_tile(tile_idx, odd_r_gen, l1_odd_r);
        noc_async_read_barrier();
        cb_push_back(cb_odd_r, 1);
        
        // Read odd (data1) imaginary
        cb_reserve_back(cb_odd_i, 1);
        uint32_t l1_odd_i = get_write_ptr(cb_odd_i);
        noc_async_read_tile(tile_idx, odd_i_gen, l1_odd_i);
        noc_async_read_barrier();
        cb_push_back(cb_odd_i, 1);
        
        // Read twiddle real
        cb_reserve_back(cb_tw_r, 1);
        uint32_t l1_tw_r = get_write_ptr(cb_tw_r);
        noc_async_read_tile(tile_idx, tw_r_gen, l1_tw_r);
        noc_async_read_barrier();
        cb_push_back(cb_tw_r, 1);
        
        // Read twiddle imaginary
        cb_reserve_back(cb_tw_i, 1);
        uint32_t l1_tw_i = get_write_ptr(cb_tw_i);
        noc_async_read_tile(tile_idx, tw_i_gen, l1_tw_i);
        noc_async_read_barrier();
        cb_push_back(cb_tw_i, 1);
    }
}