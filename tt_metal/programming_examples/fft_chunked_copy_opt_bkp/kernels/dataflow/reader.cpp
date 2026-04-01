#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t data_r_addr = get_arg_val<uint32_t>(0);
    uint32_t data_i_addr = get_arg_val<uint32_t>(1);
    uint32_t twiddle_addr = get_arg_val<uint32_t>(2);
    uint32_t data_r_bank_id = get_arg_val<uint32_t>(3);
    uint32_t data_i_bank_id = get_arg_val<uint32_t>(4);
    uint32_t twiddle_bank_id = get_arg_val<uint32_t>(5);
    uint32_t domain_size = get_arg_val<uint32_t>(6);

    uint64_t data_r_noc_addr = get_noc_addr_from_bank_id<true>(data_r_bank_id, data_r_addr);
    uint64_t data_i_noc_addr = get_noc_addr_from_bank_id<true>(data_i_bank_id, data_i_addr);
    uint64_t twiddle_noc_addr = get_noc_addr_from_bank_id<true>(twiddle_bank_id, twiddle_addr);

    constexpr auto cb_out_data_r = tt::CBIndex::c_20;
    constexpr auto cb_out_data_i = tt::CBIndex::c_21;
    constexpr auto cb_out_twiddle = tt::CBIndex::c_22;

    cb_reserve_back(cb_out_data_r, 1);
    cb_reserve_back(cb_out_data_i, 1);
    cb_reserve_back(cb_out_twiddle, 1);
    uint32_t cb_out_data_r_ptr = get_write_ptr(cb_out_data_r);
    uint32_t cb_out_data_i_ptr = get_write_ptr(cb_out_data_i);
    uint32_t cb_out_twiddle_ptr = get_write_ptr(cb_out_twiddle);

    noc_async_read(data_r_noc_addr, cb_out_data_r_ptr, domain_size * 4);
    noc_async_read(data_i_noc_addr, cb_out_data_i_ptr, domain_size * 4);
    noc_async_read(twiddle_noc_addr, cb_out_twiddle_ptr, domain_size * 4);
    noc_async_read_barrier();

    cb_push_back(cb_out_data_r, 1);
    cb_push_back(cb_out_data_i, 1);
    cb_push_back(cb_out_twiddle, 1);
}

