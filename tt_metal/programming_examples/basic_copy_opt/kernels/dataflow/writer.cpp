#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t data_r_addr = get_arg_val<uint32_t>(0);
    uint32_t data_i_addr = get_arg_val<uint32_t>(1);
    uint32_t data_r_bank_id = get_arg_val<uint32_t>(2);
    uint32_t data_i_bank_id = get_arg_val<uint32_t>(3);
    uint32_t domain_size = get_arg_val<uint32_t>(4);

    constexpr auto cb_out_data_r = tt::CBIndex::c_8;
    constexpr auto cb_out_data_i = tt::CBIndex::c_9;

    uint64_t data_r_noc_addr = get_noc_addr_from_bank_id<true>(data_r_bank_id, data_r_addr);
    uint64_t data_i_noc_addr = get_noc_addr_from_bank_id<true>(data_i_bank_id, data_i_addr);

    cb_wait_front(cb_out_data_r, 1);
    cb_wait_front(cb_out_data_i, 1);

    uint32_t cb_out_data_r_addr = get_read_ptr(cb_out_data_r);
    uint32_t cb_out_data_i_addr = get_read_ptr(cb_out_data_i);

    noc_async_write(cb_out_data_r_addr, data_r_noc_addr, domain_size * 4);
    noc_async_write(cb_out_data_i_addr, data_i_noc_addr, domain_size * 4);
    noc_async_write_barrier();

    cb_pop_front(cb_out_data_r, 1);
    cb_pop_front(cb_out_data_i, 1);
}

