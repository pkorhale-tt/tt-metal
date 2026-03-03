#include <stdint.h>
#include "dataflow_api.h"

void read_cb_and_arange_data(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void read_external_and_arrange_data(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void read_stage_data(uint32_t cb_data0_id, uint32_t, float*, uint32_t, float*, uint32_t, uint32_t, uint32_t, uint32_t);
void bitreverse(float*, int);
int getLog(int);

void kernel_main() {
    uint32_t data_r_addr = get_arg_val<uint32_t>(0);
    uint32_t data_i_addr = get_arg_val<uint32_t>(1);
    uint32_t twiddle_addr = get_arg_val<uint32_t>(2);
    uint32_t data_r_bank_id = get_arg_val<uint32_t>(3);
    uint32_t data_i_bank_id = get_arg_val<uint32_t>(4);
    uint32_t twiddle_bank_id = get_arg_val<uint32_t>(5);
    uint32_t read_in_buffer_addr = get_arg_val<uint32_t>(6);
    uint32_t twiddle_buffer_addr = get_arg_val<uint32_t>(7);
    uint32_t domain_size = get_arg_val<uint32_t>(8);

    uint64_t data_r_noc_addr = get_noc_addr_from_bank_id<true>(data_r_bank_id, data_r_addr);
    uint64_t data_i_noc_addr = get_noc_addr_from_bank_id<true>(data_i_bank_id, data_i_addr);
    uint64_t twiddle_noc_addr = get_noc_addr_from_bank_id<true>(twiddle_bank_id, twiddle_addr);

    constexpr auto cb_data0_r = tt::CBIndex::c_0;
    constexpr auto cb_data0_i = tt::CBIndex::c_1;
    constexpr auto cb_data1_r = tt::CBIndex::c_2;
    constexpr auto cb_data1_i = tt::CBIndex::c_3;

    constexpr auto cb_twiddle_r = tt::CBIndex::c_4;
    constexpr auto cb_twiddle_i = tt::CBIndex::c_5;

    constexpr auto cb_out_data_r = tt::CBIndex::c_20;
    constexpr auto cb_out_data_i = tt::CBIndex::c_21;

    noc_async_read(twiddle_noc_addr, twiddle_buffer_addr, domain_size * 4);
    noc_async_read_barrier();

    int num_steps=getLog(domain_size);

    read_external_and_arrange_data(data_r_noc_addr, read_in_buffer_addr, cb_data0_r, cb_data1_r, cb_twiddle_r, twiddle_buffer_addr, 0, domain_size, num_steps);
    read_external_and_arrange_data(data_i_noc_addr, read_in_buffer_addr, cb_data0_i, cb_data1_i, cb_twiddle_i, twiddle_buffer_addr, 1, domain_size, num_steps);
    for (int step=1; step <= num_steps; step++) {
        read_cb_and_arange_data(cb_out_data_r, cb_data0_r, cb_data1_r, cb_twiddle_r, twiddle_buffer_addr, 0, domain_size, num_steps, step);
        read_cb_and_arange_data(cb_out_data_i, cb_data0_i, cb_data1_i, cb_twiddle_i, twiddle_buffer_addr, 1, domain_size, num_steps, step);
    }
}

void read_cb_and_arange_data(uint32_t cb_data_id, uint32_t cb_data0_id, uint32_t cb_data1_id, uint32_t cb_twiddle, uint32_t twiddle_data, uint32_t twiddle_offset, uint32_t domain_size, uint32_t num_steps, uint32_t step) {
    cb_wait_front(cb_data_id, 1);
    float * read_cb_data_addr = (float*) get_read_ptr(cb_data_id);
    read_stage_data(cb_data0_id, cb_data1_id, read_cb_data_addr, cb_twiddle, (float*) twiddle_data, twiddle_offset, domain_size, num_steps, step);
    cb_pop_front(cb_data_id, 1);
}

void read_external_and_arrange_data(uint64_t data_noc_addr, uint32_t read_in_buffer_addr, uint32_t cb_data0_id, uint32_t cb_data1_id, uint32_t cb_twiddle, uint32_t twiddle_data, uint32_t twiddle_offset, uint32_t domain_size, uint32_t num_steps) {
    noc_async_read(data_noc_addr, read_in_buffer_addr, domain_size * 4);
    noc_async_read_barrier();
    float* in_data=(float*) read_in_buffer_addr;
    // Bit reverse on the input data that we have just read
    bitreverse(in_data, domain_size);
    // Step is zero here a this is the first read
    read_stage_data(cb_data0_id, cb_data1_id, in_data, cb_twiddle, (float*) twiddle_data, twiddle_offset, domain_size, num_steps, 0);
}

void read_stage_data(uint32_t cb_data0_id, uint32_t cb_data1_id, float * in_data, uint32_t cb_twiddle, float * twiddle_data, uint32_t twiddle_offset, uint32_t domain_size, uint32_t num_steps, uint32_t step) {
    cb_reserve_back(cb_data0_id, 1);
    cb_reserve_back(cb_data1_id, 1);
    cb_reserve_back(cb_twiddle, 1);
    float * write_cb_data0_addr = (float*) get_write_ptr(cb_data0_id);
    float * write_cb_data1_addr = (float*) get_write_ptr(cb_data1_id);
    float * twiddle_addr = (float*) get_write_ptr(cb_twiddle);

    uint32_t num_spectra_in_step=step == 0 ? 1 : 2 << (step-1);
    uint32_t increment_next_point_in_step=2 << step;
    uint32_t matching_second_point=increment_next_point_in_step/2;

    uint32_t tgt_data_idx=0;
    for (uint32_t spectra=0; spectra < num_spectra_in_step; spectra++) {
        int twiddle_index=spectra << (num_steps-step);
        for (uint32_t point=0; point < domain_size; point+=increment_next_point_in_step) {
            uint32_t d0_data_index=spectra + point;
            uint32_t d1_data_index=spectra + point + matching_second_point;
            write_cb_data0_addr[tgt_data_idx]=in_data[d0_data_index];
            //DPRINT << "Data step="<<U32(step)<<" spectra="<<U32(spectra)<<" point="<<U32(point)<<" is "<<in_data[d0_data_index] << "\n";
            write_cb_data1_addr[tgt_data_idx]=in_data[d1_data_index];            
            twiddle_addr[tgt_data_idx]=twiddle_data[(twiddle_index*2)+twiddle_offset];
            //DPRINT << "Target twiddle="<<tgt_data_idx<< ENDL();
            //DPRINT << "STEP="<<step<<" twiddle index " << twiddle_index << " offset "<< twiddle_offset << " val = "<< twiddle_addr[tgt_data_idx] <<ENDL();
            tgt_data_idx++;
        }
    }
    cb_push_back(cb_data0_id, 1);
    cb_push_back(cb_data1_id, 1);
    cb_push_back(cb_twiddle, 1);
}

void bitreverse(float * data, int n) {
  int j=0;
  for (int i=0;i<n-1;i++) {
    if (i < j) {
      float temp_val=data[i];
      data[i]=data[j];
      data[j]=temp_val;
    }
    int k=n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j+=k;
  }
}

int getLog(int n) {
   int logn=0;
   n >>= 1;
   while ((n >>=1) > 0) {
      logn++;
   }
   return logn;
}
