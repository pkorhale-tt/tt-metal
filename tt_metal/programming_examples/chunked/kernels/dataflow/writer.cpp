#include "dataflow_api.h"
#include "../constants.h"

void write_data_to_external(uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void write_data_to_CB(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void write_stage_data(uint32_t, uint32_t, uint32_t, uint32_t, float*, float*, uint32_t, uint32_t, uint32_t);
inline void popfront_cbs(uint32_t, uint32_t, uint32_t, uint32_t);
inline void waitfront_cbs(uint32_t, uint32_t, uint32_t, uint32_t, float**, float**, float**, float**);
int getLog(int);

void kernel_main() {
    uint32_t data_r_addr = get_arg_val<uint32_t>(0);
    uint32_t data_i_addr = get_arg_val<uint32_t>(1);
    uint32_t data_r_bank_id = get_arg_val<uint32_t>(2);
    uint32_t data_i_bank_id = get_arg_val<uint32_t>(3);
    uint32_t domain_size = get_arg_val<uint32_t>(4);

    constexpr auto cb_out_data0_r = tt::CBIndex::c_6;
    constexpr auto cb_out_data0_i = tt::CBIndex::c_7;
    constexpr auto cb_out_data1_r = tt::CBIndex::c_8;
    constexpr auto cb_out_data1_i = tt::CBIndex::c_9;

    constexpr auto cb_out_data_r = tt::CBIndex::c_10;
    constexpr auto cb_out_data_i = tt::CBIndex::c_11;

    uint32_t number_chunks = (domain_size /2) / CHUNK_SIZE;
    if (number_chunks * CHUNK_SIZE < domain_size/2) number_chunks++;

    int num_steps=getLog(domain_size);
    for (int step=0; step < num_steps; step++) {
        write_data_to_CB(cb_out_data_r, cb_out_data_i, 
                            cb_out_data0_r, cb_out_data0_i, cb_out_data1_r, cb_out_data1_i, domain_size, number_chunks, step);
    }

    uint64_t data_r_noc_addr = get_noc_addr_from_bank_id<true>(data_r_bank_id, data_r_addr);
    uint64_t data_i_noc_addr = get_noc_addr_from_bank_id<true>(data_i_bank_id, data_i_addr);

    write_data_to_external(data_r_noc_addr, data_i_noc_addr, cb_out_data_r, cb_out_data_i, 
                            cb_out_data0_r, cb_out_data0_i, cb_out_data1_r, cb_out_data1_i, domain_size, number_chunks, num_steps);
}

void write_data_to_external(uint64_t data_r_noc_addr, uint64_t data_i_noc_addr, uint32_t cb_target_r_id, uint32_t cb_target_i_id, 
                                uint32_t cb_data0_r_id, uint32_t cb_data0_i_id, uint32_t cb_data1_r_id, uint32_t cb_data1_i_id, 
                                uint32_t domain_size, uint32_t number_chunks, uint32_t step) {
    // We use the target CB as a memory staging area to use for data reordering, then write out to DDR
    cb_reserve_back(cb_target_r_id, 1);
    cb_reserve_back(cb_target_i_id, 1);
    float * write_cb_target_r_addr = (float*) get_write_ptr(cb_target_r_id);
    float * write_cb_target_i_addr = (float*) get_write_ptr(cb_target_i_id);

    write_stage_data(cb_data0_r_id, cb_data0_i_id, cb_data1_r_id, cb_data1_i_id, 
                        write_cb_target_r_addr, write_cb_target_i_addr, domain_size, number_chunks, step); 

    noc_async_write((uint32_t) write_cb_target_r_addr, data_r_noc_addr, domain_size * 4);
    noc_async_write((uint32_t) write_cb_target_i_addr, data_i_noc_addr, domain_size * 4);
    noc_async_write_barrier();

    cb_push_back(cb_target_r_id, 1);
    cb_push_back(cb_target_i_id, 1);
}

void write_data_to_CB(uint32_t cb_target_r_id, uint32_t cb_target_i_id, uint32_t cb_data0_r_id, uint32_t cb_data0_i_id, uint32_t cb_data1_r_id, uint32_t cb_data1_i_id, 
                        uint32_t domain_size, uint32_t number_chunks, uint32_t step) {
    cb_reserve_back(cb_target_r_id, 1);
    cb_reserve_back(cb_target_i_id, 1);
    float * write_cb_target_r_addr = (float*) get_write_ptr(cb_target_r_id);
    float * write_cb_target_i_addr = (float*) get_write_ptr(cb_target_i_id);
    write_stage_data(cb_data0_r_id, cb_data0_i_id, cb_data1_r_id, cb_data1_i_id, write_cb_target_r_addr, write_cb_target_i_addr, domain_size, number_chunks, step);
    cb_push_back(cb_target_r_id, 1);
    cb_push_back(cb_target_i_id, 1);
}

void write_stage_data(uint32_t cb_data0_r_id, uint32_t cb_data0_i_id, uint32_t cb_data1_r_id, uint32_t cb_data1_i_id, float * out_r_data, float * out_i_data, 
                        uint32_t domain_size, uint32_t number_chunks, uint32_t step) {
    float *read_cb_data0_r_addr, *read_cb_data0_i_addr, *read_cb_data1_r_addr, *read_cb_data1_i_addr;

    waitfront_cbs(cb_data0_r_id, cb_data0_i_id, cb_data1_r_id, cb_data1_i_id, 
                    &read_cb_data0_r_addr, &read_cb_data0_i_addr, &read_cb_data1_r_addr, &read_cb_data1_i_addr);

    uint32_t num_spectra_in_step=step == 0 ? 1 : 2 << (step-1);
    uint32_t increment_next_point_in_step=2 << step;
    uint32_t matching_second_point=increment_next_point_in_step/2;

    uint32_t tgt_data_idx=0, chunks_computed=0;
    for (uint32_t spectra=0; spectra < num_spectra_in_step; spectra++) {
        for (uint32_t point=0; point < domain_size; point+=increment_next_point_in_step) {
            uint32_t d0_data_index=spectra + point;
            uint32_t d1_data_index=spectra + point + matching_second_point;
            //DPRINT << "Data 0 step="<<U32(step)<<" spectra="<<U32(spectra)<<" point="<<U32(point)<<" is "<<read_cb_data0_addr[tgt_data_idx] << " at index " <<tgt_data_idx << " write to "<<d0_data_index<<ENDL();
            //DPRINT << "Data 1 step="<<U32(step)<<" spectra="<<U32(spectra)<<" point="<<U32(point)<<" is "<<read_cb_data1_addr[tgt_data_idx] << " at index " << tgt_data_idx << " write to "<<d1_data_index<<ENDL();
            out_r_data[d0_data_index]=read_cb_data0_r_addr[tgt_data_idx];
            out_r_data[d1_data_index]=read_cb_data1_r_addr[tgt_data_idx];
            out_i_data[d0_data_index]=read_cb_data0_i_addr[tgt_data_idx];
            out_i_data[d1_data_index]=read_cb_data1_i_addr[tgt_data_idx];
            //DPRINT << "Data 0 step="<<U32(step)<<" spectra="<<U32(spectra)<<" point="<<U32(point)<<" is "<<out_data[d0_data_index] << " at index " << d0_data_index << "\n";
            //DPRINT << "Data 1 step="<<U32(step)<<" spectra="<<U32(spectra)<<" point="<<U32(point)<<" is "<<out_data[d1_data_index] << " at index " << d1_data_index << "\n";
            tgt_data_idx++;
            if (tgt_data_idx == CHUNK_SIZE) {
                chunks_computed++;
                if (chunks_computed < number_chunks) {
                    popfront_cbs(cb_data0_r_id, cb_data0_i_id, cb_data1_r_id, cb_data1_i_id);                    
                    waitfront_cbs(cb_data0_r_id, cb_data0_i_id, cb_data1_r_id, cb_data1_i_id,
                                    &read_cb_data0_r_addr, &read_cb_data0_i_addr, &read_cb_data1_r_addr, &read_cb_data1_i_addr);
                    tgt_data_idx=0;
                }                
            }
        }
    }
    popfront_cbs(cb_data0_r_id, cb_data0_i_id, cb_data1_r_id, cb_data1_i_id);
}

inline void popfront_cbs(uint32_t cb_data0_r_id, uint32_t cb_data0_i_id, uint32_t cb_data1_r_id, uint32_t cb_data1_i_id) {
    cb_pop_front(cb_data1_r_id, 1);
    cb_pop_front(cb_data1_i_id, 1);
    cb_pop_front(cb_data0_r_id, 1);
    cb_pop_front(cb_data0_i_id, 1);
}

inline void waitfront_cbs(uint32_t cb_data0_r_id, uint32_t cb_data0_i_id, uint32_t cb_data1_r_id, uint32_t cb_data1_i_id,
                            float ** read_cb_data0_r_addr, float ** read_cb_data0_i_addr, float ** read_cb_data1_r_addr, float ** read_cb_data1_i_addr) {
    cb_wait_front(cb_data1_r_id, 1);
    cb_wait_front(cb_data1_i_id, 1);
    cb_wait_front(cb_data0_r_id, 1);
    cb_wait_front(cb_data0_i_id, 1);

    *read_cb_data0_r_addr = (float*) get_read_ptr(cb_data0_r_id);
    *read_cb_data0_i_addr = (float*) get_read_ptr(cb_data0_i_id);
    *read_cb_data1_r_addr = (float*) get_read_ptr(cb_data1_r_id);
    *read_cb_data1_i_addr = (float*) get_read_ptr(cb_data1_i_id);
}


int getLog(int n) {
   int logn=0;
   n >>= 1;
   while ((n >>=1) > 0) {
      logn++;
   }
   return logn;
}

