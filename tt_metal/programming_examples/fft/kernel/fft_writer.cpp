// fft_writer.cpp — NCRISC
// Local stages: wait CB_OUT, copy to LHS L1, push CB_LHS (for all but last local stage).
// Last local stage (if no NOC stages): leave OUT for final DRAM write.
// NOC stages: send/signal/wait/sync.
// Final: write output to DRAM.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

struct PeerInfo { uint32_t noc_x, noc_y, scratch_r, scratch_i, sem_addr; };

FORCE_INLINE void send_range(
    uint32_t my_id, uint32_t dst_id, uint32_t stage, uint32_t local_N,
    uint32_t elem_bytes, uint32_t& off, uint32_t& cnt)
{
    uint32_t stride = 1u<<stage, base = my_id*local_N;
    uint32_t first = local_N, count = 0;
    for (uint32_t i=0; i<local_N; i++) {
        if (((base+i)^stride)/local_N == dst_id) {
            if (first==local_N) first=i; count++;
        }
    }
    off = first*elem_bytes; cnt = count*elem_bytes;
}

void kernel_main() {
    uint32_t lhs_r_addr    = get_arg_val<uint32_t>(RT_CB_R);
    uint32_t lhs_i_addr    = get_arg_val<uint32_t>(RT_CB_I);
    uint32_t scratch_r_addr= get_arg_val<uint32_t>(RT_SCRATCH_R);
    uint32_t scratch_i_addr= get_arg_val<uint32_t>(RT_SCRATCH_I);
    uint32_t dram_out_addr = get_arg_val<uint32_t>(RT_TWIDDLE_DRAM);
    uint32_t num_cores     = get_arg_val<uint32_t>(RT_NUM_CORES);
    uint32_t my_id         = get_arg_val<uint32_t>(RT_MY_CORE_ID);
    uint32_t first_noc_stg = get_arg_val<uint32_t>(RT_FIRST_NOC_STG);
    uint32_t sem_id        = get_arg_val<uint32_t>(RT_SEM_ID);

    uint32_t local_N    = get_compile_time_arg_val(CT_LOCAL_N);
    uint32_t num_stages = get_compile_time_arg_val(CT_NUM_STAGES);
    uint32_t elem_bytes = 4;
    uint32_t num_peers  = num_cores - 1;
    uint32_t num_noc    = num_stages - first_noc_stg;

    volatile tt_l1_ptr uint32_t* local_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    PeerInfo peers[120];
    for (uint32_t p=0; p<num_peers; p++) {
        uint32_t base     = RT_PEER_BASE + p*RT_PEER_STRIDE;
        peers[p].noc_x    = get_arg_val<uint32_t>(base+PEER_NOC_X);
        peers[p].noc_y    = get_arg_val<uint32_t>(base+PEER_NOC_Y);
        peers[p].scratch_r= get_arg_val<uint32_t>(base+PEER_SCRATCH_R);
        peers[p].scratch_i= get_arg_val<uint32_t>(base+PEER_SCRATCH_I);
        peers[p].sem_addr = get_arg_val<uint32_t>(base+PEER_SEM_ADDR);
    }

    // ── Local stages ─────────────────────────────────────────
    for (uint32_t s=0; s<first_noc_stg; s++) {
        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);

        bool is_last = (s == first_noc_stg - 1) && (num_noc == 0);

        if (!is_last) {
            // Copy OUT into LHS L1, then push CB_LHS for next stage's reader
            const uint32_t* out_r = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_OUT_R));
            const uint32_t* out_i = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_OUT_I));
            uint32_t* lhs_r = reinterpret_cast<uint32_t*>(lhs_r_addr);
            uint32_t* lhs_i = reinterpret_cast<uint32_t*>(lhs_i_addr);
            for (uint32_t i=0; i<local_N; i++) { lhs_r[i]=out_r[i]; lhs_i[i]=out_i[i]; }

            cb_pop_front(CB_OUT_R, 1);
            cb_pop_front(CB_OUT_I, 1);

            cb_reserve_back(CB_LHS_R, 1);
            cb_reserve_back(CB_LHS_I, 1);
            cb_push_back(CB_LHS_R, 1);
            cb_push_back(CB_LHS_I, 1);
        }
        // If last local stage with no NOC stages: leave OUT for final write below
    }

    // ── NOC stages ───────────────────────────────────────────
    for (uint32_t s=first_noc_stg; s<num_stages; s++) {
        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);

        uint32_t out_r = get_read_ptr(CB_OUT_R);
        uint32_t out_i = get_read_ptr(CB_OUT_I);

        for (uint32_t p=0; p<num_peers; p++) {
            uint32_t dst = (p<my_id)?p:p+1;
            uint32_t off, cnt;
            send_range(my_id, dst, s, local_N, elem_bytes, off, cnt);
            if (!cnt) continue;
            noc_async_write(out_r+off,
                get_noc_addr(peers[p].noc_x,peers[p].noc_y,peers[p].scratch_r+off),cnt);
            noc_async_write(out_i+off,
                get_noc_addr(peers[p].noc_x,peers[p].noc_y,peers[p].scratch_i+off),cnt);
        }
        noc_async_write_barrier();

        for (uint32_t p=0; p<num_peers; p++)
            noc_semaphore_inc(
                get_noc_addr(peers[p].noc_x,peers[p].noc_y,peers[p].sem_addr), 1);

        noc_semaphore_wait(local_sem, num_peers);
        noc_semaphore_set(local_sem, 0);

        cb_reserve_back(CB_SYNC, 1);
        cb_push_back(CB_SYNC, 1);

        cb_pop_front(CB_OUT_R, 1);
        cb_pop_front(CB_OUT_I, 1);
    }

    // ── Final: write to DRAM ──────────────────────────────────
    // If last local stage had no NOC stages, OUT is still waiting.
    // If NOC stages ran, OUT has final result waiting.
    cb_wait_front(CB_OUT_R, 1);
    cb_wait_front(CB_OUT_I, 1);

    const DataFormat df  = get_dataformat(CB_OUT_R);
    const uint32_t tsize = get_tile_size(CB_OUT_R);
    InterleavedAddrGenFast<true> gen = {
        .bank_base_address = dram_out_addr,
        .page_size         = tsize,
        .data_format       = df
    };
    noc_async_write_tile(my_id,            gen, get_read_ptr(CB_OUT_R));
    noc_async_write_tile(my_id + num_cores, gen, get_read_ptr(CB_OUT_I));
    noc_async_write_barrier();

    cb_pop_front(CB_OUT_R, 1);
    cb_pop_front(CB_OUT_I, 1);
}