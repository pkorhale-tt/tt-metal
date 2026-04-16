// fft_writer.cpp — NCRISC
// Local stages: OUT0→EVEN, OUT1→ODD for next stage.
// Final local stage (no NOC): write OUT0 to DRAM output.
// NOC stages: send/sync, push CB_SYNC.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

struct PeerInfo { uint32_t noc_x,noc_y,scratch_r,scratch_i,sem_addr; };

FORCE_INLINE void send_range(
    uint32_t my_id, uint32_t dst_id, uint32_t stage, uint32_t local_N,
    uint32_t elem_bytes, uint32_t& off, uint32_t& cnt)
{
    uint32_t stride=1u<<stage, base=my_id*local_N;
    uint32_t first=local_N, count=0;
    for (uint32_t i=0; i<local_N; i++) {
        if (((base+i)^stride)/local_N==dst_id) {
            if (first==local_N) { first=i; } count++;
        }
    }
    off=first*elem_bytes; cnt=count*elem_bytes;
}

void kernel_main() {
    uint32_t dram_out_r    = get_arg_val<uint32_t>(0);
    uint32_t dram_out_i    = get_arg_val<uint32_t>(1);
    uint32_t scratch_r_addr= get_arg_val<uint32_t>(2);
    uint32_t scratch_i_addr= get_arg_val<uint32_t>(3);
    uint32_t num_cores     = get_arg_val<uint32_t>(4);
    uint32_t my_id         = get_arg_val<uint32_t>(5);
    uint32_t first_noc_stg = get_arg_val<uint32_t>(6);
    uint32_t sem_id        = get_arg_val<uint32_t>(7);
    uint32_t num_tiles     = get_arg_val<uint32_t>(8);

    uint32_t local_N    = get_compile_time_arg_val(0);
    uint32_t num_stages = get_compile_time_arg_val(1);
    uint32_t elem_bytes = 4;
    uint32_t num_peers  = num_cores-1;
    uint32_t num_noc    = num_stages-first_noc_stg;

    volatile tt_l1_ptr uint32_t* local_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    constexpr auto CB_EVEN_R = tt::CBIndex::c_0;
    constexpr auto CB_EVEN_I = tt::CBIndex::c_1;
    constexpr auto CB_ODD_R  = tt::CBIndex::c_2;
    constexpr auto CB_ODD_I  = tt::CBIndex::c_3;
    constexpr auto CB_OUT0_R = tt::CBIndex::c_6;
    constexpr auto CB_OUT0_I = tt::CBIndex::c_7;
    constexpr auto CB_OUT1_R = tt::CBIndex::c_8;
    constexpr auto CB_OUT1_I = tt::CBIndex::c_9;
    constexpr auto CB_SYNC   = tt::CBIndex::c_10;

    PeerInfo peers[120];
    for (uint32_t p=0; p<num_peers; p++) {
        uint32_t base=9+p*5;
        peers[p].noc_x    =get_arg_val<uint32_t>(base+0);
        peers[p].noc_y    =get_arg_val<uint32_t>(base+1);
        peers[p].scratch_r=get_arg_val<uint32_t>(base+2);
        peers[p].scratch_i=get_arg_val<uint32_t>(base+3);
        peers[p].sem_addr =get_arg_val<uint32_t>(base+4);
    }

    // ── Local stages ─────────────────────────────────────────
    for (uint32_t s=0; s<first_noc_stg; s++) {
        cb_wait_front(CB_OUT0_R,1); cb_wait_front(CB_OUT0_I,1);
        cb_wait_front(CB_OUT1_R,1); cb_wait_front(CB_OUT1_I,1);

        bool has_next = (s < first_noc_stg-1) || (num_noc > 0);

        if (has_next) {
            // Recycle: OUT0 → EVEN, OUT1 → ODD for next stage
            uint32_t* o0r = reinterpret_cast<uint32_t*>(get_read_ptr(CB_OUT0_R));
            uint32_t* o0i = reinterpret_cast<uint32_t*>(get_read_ptr(CB_OUT0_I));
            uint32_t* o1r = reinterpret_cast<uint32_t*>(get_read_ptr(CB_OUT1_R));
            uint32_t* o1i = reinterpret_cast<uint32_t*>(get_read_ptr(CB_OUT1_I));

            cb_reserve_back(CB_EVEN_R,1); cb_reserve_back(CB_EVEN_I,1);
            cb_reserve_back(CB_ODD_R,1);  cb_reserve_back(CB_ODD_I,1);
            uint32_t* er = reinterpret_cast<uint32_t*>(get_write_ptr(CB_EVEN_R));
            uint32_t* ei = reinterpret_cast<uint32_t*>(get_write_ptr(CB_EVEN_I));
            uint32_t* or_ = reinterpret_cast<uint32_t*>(get_write_ptr(CB_ODD_R));
            uint32_t* oi = reinterpret_cast<uint32_t*>(get_write_ptr(CB_ODD_I));

            uint32_t half = local_N/2;
            // Lower half: OUT0 (even + W*odd)
            for (uint32_t i=0;i<half;i++){er[i]=o0r[i];ei[i]=o0i[i];}
            // Upper half: OUT1 (even - W*odd)
            for (uint32_t i=0;i<half;i++){er[half+i]=o1r[i];ei[half+i]=o1i[i];}
            // Next stage's odd
            for (uint32_t i=0;i<half;i++){or_[i]=o0r[half+i];oi[i]=o0i[half+i];}
            for (uint32_t i=0;i<half;i++){or_[half+i]=o1r[half+i];oi[half+i]=o1i[half+i];}

            cb_pop_front(CB_OUT0_R,1); cb_pop_front(CB_OUT0_I,1);
            cb_pop_front(CB_OUT1_R,1); cb_pop_front(CB_OUT1_I,1);

            cb_push_back(CB_EVEN_R,1); cb_push_back(CB_EVEN_I,1);
            cb_push_back(CB_ODD_R,1);  cb_push_back(CB_ODD_I,1);
        }
        // If last stage with no NOC: leave OUT0/1 for final write
    }

    // ── NOC stages ───────────────────────────────────────────
    for (uint32_t s=first_noc_stg; s<num_stages; s++) {
        cb_wait_front(CB_OUT0_R,1); cb_wait_front(CB_OUT0_I,1);
        uint32_t out_r=get_read_ptr(CB_OUT0_R), out_i=get_read_ptr(CB_OUT0_I);

        for (uint32_t p=0; p<num_peers; p++) {
            uint32_t dst=(p<my_id)?p:p+1;
            uint32_t off,cnt;
            send_range(my_id,dst,s,local_N,elem_bytes,off,cnt);
            if (!cnt) continue;
            noc_async_write(out_r+off, get_noc_addr(peers[p].noc_x,peers[p].noc_y,peers[p].scratch_r+off),cnt);
            noc_async_write(out_i+off, get_noc_addr(peers[p].noc_x,peers[p].noc_y,peers[p].scratch_i+off),cnt);
        }
        noc_async_write_barrier();
        for (uint32_t p=0; p<num_peers; p++)
            noc_semaphore_inc(get_noc_addr(peers[p].noc_x,peers[p].noc_y,peers[p].sem_addr),1);
        noc_semaphore_wait(local_sem,num_peers);
        noc_semaphore_set(local_sem,0);

        cb_reserve_back(CB_SYNC,1); cb_push_back(CB_SYNC,1);
        cb_pop_front(CB_OUT0_R,1); cb_pop_front(CB_OUT0_I,1);
    }

    // ── Final: write OUT0 to DRAM ─────────────────────────────
    cb_wait_front(CB_OUT0_R,1); cb_wait_front(CB_OUT0_I,1);

    const DataFormat df=get_dataformat(CB_OUT0_R);
    const uint32_t tsize=get_tile_size(CB_OUT0_R);
    InterleavedAddrGenFast<true> gen_r={.bank_base_address=dram_out_r,.page_size=tsize,.data_format=df};
    InterleavedAddrGenFast<true> gen_i={.bank_base_address=dram_out_i,.page_size=tsize,.data_format=df};
    noc_async_write_tile(my_id, gen_r, get_read_ptr(CB_OUT0_R));
    noc_async_write_tile(my_id, gen_i, get_read_ptr(CB_OUT0_I));
    noc_async_write_barrier();

    cb_pop_front(CB_OUT0_R,1); cb_pop_front(CB_OUT0_I,1);
}