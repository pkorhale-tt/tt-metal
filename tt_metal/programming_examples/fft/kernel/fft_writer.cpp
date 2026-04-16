// ============================================================
//  fft_writer.cpp  –  NCRISC (data mover out / NOC sender)
//
//  Local stages: pop CB_OUT, bounce back for next stage reader.
//  NOC stages:
//    1. Send local output chunk to each peer's scratch CB
//    2. noc_async_write_barrier()
//    3. noc_semaphore_inc() each peer  (data landed signal)
//    4. noc_semaphore_wait() until all peers signal back
//    5. cb_push_back(CB_SYNC) to unblock compute
//  Final stage: write results to DRAM output buffer.
// ============================================================

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

// Peer table entry (packed from runtime args)
struct PeerInfo {
    uint32_t noc_x, noc_y;
    uint32_t scratch_r, scratch_i;
    uint32_t sem_addr;
};

// Compute which contiguous block of local elements go to dst_core_id
// at NOC stage s.  For power-of-2 layout: exactly local_N/2 elements,
// starting at either 0 or local_N/2 depending on XOR pattern.
FORCE_INLINE void send_range(
    uint32_t my_id, uint32_t dst_id, uint32_t stage, uint32_t local_N,
    uint32_t elem_bytes,
    uint32_t& off_bytes, uint32_t& cnt_bytes)
{
    uint32_t stride      = 1u << stage;
    uint32_t global_base = my_id * local_N;
    uint32_t first = local_N, count = 0;
    for (uint32_t i = 0; i < local_N; i++) {
        uint32_t g       = global_base + i;
        uint32_t partner = g ^ stride;
        uint32_t owner   = partner / local_N;
        if (owner == dst_id) {
            if (first == local_N) first = i;
            count++;
        }
    }
    off_bytes = first * elem_bytes;
    cnt_bytes = count * elem_bytes;
}

void kernel_main() {
    // Runtime args
    uint32_t lhs_r_addr      = get_arg_val<uint32_t>(RT_CB_R);
    uint32_t lhs_i_addr      = get_arg_val<uint32_t>(RT_CB_I);
    uint32_t scratch_r_addr  = get_arg_val<uint32_t>(RT_SCRATCH_R);
    uint32_t scratch_i_addr  = get_arg_val<uint32_t>(RT_SCRATCH_I);
    uint32_t dram_out_addr   = get_arg_val<uint32_t>(RT_TWIDDLE_DRAM);
    uint32_t num_cores       = get_arg_val<uint32_t>(RT_NUM_CORES);
    uint32_t my_id           = get_arg_val<uint32_t>(RT_MY_CORE_ID);
    uint32_t first_noc_stage = get_arg_val<uint32_t>(RT_FIRST_NOC_STG);
    uint32_t sem_id          = get_arg_val<uint32_t>(RT_SEM_ID);

    // Compile-time args
    uint32_t local_N    = get_compile_time_arg_val(CT_LOCAL_N);
    uint32_t num_stages = get_compile_time_arg_val(CT_NUM_STAGES);
    uint32_t elem_bytes = 4; // fp32

    uint32_t num_local_stages = first_noc_stage;
    uint32_t num_noc_stages   = num_stages - first_noc_stage;
    uint32_t num_peers        = num_cores - 1;

    // Semaphore: resolve from id to volatile pointer
    volatile tt_l1_ptr uint32_t* local_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    // Build peer table
    PeerInfo peers[120];
    for (uint32_t p = 0; p < num_peers; p++) {
        uint32_t base        = RT_PEER_BASE + p * RT_PEER_STRIDE;
        peers[p].noc_x       = get_arg_val<uint32_t>(base + PEER_NOC_X);
        peers[p].noc_y       = get_arg_val<uint32_t>(base + PEER_NOC_Y);
        peers[p].scratch_r   = get_arg_val<uint32_t>(base + PEER_SCRATCH_R);
        peers[p].scratch_i   = get_arg_val<uint32_t>(base + PEER_SCRATCH_I);
        peers[p].sem_addr    = get_arg_val<uint32_t>(base + PEER_SEM_ADDR);
    }

    // ── Local stages: pop OUT, pipeline flows to reader ──────
    for (uint32_t s = 0; s < num_local_stages; s++) {
        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);
        cb_pop_front(CB_OUT_R, 1);
        cb_pop_front(CB_OUT_I, 1);
    }

    // ── NOC stages ───────────────────────────────────────────
    for (uint32_t s = first_noc_stage; s < num_stages; s++) {
        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);

        uint32_t out_r = get_read_ptr(CB_OUT_R);
        uint32_t out_i = get_read_ptr(CB_OUT_I);

        // Send chunk to each peer
        for (uint32_t p = 0; p < num_peers; p++) {
            uint32_t dst_id = (p < my_id) ? p : p + 1;
            uint32_t off, cnt;
            send_range(my_id, dst_id, s, local_N, elem_bytes, off, cnt);
            if (cnt == 0) continue;

            uint64_t dst_r = get_noc_addr(peers[p].noc_x, peers[p].noc_y,
                                          peers[p].scratch_r + off);
            uint64_t dst_i = get_noc_addr(peers[p].noc_x, peers[p].noc_y,
                                          peers[p].scratch_i + off);
            noc_async_write(out_r + off, dst_r, cnt);
            noc_async_write(out_i + off, dst_i, cnt);
        }

        // Barrier: all writes committed before signaling
        noc_async_write_barrier();

        // Signal each peer: data is in their scratch
        for (uint32_t p = 0; p < num_peers; p++) {
            uint64_t peer_sem = get_noc_addr(peers[p].noc_x, peers[p].noc_y,
                                             peers[p].sem_addr);
            noc_semaphore_inc(peer_sem, 1);
        }

        // Wait until all peers have signaled us back
        noc_semaphore_wait(local_sem, num_peers);
        noc_semaphore_set(local_sem, 0);

        // Unblock compute: scratch data is valid
        cb_reserve_back(CB_SYNC, 1);
        cb_push_back(CB_SYNC, 1);

        cb_pop_front(CB_OUT_R, 1);
        cb_pop_front(CB_OUT_I, 1);
    }

    // ── Final: write results to DRAM ─────────────────────────
    cb_wait_front(CB_OUT_R, 1);
    cb_wait_front(CB_OUT_I, 1);

    uint32_t final_r = get_read_ptr(CB_OUT_R);
    uint32_t final_i = get_read_ptr(CB_OUT_I);

    // Use InterleavedAddrGenFast for tile-based writes
    const DataFormat df        = get_dataformat(CB_OUT_R);
    const uint32_t   tile_size = get_tile_size(CB_OUT_R);

    InterleavedAddrGenFast<true> dram_out = {
        .bank_base_address = dram_out_addr,
        .page_size         = tile_size,
        .data_format       = df
    };

    uint32_t global_offset = my_id * local_N;

    // Write real and imag output tiles to DRAM
    // Real elements at tile index my_id*2, imag at my_id*2+1
    // (host reads back as interleaved pairs)
    noc_async_write_tile(my_id * 2,     dram_out, final_r);
    noc_async_write_tile(my_id * 2 + 1, dram_out, final_i);
    noc_async_write_barrier();

    cb_pop_front(CB_OUT_R, 1);
    cb_pop_front(CB_OUT_I, 1);
}