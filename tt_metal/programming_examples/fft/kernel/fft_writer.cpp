// ============================================================
//  fft_writer.cpp  –  BRISC-1 (data mover out)
//
//  Responsibilities:
//    1. Local stages: write compute results back into LHS CBs
//       so the next stage's reader can reorder from them
//    2. NOC stages:
//       a. Send our chunk to each peer core (noc_async_write)
//       b. Barrier + semaphore-inc to each peer (data ready)
//       c. Wait for (num_cores-1) incoming sems (all peers sent to us)
//       d. Push CB_SYNC tile to unblock compute kernel
//    3. Final stage: write results to DRAM output buffer
// ============================================================

#include "dataflow_api.h"
#include "fft_common.h"

// ── NOC address helper ───────────────────────────────────────
FORCE_INLINE uint64_t peer_noc_addr(
    uint32_t peer_noc_x, uint32_t peer_noc_y, uint32_t l1_addr)
{
    return get_noc_addr(peer_noc_x, peer_noc_y, l1_addr);
}

// ── Compute which chunk of local data to send to a given peer
// at a given NOC stage.
//
// For stage s (s >= log2(local_N)):
//   stride = 1 << s
//   Each core owns global indices [my_id*local_N .. (my_id+1)*local_N - 1].
//   The butterfly partner for element g at stage s is:
//     partner_g = g ^ stride   (XOR with stride flips the bit at position s)
//   The core that owns partner_g is: partner_g / local_N
//
// So we need to send all our elements whose partner lives on `dst_core_id`.
// Since stride >= local_N at NOC stages, exactly half our elements go to
// exactly one partner core per stage (for power-of-2 N and cores).
//
// Returns: offset into local L1 buffer of the first element to send,
//          and the count (always local_N/2 for clean power-of-2 case).
FORCE_INLINE void compute_send_range(
    uint32_t my_core_id,
    uint32_t dst_core_id,
    uint32_t stage,
    uint32_t local_N,
    uint32_t& out_offset_bytes,  // byte offset into local CB
    uint32_t& out_count_bytes,   // byte count to send
    uint32_t  elem_bytes)
{
    uint32_t stride       = 1u << stage;
    uint32_t global_base  = my_core_id * local_N;

    // Walk our elements and collect those whose butterfly partner
    // lives on dst_core_id.  For power-of-2 layout these are always
    // a contiguous block of local_N/2 elements starting at either
    // offset 0 or offset local_N/2 depending on the XOR pattern.

    uint32_t first_elem = local_N; // sentinel
    uint32_t count      = 0;

    for (uint32_t i = 0; i < local_N; i++) {
        uint32_t g         = global_base + i;
        uint32_t partner_g = g ^ stride;
        uint32_t owner     = partner_g / local_N;
        if (owner == dst_core_id) {
            if (first_elem == local_N) first_elem = i;
            count++;
        }
    }

    out_offset_bytes = first_elem * elem_bytes;
    out_count_bytes  = count * elem_bytes;
}

void kernel_main() {
    // ── Runtime args ────────────────────────────────────────
    uint32_t local_cb_r_addr   = get_arg_val<uint32_t>(RT_CB_R);
    uint32_t local_cb_i_addr   = get_arg_val<uint32_t>(RT_CB_I);
    uint32_t scratch_r_addr    = get_arg_val<uint32_t>(RT_SCRATCH_R);
    uint32_t scratch_i_addr    = get_arg_val<uint32_t>(RT_SCRATCH_I);
    uint32_t dram_output_addr  = get_arg_val<uint32_t>(RT_TWIDDLE_DRAM);  // reused slot
    uint32_t dram_output_bank  = get_arg_val<uint32_t>(RT_TWIDDLE_BANK);
    uint32_t num_cores         = get_arg_val<uint32_t>(RT_NUM_CORES);
    uint32_t my_core_id        = get_arg_val<uint32_t>(RT_MY_CORE_ID);
    uint32_t first_noc_stage   = get_arg_val<uint32_t>(RT_FIRST_NOC_STG);
    uint32_t sem_id            = get_arg_val<uint32_t>(RT_SEM_ID);

    uint32_t use_bf16    = get_compile_time_arg_val(CT_USE_BF16);
    uint32_t local_N     = get_compile_time_arg_val(CT_LOCAL_N);
    uint32_t num_cores_  = get_compile_time_arg_val(CT_NUM_CORES);
    uint32_t num_stages  = get_compile_time_arg_val(CT_NUM_STAGES);

    uint32_t elem_bytes  = use_bf16 ? 2 : 4;
    uint32_t num_local_stages = first_noc_stage;        // = log2(local_N)
    uint32_t num_noc_stages   = num_stages - first_noc_stage;

    // Resolve semaphore L1 address for this core
    uint32_t local_sem_addr = get_semaphore(sem_id);

    // ── Build peer table from runtime args ───────────────────
    // Peers are listed in order, skipping my_core_id.
    // Each peer: [noc_x, noc_y, scratch_r, scratch_i, sem_addr]
    struct PeerInfo {
        uint32_t noc_x, noc_y;
        uint32_t scratch_r, scratch_i;
        uint32_t sem_addr;
    };
    // Max 120 Tensix cores on n300
    PeerInfo peers[120];
    uint32_t num_peers = 0;
    for (uint32_t p = 0; p < num_cores - 1; p++) {
        uint32_t base = RT_PEER_BASE + p * RT_PEER_STRIDE;
        peers[p].noc_x     = get_arg_val<uint32_t>(base + PEER_NOC_X);
        peers[p].noc_y     = get_arg_val<uint32_t>(base + PEER_NOC_Y);
        peers[p].scratch_r = get_arg_val<uint32_t>(base + PEER_SCRATCH_R);
        peers[p].scratch_i = get_arg_val<uint32_t>(base + PEER_SCRATCH_I);
        peers[p].sem_addr  = get_arg_val<uint32_t>(base + PEER_SEM_ADDR);
        num_peers++;
    }

    // ── LOCAL STAGES: pass-through ───────────────────────────
    // During local stages, compute kernel writes results back into
    // CB_OUT_R/I. Writer just waits for those and bounces them
    // back into CB_LHS_R/I for the next stage's reader.
    // (In a chunked pipeline this is handled by CB double-buffering;
    //  here we do the explicit wait+push pattern for clarity.)
    for (uint32_t s = 0; s < num_local_stages; s++) {
        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);

        // The output is now the input for the next stage.
        // In practice the compute kernel writes results into a
        // separate output CB and the reader picks them up.
        // We just release them here so the pipeline can proceed.
        cb_pop_front(CB_OUT_R, 1);
        cb_pop_front(CB_OUT_I, 1);
    }

    // ── NOC STAGES ───────────────────────────────────────────
    for (uint32_t s = first_noc_stage; s < num_stages; s++) {

        // Wait for compute to finish the previous stage and
        // write its results into the local output CB
        cb_wait_front(CB_OUT_R, 1);
        cb_wait_front(CB_OUT_I, 1);

        uint32_t out_r_ptr = get_read_ptr(CB_OUT_R);
        uint32_t out_i_ptr = get_read_ptr(CB_OUT_I);

        // ── SEND to all peers ────────────────────────────────
        for (uint32_t p = 0; p < num_peers; p++) {
            uint32_t dst_core_id = (p < my_core_id) ? p : p + 1;

            uint32_t offset_bytes, count_bytes;
            compute_send_range(
                my_core_id, dst_core_id, s,
                local_N, offset_bytes, count_bytes, elem_bytes);

            if (count_bytes == 0) continue;  // no data for this peer at this stage

            // Send real part
            uint64_t dst_r = peer_noc_addr(
                peers[p].noc_x, peers[p].noc_y,
                peers[p].scratch_r + offset_bytes);
            noc_async_write(out_r_ptr + offset_bytes, dst_r, count_bytes);

            // Send imag part
            uint64_t dst_i = peer_noc_addr(
                peers[p].noc_x, peers[p].noc_y,
                peers[p].scratch_i + offset_bytes);
            noc_async_write(out_i_ptr + offset_bytes, dst_i, count_bytes);
        }

        // ── BARRIER: wait until ALL writes have committed ────
        // This is mandatory before semaphore signals.
        // Without this, the semaphore can arrive before the data.
        noc_async_write_barrier();

        // ── SIGNAL all peers: data is in their scratch ───────
        for (uint32_t p = 0; p < num_peers; p++) {
            uint64_t peer_sem = peer_noc_addr(
                peers[p].noc_x, peers[p].noc_y,
                peers[p].sem_addr);
            noc_semaphore_inc(peer_sem, 1);
            // NOC semaphore increment is atomic at destination.
            // This is the only safe signal mechanism across cores.
        }

        // ── WAIT: block until all peers have signaled us ─────
        // Each peer increments our semaphore once → wait for count
        // to reach num_peers (= num_cores - 1).
        noc_semaphore_wait(local_sem_addr, num_peers);

        // Reset for next stage BEFORE unblocking compute.
        // If we reset after cb_push_back, compute might race ahead
        // and wait on a semaphore that hasn't been reset yet.
        noc_semaphore_set(local_sem_addr, 0);

        // ── UNBLOCK compute: all scratch data is valid ───────
        // Push a sync tile so the compute kernel's cb_wait_front
        // on CB_SYNC returns and it can proceed with cross-core butterfly.
        cb_reserve_back(CB_SYNC, 1);
        cb_push_back(CB_SYNC, 1);

        // Release previous stage output CBs
        cb_pop_front(CB_OUT_R, 1);
        cb_pop_front(CB_OUT_I, 1);
    }

    // ── FINAL: write results to DRAM ─────────────────────────
    // Wait for last compute stage to finish
    cb_wait_front(CB_OUT_R, 1);
    cb_wait_front(CB_OUT_I, 1);

    uint32_t final_r_ptr = get_read_ptr(CB_OUT_R);
    uint32_t final_i_ptr = get_read_ptr(CB_OUT_I);

    InterleavedAddrGen<true> dram_output = {
        .bank_base_address = dram_output_addr,
        .page_size         = elem_bytes
    };

    uint32_t global_offset = my_core_id * local_N;

    // Write interleaved [real_0, imag_0, real_1, imag_1, ...]
    for (uint32_t i = 0; i < local_N; i++) {
        uint32_t g = global_offset + i;

        uint64_t dst_real = get_noc_addr(
            dram_output.get_bank_id(g * 2),
            dram_output.get_bank_addr(g * 2));
        noc_async_write(final_r_ptr + i * elem_bytes, dst_real, elem_bytes);

        uint64_t dst_imag = get_noc_addr(
            dram_output.get_bank_id(g * 2 + 1),
            dram_output.get_bank_addr(g * 2 + 1));
        noc_async_write(final_i_ptr + i * elem_bytes, dst_imag, elem_bytes);
    }

    noc_async_write_barrier();

    cb_pop_front(CB_OUT_R, 1);
    cb_pop_front(CB_OUT_I, 1);
}
