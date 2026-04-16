// fft_reader.cpp — BRISC
// Loads input once, then pushes RHS+twiddles each local stage,
// twiddles only each NOC stage.
// Does NOT manage LHS — that's between compute and writer.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

void kernel_main() {
    uint32_t input_addr    = get_arg_val<uint32_t>(0);
    uint32_t twiddle_addr  = get_arg_val<uint32_t>(2);
    uint32_t local_N       = get_arg_val<uint32_t>(4);
    uint32_t my_id         = get_arg_val<uint32_t>(5);
    uint32_t total_N       = get_arg_val<uint32_t>(6);
    uint32_t num_local_stg = get_arg_val<uint32_t>(7);
    uint32_t num_stages    = get_arg_val<uint32_t>(8);

    const DataFormat df  = get_dataformat(CB_LHS_R);
    const uint32_t tsize = get_tile_size(CB_LHS_R);
    uint32_t num_cores   = total_N / local_N;
    uint32_t tw_imag_base = num_stages * num_cores;

    InterleavedAddrGenFast<true> input_gen = {
        .bank_base_address = input_addr,
        .page_size         = tsize,
        .data_format       = df
    };
    InterleavedAddrGenFast<true> twiddle_gen = {
        .bank_base_address = twiddle_addr,
        .page_size         = tsize,
        .data_format       = df
    };

    // ── Load input → CB_LHS_R/I (stage 0 seed) ───────────────
    cb_reserve_back(CB_LHS_R, 1);
    cb_reserve_back(CB_LHS_I, 1);
    noc_async_read_tile(my_id,            input_gen, get_write_ptr(CB_LHS_R));
    noc_async_read_tile(my_id + num_cores, input_gen, get_write_ptr(CB_LHS_I));
    noc_async_read_barrier();
    cb_push_back(CB_LHS_R, 1);
    cb_push_back(CB_LHS_I, 1);

    // ── Local stages: push RHS (from CB_LHS snapshot) + twiddles
    // Reader waits for LHS to be available, snapshots it into RHS,
    // then immediately releases LHS back (pop) so writer can recycle.
    // Actually: reader just reads LHS without popping — compute pops it.
    // But reader needs to BUILD RHS from LHS data.
    // Solution: reader peeks LHS (wait_front but no pop), builds RHS,
    // then compute will pop LHS in butterfly.
    for (uint32_t s=0; s<num_local_stg; s++) {
        uint32_t stride = 1u << s;

        // Wait for LHS data to be present (pushed by initial load or writer recycle)
        cb_wait_front(CB_LHS_R, 1);
        cb_wait_front(CB_LHS_I, 1);

        // Build RHS by copying butterfly partners from LHS
        cb_reserve_back(CB_RHS_R, 1);
        cb_reserve_back(CB_RHS_I, 1);

        const uint32_t* lhs_r = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_LHS_R));
        const uint32_t* lhs_i = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_LHS_I));
        uint32_t*       rhs_r = reinterpret_cast<uint32_t*>(get_write_ptr(CB_RHS_R));
        uint32_t*       rhs_i = reinterpret_cast<uint32_t*>(get_write_ptr(CB_RHS_I));

        for (uint32_t i=0; i<local_N/2; i++) {
            uint32_t grp = i / stride;
            uint32_t pos = i % stride;
            uint32_t lo  = grp*(2*stride) + pos;
            uint32_t hi  = lo + stride;
            rhs_r[lo] = lhs_r[hi];
            rhs_i[lo] = lhs_i[hi];
        }
        cb_push_back(CB_RHS_R, 1);
        cb_push_back(CB_RHS_I, 1);

        // Load twiddles
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);
        uint32_t tw_r = s * num_cores + my_id;
        uint32_t tw_i = tw_imag_base + tw_r;
        noc_async_read_tile(tw_r, twiddle_gen, get_write_ptr(CB_TWIDDLE_R));
        noc_async_read_tile(tw_i, twiddle_gen, get_write_ptr(CB_TWIDDLE_I));
        noc_async_read_barrier();
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);

        // Wait for compute to consume LHS (pop it) before writer recycles
        // Actually the writer waits on CB_OUT then recycles LHS.
        // We need to wait for the PREVIOUS stage's OUT to have been recycled
        // before we try to wait_front on CB_LHS again.
        // Since we already did cb_wait_front(CB_LHS) above, and compute will
        // pop it, we just need to wait for CB_LHS to be empty before looping.
        // Use CB_OUT_R as a signal: wait for it to have a tile (compute done),
        // then the writer will recycle.
        // SIMPLER: just use a sync tile: writer pushes CB_SYNC after recycling LHS.
        // For now: wait for CB_LHS to be consumed (count goes to 0).
        // There's no direct "wait_empty" — instead we'll use a handshake via
        // a dedicated sync CB. But that adds complexity.
        // SIMPLEST: after pushing RHS+twiddles, wait for compute to signal via CB_SYNC.
        // But CB_SYNC is used for NOC sync...
        // ACTUAL SOLUTION: just don't wait here. The pipeline self-regulates:
        // - Reader peeks LHS (cb_wait_front without pop)
        // - Compute pops LHS when it does the butterfly
        // - Writer then recycles by pushing new LHS
        // - Reader's next cb_wait_front(CB_LHS) blocks until writer pushes
        // This is correct! The reader's cb_wait_front at the TOP of the loop
        // will block until writer pushes the new LHS after recycling.
        // No extra sync needed.
    }

    // ── NOC stages: twiddles only ────────────────────────────
    for (uint32_t s=num_local_stg; s<num_stages; s++) {
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);
        uint32_t tw_r = s * num_cores + my_id;
        uint32_t tw_i = tw_imag_base + tw_r;
        noc_async_read_tile(tw_r, twiddle_gen, get_write_ptr(CB_TWIDDLE_R));
        noc_async_read_tile(tw_i, twiddle_gen, get_write_ptr(CB_TWIDDLE_I));
        noc_async_read_barrier();
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }
}