// fft_reader.cpp — BRISC data mover in
// Stage 0: load input tiles from DRAM into CB_LHS_R/I.
// Each local stage: compute RHS by copying butterfly partners
//   within L1, load twiddles from DRAM, push RHS+twiddles.
// NOC stages: push twiddles only (scratch arrives via writer).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

FORCE_INLINE uint32_t bit_rev(uint32_t x, uint32_t b) {
    uint32_t r=0;
    for (uint32_t i=0;i<b;i++){r=(r<<1)|(x&1);x>>=1;}
    return r;
}

void kernel_main() {
    uint32_t input_addr    = get_arg_val<uint32_t>(0);
    // 1 = unused
    uint32_t twiddle_addr  = get_arg_val<uint32_t>(2);
    // 3 = unused
    uint32_t local_N       = get_arg_val<uint32_t>(4);
    uint32_t my_id         = get_arg_val<uint32_t>(5);
    uint32_t total_N       = get_arg_val<uint32_t>(6);
    uint32_t num_local_stg = get_arg_val<uint32_t>(7);
    uint32_t num_stages    = get_arg_val<uint32_t>(8);

    const DataFormat df    = get_dataformat(CB_LHS_R);
    const uint32_t tsize   = get_tile_size(CB_LHS_R);
    uint32_t num_cores     = total_N / local_N;
    // Twiddle layout: S*C tiles real, then S*C tiles imag
    uint32_t tw_imag_base  = num_stages * num_cores; // tile index offset

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

    // ── Load input → CB_LHS_R/I ───────────────────────────────
    // Host writes tiles pre-bit-reversed: tile[bit_rev(c)] = data[c]
    // so we just read our own tile index.
    {
        cb_reserve_back(CB_LHS_R, 1);
        cb_reserve_back(CB_LHS_I, 1);
        noc_async_read_tile(my_id,           input_gen, get_write_ptr(CB_LHS_R));
        noc_async_read_tile(my_id + num_cores, input_gen, get_write_ptr(CB_LHS_I));
        noc_async_read_barrier();
        cb_push_back(CB_LHS_R, 1);
        cb_push_back(CB_LHS_I, 1);
    }

    // ── Local stages ──────────────────────────────────────────
    // For each stage: wait for LHS to be ready (pushed by writer recycle
    // or the initial load above), compute RHS from LHS in L1, push twiddles.
    for (uint32_t s=0; s<num_local_stg; s++) {
        uint32_t stride = 1u << s;

        // Wait for LHS to be available
        cb_wait_front(CB_LHS_R, 1);
        cb_wait_front(CB_LHS_I, 1);

        // Build RHS: copy butterfly partners from LHS in L1
        cb_reserve_back(CB_RHS_R, 1);
        cb_reserve_back(CB_RHS_I, 1);

        uint32_t* lhs_r = reinterpret_cast<uint32_t*>(get_read_ptr(CB_LHS_R));
        uint32_t* lhs_i = reinterpret_cast<uint32_t*>(get_read_ptr(CB_LHS_I));
        uint32_t* rhs_r = reinterpret_cast<uint32_t*>(get_write_ptr(CB_RHS_R));
        uint32_t* rhs_i = reinterpret_cast<uint32_t*>(get_write_ptr(CB_RHS_I));

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

        // Load twiddles for this stage
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);
        uint32_t tw_r_tile = s * num_cores + my_id;
        uint32_t tw_i_tile = tw_imag_base + tw_r_tile;
        noc_async_read_tile(tw_r_tile, twiddle_gen, get_write_ptr(CB_TWIDDLE_R));
        noc_async_read_tile(tw_i_tile, twiddle_gen, get_write_ptr(CB_TWIDDLE_I));
        noc_async_read_barrier();
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }

    // ── NOC stages: twiddles only ────────────────────────────
    for (uint32_t s=num_local_stg; s<num_stages; s++) {
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);
        uint32_t tw_r_tile = s * num_cores + my_id;
        uint32_t tw_i_tile = tw_imag_base + tw_r_tile;
        noc_async_read_tile(tw_r_tile, twiddle_gen, get_write_ptr(CB_TWIDDLE_R));
        noc_async_read_tile(tw_i_tile, twiddle_gen, get_write_ptr(CB_TWIDDLE_I));
        noc_async_read_barrier();
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }
}