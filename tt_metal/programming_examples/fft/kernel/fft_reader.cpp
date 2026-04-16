// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_reader.cpp — BRISC0 / reader (multi-core capable)
//
// Each core owns one 1024-element tile of state in CB_STATE_{R,I}. Across
// log2(N) butterfly stages it:
//
//   1. (stage 0 only) loads its shard of the bit-reversed input from DRAM
//      into CB_STATE.
//   2. for each LOCAL stage (stride < 1024):
//        scatter state -> EVEN/ODD,
//        stream this stage's twiddle tile from DRAM,
//        wait for OUT0/OUT1 from compute,
//        gather back into state.
//   3. for each CROSS-CORE stage (stride >= 1024):
//        NoC-write our state into the partner's CB_RECV buffers and
//          inc the partner's semaphore,
//        wait on our own semaphore for the partner's tile to land,
//        push EVEN/ODD (our state / the received tile, order depends on
//          whether we are c_even or c_odd in this stage's pair),
//        stream per-core twiddle tile from DRAM,
//        wait for OUT0/OUT1 from compute,
//        keep OUT0 (c_even) or OUT1 (c_odd) as our new state.
//
// The compute kernel is identical in both cases — it sees EVEN/ODD/TW
// tiles and produces OUT0=E+W*O, OUT1=E-W*O.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

// Copy TILE_ELEMS floats from L1 src to L1 dst. Used for state<->EVEN/ODD/OUT
// scalar scatter/gather and for keeping one of the two compute outputs as the
// new state in cross-core stages.
FORCE_INLINE void copy_tile_l1(
    volatile tt_l1_ptr float* dst,
    volatile tt_l1_ptr float* src)
{
    for (uint32_t j = 0; j < TILE_ELEMS; ++j) dst[j] = src[j];
}

void kernel_main() {
    // Runtime args
    const uint32_t in_r_addr  = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr  = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr  = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr  = get_arg_val<uint32_t>(3);
    const uint32_t my_core    = get_arg_val<uint32_t>(4);
    const uint32_t sem_id     = get_arg_val<uint32_t>(5);
    // args 6..6+2P-1 : noc_x[c], noc_y[c] interleaved for c=0..P-1

    // Compile-time args
    constexpr uint32_t N             = get_compile_time_arg_val(0);
    constexpr uint32_t LOG2N         = get_compile_time_arg_val(1);
    constexpr uint32_t P             = get_compile_time_arg_val(2);
    constexpr uint32_t LOG2P         = get_compile_time_arg_val(3);
    constexpr uint32_t LOG2N_LOCAL   = LOG2N - LOG2P;            // 10 if P>1 else LOG2N
    // Valid pairs per local stage: for the single-core (P=1) case this is
    // N/2 (tile may be partially filled). For the multi-core case the tile
    // is always full (TILE_ELEMS) so every local stage has TILE_ELEMS/2 pairs.
    constexpr uint32_t LOCAL_PAIRS   = (P == 1) ? (N / 2) : (TILE_ELEMS / 2);

    const DataFormat df = get_dataformat(CB_EVEN_R);
    const uint32_t   ts = get_tile_size(CB_EVEN_R);

    InterleavedAddrGenFast<true> in_r_gen = {
        .bank_base_address = in_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> in_i_gen = {
        .bank_base_address = in_i_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr, .page_size = ts, .data_format = df};

    // ── Load this core's shard of the bit-reversed input into state ───────
    cb_reserve_back(CB_STATE_R, 1);
    cb_reserve_back(CB_STATE_I, 1);
    const uint32_t state_r_l1 = get_write_ptr(CB_STATE_R);
    const uint32_t state_i_l1 = get_write_ptr(CB_STATE_I);
    noc_async_read_tile(my_core, in_r_gen, state_r_l1);
    noc_async_read_tile(my_core, in_i_gen, state_i_l1);
    noc_async_read_barrier();
    cb_push_back(CB_STATE_R, 1);
    cb_push_back(CB_STATE_I, 1);

    volatile tt_l1_ptr float* const state_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
    volatile tt_l1_ptr float* const state_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);

    // ── Prep cross-core bookkeeping (unused if P == 1) ────────────────────
    const uint32_t sem_l1 = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* const sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);

    // Reserve our receive slots once; we reuse the same L1 region for every
    // cross-core stage. We never cb_push_back on these CBs — they are a
    // raw L1 landing zone; the partner writes directly into them and signals
    // via the semaphore, which is the real ordering primitive.
    const uint32_t recv_r_l1 = get_write_ptr(CB_RECV_R);
    const uint32_t recv_i_l1 = get_write_ptr(CB_RECV_I);
    volatile tt_l1_ptr float* const recv_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(recv_r_l1);
    volatile tt_l1_ptr float* const recv_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(recv_i_l1);

    // ── LOCAL stages (0 .. LOG2N_LOCAL-1) ─────────────────────────────────
    for (uint32_t s = 0; s < LOG2N_LOCAL; ++s) {
        const uint32_t stride     = 1u << s;
        const uint32_t group_size = stride << 1;
        const uint32_t mask       = stride - 1;

        // Twiddle tile: page index (s * P + my_core). For s < LOG2N_LOCAL all
        // P pages at stage s hold the same twiddle tile (host replicates).
        cb_reserve_back(CB_TW_R, 1);
        cb_reserve_back(CB_TW_I, 1);
        noc_async_read_tile(s * P + my_core, tw_r_gen, get_write_ptr(CB_TW_R));
        noc_async_read_tile(s * P + my_core, tw_i_gen, get_write_ptr(CB_TW_I));
        noc_async_read_barrier();
        cb_push_back(CB_TW_R, 1);
        cb_push_back(CB_TW_I, 1);

        // Scatter state -> EVEN/ODD
        cb_reserve_back(CB_EVEN_R, 1);
        cb_reserve_back(CB_EVEN_I, 1);
        cb_reserve_back(CB_ODD_R,  1);
        cb_reserve_back(CB_ODD_I,  1);
        volatile tt_l1_ptr float* const even_r =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_EVEN_R));
        volatile tt_l1_ptr float* const even_i =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_EVEN_I));
        volatile tt_l1_ptr float* const odd_r =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_ODD_R));
        volatile tt_l1_ptr float* const odd_i =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_ODD_I));

        for (uint32_t i = 0; i < LOCAL_PAIRS; ++i) {
            const uint32_t group = i >> s;
            const uint32_t pos   = i & mask;
            const uint32_t lo    = group * group_size + pos;
            const uint32_t hi    = lo + stride;
            even_r[i] = state_r[lo];
            even_i[i] = state_i[lo];
            odd_r[i]  = state_r[hi];
            odd_i[i]  = state_i[hi];
        }

        cb_push_back(CB_EVEN_R, 1);
        cb_push_back(CB_EVEN_I, 1);
        cb_push_back(CB_ODD_R,  1);
        cb_push_back(CB_ODD_I,  1);

        // Gather OUT0/OUT1 back into state
        cb_wait_front(CB_OUT0_R, 1);
        cb_wait_front(CB_OUT0_I, 1);
        cb_wait_front(CB_OUT1_R, 1);
        cb_wait_front(CB_OUT1_I, 1);

        volatile tt_l1_ptr float* const o0r =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT0_R));
        volatile tt_l1_ptr float* const o0i =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT0_I));
        volatile tt_l1_ptr float* const o1r =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT1_R));
        volatile tt_l1_ptr float* const o1i =
            reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT1_I));

        for (uint32_t i = 0; i < LOCAL_PAIRS; ++i) {
            const uint32_t group = i >> s;
            const uint32_t pos   = i & mask;
            const uint32_t lo    = group * group_size + pos;
            const uint32_t hi    = lo + stride;
            state_r[lo] = o0r[i]; state_i[lo] = o0i[i];
            state_r[hi] = o1r[i]; state_i[hi] = o1i[i];
        }

        cb_pop_front(CB_OUT0_R, 1);
        cb_pop_front(CB_OUT0_I, 1);
        cb_pop_front(CB_OUT1_R, 1);
        cb_pop_front(CB_OUT1_I, 1);
    }

    // ── CROSS-CORE stages (LOG2N_LOCAL .. LOG2N-1) ────────────────────────
    // Only entered when P > 1; otherwise LOG2P == 0 and this loop is empty.
    if constexpr (P > 1) {
        for (uint32_t k = 0; k < LOG2P; ++k) {
            const uint32_t s           = LOG2N_LOCAL + k;         // absolute stage
            const uint32_t bit         = 1u << k;                 // XOR bit for partner
            const uint32_t partner     = my_core ^ bit;
            const bool     is_c_even   = (my_core & bit) == 0;

            // Look up partner's physical NoC coords in the runtime arg table.
            const uint32_t partner_x = get_arg_val<uint32_t>(6 + partner * 2);
            const uint32_t partner_y = get_arg_val<uint32_t>(6 + partner * 2 + 1);

            // Compute 64-bit NoC addresses for partner's RECV_R, RECV_I, and
            // semaphore. Recompute each stage in case of future grid changes.
            const uint64_t partner_recv_r_addr = get_noc_addr(partner_x, partner_y, recv_r_l1);
            const uint64_t partner_recv_i_addr = get_noc_addr(partner_x, partner_y, recv_i_l1);
            const uint64_t partner_sem_addr    = get_noc_addr(partner_x, partner_y, sem_l1);

            // Send our state to the partner. Two NoC writes (R tile, I tile),
            // then a barrier so our local state isn't modified while writes
            // are still in flight, then a single semaphore inc on the partner.
            noc_async_write(state_r_l1, partner_recv_r_addr, TILE_SIZE_FP32);
            noc_async_write(state_i_l1, partner_recv_i_addr, TILE_SIZE_FP32);
            noc_async_write_barrier();
            noc_semaphore_inc(partner_sem_addr, 1);

            // Wait for the cumulative count: after k+1 cross-core stages we
            // expect k+1 incs (one per stage, each from whoever our partner
            // for that stage was). Using a monotonic threshold avoids a race
            // where a fast core incs us for stage k+1 before we could reset
            // the sem from stage k -- since partners differ between stages
            // there's no other synchronisation linking us to the next-stage
            // partner. The sem stays monotonically increasing for the life
            // of the kernel; no reset needed.
            noc_semaphore_wait(sem_ptr, k + 1);

            // Twiddle tile for this core at this stage: page (s * P + my_core).
            cb_reserve_back(CB_TW_R, 1);
            cb_reserve_back(CB_TW_I, 1);
            noc_async_read_tile(s * P + my_core, tw_r_gen, get_write_ptr(CB_TW_R));
            noc_async_read_tile(s * P + my_core, tw_i_gen, get_write_ptr(CB_TW_I));
            noc_async_read_barrier();
            cb_push_back(CB_TW_R, 1);
            cb_push_back(CB_TW_I, 1);

            // Feed EVEN/ODD. For c_even: EVEN = state (lower), ODD = recv (upper).
            // For c_odd:  EVEN = recv (partner's state, which is the lower side),
            //             ODD  = state (our own, the upper side).
            cb_reserve_back(CB_EVEN_R, 1);
            cb_reserve_back(CB_EVEN_I, 1);
            cb_reserve_back(CB_ODD_R,  1);
            cb_reserve_back(CB_ODD_I,  1);

            volatile tt_l1_ptr float* const even_r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_EVEN_R));
            volatile tt_l1_ptr float* const even_i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_EVEN_I));
            volatile tt_l1_ptr float* const odd_r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_ODD_R));
            volatile tt_l1_ptr float* const odd_i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(CB_ODD_I));

            if (is_c_even) {
                copy_tile_l1(even_r, state_r); copy_tile_l1(even_i, state_i);
                copy_tile_l1(odd_r,  recv_r);  copy_tile_l1(odd_i,  recv_i);
            } else {
                copy_tile_l1(even_r, recv_r);  copy_tile_l1(even_i, recv_i);
                copy_tile_l1(odd_r,  state_r); copy_tile_l1(odd_i,  state_i);
            }

            cb_push_back(CB_EVEN_R, 1);
            cb_push_back(CB_EVEN_I, 1);
            cb_push_back(CB_ODD_R,  1);
            cb_push_back(CB_ODD_I,  1);

            // Compute produces OUT0=E+W*O, OUT1=E-W*O over the whole tile.
            cb_wait_front(CB_OUT0_R, 1);
            cb_wait_front(CB_OUT0_I, 1);
            cb_wait_front(CB_OUT1_R, 1);
            cb_wait_front(CB_OUT1_I, 1);

            volatile tt_l1_ptr float* const o0r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT0_R));
            volatile tt_l1_ptr float* const o0i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT0_I));
            volatile tt_l1_ptr float* const o1r =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT1_R));
            volatile tt_l1_ptr float* const o1i =
                reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(CB_OUT1_I));

            if (is_c_even) { copy_tile_l1(state_r, o0r); copy_tile_l1(state_i, o0i); }
            else           { copy_tile_l1(state_r, o1r); copy_tile_l1(state_i, o1i); }

            cb_pop_front(CB_OUT0_R, 1);
            cb_pop_front(CB_OUT0_I, 1);
            cb_pop_front(CB_OUT1_R, 1);
            cb_pop_front(CB_OUT1_I, 1);
        }
    }

    // Signal writer: final state is in place.
    cb_reserve_back(CB_SYNC, 1);
    cb_push_back(CB_SYNC, 1);
}
