// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_reader.cpp — BRISC0 / reader
//
// Owns CB_STATE_{R,I}, the persistent single-tile state that holds the FFT
// working set across all stages. For each stage it:
//   1. streams the stage's twiddle tile from DRAM,
//   2. scatters state -> EVEN/ODD tiles according to the stage's pair stride,
//   3. waits for OUT0/OUT1 from compute and gathers them back into state.
//
// After the last stage it signals the writer via CB_SYNC; the writer then
// reads CB_STATE directly from L1 and pushes it to DRAM.
//
// Performance note: the scalar gather/scatter is O(N) per stage on BRISC. For
// N <= 1024 this costs a few thousand cycles per stage, which is dominated by
// the SFPU butterfly latency. The code is written to be branchless in the hot
// loop (cheap div/mod by a power of two).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

void kernel_main() {
    const uint32_t in_r_addr = get_arg_val<uint32_t>(0);
    const uint32_t in_i_addr = get_arg_val<uint32_t>(1);
    const uint32_t tw_r_addr = get_arg_val<uint32_t>(2);
    const uint32_t tw_i_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t N         = get_compile_time_arg_val(0);
    constexpr uint32_t LOG2N     = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_PAIRS = N / 2;

    const DataFormat df  = get_dataformat(CB_EVEN_R);
    const uint32_t   ts  = get_tile_size(CB_EVEN_R);

    InterleavedAddrGenFast<true> in_r_gen = {
        .bank_base_address = in_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> in_i_gen = {
        .bank_base_address = in_i_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_r_gen = {
        .bank_base_address = tw_r_addr, .page_size = ts, .data_format = df};
    InterleavedAddrGenFast<true> tw_i_gen = {
        .bank_base_address = tw_i_addr, .page_size = ts, .data_format = df};

    // ── Load bit-reversed input into persistent state ─────────────────────
    cb_reserve_back(CB_STATE_R, 1);
    cb_reserve_back(CB_STATE_I, 1);
    const uint32_t state_r_l1 = get_write_ptr(CB_STATE_R);
    const uint32_t state_i_l1 = get_write_ptr(CB_STATE_I);
    noc_async_read_tile(0, in_r_gen, state_r_l1);
    noc_async_read_tile(0, in_i_gen, state_i_l1);
    noc_async_read_barrier();
    cb_push_back(CB_STATE_R, 1);
    cb_push_back(CB_STATE_I, 1);

    // Raw float views of the state tiles. The state stays put for the whole
    // kernel; we just read and write in-place.
    volatile tt_l1_ptr float* const state_r =
        reinterpret_cast<volatile tt_l1_ptr float*>(state_r_l1);
    volatile tt_l1_ptr float* const state_i =
        reinterpret_cast<volatile tt_l1_ptr float*>(state_i_l1);

    // ── Butterfly stages ──────────────────────────────────────────────────
    for (uint32_t s = 0; s < LOG2N; ++s) {
        const uint32_t stride     = 1u << s;         // pair distance at stage s
        const uint32_t group_size = stride << 1;     // 2*stride = elements per pair-group
        const uint32_t mask       = stride - 1;      // pos = i & mask  (stride is pow2)

        // --- stage twiddle tile -----------------------------------------
        cb_reserve_back(CB_TW_R, 1);
        cb_reserve_back(CB_TW_I, 1);
        noc_async_read_tile(s, tw_r_gen, get_write_ptr(CB_TW_R));
        noc_async_read_tile(s, tw_i_gen, get_write_ptr(CB_TW_I));
        noc_async_read_barrier();
        cb_push_back(CB_TW_R, 1);
        cb_push_back(CB_TW_I, 1);

        // --- scatter: state -> EVEN/ODD ---------------------------------
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

        // For each pair index i at stage s:
        //   group = i >> s,  pos = i & (stride-1)
        //   lo    = group*2*stride + pos
        //   hi    = lo + stride
        for (uint32_t i = 0; i < NUM_PAIRS; ++i) {
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

        // --- gather: OUT0/OUT1 -> state ---------------------------------
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

        for (uint32_t i = 0; i < NUM_PAIRS; ++i) {
            const uint32_t group = i >> s;
            const uint32_t pos   = i & mask;
            const uint32_t lo    = group * group_size + pos;
            const uint32_t hi    = lo + stride;
            state_r[lo] = o0r[i];
            state_i[lo] = o0i[i];
            state_r[hi] = o1r[i];
            state_i[hi] = o1i[i];
        }

        cb_pop_front(CB_OUT0_R, 1);
        cb_pop_front(CB_OUT0_I, 1);
        cb_pop_front(CB_OUT1_R, 1);
        cb_pop_front(CB_OUT1_I, 1);
    }

    // ── Signal writer: state is final ─────────────────────────────────────
    cb_reserve_back(CB_SYNC, 1);
    cb_push_back(CB_SYNC, 1);
}
