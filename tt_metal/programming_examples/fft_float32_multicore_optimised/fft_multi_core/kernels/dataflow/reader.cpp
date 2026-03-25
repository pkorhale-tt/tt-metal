// reader.cpp — RISCV_0 dataflow kernel
// Wormhole 1-D Cooley-Tukey FFT (decimation-in-time, radix-2)
// ═══════════════════════════════════════════════════════════════════════
//
//  CB ownership:
//    CB  0  even_r   stage-0: reader→compute   stage 1+: writer→compute
//    CB  1  even_i   stage-0: reader→compute   stage 1+: writer→compute
//    CB  2  odd_r    stage-0: reader→compute   stage 1+: writer→compute
//    CB  3  odd_i    stage-0: reader→compute   stage 1+: writer→compute
//    CB  4  tw_r     reader→compute (all stages)
//    CB  5  tw_i     reader→compute (all stages)
//    CB  6  out_even_r  compute→writer
//    CB  7  out_even_i  compute→writer
//    CB  8  out_odd_r   compute→writer
//    CB  9  out_odd_i   compute→writer
//    CB 10-13  scratch (compute, depth=1)
//    CB 14-15  compact twiddle table (reader L1, depth=1, never popped mid-row)
//    CB 16-19  writer L1 shuffle scratch (plain memory, depth=1)
//
//  Reader responsibilities:
//    1. Load compact twiddle table (N/2 entries) from DRAM once.
//    2. For each assigned row:
//       a. Stage 0: DMA bit-reversed even/odd split from DRAM → CB 0-3.
//       b. Every stage: scatter correct twiddles → CB 4-5.
//          cb_reserve_back on CB 4/5 provides natural back-pressure.
//
//  Argument map:
//    [0]  even_r_addr     DRAM: bit-reversed even elements, real part
//    [1]  even_i_addr     DRAM: bit-reversed even elements, imag part
//    [2]  odd_r_addr      DRAM: bit-reversed odd  elements, real part
//    [3]  odd_i_addr      DRAM: bit-reversed odd  elements, imag part
//    [4]  ctw_r_addr      DRAM: compact twiddle table real  (one tile)
//    [5]  ctw_i_addr      DRAM: compact twiddle table imag  (one tile)
//    [6]  tiles_per_row   tiles covering N/2 elements
//    [7]  tile_offset     first tile index owned by this core
//    [8]  num_stages      log2(N)
//    [9]  half_N          N/2
//    [10] rows_per_core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t even_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t even_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t odd_r_addr    = get_arg_val<uint32_t>(2);
    const uint32_t odd_i_addr    = get_arg_val<uint32_t>(3);
    const uint32_t ctw_r_addr    = get_arg_val<uint32_t>(4);
    const uint32_t ctw_i_addr    = get_arg_val<uint32_t>(5);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset   = get_arg_val<uint32_t>(7);
    const uint32_t num_stages    = get_arg_val<uint32_t>(8);
    const uint32_t half_N        = get_arg_val<uint32_t>(9);
    const uint32_t rows_per_core = get_arg_val<uint32_t>(10);

    if (tiles_per_row == 0 || num_stages == 0 || rows_per_core == 0) return;

    constexpr uint32_t CB_EVEN_R = 0;
    constexpr uint32_t CB_EVEN_I = 1;
    constexpr uint32_t CB_ODD_R  = 2;
    constexpr uint32_t CB_ODD_I  = 3;
    constexpr uint32_t CB_TW_R   = 4;
    constexpr uint32_t CB_TW_I   = 5;
    constexpr uint32_t CB_CTW_R  = 14;  // compact twiddle real  (L1 permanent)
    constexpr uint32_t CB_CTW_I  = 15;  // compact twiddle imag  (L1 permanent)

    const uint32_t tile_bytes    = get_tile_size(CB_EVEN_R);
    const DataFormat fmt         = get_dataformat(CB_EVEN_R);
    constexpr uint32_t F         = sizeof(float);
    const uint32_t elems_per_row = (tile_bytes / F) * tiles_per_row;

    const InterleavedAddrGenFast<true> gen_er = {
        .bank_base_address = even_r_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ei = {
        .bank_base_address = even_i_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_or = {
        .bank_base_address = odd_r_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_oi = {
        .bank_base_address = odd_i_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_cr = {
        .bank_base_address = ctw_r_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ci = {
        .bank_base_address = ctw_i_addr,  .page_size = tile_bytes, .data_format = fmt };

    auto rd32 = [](uint32_t a) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(a); };
    auto wr32 = [](uint32_t a, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(a) = v; };

    // ── 1. Load compact twiddle table once ───────────────────────────
    // The table has half_N entries; it always fits in one tile because
    // the host pads it to TILE_SIZE floats.
    cb_reserve_back(CB_CTW_R, 1);
    cb_reserve_back(CB_CTW_I, 1);
    noc_async_read_tile(0, gen_cr, get_write_ptr(CB_CTW_R));
    noc_async_read_tile(0, gen_ci, get_write_ptr(CB_CTW_I));
    noc_async_read_barrier();
    cb_push_back(CB_CTW_R, 1);
    cb_push_back(CB_CTW_I, 1);
    cb_wait_front(CB_CTW_R, 1);
    cb_wait_front(CB_CTW_I, 1);
    const uint32_t ctw_r = get_read_ptr(CB_CTW_R);
    const uint32_t ctw_i = get_read_ptr(CB_CTW_I);

    // ── 2. Per-row loop ───────────────────────────────────────────────
    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;
        const uint32_t row_elem_base = row_tile_base * (tile_bytes / F);

        // ── Stage 0: DMA even/odd split from DRAM ────────────────────
        // Issue all four DMA streams before the barrier so they run in
        // parallel on the NoC fabric.
        cb_reserve_back(CB_EVEN_R, tiles_per_row);
        cb_reserve_back(CB_EVEN_I, tiles_per_row);
        cb_reserve_back(CB_ODD_R,  tiles_per_row);
        cb_reserve_back(CB_ODD_I,  tiles_per_row);
        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            const uint32_t gt = row_tile_base + t;
            noc_async_read_tile(gt, gen_er,
                get_write_ptr(CB_EVEN_R) + t * tile_bytes);
            noc_async_read_tile(gt, gen_ei,
                get_write_ptr(CB_EVEN_I) + t * tile_bytes);
            noc_async_read_tile(gt, gen_or,
                get_write_ptr(CB_ODD_R)  + t * tile_bytes);
            noc_async_read_tile(gt, gen_oi,
                get_write_ptr(CB_ODD_I)  + t * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(CB_EVEN_R, tiles_per_row);
        cb_push_back(CB_EVEN_I, tiles_per_row);
        cb_push_back(CB_ODD_R,  tiles_per_row);
        cb_push_back(CB_ODD_I,  tiles_per_row);

        // ── All stages: scatter twiddle factors ──────────────────────
        // Stage s: twiddle index for element p is (p & (2^s - 1)) * (N >> (s+1))
        // This maps element p to the correct root-of-unity from the compact table.
        // cb_reserve_back blocks until compute drains stage s-1 twiddles,
        // naturally throttling the reader to one stage ahead.
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const uint32_t half_m = 1u << stage;       // 1, 2, 4, 8 ...
            const uint32_t stride = half_N >> stage;   // N/2, N/4, N/8 ...
            const uint32_t mask   = half_m - 1u;

            cb_reserve_back(CB_TW_R, tiles_per_row);
            cb_reserve_back(CB_TW_I, tiles_per_row);
            const uint32_t dst_r = get_write_ptr(CB_TW_R);
            const uint32_t dst_i = get_write_ptr(CB_TW_I);

            for (uint32_t lp = 0; lp < elems_per_row; ++lp) {
                // Element index in the global butterfly ordering
                const uint32_t p   = row_elem_base + lp;
                // Which twiddle factor this element needs
                const uint32_t idx = (p & mask) * stride;
                wr32(dst_r + lp * F, rd32(ctw_r + idx * F));
                wr32(dst_i + lp * F, rd32(ctw_i + idx * F));
            }

            cb_push_back(CB_TW_R, tiles_per_row);
            cb_push_back(CB_TW_I, tiles_per_row);
        }
        // Reader finished this row. Compute + writer pipeline the rest.
    }

    // Release compact twiddle table
    cb_pop_front(CB_CTW_R, 1);
    cb_pop_front(CB_CTW_I, 1);
}