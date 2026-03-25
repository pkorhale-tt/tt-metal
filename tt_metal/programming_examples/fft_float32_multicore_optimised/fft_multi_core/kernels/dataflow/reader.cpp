// reader.cpp  — Wormhole 1-D FFT data mover (RISCV_0)
//
// FIXES vs reader_debug.cpp:
//   1. Twiddle push count was tiles_per_row but only 1 tile was written → stall.
//      Now always pushes exactly 1 twiddle tile per stage (the compact table
//      is one tile; compute pops exactly 1 per butterfly tile iteration).
//   2. Twiddle tile index for each stage is always 0 (the compact table holds
//      all half_N entries in one tile, indexed by element position inside it).
//      This is correct — left unchanged but made explicit.
//   3. Even/odd inputs are pushed once per row (not once total), matching the
//      compute loop which iterates row × stage × tile.
//   4. Twiddle push is now inside the tile loop so compute receives one twiddle
//      tile per butterfly tile, instead of tiles_per_row twiddles per stage.
//
// CB map (matches host + compute):
//   0  even_r    depth=tiles_per_row
//   1  even_i    depth=tiles_per_row
//   2  odd_r     depth=tiles_per_row
//   3  odd_i     depth=tiles_per_row
//   4  tw_r      depth=tiles_per_row   (twiddle, 1 tile per butterfly)
//   5  tw_i      depth=tiles_per_row
//
// Args:
//   [0]  even_r DRAM base address
//   [1]  even_i DRAM base address
//   [2]  odd_r  DRAM base address
//   [3]  odd_i  DRAM base address
//   [4]  compact twiddle real DRAM address  (1 tile, half_N entries)
//   [5]  compact twiddle imag DRAM address
//   [6]  tiles_per_row
//   [7]  tile_offset  (first tile index for this core)
//   [8]  num_stages   (= log2(N))
//   [9]  half_N       (= N/2, informational — not used for addressing here)
//   [10] rows_per_core

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
    // arg[9] half_N not needed for addressing
    const uint32_t rows_per_core = get_arg_val<uint32_t>(10);

    if (tiles_per_row == 0 || num_stages == 0 || rows_per_core == 0) return;

    constexpr uint32_t CB_ER  = 0;
    constexpr uint32_t CB_EI  = 1;
    constexpr uint32_t CB_OR  = 2;
    constexpr uint32_t CB_OI  = 3;
    constexpr uint32_t CB_TWR = 4;
    constexpr uint32_t CB_TWI = 5;

    const uint32_t tile_bytes = get_tile_size(CB_ER);
    const DataFormat fmt      = get_dataformat(CB_ER);

    const InterleavedAddrGenFast<true> gen_er  = {
        .bank_base_address = even_r_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ei  = {
        .bank_base_address = even_i_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_or  = {
        .bank_base_address = odd_r_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_oi  = {
        .bank_base_address = odd_i_addr,  .page_size = tile_bytes, .data_format = fmt };
    // Compact twiddle table — always tile index 0, read repeatedly
    const InterleavedAddrGenFast<true> gen_cr  = {
        .bank_base_address = ctw_r_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ci  = {
        .bank_base_address = ctw_i_addr,  .page_size = tile_bytes, .data_format = fmt };

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        // Global tile base for this row on this core
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        // ── Stage 0: push even/odd inputs from DRAM ──────────────────
        // For stage 0, inputs come from DRAM (bit-reversed, pre-split).
        // For stages 1+, inputs are fed back by the writer via CBs 0–3;
        // the reader does NOT push anything for those stages — the writer
        // is responsible for re-filling the input CBs.
        cb_reserve_back(CB_ER, tiles_per_row);
        cb_reserve_back(CB_EI, tiles_per_row);
        cb_reserve_back(CB_OR, tiles_per_row);
        cb_reserve_back(CB_OI, tiles_per_row);
        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            const uint32_t gt = row_tile_base + t;
            noc_async_read_tile(gt, gen_er, get_write_ptr(CB_ER) + t * tile_bytes);
            noc_async_read_tile(gt, gen_ei, get_write_ptr(CB_EI) + t * tile_bytes);
            noc_async_read_tile(gt, gen_or, get_write_ptr(CB_OR) + t * tile_bytes);
            noc_async_read_tile(gt, gen_oi, get_write_ptr(CB_OI) + t * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(CB_ER, tiles_per_row);
        cb_push_back(CB_EI, tiles_per_row);
        cb_push_back(CB_OR, tiles_per_row);
        cb_push_back(CB_OI, tiles_per_row);

        // ── All stages: push twiddle tiles ───────────────────────────
        // The compute kernel's inner loop is:
        //   for (stage) for (tile t in 0..tiles_per_row-1) { consume 1 twiddle }
        // So we push exactly 1 twiddle tile per butterfly tile per stage.
        // The compact table (one DRAM tile, half_N entries) covers all stages;
        // tile index 0 is always correct because the kernel addresses into
        // the tile by element index, not by DRAM tile index.
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                // FIX: reserve exactly 1 slot and push exactly 1 tile.
                // The original code reserved tiles_per_row slots but only
                // wrote 1 tile, causing the compute kernel to stall forever.
                cb_reserve_back(CB_TWR, 1);
                cb_reserve_back(CB_TWI, 1);
                noc_async_read_tile(0, gen_cr, get_write_ptr(CB_TWR));
                noc_async_read_tile(0, gen_ci, get_write_ptr(CB_TWI));
                noc_async_read_barrier();
                cb_push_back(CB_TWR, 1);
                cb_push_back(CB_TWI, 1);
            }
        }
    }
}