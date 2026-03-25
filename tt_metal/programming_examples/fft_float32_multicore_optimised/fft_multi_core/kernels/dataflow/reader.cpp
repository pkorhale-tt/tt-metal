// reader_debug.cpp - minimal: just push even/odd from DRAM and one twiddle tile
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

    constexpr uint32_t CB_ER  = 0;
    constexpr uint32_t CB_EI  = 1;
    constexpr uint32_t CB_OR  = 2;
    constexpr uint32_t CB_OI  = 3;
    constexpr uint32_t CB_TWR = 4;
    constexpr uint32_t CB_TWI = 5;

    const uint32_t tile_bytes = get_tile_size(CB_ER);
    const DataFormat fmt      = get_dataformat(CB_ER);
    constexpr uint32_t F      = sizeof(float);

    const InterleavedAddrGenFast<true> gen_er  = { .bank_base_address = even_r_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ei  = { .bank_base_address = even_i_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_or  = { .bank_base_address = odd_r_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_oi  = { .bank_base_address = odd_i_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_cr  = { .bank_base_address = ctw_r_addr,  .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ci  = { .bank_base_address = ctw_i_addr,  .page_size = tile_bytes, .data_format = fmt };

    auto wr32 = [](uint32_t a, uint32_t v) { *reinterpret_cast<volatile uint32_t*>(a) = v; };
    auto rd32 = [](uint32_t a) -> uint32_t { return *reinterpret_cast<volatile uint32_t*>(a); };

    // Load compact twiddle into a TEMP local buffer on L1 via CB_TWR (reuse write ptr)
    // Actually just load ctw directly into CB_TWR/TWI for stage 0 — no separate CB needed
    // For 1 stage, just push one set of twiddles directly

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        // Push even/odd inputs
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

        // Push twiddles for each stage
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const uint32_t half_m = 1u << stage;
            const uint32_t stride = half_N >> stage;
            const uint32_t mask   = half_m - 1u;
            const uint32_t elems  = tiles_per_row * (tile_bytes / F);

            cb_reserve_back(CB_TWR, tiles_per_row);
            cb_reserve_back(CB_TWI, tiles_per_row);

            // For stage 0 with N=2: half_m=1, stride=1, mask=0
            // All elements get twiddle index 0 = W^0 = (1, 0)
            // Just load from ctw DRAM tile directly
            noc_async_read_tile(0, gen_cr, get_write_ptr(CB_TWR));
            noc_async_read_tile(0, gen_ci, get_write_ptr(CB_TWI));
            noc_async_read_barrier();

            cb_push_back(CB_TWR, tiles_per_row);
            cb_push_back(CB_TWI, tiles_per_row);
        }
    }
}