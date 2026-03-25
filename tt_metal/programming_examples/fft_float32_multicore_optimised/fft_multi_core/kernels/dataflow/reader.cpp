// reader.cpp — Wormhole 1-D FFT data mover (RISCV_0)
//
// Stage 0 inputs come from DRAM. Twiddles are precomputed per stage by the
// host and laid out as [stage][tile]. The reader streams exactly one twiddle
// tile per butterfly tile per stage.

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

    const InterleavedAddrGenFast<true> gen_er = {
        .bank_base_address = even_r_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ei = {
        .bank_base_address = even_i_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_or = {
        .bank_base_address = odd_r_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_oi = {
        .bank_base_address = odd_i_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_tr = {
        .bank_base_address = ctw_r_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ti = {
        .bank_base_address = ctw_i_addr, .page_size = tile_bytes, .data_format = fmt };

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        // Stage 0 input tiles
        cb_reserve_back(CB_ER, tiles_per_row);
        cb_reserve_back(CB_EI, tiles_per_row);
        cb_reserve_back(CB_OR, tiles_per_row);
        cb_reserve_back(CB_OI, tiles_per_row);
        const uint32_t dst_er = get_write_ptr(CB_ER);
        const uint32_t dst_ei = get_write_ptr(CB_EI);
        const uint32_t dst_or = get_write_ptr(CB_OR);
        const uint32_t dst_oi = get_write_ptr(CB_OI);

        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            const uint32_t gt = row_tile_base + t;
            const uint32_t off = t * tile_bytes;
            noc_async_read_tile(gt, gen_er, dst_er + off);
            noc_async_read_tile(gt, gen_ei, dst_ei + off);
            noc_async_read_tile(gt, gen_or, dst_or + off);
            noc_async_read_tile(gt, gen_oi, dst_oi + off);
        }
        noc_async_read_barrier();
        cb_push_back(CB_ER, tiles_per_row);
        cb_push_back(CB_EI, tiles_per_row);
        cb_push_back(CB_OR, tiles_per_row);
        cb_push_back(CB_OI, tiles_per_row);

        // Stage-specific twiddles
        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const uint32_t stage_tile_base = stage * tiles_per_row;
            for (uint32_t t = 0; t < tiles_per_row; ++t) {
                cb_reserve_back(CB_TWR, 1);
                cb_reserve_back(CB_TWI, 1);
                const uint32_t tw_dst_r = get_write_ptr(CB_TWR);
                const uint32_t tw_dst_i = get_write_ptr(CB_TWI);
                noc_async_read_tile(stage_tile_base + t, gen_tr, tw_dst_r);
                noc_async_read_tile(stage_tile_base + t, gen_ti, tw_dst_i);
                noc_async_read_barrier();
                cb_push_back(CB_TWR, 1);
                cb_push_back(CB_TWI, 1);
            }
        }
    }
}
