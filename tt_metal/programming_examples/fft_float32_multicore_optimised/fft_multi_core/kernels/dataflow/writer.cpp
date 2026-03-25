// writer.cpp — Wormhole 1-D FFT output / feedback kernel (RISCV_1)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_er_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out_ei_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out_or_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out_oi_addr   = get_arg_val<uint32_t>(3);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t half_N        = get_arg_val<uint32_t>(6);
    const uint32_t tile_offset   = get_arg_val<uint32_t>(7);
    const uint32_t rows_per_core = get_arg_val<uint32_t>(8);

    if (tiles_per_row == 0 || num_stages == 0 || rows_per_core == 0) return;

    constexpr uint32_t CB_OER    = 6;
    constexpr uint32_t CB_OEI    = 7;
    constexpr uint32_t CB_OOR    = 8;
    constexpr uint32_t CB_OOI    = 9;
    constexpr uint32_t CB_ER     = 0;
    constexpr uint32_t CB_EI     = 1;
    constexpr uint32_t CB_OR     = 2;
    constexpr uint32_t CB_OI     = 3;
    constexpr uint32_t CB_SCR_ER = 16;
    constexpr uint32_t CB_SCR_EI = 17;
    constexpr uint32_t CB_SCR_OR = 18;
    constexpr uint32_t CB_SCR_OI = 19;

    const uint32_t tile_bytes = get_tile_size(CB_OER);
    const DataFormat fmt      = get_dataformat(CB_OER);
    constexpr uint32_t F = 4u;
    const uint32_t elems_padded = tiles_per_row * (tile_bytes / F);
    const uint32_t valid_elems  = half_N;

    const InterleavedAddrGenFast<true> gen_er = {
        .bank_base_address = out_er_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_ei = {
        .bank_base_address = out_ei_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_or = {
        .bank_base_address = out_or_addr, .page_size = tile_bytes, .data_format = fmt };
    const InterleavedAddrGenFast<true> gen_oi = {
        .bank_base_address = out_oi_addr, .page_size = tile_bytes, .data_format = fmt };

    auto rd32 = [](uint32_t a) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(a); };
    auto wr32 = [](uint32_t a, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(a) = v; };

    const uint32_t scr_er = get_write_ptr(CB_SCR_ER);
    const uint32_t scr_ei = get_write_ptr(CB_SCR_EI);
    const uint32_t scr_or = get_write_ptr(CB_SCR_OR);
    const uint32_t scr_oi = get_write_ptr(CB_SCR_OI);

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const bool is_last = (stage == num_stages - 1);

            cb_wait_front(CB_OER, tiles_per_row);
            cb_wait_front(CB_OEI, tiles_per_row);
            cb_wait_front(CB_OOR, tiles_per_row);
            cb_wait_front(CB_OOI, tiles_per_row);

            if (is_last) {
                const uint32_t src_er = get_read_ptr(CB_OER);
                const uint32_t src_ei = get_read_ptr(CB_OEI);
                const uint32_t src_or = get_read_ptr(CB_OOR);
                const uint32_t src_oi = get_read_ptr(CB_OOI);

                for (uint32_t t = 0; t < tiles_per_row; ++t) {
                    const uint32_t off = t * tile_bytes;
                    noc_async_write_tile(row_tile_base + t, gen_er, src_er + off);
                    noc_async_write_tile(row_tile_base + t, gen_ei, src_ei + off);
                    noc_async_write_tile(row_tile_base + t, gen_or, src_or + off);
                    noc_async_write_tile(row_tile_base + t, gen_oi, src_oi + off);
                }
                noc_async_write_barrier();

                cb_pop_front(CB_OER, tiles_per_row);
                cb_pop_front(CB_OEI, tiles_per_row);
                cb_pop_front(CB_OOR, tiles_per_row);
                cb_pop_front(CB_OOI, tiles_per_row);
            } else {
                const uint32_t src_er = get_read_ptr(CB_OER);
                const uint32_t src_ei = get_read_ptr(CB_OEI);
                const uint32_t src_or = get_read_ptr(CB_OOR);
                const uint32_t src_oi = get_read_ptr(CB_OOI);

                // zero entire padded region first
                for (uint32_t lp = 0; lp < elems_padded; ++lp) {
                    const uint32_t off = lp * F;
                    wr32(scr_er + off, 0u);
                    wr32(scr_ei + off, 0u);
                    wr32(scr_or + off, 0u);
                    wr32(scr_oi + off, 0u);
                }

                const uint32_t m       = 1u << (stage + 1);
                const uint32_t half_m  = m >> 1;
                const uint32_t next_m  = m << 1;
                const uint32_t next_hm = next_m >> 1;
                const uint32_t num_groups_next = (next_hm <= valid_elems) ? (valid_elems / next_hm) : 1u;
                const uint32_t log2m   = stage + 1;
                const uint32_t m_mask  = m - 1u;

                uint32_t dst_idx = 0;
                for (uint32_t g2 = 0; g2 < num_groups_next; ++g2) {
                    const uint32_t base_e = g2 * next_m;
                    const uint32_t base_o = base_e + next_hm;

                    for (uint32_t j2 = 0; j2 < next_hm && dst_idx < valid_elems; ++j2, ++dst_idx) {
                        {
                            const uint32_t f = base_e + j2;
                            const uint32_t g_cur = f >> log2m;
                            const uint32_t offset = f & m_mask;
                            uint32_t idx, sr, si;
                            if (offset < half_m) {
                                idx = g_cur * half_m + offset;
                                sr = src_er; si = src_ei;
                            } else {
                                idx = g_cur * half_m + (offset - half_m);
                                sr = src_or; si = src_oi;
                            }
                            wr32(scr_er + dst_idx * F, rd32(sr + idx * F));
                            wr32(scr_ei + dst_idx * F, rd32(si + idx * F));
                        }
                        {
                            const uint32_t f = base_o + j2;
                            const uint32_t g_cur = f >> log2m;
                            const uint32_t offset = f & m_mask;
                            uint32_t idx, sr, si;
                            if (offset < half_m) {
                                idx = g_cur * half_m + offset;
                                sr = src_er; si = src_ei;
                            } else {
                                idx = g_cur * half_m + (offset - half_m);
                                sr = src_or; si = src_oi;
                            }
                            wr32(scr_or + dst_idx * F, rd32(sr + idx * F));
                            wr32(scr_oi + dst_idx * F, rd32(si + idx * F));
                        }
                    }
                }

                cb_pop_front(CB_OER, tiles_per_row);
                cb_pop_front(CB_OEI, tiles_per_row);
                cb_pop_front(CB_OOR, tiles_per_row);
                cb_pop_front(CB_OOI, tiles_per_row);

                cb_reserve_back(CB_ER, tiles_per_row);
                cb_reserve_back(CB_EI, tiles_per_row);
                cb_reserve_back(CB_OR, tiles_per_row);
                cb_reserve_back(CB_OI, tiles_per_row);

                const uint32_t dst_er = get_write_ptr(CB_ER);
                const uint32_t dst_ei = get_write_ptr(CB_EI);
                const uint32_t dst_or = get_write_ptr(CB_OR);
                const uint32_t dst_oi = get_write_ptr(CB_OI);

                for (uint32_t lp = 0; lp < elems_padded; ++lp) {
                    const uint32_t off = lp * F;
                    wr32(dst_er + off, rd32(scr_er + off));
                    wr32(dst_ei + off, rd32(scr_ei + off));
                    wr32(dst_or + off, rd32(scr_or + off));
                    wr32(dst_oi + off, rd32(scr_oi + off));
                }

                cb_push_back(CB_ER, tiles_per_row);
                cb_push_back(CB_EI, tiles_per_row);
                cb_push_back(CB_OR, tiles_per_row);
                cb_push_back(CB_OI, tiles_per_row);
            }
        }
    }
}
