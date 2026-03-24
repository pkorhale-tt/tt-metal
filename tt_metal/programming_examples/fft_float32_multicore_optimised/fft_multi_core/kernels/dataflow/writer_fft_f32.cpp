#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out0_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr   = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles     = get_arg_val<uint32_t>(4);   // tiles_per_row
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);   // log2_row
    const uint32_t half_N        = get_arg_val<uint32_t>(6);   // N_row/2
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);  // starting tile for this core
    const uint32_t rows_per_core = get_arg_val<uint32_t>(13);  // rows handled by this core

    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;

    constexpr uint32_t cb_next_even_r = 6;
    constexpr uint32_t cb_next_even_i = 7;
    constexpr uint32_t cb_next_odd_r  = 8;
    constexpr uint32_t cb_next_odd_i  = 9;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);
    constexpr uint32_t ELEM      = sizeof(float);
    const uint32_t elems_per_tile = tile_bytes / ELEM;
    const uint32_t elems_total    = elems_per_tile * num_tiles;

    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    if (num_tiles == 0 || num_stages == 0 || rows_per_core == 0) {
        return;
    }

    auto rd = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr = [](uint32_t addr, uint32_t raw) {
        *reinterpret_cast<volatile uint32_t*>(addr) = raw;
    };

    for (uint32_t row = 0; row < rows_per_core; row++) {
        const uint32_t row_tile_base = tile_offset + row * num_tiles;

        for (uint32_t stage = 0; stage < num_stages; stage++) {
            const bool is_last = (stage == num_stages - 1);

            cb_wait_front(cb_out0_r, num_tiles);
            cb_wait_front(cb_out0_i, num_tiles);
            cb_wait_front(cb_out1_r, num_tiles);
            cb_wait_front(cb_out1_i, num_tiles);

            if (is_last) {
                for (uint32_t t = 0; t < num_tiles; t++) {
                    noc_async_write_tile(row_tile_base + t, out0_r_gen,
                        get_read_ptr(cb_out0_r) + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out0_i_gen,
                        get_read_ptr(cb_out0_i) + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_r_gen,
                        get_read_ptr(cb_out1_r) + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, out1_i_gen,
                        get_read_ptr(cb_out1_i) + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(cb_out0_r, num_tiles);
                cb_pop_front(cb_out0_i, num_tiles);
                cb_pop_front(cb_out1_r, num_tiles);
                cb_pop_front(cb_out1_i, num_tiles);
                continue;
            }

            const uint32_t src0r = get_read_ptr(cb_out0_r);
            const uint32_t src0i = get_read_ptr(cb_out0_i);
            const uint32_t src1r = get_read_ptr(cb_out1_r);
            const uint32_t src1i = get_read_ptr(cb_out1_i);

            cb_reserve_back(cb_next_even_r, num_tiles);
            cb_reserve_back(cb_next_even_i, num_tiles);
            cb_reserve_back(cb_next_odd_r,  num_tiles);
            cb_reserve_back(cb_next_odd_i,  num_tiles);

            const uint32_t dst_er = get_write_ptr(cb_next_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_next_even_i);
            const uint32_t dst_or = get_write_ptr(cb_next_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_next_odd_i);

            for (uint32_t i = 0; i < elems_total; i++) {
                wr(dst_er + i * ELEM, 0u);
                wr(dst_ei + i * ELEM, 0u);
                wr(dst_or + i * ELEM, 0u);
                wr(dst_oi + i * ELEM, 0u);
            }

            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = (half_N >= half_m2) ? (half_N / half_m2) : 0u;
            const uint32_t log2m   = stage + 1;
            const uint32_t m_mask  = m - 1u;

            uint32_t dst = 0;
            for (uint32_t g2 = 0; g2 < G2; g2++) {
                const uint32_t base_e = g2 * m2;
                const uint32_t base_o = base_e + half_m2;
                for (uint32_t j2 = 0; j2 < half_m2; j2++) {
                    {
                        const uint32_t f      = base_e + j2;
                        const uint32_t g_old  = f >> log2m;
                        const uint32_t offset = f & m_mask;
                        uint32_t idx, srcr, srci;
                        if (offset < half_m) {
                            idx = g_old * half_m + offset;
                            srcr = src0r; srci = src0i;
                        } else {
                            idx = g_old * half_m + (offset - half_m);
                            srcr = src1r; srci = src1i;
                        }
                        wr(dst_er + dst * ELEM, rd(srcr + idx * ELEM));
                        wr(dst_ei + dst * ELEM, rd(srci + idx * ELEM));
                    }
                    {
                        const uint32_t f      = base_o + j2;
                        const uint32_t g_old  = f >> log2m;
                        const uint32_t offset = f & m_mask;
                        uint32_t idx, srcr, srci;
                        if (offset < half_m) {
                            idx = g_old * half_m + offset;
                            srcr = src0r; srci = src0i;
                        } else {
                            idx = g_old * half_m + (offset - half_m);
                            srcr = src1r; srci = src1i;
                        }
                        wr(dst_or + dst * ELEM, rd(srcr + idx * ELEM));
                        wr(dst_oi + dst * ELEM, rd(srci + idx * ELEM));
                    }
                    dst++;
                }
            }

            cb_push_back(cb_next_even_r, num_tiles);
            cb_push_back(cb_next_even_i, num_tiles);
            cb_push_back(cb_next_odd_r,  num_tiles);
            cb_push_back(cb_next_odd_i,  num_tiles);

            cb_pop_front(cb_out0_r, num_tiles);
            cb_pop_front(cb_out0_i, num_tiles);
            cb_pop_front(cb_out1_r, num_tiles);
            cb_pop_front(cb_out1_i, num_tiles);
        }
    }
}
