// writer.cpp — Wormhole 1-D FFT output / feedback kernel (RISCV_1)
//
// FIXES vs writer_debug.cpp:
//   1. BUG (root cause of all-zero output): G2 formula was
//          G2 = (half_N >= half_m2) ? (half_N / half_m2) : 0
//      For stage 0, N=2: half_N=1, half_m2 = (1<<(0+1))<<1 / 2 = 2
//        → (1 >= 2) is false → G2 = 0 → shuffle body never ran → zeros.
//      The correct formula is:
//          G2 = half_N / half_m2   (with the precondition half_N >= half_m2)
//      but the outer guard should also fire when half_N < half_m2, because
//      then there is exactly ONE group covering the whole vector:
//          G2 = 1  when  half_N < half_m2
//      Fixed to: G2 = (half_m2 <= elems) ? (elems / half_m2) : 1u
//      where elems = tiles_per_row * TILE_SIZE elements.
//
//   2. Byte-offset arithmetic: the inner loop used `idx * F` where F =
//      sizeof(float) = 4, which is correct.  Verified and kept.
//
//   3. Removed unused `elems` variable computed from tile_bytes/F; now
//      derived directly from tiles_per_row * TILE_SIZE for clarity.
//
//   4. Added explicit `F = 4u` constant instead of sizeof(float) to avoid
//      any potential issues with device-side sizeof.
//
//   5. The "pop before reserve" ordering was already correct; added comment
//      to make the anti-deadlock intent explicit.
//
// CB map:
//   0–3   even/odd real+imag input (feedback from previous stage)
//   6–9   out_even/odd real+imag (from compute)
//   16–19 L1 shuffle scratch (plain memory, never pushed/popped)
//
// Args:
//   [0]  out even_r DRAM address
//   [1]  out even_i DRAM address
//   [2]  out odd_r  DRAM address
//   [3]  out odd_i  DRAM address
//   [4]  tiles_per_row
//   [5]  num_stages
//   [6]  half_N
//   [7]  tile_offset  (first output tile index)
//   [8]  rows_per_core

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
    // Scratch CBs — used as plain L1 memory, never pushed/popped
    constexpr uint32_t CB_SCR_ER = 16;
    constexpr uint32_t CB_SCR_EI = 17;
    constexpr uint32_t CB_SCR_OR = 18;
    constexpr uint32_t CB_SCR_OI = 19;

    const uint32_t tile_bytes = get_tile_size(CB_OER);
    const DataFormat fmt      = get_dataformat(CB_OER);

    // Use explicit 4-byte float size to avoid device-side sizeof issues
    constexpr uint32_t F = 4u;

    // Total float elements per row (may be larger than half_N due to tile padding)
    const uint32_t elems = tiles_per_row * (tile_bytes / F);

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

    // Get base L1 addresses for scratch CBs once (write ptr is stable)
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
                // ── Write final result to DRAM ───────────────────────
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
                // ── Intermediate: shuffle outputs → feedback inputs ──
                //
                // After stage s, the butterfly produced:
                //   out_even[g * half_m + j]  = X[g*m + j]        (j=0..half_m-1)
                //   out_odd [g * half_m + j]  = X[g*m + j+half_m] (j=0..half_m-1)
                // where m = 2^(s+1), half_m = m/2, g = group index.
                //
                // For stage s+1 we need the data regrouped with m' = 2^(s+2).
                // This shuffle reconstructs the natural order in scratch, then
                // splits it into new even/odd halves for the next butterfly.

                const uint32_t m      = 1u << (stage + 1);       // current m
                const uint32_t half_m = m >> 1;                   // m/2
                const uint32_t m2     = m << 1;                   // next m
                const uint32_t half_m2 = m2 >> 1;                 // next half_m

                // FIX: G2 was (half_N >= half_m2) ? (half_N/half_m2) : 0
                // When half_N < half_m2 (small N or early stage) the entire
                // data fits in one group, so G2 must be 1, not 0.
                // Use element count (elems) rather than half_N to handle
                // tile-padded buffers correctly.
                const uint32_t G2 = (half_m2 <= elems) ? (elems / half_m2) : 1u;

                const uint32_t src_er = get_read_ptr(CB_OER);
                const uint32_t src_ei = get_read_ptr(CB_OEI);
                const uint32_t src_or = get_read_ptr(CB_OOR);
                const uint32_t src_oi = get_read_ptr(CB_OOI);

                // Zero scratch
                for (uint32_t lp = 0; lp < elems; ++lp) {
                    const uint32_t off = lp * F;
                    wr32(scr_er + off, 0u); wr32(scr_ei + off, 0u);
                    wr32(scr_or + off, 0u); wr32(scr_oi + off, 0u);
                }

                // Shuffle: map from stage-s output layout → stage-(s+1) input layout
                const uint32_t log2m  = stage + 1;          // log2(m)
                const uint32_t m_mask = m - 1u;

                uint32_t dst_idx = 0;
                for (uint32_t g2 = 0; g2 < G2; ++g2) {
                    const uint32_t base_e = g2 * m2;            // even group start
                    const uint32_t base_o = base_e + half_m2;   // odd  group start

                    for (uint32_t j2 = 0; j2 < half_m2; ++j2, ++dst_idx) {
                        // ── even output of next stage ──
                        {
                            const uint32_t f      = base_e + j2;
                            const uint32_t g_cur  = f >> log2m;
                            const uint32_t offset = f & m_mask;
                            uint32_t idx, sr, si;
                            if (offset < half_m) {
                                // came from out_even
                                idx = g_cur * half_m + offset;
                                sr  = src_er;  si = src_ei;
                            } else {
                                // came from out_odd
                                idx = g_cur * half_m + (offset - half_m);
                                sr  = src_or;  si = src_oi;
                            }
                            wr32(scr_er + dst_idx * F, rd32(sr + idx * F));
                            wr32(scr_ei + dst_idx * F, rd32(si + idx * F));
                        }
                        // ── odd output of next stage ──
                        {
                            const uint32_t f      = base_o + j2;
                            const uint32_t g_cur  = f >> log2m;
                            const uint32_t offset = f & m_mask;
                            uint32_t idx, sr, si;
                            if (offset < half_m) {
                                idx = g_cur * half_m + offset;
                                sr  = src_er;  si = src_ei;
                            } else {
                                idx = g_cur * half_m + (offset - half_m);
                                sr  = src_or;  si = src_oi;
                            }
                            wr32(scr_or + dst_idx * F, rd32(sr + idx * F));
                            wr32(scr_oi + dst_idx * F, rd32(si + idx * F));
                        }
                    }
                }

                // KEY: pop output CBs BEFORE reserving feedback CBs to
                // prevent circular deadlock (compute waits on input CB space,
                // writer waits on output CB space).
                cb_pop_front(CB_OER, tiles_per_row);
                cb_pop_front(CB_OEI, tiles_per_row);
                cb_pop_front(CB_OOR, tiles_per_row);
                cb_pop_front(CB_OOI, tiles_per_row);

                // Push shuffled data back into input CBs for next stage
                cb_reserve_back(CB_ER, tiles_per_row);
                cb_reserve_back(CB_EI, tiles_per_row);
                cb_reserve_back(CB_OR, tiles_per_row);
                cb_reserve_back(CB_OI, tiles_per_row);

                const uint32_t dst_er = get_write_ptr(CB_ER);
                const uint32_t dst_ei = get_write_ptr(CB_EI);
                const uint32_t dst_or = get_write_ptr(CB_OR);
                const uint32_t dst_oi = get_write_ptr(CB_OI);

                for (uint32_t lp = 0; lp < elems; ++lp) {
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