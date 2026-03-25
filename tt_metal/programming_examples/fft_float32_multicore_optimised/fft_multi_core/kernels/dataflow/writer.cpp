// writer.cpp — RISCV_1 dataflow kernel
// Wormhole 1-D Cooley-Tukey FFT (decimation-in-time, radix-2)
// ═══════════════════════════════════════════════════════════════════════
//
//  Responsibilities:
//    - Last stage:  DMA output even/odd (CB 6-9) to DRAM.
//    - Other stages: shuffle butterfly outputs into the correct even/odd
//      order for the next stage, then push into CB 0-3 for compute.
//
//  DEADLOCK-FREE TWO-PASS PROTOCOL:
//    Pass 1 (drain):  Wait for CB 6-9, run shuffle, write into L1 scratch
//                     (CB 16-19 write pointers used as plain memory),
//                     then POP CB 6-9 immediately.
//    Pass 2 (fill):   Reserve CB 0-3, copy from L1 scratch, push.
//
//  Popping CB 6-9 before reserving CB 0-3 eliminates the circular
//  dependency that would otherwise occur:
//    compute waits on CB 0-3 ← writer fills 0-3
//    writer waits on CB 6-9  ← compute fills 6-9
//    (would deadlock without Pass 1 popping 6-9 first)
//
//  STACK DISCIPLINE:
//    No VLAs.  No large stack arrays.  L1 scratch is backed by four
//    depth-1 CBs (16-19) whose write pointers act as flat 4 KB buffers.
//    Stack use is O(1) — only loop counters and pointers.
//
//  CB map (for reference):
//    CB  0-3   even/odd input  (compute reads, writer writes for stage 1+)
//    CB  4-5   twiddles        (reader writes, compute reads — writer ignores)
//    CB  6-9   butterfly out   (compute writes, writer reads)
//    CB 10-13  compute scratch (writer ignores)
//    CB 14-15  reader L1 ctw   (writer ignores)
//    CB 16-19  writer L1 scratch (plain memory, never pushed/popped)
//
//  Argument map:
//    [0]  out_even_r_addr  DRAM output for last-stage even real
//    [1]  out_even_i_addr  DRAM output for last-stage even imag
//    [2]  out_odd_r_addr   DRAM output for last-stage odd  real
//    [3]  out_odd_i_addr   DRAM output for last-stage odd  imag
//    [4]  tiles_per_row    tiles per butterfly group
//    [5]  num_stages       log2(N)
//    [6]  half_N           N/2
//    [7]  tile_offset      first output tile index for this core
//    [8]  rows_per_core

#include <cstdint>
#include "dataflow_api.h"

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

    // Compute output CBs (writer reads)
    constexpr uint32_t CB_OUT_ER = 6;
    constexpr uint32_t CB_OUT_EI = 7;
    constexpr uint32_t CB_OUT_OR = 8;
    constexpr uint32_t CB_OUT_OI = 9;

    // Next-stage input CBs (writer pushes, compute reads)
    constexpr uint32_t CB_EVEN_R = 0;
    constexpr uint32_t CB_EVEN_I = 1;
    constexpr uint32_t CB_ODD_R  = 2;
    constexpr uint32_t CB_ODD_I  = 3;

    // L1 scratch CBs — used as plain memory, NEVER pushed/popped.
    // Host must create these with depth=1.
    constexpr uint32_t CB_SCR_ER = 16;
    constexpr uint32_t CB_SCR_EI = 17;
    constexpr uint32_t CB_SCR_OR = 18;
    constexpr uint32_t CB_SCR_OI = 19;

    const uint32_t tile_bytes      = get_tile_size(CB_OUT_ER);
    const DataFormat fmt           = get_dataformat(CB_OUT_ER);
    constexpr uint32_t F           = sizeof(float);
    const uint32_t elems_per_batch = tiles_per_row * (tile_bytes / F);

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

    // Stable L1 scratch pointers — write ptrs of depth-1 CBs.
    // Never change because we never push/pop the scratch CBs.
    const uint32_t scr_er = get_write_ptr(CB_SCR_ER);
    const uint32_t scr_ei = get_write_ptr(CB_SCR_EI);
    const uint32_t scr_or = get_write_ptr(CB_SCR_OR);
    const uint32_t scr_oi = get_write_ptr(CB_SCR_OI);

    for (uint32_t row = 0; row < rows_per_core; ++row) {
        const uint32_t row_tile_base = tile_offset + row * tiles_per_row;

        for (uint32_t stage = 0; stage < num_stages; ++stage) {
            const bool is_last = (stage == num_stages - 1);

            // ── Wait for compute to produce butterfly outputs ─────────
            cb_wait_front(CB_OUT_ER, tiles_per_row);
            cb_wait_front(CB_OUT_EI, tiles_per_row);
            cb_wait_front(CB_OUT_OR, tiles_per_row);
            cb_wait_front(CB_OUT_OI, tiles_per_row);

            if (is_last) {
                // ── Final stage: DMA to DRAM ──────────────────────────
                // The full FFT output is in out_even (first N/2 bins) and
                // out_odd (last N/2 bins).  Write them to contiguous DRAM.
                const uint32_t src_er = get_read_ptr(CB_OUT_ER);
                const uint32_t src_ei = get_read_ptr(CB_OUT_EI);
                const uint32_t src_or = get_read_ptr(CB_OUT_OR);
                const uint32_t src_oi = get_read_ptr(CB_OUT_OI);

                for (uint32_t t = 0; t < tiles_per_row; ++t) {
                    noc_async_write_tile(row_tile_base + t, gen_er,
                        src_er + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, gen_ei,
                        src_ei + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, gen_or,
                        src_or + t * tile_bytes);
                    noc_async_write_tile(row_tile_base + t, gen_oi,
                        src_oi + t * tile_bytes);
                }
                noc_async_write_barrier();

                cb_pop_front(CB_OUT_ER, tiles_per_row);
                cb_pop_front(CB_OUT_EI, tiles_per_row);
                cb_pop_front(CB_OUT_OR, tiles_per_row);
                cb_pop_front(CB_OUT_OI, tiles_per_row);

            } else {
                // ── Intermediate stage: shuffle into next-stage even/odd ─
                //
                // After a DIT butterfly at stage s, the two output groups
                // (even and odd) must be re-split into the new even/odd
                // partition for stage s+1.
                //
                // At stage s, output group g (0-indexed) spans elements:
                //   even: [g * m, g*m + m/2)     with m = 2^(s+1)
                //   odd:  [g * m + m/2, g*m + m)
                //
                // For stage s+1 with m2 = 2*m, new groups of size m2:
                //   new_even of group g2: elements at 2-step positions
                //   new_odd  of group g2: elements at 2-step + offset
                //
                // The shuffle below re-indexes from the stage-s layout
                // (even[0..N/2-1], odd[0..N/2-1]) to the stage-(s+1)
                // layout (new_even[0..N/2-1], new_odd[0..N/2-1]).
                //
                // ── PASS 1: shuffle into L1 scratch ──────────────────
                const uint32_t m       = 1u << (stage + 1);   // m = 2^(s+1)
                const uint32_t half_m  = m >> 1;
                const uint32_t m2      = m << 1;               // next-stage block size
                const uint32_t half_m2 = m2 >> 1;
                // Number of complete next-stage groups that fit in N/2 elements
                const uint32_t G2 = (half_N >= half_m2) ? (half_N / half_m2) : 0u;

                const uint32_t src_er = get_read_ptr(CB_OUT_ER);
                const uint32_t src_ei = get_read_ptr(CB_OUT_EI);
                const uint32_t src_or = get_read_ptr(CB_OUT_OR);
                const uint32_t src_oi = get_read_ptr(CB_OUT_OI);

                // Zero-initialise scratch (clears unused tile tail).
                for (uint32_t lp = 0; lp < elems_per_batch; ++lp) {
                    const uint32_t off = lp * F;
                    wr32(scr_er + off, 0u);
                    wr32(scr_ei + off, 0u);
                    wr32(scr_or + off, 0u);
                    wr32(scr_oi + off, 0u);
                }

                if (G2 > 0) {
                    const uint32_t log2m  = stage + 1;   // = log2(m)
                    const uint32_t m_mask = m - 1u;
                    uint32_t dst_idx = 0;

                    for (uint32_t g2 = 0; g2 < G2; ++g2) {
                        const uint32_t base_e = g2 * m2;
                        const uint32_t base_o = base_e + half_m2;

                        for (uint32_t j2 = 0; j2 < half_m2; ++j2, ++dst_idx) {
                            // ── even slot of next stage ───────────────
                            {
                                const uint32_t f      = base_e + j2;
                                const uint32_t g_cur  = f >> log2m;
                                const uint32_t offset = f & m_mask;
                                uint32_t idx, sr, si;
                                if (offset < half_m) {
                                    // came from even output of stage s
                                    idx = g_cur * half_m + offset;
                                    sr = src_er; si = src_ei;
                                } else {
                                    // came from odd output of stage s
                                    idx = g_cur * half_m + (offset - half_m);
                                    sr = src_or; si = src_oi;
                                }
                                wr32(scr_er + dst_idx * F, rd32(sr + idx * F));
                                wr32(scr_ei + dst_idx * F, rd32(si + idx * F));
                            }
                            // ── odd slot of next stage ────────────────
                            {
                                const uint32_t f      = base_o + j2;
                                const uint32_t g_cur  = f >> log2m;
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
                }

                // KEY: pop output CBs BEFORE reserving input CBs.
                // This breaks the circular deadlock:
                //   compute waiting on CB 0-3 ← writer must fill
                //   writer waiting on CB 6-9  ← compute must fill
                // By popping 6-9 first, compute is unblocked and can
                // start filling 6-9 for stage s+1 while writer fills 0-3.
                cb_pop_front(CB_OUT_ER, tiles_per_row);
                cb_pop_front(CB_OUT_EI, tiles_per_row);
                cb_pop_front(CB_OUT_OR, tiles_per_row);
                cb_pop_front(CB_OUT_OI, tiles_per_row);

                // ── PASS 2: copy L1 scratch → CB 0-3 ─────────────────
                cb_reserve_back(CB_EVEN_R, tiles_per_row);
                cb_reserve_back(CB_EVEN_I, tiles_per_row);
                cb_reserve_back(CB_ODD_R,  tiles_per_row);
                cb_reserve_back(CB_ODD_I,  tiles_per_row);

                const uint32_t dst_er_cb = get_write_ptr(CB_EVEN_R);
                const uint32_t dst_ei_cb = get_write_ptr(CB_EVEN_I);
                const uint32_t dst_or_cb = get_write_ptr(CB_ODD_R);
                const uint32_t dst_oi_cb = get_write_ptr(CB_ODD_I);

                for (uint32_t lp = 0; lp < elems_per_batch; ++lp) {
                    const uint32_t off = lp * F;
                    wr32(dst_er_cb + off, rd32(scr_er + off));
                    wr32(dst_ei_cb + off, rd32(scr_ei + off));
                    wr32(dst_or_cb + off, rd32(scr_or + off));
                    wr32(dst_oi_cb + off, rd32(scr_oi + off));
                }

                cb_push_back(CB_EVEN_R, tiles_per_row);
                cb_push_back(CB_EVEN_I, tiles_per_row);
                cb_push_back(CB_ODD_R,  tiles_per_row);
                cb_push_back(CB_ODD_I,  tiles_per_row);
            }
        }
    }
}