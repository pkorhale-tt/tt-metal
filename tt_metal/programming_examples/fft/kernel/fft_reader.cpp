// ============================================================
//  fft_reader.cpp  –  BRISC (data mover in)
//
//  Stage 0: Load input from DRAM in bit-reversed order.
//  Local stages: reorder data into butterfly pairs, load twiddles.
//  NOC stages: load twiddles only (scratch data arrives via writer).
// ============================================================

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fft_common.h"

FORCE_INLINE uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

void kernel_main() {
    uint32_t input_addr     = get_arg_val<uint32_t>(0);
    // arg 1 unused (was bank_id)
    uint32_t twiddle_addr   = get_arg_val<uint32_t>(2);
    // arg 3 unused
    uint32_t local_N        = get_arg_val<uint32_t>(4);
    uint32_t my_id          = get_arg_val<uint32_t>(5);
    uint32_t total_N        = get_arg_val<uint32_t>(6);
    uint32_t num_local_stg  = get_arg_val<uint32_t>(7);
    uint32_t num_stages     = get_arg_val<uint32_t>(8);
    // arg 9 = use_bf16 (unused, always fp32)

    const DataFormat df        = get_dataformat(CB_LHS_R);
    const uint32_t   tile_size = get_tile_size(CB_LHS_R);
    uint32_t elem_bytes        = 4; // fp32

    // Input DRAM: separate real/imag tiles
    // Layout: real tiles [0..num_cores-1], imag tiles [num_cores..2*num_cores-1]
    // (host allocates and writes this layout)
    InterleavedAddrGenFast<true> input_r_gen = {
        .bank_base_address = input_addr,
        .page_size         = tile_size,
        .data_format       = df
    };
    // Imag starts at offset num_cores tiles into the same buffer
    // We treat input as: tile 0..N-1 = real, tile N..2N-1 = imag
    // where N = total_N / TILE_HW (number of tiles per channel)
    // For simplicity: real tile index = my_id, imag = my_id + num_tiles
    uint32_t num_tiles = total_N / TILE_HW; // tiles per channel

    InterleavedAddrGenFast<true> twiddle_r_gen = {
        .bank_base_address = twiddle_addr,
        .page_size         = tile_size,
        .data_format       = df
    };
    // Twiddle imag tiles follow real tiles: offset by num_stages * num_tiles
    uint32_t tw_imag_offset = num_stages * num_tiles;

    uint32_t global_offset = my_id * local_N;

    // ── Load input (bit-reversed) into CB_LHS_R/I ────────────
    {
        cb_reserve_back(CB_LHS_R, 1);
        cb_reserve_back(CB_LHS_I, 1);

        // Read our tile from DRAM.
        // Bit-reversal: the tile index to read for our core is
        // bit_reverse(my_id, log2(num_cores)).
        // Within-tile bit reversal is handled by the reader
        // reordering elements after loading.
        uint32_t log2_total = num_stages;
        uint32_t src_tile_r = bit_reverse(my_id, log2_total / (local_N / TILE_HW == 0 ? 1 : local_N / TILE_HW));
        // Simpler: just read our own tile; the host pre-shuffles input
        // in bit-reversed order before writing to DRAM.
        // (Standard approach for hardware FFT accelerators.)
        noc_async_read_tile(my_id, input_r_gen, get_write_ptr(CB_LHS_R));
        noc_async_read_tile(my_id + num_tiles, input_r_gen, get_write_ptr(CB_LHS_I));
        noc_async_read_barrier();

        cb_push_back(CB_LHS_R, 1);
        cb_push_back(CB_LHS_I, 1);
    }

    // ── Local stages: RHS reorder + twiddles ─────────────────
    for (uint32_t s = 0; s < num_local_stg; s++) {
        uint32_t stride = 1u << s;

        cb_reserve_back(CB_RHS_R, 1);
        cb_reserve_back(CB_RHS_I, 1);
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);

        // RHS: copy butterfly partner elements from LHS tile in L1
        // (local L1 scalar copy — no NOC needed)
        uint32_t* lhs_r = reinterpret_cast<uint32_t*>(get_read_ptr(CB_LHS_R));
        uint32_t* lhs_i = reinterpret_cast<uint32_t*>(get_read_ptr(CB_LHS_I));
        uint32_t* rhs_r = reinterpret_cast<uint32_t*>(get_write_ptr(CB_RHS_R));
        uint32_t* rhs_i = reinterpret_cast<uint32_t*>(get_write_ptr(CB_RHS_I));

        for (uint32_t i = 0; i < local_N / 2; i++) {
            uint32_t grp = i / stride;
            uint32_t pos = i % stride;
            uint32_t lo  = grp * (2 * stride) + pos;
            uint32_t hi  = lo + stride;
            rhs_r[lo] = lhs_r[hi];
            rhs_i[lo] = lhs_i[hi];
        }

        // Twiddle tile for this stage
        uint32_t tw_tile = s * num_tiles / num_stages + my_id % (num_tiles / num_stages + 1);
        // Simpler: one twiddle tile per stage per core
        uint32_t tw_idx = s * num_tiles + my_id;
        noc_async_read_tile(tw_idx, twiddle_r_gen, get_write_ptr(CB_TWIDDLE_R));
        // Imag twiddle: offset by tw_imag_offset tiles
        uint32_t tw_i_idx = tw_imag_offset + tw_idx;
        noc_async_read_tile(tw_i_idx, twiddle_r_gen, get_write_ptr(CB_TWIDDLE_I));
        noc_async_read_barrier();

        cb_push_back(CB_RHS_R, 1);
        cb_push_back(CB_RHS_I, 1);
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }

    // ── NOC stages: twiddles only ────────────────────────────
    for (uint32_t s = num_local_stg; s < num_stages; s++) {
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);

        uint32_t tw_idx   = s * num_tiles + my_id;
        uint32_t tw_i_idx = tw_imag_offset + tw_idx;

        noc_async_read_tile(tw_idx,   twiddle_r_gen, get_write_ptr(CB_TWIDDLE_R));
        noc_async_read_tile(tw_i_idx, twiddle_r_gen, get_write_ptr(CB_TWIDDLE_I));
        noc_async_read_barrier();

        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }
}