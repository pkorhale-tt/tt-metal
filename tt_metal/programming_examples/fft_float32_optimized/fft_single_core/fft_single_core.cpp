// fft_single_core_opt.cpp — UPDATE ONLY CB CREATION

void create_cb(Program& p, CoreCoord c, uint32_t id, uint32_t n, uint32_t b) {
    CircularBufferConfig cfg = CircularBufferConfig(n*b, {{id, tt::DataFormat::Float32}})
        .set_page_size(id, b);
    CreateCircularBuffer(p, c, cfg);
}

// In main():

// Input CBs: depth = tiles (stage-0 from reader, L1 shuffle from writer)
create_cb(prog, core, 0, tiles, TILE_BYTES);  // even_r
create_cb(prog, core, 1, tiles, TILE_BYTES);  // even_i
create_cb(prog, core, 2, tiles, TILE_BYTES);  // odd_r
create_cb(prog, core, 3, tiles, TILE_BYTES);  // odd_i

// ✅ OPTIMIZED: Twiddle CBs depth = tiles (was log2N*tiles)
create_cb(prog, core, 4, tiles, TILE_BYTES);  // tw_r — STREAMING
create_cb(prog, core, 5, tiles, TILE_BYTES);  // tw_i — STREAMING

// Output CBs: depth = tiles
create_cb(prog, core, 16, tiles, TILE_BYTES);  // out0_r
create_cb(prog, core, 17, tiles, TILE_BYTES);  // out0_i
create_cb(prog, core, 18, tiles, TILE_BYTES);  // out1_r
create_cb(prog, core, 19, tiles, TILE_BYTES);  // out1_i

// ✅ REMOVED: No tmp CBs needed (compute uses DST regs directly)
// Old code had CB 20-23 (tmp0, tmp1, tw_odd_r, tw_odd_i) — now deleted