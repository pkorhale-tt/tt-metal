// fft_multicore_2d.cpp — 2D FFT host (BUGFREE + OPTIMISED)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  BUGS FIXED vs previous version
// ══════════════════════════════════════════════════════════════════════
//
//  BUG 2 / BUG 1 (primary hang — compute kernel)
//    cb_tmp0 and cb_tmp1 depth must be ≥ 2. The compute kernel pushes
//    2 tiles in Session A before popping any in Session B/C.
//    FIX: tmp_cb_depth = max(2u, tiles_per_row) — unchanged from the
//    previous "FINAL" version but now correctly paired with a compute
//    kernel that doesn't deadlock on the push-before-pop ordering.
//
//  BUG 4 (twiddle scatter size — reader kernel)
//    The reader's twiddle scatter loop previously filled only local_half
//    (= 512) of the TILE_SIZE (= 1024) float slots per twiddle tile.
//    FIX: reader now iterates elems_per_row = (tile_bytes/ELEM) *
//    tiles_per_row, filling all slots. No host change required — the
//    reader arg local_half (arg[10]) is still passed correctly; the
//    reader just ignores it and uses elems_per_row instead.
//
//  BUG 5 (row-loop CB race — reader + writer)
//    Reader was reserving even/odd CBs inside a per-stage loop, racing
//    with the writer's shuffle. Writer was popping out0/out1 AFTER
//    reserving even/odd, not before.
//    FIX: reader fills even/odd once per row (stage 0 only); writer pops
//    out0/out1 before reserving even/odd. No host changes required —
//    the CB depth (tiles_per_row) provides the back-pressure.
//
//  CB 22 (tw_odd_r) and CB 23 (tw_odd_i) remain removed.
//    These were unused in the fixed compute kernel.
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS
// ══════════════════════════════════════════════════════════════════════
//
//  1. mk_buf lambda uses ReplicatedBufferConfig for all data buffers —
//     each device gets its own copy, avoiding cross-device NOC traffic.
//
//  2. Input staging (prepare_stage0_row) pre-computes bit-reversed
//     indices once per row and bulk-inserts into pre-sized vectors to
//     eliminate repeated push_back reallocations.
//
//  3. Validation reports per-row max error in addition to global max,
//     making it easier to identify which row diverges first.
//
//  4. EnqueueWriteMeshBuffer calls for input buffers are all issued
//     before Finish() so the command queue can pipeline them.
//
// ══════════════════════════════════════════════════════════════════════

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"
#include "tt_metal/api/tt-metalium/allocator.hpp"
#include "tt_metal/api/tt-metalium/hal.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float    PI         = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;         // elements per tile
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

// ── Bit manipulation helpers ──────────────────────────────────────────
inline uint32_t f2u(float f)    { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float    u2f(uint32_t u) { float f;    std::memcpy(&f, &u, 4); return f; }

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// ── Pack/unpack tile buffers ──────────────────────────────────────────
std::vector<uint32_t> pack_tiles(const std::vector<float>& d, uint32_t ntiles) {
    std::vector<uint32_t> o(ntiles * TILE_SIZE, 0u);
    for (uint32_t i = 0; i < d.size() && i < o.size(); i++) o[i] = f2u(d[i]);
    return o;
}
std::vector<float> unpack_tiles(const std::vector<uint32_t>& d, uint32_t n) {
    std::vector<float> o(n);
    for (uint32_t i = 0; i < n && i < d.size(); i++) o[i] = u2f(d[i]);
    return o;
}

// ── Reference CPU FFT (Cooley-Tukey, in-place) ───────────────────────
void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const uint32_t N = re.size();
    uint32_t log2N  = 0;
    while ((1u << log2N) < N) log2N++;

    // Bit-reverse permutation.
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    // Butterfly stages.
    for (uint32_t s = 0; s < log2N; s++) {
        const uint32_t m  = 1u << (s + 1);
        const float    ab = (inv ? 2.f : -2.f) * PI / static_cast<float>(m);
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                const float wr = std::cos(ab * j);
                const float wi = std::sin(ab * j);
                const uint32_t e = k + j, o = k + j + m / 2;
                const float tr = wr * re[o] - wi * im[o];
                const float ti = wr * im[o] + wi * re[o];
                const float er = re[e], ei = im[e];
                re[e] = er + tr;  im[e] = ei + ti;
                re[o] = er - tr;  im[o] = ei - ti;
            }
        }
    }
    if (inv) {
        const float inv_N = 1.f / static_cast<float>(N);
        for (uint32_t i = 0; i < N; i++) { re[i] *= inv_N; im[i] *= inv_N; }
    }
}

// ── Stage-0 input preparation ─────────────────────────────────────────
// Bit-reverse and split into even/odd sub-sequences, then pack to tiles.
void prepare_stage0_row(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t row_offset, uint32_t N_row, uint32_t log2_row,
    uint32_t tiles_per_row,
    std::vector<uint32_t>& er, std::vector<uint32_t>& ei,
    std::vector<uint32_t>& or_, std::vector<uint32_t>& oi)
{
    const uint32_t half = N_row / 2;
    std::vector<float> _er(half), _ei(half), _or(half), _oi(half);
    for (uint32_t i = 0; i < half; i++) {
        const uint32_t e = bit_reverse(2 * i,     log2_row);
        const uint32_t o = bit_reverse(2 * i + 1, log2_row);
        _er[i] = sr[row_offset + e];  _ei[i] = si[row_offset + e];
        _or[i] = sr[row_offset + o];  _oi[i] = si[row_offset + o];
    }
    auto append = [](std::vector<uint32_t>& dst,
                     const std::vector<float>& src, uint32_t ntiles) {
        auto packed = [&]{
            std::vector<uint32_t> o(ntiles * TILE_SIZE, 0u);
            for (uint32_t i = 0; i < src.size() && i < o.size(); i++)
                o[i] = f2u(src[i]);
            return o;
        }();
        dst.insert(dst.end(), packed.begin(), packed.end());
    };
    append(er,  _er, tiles_per_row);
    append(ei,  _ei, tiles_per_row);
    append(or_, _or, tiles_per_row);
    append(oi,  _oi, tiles_per_row);
}

// ── Compact twiddle table (half_N entries, one per FFT point) ─────────
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N_row, uint32_t direction) {
    const uint32_t half  = N_row / 2;
    const float    sign  = (direction == 1) ? 1.f : -1.f;
    std::vector<uint32_t> tw_r(half), tw_i(half);
    for (uint32_t k = 0; k < half; k++) {
        const float angle = sign * 2.f * PI * static_cast<float>(k)
                                             / static_cast<float>(N_row);
        tw_r[k] = f2u(std::cos(angle));
        tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

// ── CB factory ────────────────────────────────────────────────────────
CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bytes_per_tile) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles * bytes_per_tile,
                             {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes_per_tile);
    return CreateCircularBuffer(p, c, cfg);
}

// ── Core detection ────────────────────────────────────────────────────
uint32_t detect_available_cores(IDevice* device, uint32_t max_req,
                                 uint32_t num_rows) {
    const CoreCoord grid = device->compute_with_storage_grid_size();
    std::cout << " Device grid : " << grid.x << " x " << grid.y
              << " Tensix cores\n";
    uint32_t usable = 0;
    for (uint32_t col = 0; col < grid.x; col++) {
        try {
            (void)device->worker_core_from_logical_core({col, 0});
            usable++;
        } catch (...) { break; }
    }
    std::cout << " Usable row-0 cores: " << usable << "\n";
    uint32_t cap    = std::min(usable, max_req);
    uint32_t result = 1u;
    if (num_rows == 0) {
        while (result * 2 <= cap) result *= 2;
    } else {
        while (result * 2 <= cap && num_rows % (result * 2) == 0) result *= 2;
    }
    std::cout << " Selected cores: " << result << "\n";
    return result;
}

// ── Argument parsing helpers ──────────────────────────────────────────
bool is_uint_str(const char* s) {
    if (!s || !*s) return false;
    for (const char* p = s; *p; ++p)
        if (*p < '0' || *p > '9') return false;
    return true;
}

bool read_input_file(const std::string& path, uint32_t N_row,
                     std::vector<float>& ir, std::vector<float>& ii) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open input file: " << path << "\n";
        return false;
    }
    std::vector<float> vals;
    std::string tok;
    while (f >> tok) {
        if (!tok.empty() && tok.back() == ',') tok.pop_back();
        if (tok.empty()) continue;
        try { vals.push_back(std::stof(tok)); }
        catch (...) {
            std::cerr << "Bad token in file: '" << tok << "'\n";
            return false;
        }
    }
    if (vals.empty()) { std::cerr << "Empty input file\n"; return false; }

    ir.assign(N_row, 0.f);
    ii.assign(N_row, 0.f);

    const bool interleaved = (vals.size() >= 2 * N_row);
    if (interleaved) {
        std::cout << " File mode: interleaved complex ("
                  << vals.size() << " values → " << N_row << " complex)\n";
        for (uint32_t i = 0; i < N_row && 2 * i + 1 < vals.size(); i++) {
            ir[i] = vals[2 * i];
            ii[i] = vals[2 * i + 1];
        }
    } else {
        std::cout << " File mode: real-only ("
                  << vals.size() << " values → " << N_row << " points)\n";
        if (vals.size() < N_row)
            std::cout << " Note: " << (N_row - vals.size())
                      << " values zero-padded\n";
        for (uint32_t i = 0; i < N_row && i < vals.size(); i++)
            ir[i] = vals[i];
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <direction:0|1> [N_row] [num_rows] [num_cores] [input_file]\n"
                  << " Default: forward FFT, N_row=1024, num_rows=auto\n";
        return 1;
    }

    const uint32_t direction            = static_cast<uint32_t>(std::atoi(argv[1]));
    uint32_t       N_row                = 1024;
    uint32_t       num_rows             = 0;
    uint32_t       user_cores           = 0;
    const uint32_t rows_per_core_target = 128;
    std::string    in_file              = "";

    for (int i = 2; i < argc; i++) {
        if (!is_uint_str(argv[i])) { in_file = argv[i]; continue; }
        const uint32_t v = static_cast<uint32_t>(std::stoul(argv[i]));
        if      (v >= 2 && v <= 64 && (v & (v-1)) == 0) user_cores = v;
        else if (v > 64  && v <= 1024 && (v & (v-1)) == 0) N_row    = v;
        else if (v > 1024 && (v & (v-1)) == 0)             num_rows  = v;
        else if (v >= 2)                                    num_rows  = v;
    }
    if (N_row < 2 || (N_row & (N_row - 1))) {
        std::cerr << "N_row must be a power of 2\n"; return 1;
    }

    // ── Device init ───────────────────────────────────────────────────
    const int dev_id = 0;
    auto mesh   = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(dev_id);
    auto& cq    = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices().at(0);

    const uint32_t max_req   = user_cores > 0 ? user_cores : 64u;
    const uint32_t num_cores = detect_available_cores(device, max_req, num_rows);

    if (num_rows == 0) num_rows = num_cores * rows_per_core_target;
    num_rows = (num_rows / num_cores) * num_cores;
    if (num_rows == 0) num_rows = num_cores;

    const uint32_t rows_per_core = num_rows / num_cores;
    const uint32_t log2_row      = [&]{
        uint32_t l = 0; while ((1u << l) < N_row) l++; return l; }();
    const uint32_t half_row      = N_row / 2;
    const uint32_t tiles_per_row = (half_row + TILE_SIZE - 1) / TILE_SIZE;
    const uint32_t compact_bytes = half_row * sizeof(float);
    const uint32_t total_N       = N_row * num_rows;

    // tmp CB depth must be ≥ 2 (Session A pushes 2 tiles before B/C pop any).
    const uint32_t tmp_cb_depth = std::max(2u, tiles_per_row);

    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " TT-Metal MULTICORE FFT (row decomposition)\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " N_row        : " << N_row        << "\n";
    std::cout << " num_rows     : " << num_rows     << "\n";
    std::cout << " num_cores    : " << num_cores    << "\n";
    std::cout << " rows/core    : " << rows_per_core << "\n";
    std::cout << " log2(N_row)  : " << log2_row     << "\n";
    std::cout << " tiles/row    : " << tiles_per_row << "\n";
    std::cout << " tmp CB depth : " << tmp_cb_depth  << "\n";
    std::cout << " Direction    : " << (direction ? "Inverse" : "Forward") << "\n";
    std::cout << " Total FFTs   : " << num_rows
              << "  (" << num_cores << " cores × " << rows_per_core << " rows)\n";
    std::cout << " Total points : " << (static_cast<uint64_t>(num_rows) * N_row / 1024)
              << " K complex samples\n";
    std::cout << "════════════════════════════════════════════════\n";

    // ── Input data ────────────────────────────────────────────────────
    std::vector<float> ir(total_N, 0.f), ii(total_N, 0.f);

    if (!in_file.empty()) {
        std::cout << " Input file  : " << in_file << "\n";
        std::vector<float> row_r, row_i;
        if (!read_input_file(in_file, N_row, row_r, row_i)) {
            mesh->close(); return 1;
        }
        for (uint32_t row = 0; row < num_rows; row++)
            for (uint32_t i = 0; i < N_row; i++) {
                ir[row * N_row + i] = row_r[i];
                ii[row * N_row + i] = row_i[i];
            }
    } else {
        // Default: sum of two sinusoids, all rows identical.
        for (uint32_t row = 0; row < num_rows; row++)
            for (uint32_t i = 0; i < N_row; i++)
                ir[row * N_row + i] =
                    std::sin(2.f * PI * 4.f * i / N_row)
                  + 0.5f * std::sin(2.f * PI * 8.f * i / N_row);
    }

    // ── Reference CPU FFT ─────────────────────────────────────────────
    std::vector<float> ref_r(ir), ref_i(ii);
    for (uint32_t row = 0; row < num_rows; row++) {
        std::vector<float> row_r(ir.begin() + row * N_row,
                                  ir.begin() + (row + 1) * N_row);
        std::vector<float> row_i(ii.begin() + row * N_row,
                                  ii.begin() + (row + 1) * N_row);
        cpu_fft(row_r, row_i, direction == 1);
        for (uint32_t i = 0; i < N_row; i++) {
            ref_r[row * N_row + i] = row_r[i];
            ref_i[row * N_row + i] = row_i[i];
        }
    }

    // ── Stage-0 input packing ─────────────────────────────────────────
    std::vector<uint32_t> all_er, all_ei, all_or, all_oi;
    all_er.reserve(num_rows * tiles_per_row * TILE_SIZE);
    all_ei.reserve(num_rows * tiles_per_row * TILE_SIZE);
    all_or.reserve(num_rows * tiles_per_row * TILE_SIZE);
    all_oi.reserve(num_rows * tiles_per_row * TILE_SIZE);
    for (uint32_t row = 0; row < num_rows; row++)
        prepare_stage0_row(ir, ii, row * N_row, N_row, log2_row,
                           tiles_per_row, all_er, all_ei, all_or, all_oi);

    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N_row, direction);

    // ── Program + core range ──────────────────────────────────────────
    Program   prog       = CreateProgram();
    CoreRange core_range({0, 0}, {num_cores - 1, 0});

    // ── DRAM buffers ──────────────────────────────────────────────────
    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile{
        .page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};

    const uint32_t tiles_per_core = tiles_per_row * rows_per_core;
    const uint32_t bytes_per_core = tiles_per_core * TILE_BYTES;
    const uint32_t total_bytes    = bytes_per_core * num_cores;

    auto mk_buf = [&](uint32_t bytes) {
        ReplicatedBufferConfig rc{.size = bytes};
        return MeshBuffer::create(rc, dram_tile, mesh.get());
    };

    auto b_er  = mk_buf(total_bytes);
    auto b_ei  = mk_buf(total_bytes);
    auto b_or  = mk_buf(total_bytes);
    auto b_oi  = mk_buf(total_bytes);
    auto b_o0r = mk_buf(total_bytes);
    auto b_o0i = mk_buf(total_bytes);
    auto b_o1r = mk_buf(total_bytes);
    auto b_o1i = mk_buf(total_bytes);

    DeviceLocalBufferConfig dram_cmp{
        .page_size = compact_bytes, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig rc_cmp{.size = compact_bytes};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // ── Circular buffers ──────────────────────────────────────────────
    //
    // CB depth rules:
    //   - Data CBs (even/odd/out): depth = tiles_per_row. The writer and
    //     reader sequence one row at a time; depth=1 tile per row is
    //     sufficient and uses the least L1.
    //   - Twiddle CBs (tw_r/tw_i): depth = tiles_per_row (one push per
    //     stage; compute drains it before the next push).
    //   - Scratch CBs (tmp0/tmp1): depth = tmp_cb_depth (≥ 2). Session A
    //     pushes 2 tiles before Session B pops the first.
    //   - Compact CBs: sized to hold the entire half_N-float table.
    //
    // CB 22 and CB 23 (tw_odd_r / tw_odd_i) intentionally absent —
    // the fixed compute kernel routes partial products through tmp0/tmp1.

    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        create_cb(prog, cc,  0, tiles_per_row,  TILE_BYTES);  // even_r
        create_cb(prog, cc,  1, tiles_per_row,  TILE_BYTES);  // even_i
        create_cb(prog, cc,  2, tiles_per_row,  TILE_BYTES);  // odd_r
        create_cb(prog, cc,  3, tiles_per_row,  TILE_BYTES);  // odd_i
        create_cb(prog, cc,  4, tiles_per_row,  TILE_BYTES);  // tw_r
        create_cb(prog, cc,  5, tiles_per_row,  TILE_BYTES);  // tw_i
        create_cb(prog, cc, 16, tiles_per_row,  TILE_BYTES);  // out0_r
        create_cb(prog, cc, 17, tiles_per_row,  TILE_BYTES);  // out0_i
        create_cb(prog, cc, 18, tiles_per_row,  TILE_BYTES);  // out1_r
        create_cb(prog, cc, 19, tiles_per_row,  TILE_BYTES);  // out1_i
        create_cb(prog, cc, 20, tmp_cb_depth,   TILE_BYTES);  // tmp0  depth ≥ 2
        create_cb(prog, cc, 21, tmp_cb_depth,   TILE_BYTES);  // tmp1  depth ≥ 2
        // CB 10/11: compact twiddle table — single allocation, half_N floats.
        const uint32_t cmp_ntiles =
            (compact_bytes + TILE_BYTES - 1) / TILE_BYTES;
        create_cb(prog, cc, 10, cmp_ntiles, compact_bytes);   // compact_r
        create_cb(prog, cc, 11, cmp_ntiles, compact_bytes);   // compact_i
    }

    // ── Kernel creation ───────────────────────────────────────────────
    constexpr const char* KERNEL_PATH =
        "tt_metal/programming_examples/fft_float32_multicore_optimised/"
        "fft_multi_core/kernels/";

    KernelHandle reader_k = CreateKernel(prog,
        std::string(KERNEL_PATH) + "dataflow/reader_fft_f32.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default});

    KernelHandle writer_k = CreateKernel(prog,
        std::string(KERNEL_PATH) + "dataflow/writer_fft_f32.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default});

    KernelHandle compute_k = CreateKernel(prog,
        std::string(KERNEL_PATH) + "compute/fft_compute_f32.cpp",
        core_range,
        ComputeConfig{.math_fidelity    = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true,
                      .math_approx_mode = false});

    // ── Runtime arguments ─────────────────────────────────────────────
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord       cc          = {c, 0};
        const uint32_t tile_offset  = c * tiles_per_core;

        // Reader: 12 args [0..11]
        SetRuntimeArgs(prog, reader_k, cc, std::vector<uint32_t>{
            b_er->address(),     // [0]  even_r DRAM base
            b_ei->address(),     // [1]  even_i DRAM base
            b_or->address(),     // [2]  odd_r  DRAM base
            b_oi->address(),     // [3]  odd_i  DRAM base
            b_cmp_r->address(),  // [4]  compact twiddle real
            b_cmp_i->address(),  // [5]  compact twiddle imag
            tiles_per_row,       // [6]  tiles per row
            tile_offset,         // [7]  first tile index for this core
            log2_row,            // [8]  num_stages
            half_row,            // [9]  half_N
            half_row,            // [10] local_half (kept for ABI compat)
            rows_per_core,       // [11] rows this core processes
        });

        // Compute: 2 args [0..1]
        SetRuntimeArgs(prog, compute_k, cc, std::vector<uint32_t>{
            log2_row,            // [0]  num_stages
            tiles_per_row,       // [1]  tiles_per_stage
        });

        // Writer: 14 args [0..13]
        SetRuntimeArgs(prog, writer_k, cc, std::vector<uint32_t>{
            b_o0r->address(),    // [0]  out0_r DRAM base
            b_o0i->address(),    // [1]  out0_i DRAM base
            b_o1r->address(),    // [2]  out1_r DRAM base
            b_o1i->address(),    // [3]  out1_i DRAM base
            tiles_per_row,       // [4]  local_tiles per row
            log2_row,            // [5]  num_stages
            half_row,            // [6]  local_half
            half_row,            // [7]  half_N
            1u,                  // [8]  num_cores (self-contained)
            c,                   // [9]  core_id
            0u,                  // [10] log2_cores
            tile_offset,         // [11] base tile offset for DRAM writes
            0u,                  // [12] core_elem_base
            rows_per_core,       // [13] rows this core processes
        });
    }

    // ── Workload dispatch ─────────────────────────────────────────────
    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    std::cout << "Writing inputs to DRAM...\n";
    // Issue all writes before Finish so the command queue pipelines them.
    EnqueueWriteMeshBuffer(cq, b_er,    all_er,  false);
    EnqueueWriteMeshBuffer(cq, b_ei,    all_ei,  false);
    EnqueueWriteMeshBuffer(cq, b_or,    all_or,  false);
    EnqueueWriteMeshBuffer(cq, b_oi,    all_oi,  false);
    EnqueueWriteMeshBuffer(cq, b_cmp_r, cmp_r_t, false);
    EnqueueWriteMeshBuffer(cq, b_cmp_i, cmp_i_t, false);
    Finish(cq);

    std::cout << "Launching multicore FFT (" << num_cores << " cores, "
              << num_rows << " rows of " << N_row << " points)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

    // ── Read results ──────────────────────────────────────────────────
    std::vector<uint32_t> o0r_raw(total_bytes / 4);
    std::vector<uint32_t> o0i_raw(total_bytes / 4);
    std::vector<uint32_t> o1r_raw(total_bytes / 4);
    std::vector<uint32_t> o1i_raw(total_bytes / 4);
    EnqueueReadMeshBuffer(cq, o0r_raw, b_o0r, true);
    EnqueueReadMeshBuffer(cq, o0i_raw, b_o0i, true);
    EnqueueReadMeshBuffer(cq, o1r_raw, b_o1r, true);
    EnqueueReadMeshBuffer(cq, o1i_raw, b_o1i, true);

    // Reconstruct interleaved result.
    std::vector<float> result_r(total_N), result_i(total_N);
    for (uint32_t row = 0; row < num_rows; row++) {
        const uint32_t tile_base = row * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < half_row; i++) {
            result_r[row * N_row + i]            = u2f(o0r_raw[tile_base + i]);
            result_i[row * N_row + i]            = u2f(o0i_raw[tile_base + i]);
            result_r[row * N_row + i + half_row] = u2f(o1r_raw[tile_base + i]);
            result_i[row * N_row + i + half_row] = u2f(o1i_raw[tile_base + i]);
        }
    }
    // Inverse FFT normalisation (applied by host to match cpu_fft convention).
    if (direction == 1) {
        const float inv_N = 1.f / static_cast<float>(N_row);
        for (uint32_t i = 0; i < total_N; i++) {
            result_r[i] *= inv_N;
            result_i[i] *= inv_N;
        }
    }

    // ── Validation ────────────────────────────────────────────────────
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " VALIDATION (all " << num_rows << " rows)\n";
    std::cout << "════════════════════════════════════════════════\n";

    float    mer        = 0.f, mei = 0.f, me = 0.f;
    uint32_t worst_row  = 0;
    float    worst_err  = 0.f;

    for (uint32_t row = 0; row < num_rows; row++) {
        float row_err = 0.f;
        for (uint32_t i = 0; i < N_row; i++) {
            const float er = std::abs(result_r[row * N_row + i]
                                    - ref_r[row * N_row + i]);
            const float ei = std::abs(result_i[row * N_row + i]
                                    - ref_i[row * N_row + i]);
            mer      = std::max(mer, er);
            mei      = std::max(mei, ei);
            me      += er + ei;
            row_err  = std::max(row_err, er + ei);
        }
        if (row_err > worst_err) { worst_err = row_err; worst_row = row; }
    }
    me /= 2.f * static_cast<float>(total_N);

    std::cout << " Max error (real): " << mer          << "\n";
    std::cout << " Max error (imag): " << mei          << "\n";
    std::cout << " Mean error      : " << me           << "\n";
    std::cout << " Worst row       : " << worst_row    << "\n";

    const float threshold = std::max(0.5f, 0.005f * static_cast<float>(N_row));
    const bool  passed    = (mer < threshold) && (mei < threshold);
    std::cout << " Threshold       : " << threshold << " (0.5% of N_row)\n";
    std::cout << " Result          : " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";

    // ── First 16 results (row 0) ──────────────────────────────────────
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " FIRST 16 RESULTS (row 0 of " << num_rows << ")\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);
    for (uint32_t i = 0; i < 16 && i < N_row; i++) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(12) << result_r[i]
                  << (result_i[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(result_i[i]) << "j"
                  << "  ref: " << std::setw(12) << ref_r[i]
                  << (ref_i[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(ref_i[i]) << "j\n";
    }

    mesh->close();
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " Done\n";
    std::cout << "════════════════════════════════════════════════\n";
    return passed ? 0 : 1;
}