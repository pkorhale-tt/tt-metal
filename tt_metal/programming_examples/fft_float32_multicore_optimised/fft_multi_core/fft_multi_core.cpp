// fft_multicore_2d.cpp — 2D FFT host (FIXED v3)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  CHANGES vs v2
// ══════════════════════════════════════════════════════════════════════
//
//  No CB layout changes in the host — the host was already creating the
//  correct set of CBs.  Comments updated to document the fixed pipeline.
//
//  CB ownership summary (post-fix):
//  ──────────────────────────────────────────────────────────────────
//  CB  0-3  Stage-0 input even/odd r/i
//           Producer: reader kernel (RISCV_0)
//           Consumer: compute kernel (stage 0 only)
//
//  CB  4-5  Twiddle factors tw_r / tw_i
//           Producer: reader kernel (all stages)
//           Consumer: compute kernel (all stages)
//
//  CB  6-9  Next-stage input even/odd r/i
//           Producer: writer kernel (inter-stage shuffle, stages 0..N-2)
//           Consumer: compute kernel (stages 1..N-1)
//
//  CB 16-19 Butterfly outputs out0/out1 r/i
//           Producer: compute kernel (all stages)
//           Consumer: writer kernel (all stages)
//
//  CB 20-23 Compute scratch tmp0..tmp3 (depth=1 each)
//           Internal to compute kernel only
//
//  CB 10-11 Compact twiddle table (depth=1)
//           Producer: reader kernel (loaded once from DRAM)
//           Consumer: reader kernel (read pointer held for all stages)
//
//  KEY FIXES IN KERNELS (not host):
//  ──────────────────────────────────────────────────────────────────
//  reader_fft_f32_mc.cpp:
//    - Stage-0 even/odd now pushed into CB 0-3 (was incorrectly 6-9)
//
//  writer_fft_f32.cpp:
//    - Inter-stage shuffle now writes to CB 6-9 (was incorrectly 0-3)
//    - Variable rename: cb_even_r/i, cb_odd_r/i → cb_next_even_r/i,
//      cb_next_odd_r/i to make intent explicit
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
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f)    { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float    u2f(uint32_t u) { float f;    std::memcpy(&f, &u, 4); return f; }

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

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

void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const uint32_t N = re.size();
    uint32_t log2N  = 0;
    while ((1u << log2N) < N) log2N++;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
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
        std::vector<uint32_t> o(ntiles * TILE_SIZE, 0u);
        for (uint32_t i = 0; i < src.size() && i < o.size(); i++)
            o[i] = f2u(src[i]);
        dst.insert(dst.end(), o.begin(), o.end());
    };
    append(er,  _er, tiles_per_row);
    append(ei,  _ei, tiles_per_row);
    append(or_, _or, tiles_per_row);
    append(oi,  _oi, tiles_per_row);
}

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

// Compact twiddles are fetched in the reader with noc_async_read_tile().
// Store them in DRAM as one full tile page and zero-pad the unused tail.
static std::vector<uint32_t> padCompactTwiddlesToTile(
    const std::vector<uint32_t>& compactTwiddles,
    uint32_t validCount
) {
    std::vector<uint32_t> padded(TILE_SIZE, 0);
    const uint32_t count = std::min<uint32_t>(validCount, static_cast<uint32_t>(compactTwiddles.size()));
    for (uint32_t i = 0; i < count; ++i) {
        padded[i] = compactTwiddles[i];
    }
    return padded;
}

CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bytes_per_tile) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles * bytes_per_tile,
                             {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes_per_tile);
    return CreateCircularBuffer(p, c, cfg);
}

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
    uint32_t       N_row                = 4;
    uint32_t       num_rows             = 0;
    uint32_t       user_cores           = 0;
    const uint32_t rows_per_core_target = 4;
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
    const uint32_t total_N       = N_row * num_rows;

    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " TT-Metal MULTICORE FFT (row decomposition ok ok)\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " N_row        : " << N_row         << "\n";
    std::cout << " num_rows     : " << num_rows      << "\n";
    std::cout << " num_cores    : " << num_cores     << "\n";
    std::cout << " rows/core    : " << rows_per_core << "\n";
    std::cout << " log2(N_row)  : " << log2_row      << "\n";
    std::cout << " tiles/row    : " << tiles_per_row  << "\n";
    std::cout << " scratch CBs  : tmp0/1/2/3 [20-23] depth=1 each\n";
    std::cout << " Direction    : " << (direction ? "Inverse" : "Forward") << "\n";
    std::cout << " Total FFTs   : " << num_rows
              << "  (" << num_cores << " cores × " << rows_per_core << " rows)\n";
    std::cout << " Total points : "
              << (static_cast<uint64_t>(num_rows) * N_row / 1024)
              << " K complex samples\n";
    std::cout << "════════════════════════════════════════════════\n";

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
        for (uint32_t row = 0; row < num_rows; row++)
            for (uint32_t i = 0; i < N_row; i++)
                ir[row * N_row + i] =
                    std::sin(2.f * PI * 4.f * i / N_row)
                  + 0.5f * std::sin(2.f * PI * 8.f * i / N_row);
    }

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

    std::vector<uint32_t> all_er, all_ei, all_or, all_oi;
    all_er.reserve(num_rows * tiles_per_row * TILE_SIZE);
    all_ei.reserve(num_rows * tiles_per_row * TILE_SIZE);
    all_or.reserve(num_rows * tiles_per_row * TILE_SIZE);
    all_oi.reserve(num_rows * tiles_per_row * TILE_SIZE);
    for (uint32_t row = 0; row < num_rows; row++)
        prepare_stage0_row(ir, ii, row * N_row, N_row, log2_row,
                           tiles_per_row, all_er, all_ei, all_or, all_oi);

    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N_row, direction);
    auto cmp_r_tile = padCompactTwiddlesToTile(cmp_r_t, half_row);
    auto cmp_i_tile = padCompactTwiddlesToTile(cmp_i_t, half_row);

    Program   prog       = CreateProgram();
    CoreRange core_range({0, 0}, {num_cores - 1, 0});

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
        .page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig rc_cmp{.size = TILE_BYTES};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // ── CB creation ───────────────────────────────────────────────────
    //
    // CB  0-3  Stage-0 even/odd input (reader → compute)
    // CB  4-5  Twiddle factors        (reader → compute)
    // CB  6-9  Next-stage even/odd    (writer → compute, stages 1..N-1)
    // CB 16-19 Butterfly outputs      (compute → writer)
    // CB 20-23 Compute scratch        (internal, depth=1)
    // CB 10-11 Compact twiddle table  (reader internal, depth=1)
    //
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        // Stage-0 inputs: reader writes, compute reads (stage 0)
        create_cb(prog, cc,  0, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  1, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  2, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  3, tiles_per_row, TILE_BYTES);
        // Twiddle factors: reader writes, compute reads (all stages)
        create_cb(prog, cc,  4, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  5, tiles_per_row, TILE_BYTES);
        // Next-stage inputs: writer writes (shuffle), compute reads (stage 1+)
        create_cb(prog, cc,  6, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  7, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  8, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc,  9, tiles_per_row, TILE_BYTES);
        // Butterfly outputs: compute writes, writer reads
        create_cb(prog, cc, 16, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc, 17, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc, 18, tiles_per_row, TILE_BYTES);
        create_cb(prog, cc, 19, tiles_per_row, TILE_BYTES);
        // Compute scratch (depth=1): used internally by compute kernel
        create_cb(prog, cc, 20, 1, TILE_BYTES);
        create_cb(prog, cc, 21, 1, TILE_BYTES);
        create_cb(prog, cc, 22, 1, TILE_BYTES);
        create_cb(prog, cc, 23, 1, TILE_BYTES);
        // Compact twiddle table (depth=1): loaded once by reader
        create_cb(prog, cc, 10, 1, TILE_BYTES);
        create_cb(prog, cc, 11, 1, TILE_BYTES);
    }

    constexpr const char* KERNEL_PATH =
        "tt_metal/programming_examples/fft_float32_multicore_optimised/"
        "fft_multi_core/kernels/";

    KernelHandle reader_k = CreateKernel(
        prog,
        std::string(KERNEL_PATH) + "dataflow/reader_fft_f32.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default
        });

    KernelHandle writer_k = CreateKernel(
        prog,
        std::string(KERNEL_PATH) + "dataflow/writer_fft_f32.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default
        });

    KernelHandle compute_k = CreateKernel(
        prog,
        std::string(KERNEL_PATH) + "compute/fft_compute_f32.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false
        });

    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord       cc         = {c, 0};
        const uint32_t tile_offset = c * tiles_per_core;

        SetRuntimeArgs(prog, reader_k, cc, std::vector<uint32_t>{
            b_er->address(),       // [0]  even_r_addr
            b_ei->address(),       // [1]  even_i_addr
            b_or->address(),       // [2]  odd_r_addr
            b_oi->address(),       // [3]  odd_i_addr
            b_cmp_r->address(),    // [4]  compact_r_addr
            b_cmp_i->address(),    // [5]  compact_i_addr
            tiles_per_row,         // [6]  tiles_per_row
            tile_offset,           // [7]  tile_offset
            log2_row,              // [8]  num_stages
            half_row,              // [9]  half_N
            half_row,              // [10] local_half (ABI compat)
            rows_per_core,         // [11] rows_per_core
        });

        SetRuntimeArgs(prog, compute_k, cc, std::vector<uint32_t>{
            log2_row,              // [0] num_stages
            tiles_per_row,         // [1] tiles_per_stage
            rows_per_core,         // [2] rows_per_core
        });

        SetRuntimeArgs(prog, writer_k, cc, std::vector<uint32_t>{
            b_o0r->address(),      // [0]  out0_r_addr
            b_o0i->address(),      // [1]  out0_i_addr
            b_o1r->address(),      // [2]  out1_r_addr
            b_o1i->address(),      // [3]  out1_i_addr
            tiles_per_row,         // [4]  num_tiles (tiles_per_row)
            log2_row,              // [5]  num_stages
            half_row,              // [6]  half_N
            half_row,              // [7]  (padding)
            1u,                    // [8]  (padding)
            c,                     // [9]  (padding / core index, unused in kernel)
            0u,                    // [10] (padding)
            tile_offset,           // [11] tile_offset
            0u,                    // [12] (padding)
            rows_per_core,         // [13] rows_per_core
        });
    }

    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_er,    all_er, false);
    EnqueueWriteMeshBuffer(cq, b_ei,    all_ei, false);
    EnqueueWriteMeshBuffer(cq, b_or,    all_or, false);
    EnqueueWriteMeshBuffer(cq, b_oi,    all_oi, false);
    EnqueueWriteMeshBuffer(cq, b_cmp_r, cmp_r_tile, false);
    EnqueueWriteMeshBuffer(cq, b_cmp_i, cmp_i_tile, false);
    Finish(cq);

    std::cout << "Launching multicore FFT (" << num_cores << " cores, "
              << num_rows << " rows of " << N_row << " points)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

    std::vector<uint32_t> o0r_raw(total_bytes / 4);
    std::vector<uint32_t> o0i_raw(total_bytes / 4);
    std::vector<uint32_t> o1r_raw(total_bytes / 4);
    std::vector<uint32_t> o1i_raw(total_bytes / 4);
    EnqueueReadMeshBuffer(cq, o0r_raw, b_o0r, true);
    EnqueueReadMeshBuffer(cq, o0i_raw, b_o0i, true);
    EnqueueReadMeshBuffer(cq, o1r_raw, b_o1r, true);
    EnqueueReadMeshBuffer(cq, o1i_raw, b_o1i, true);

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

    if (direction == 1) {
        const float inv_N = 1.f / static_cast<float>(N_row);
        for (uint32_t i = 0; i < total_N; i++) {
            result_r[i] *= inv_N;
            result_i[i] *= inv_N;
        }
    }

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " VALIDATION (all " << num_rows << " rows)\n";
    std::cout << "════════════════════════════════════════════════\n";

    float    mer       = 0.f, mei = 0.f, me = 0.f;
    uint32_t worst_row = 0;
    float    worst_err = 0.f;

    for (uint32_t row = 0; row < num_rows; row++) {
        float row_err = 0.f;
        for (uint32_t i = 0; i < N_row; i++) {
            const float er = std::abs(result_r[row * N_row + i]
                                    - ref_r[row * N_row + i]);
            const float ei = std::abs(result_i[row * N_row + i]
                                    - ref_i[row * N_row + i]);
            mer     = std::max(mer, er);
            mei     = std::max(mei, ei);
            me     += er + ei;
            row_err = std::max(row_err, er + ei);
        }
        if (row_err > worst_err) { worst_err = row_err; worst_row = row; }
    }
    me /= 2.f * static_cast<float>(total_N);

    std::cout << " Max error (real): " << mer       << "\n";
    std::cout << " Max error (imag): " << mei       << "\n";
    std::cout << " Mean error      : " << me        << "\n";
    std::cout << " Worst row       : " << worst_row << "\n";

    const float threshold = std::max(0.5f, 0.005f * static_cast<float>(N_row));
    const bool  passed    = (mer < threshold) && (mei < threshold);
    std::cout << " Threshold       : " << threshold << " (0.5% of N_row)\n";
    std::cout << " Result          : " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";

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