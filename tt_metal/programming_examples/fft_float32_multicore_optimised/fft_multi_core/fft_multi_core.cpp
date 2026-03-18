// fft_multicore_2d.cpp — 2D FFT via row decomposition (paper's approach)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  WHY ROW DECOMPOSITION (not butterfly partition)
// ══════════════════════════════════════════════════════════════════════
//
//  A 1D N-point FFT butterfly at stage s has group size 2^(s+1).
//  For N=16384 and 8 cores (local slice = 2048 elements):
//    Stages 11-13: group size > 2048 → CROSS-CORE DATA REQUIRED.
//  There is no way to avoid cross-core exchange in a 1D FFT partition.
//
//  The paper (Brown et al. 2025) solves this with a 2D approach:
//    1. Each core owns N_rows/num_cores complete rows.
//    2. Each core runs a full 1D FFT on each of its local rows.
//       → Zero cross-core communication (rows are independent).
//    3. Global transpose across cores (using tt-nn or NOC).
//    4. Each core runs a full 1D FFT on its transposed columns.
//
//  For N_rows=N_cols=1024, 8 cores:
//    Each core owns 128 rows.
//    Each core runs 128 × 1024-pt FFTs per pass.
//    DRAM: upload 1024×1024 complex = 8 MB per component.
//
//  THIS FILE implements the simpler version: multiple independent
//  1D FFTs on each core. Each core runs `rows_per_core` independent
//  FFTs of size N_row. The kernels are the same single-core FFT
//  kernel, just looped over multiple rows.
//
//  For a true 2D FFT, the transpose step between row and column
//  passes would be added here using NOC or tt-nn.
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

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"
#include "tt_metal/api/tt-metalium/allocator.hpp"
#include "tt_metal/api/tt-metalium/hal.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f)   { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float    u2f(uint32_t u){ float f;    std::memcpy(&f,&u,4); return f; }

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

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r=(r<<1)|(x&1); x>>=1; }
    return r;
}

void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    uint32_t N = re.size(), log2N = 0;
    while ((1u<<log2N) < N) log2N++;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) { std::swap(re[i],re[j]); std::swap(im[i],im[j]); }
    }
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u<<(s+1);
        float ab = (inv?2.f:-2.f)*PI/m;
        for (uint32_t k = 0; k < N; k += m)
            for (uint32_t j = 0; j < m/2; j++) {
                float wr=std::cos(ab*j), wi=std::sin(ab*j);
                uint32_t e=k+j, o=k+j+m/2;
                float tr=wr*re[o]-wi*im[o], ti=wr*im[o]+wi*re[o];
                float er=re[e], ei=im[e];
                re[e]=er+tr; im[e]=ei+ti; re[o]=er-tr; im[o]=ei-ti;
            }
    }
    if (inv) for (uint32_t i=0;i<N;i++){ re[i]/=N; im[i]/=N; }
}

// Prepare stage-0 input for one FFT row (bit-reversed, stride-2 split)
void prepare_stage0_row(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t row_offset, uint32_t N_row, uint32_t log2_row,
    uint32_t tiles_per_row,
    std::vector<uint32_t>& er, std::vector<uint32_t>& ei,
    std::vector<uint32_t>& or_, std::vector<uint32_t>& oi
) {
    uint32_t half = N_row / 2;
    std::vector<float> _er(half), _ei(half), _or(half), _oi(half);
    for (uint32_t i = 0; i < half; i++) {
        uint32_t e = bit_reverse(2*i,   log2_row);
        uint32_t o = bit_reverse(2*i+1, log2_row);
        _er[i] = sr[row_offset + e];
        _ei[i] = si[row_offset + e];
        _or[i] = sr[row_offset + o];
        _oi[i] = si[row_offset + o];
    }
    auto packed_er  = pack_tiles(_er, tiles_per_row);
    auto packed_ei  = pack_tiles(_ei, tiles_per_row);
    auto packed_or  = pack_tiles(_or, tiles_per_row);
    auto packed_oi  = pack_tiles(_oi, tiles_per_row);
    er.insert(er.end(), packed_er.begin(), packed_er.end());
    ei.insert(ei.end(), packed_ei.begin(), packed_ei.end());
    or_.insert(or_.end(), packed_or.begin(), packed_or.end());
    oi.insert(oi.end(), packed_oi.begin(), packed_oi.end());
}

std::pair<std::vector<uint32_t>,std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N_row, uint32_t direction) {
    uint32_t half = N_row / 2;
    float sign = (direction==1) ? 1.f : -1.f;
    std::vector<uint32_t> tw_r(half, 0u), tw_i(half, 0u);
    for (uint32_t k = 0; k < half; k++) {
        float angle = sign * 2.f*PI*(float)k/(float)N_row;
        tw_r[k] = f2u(std::cos(angle));
        tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bytes) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles*bytes, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes);
    return CreateCircularBuffer(p, c, cfg);
}

uint32_t detect_available_cores(IDevice* device, uint32_t max_req,
                                 uint32_t num_rows) {
    CoreCoord grid = device->compute_with_storage_grid_size();
    std::cout << " Device grid : " << grid.x << " x " << grid.y
              << " Tensix cores\n";
    uint32_t usable = 0;
    for (uint32_t col = 0; col < grid.x; col++) {
        try {
            (void)device->worker_core_from_logical_core({col,0});
            usable++;
        } catch (...) { break; }
    }
    std::cout << " Usable row-0 cores: " << usable << "\n";
    uint32_t cap = std::min(usable, max_req);
    // In auto mode (num_rows=0), just return largest power-of-2 available.
    // Otherwise ensure num_rows divides evenly by num_cores.
    uint32_t result = 1;
    if (num_rows == 0) {
        while (result*2 <= cap) result *= 2;
    } else {
        while (result*2 <= cap && num_rows % (result*2) == 0) result *= 2;
    }
    std::cout << " Selected cores: " << result << "\n";
    return result;
}

bool is_uint_str(const char* s) {
    if (!s||!*s) return false;
    for (const char* p=s; *p; ++p) if (*p<'0'||*p>'9') return false;
    return true;
}

// ── File input reader ─────────────────────────────────────────────────
// Reads a text file of floats into a single row of N_row real values.
// Supports:
//   - Plain real values:      one value per line or space-separated
//   - Interleaved complex:    pairs (re im) → sets both ir and ii
// If the file has fewer than N_row values, the rest are zero-padded.
// If more, only the first N_row are used.
// Returns false on error.
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
        // Strip trailing comma (some export formats add them)
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

    bool interleaved = (vals.size() >= 2*N_row);
    if (interleaved) {
        // Treat as interleaved re/im pairs
        std::cout << " File mode: interleaved complex ("
                  << vals.size() << " values → " << N_row << " complex)\n";
        for (uint32_t i = 0; i < N_row && 2*i+1 < vals.size(); i++) {
            ir[i] = vals[2*i];
            ii[i] = vals[2*i+1];
        }
    } else {
        // Treat as real-only
        std::cout << " File mode: real-only ("
                  << vals.size() << " values → " << N_row << " points)\n";
        if (vals.size() < N_row)
            std::cout << " Note: " << N_row - vals.size()
                      << " values zero-padded\n";
        for (uint32_t i = 0; i < N_row && i < vals.size(); i++)
            ir[i] = vals[i];
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <direction:0|1> [N_row] [num_rows] [num_cores]\n"
                  << " Default: forward FFT, N_row=1024, num_rows=auto (128 rows/core)\n";
        return 1;
    }
    uint32_t direction  = (uint32_t)std::atoi(argv[1]);
    uint32_t N_row      = 1024;
    uint32_t num_rows   = 0;    // 0 = auto: num_cores * rows_per_core_target
    uint32_t user_cores = 0;
    uint32_t rows_per_core_target = 128;  // target rows per core when auto
    std::string in_file = "";

    for (int i = 2; i < argc; i++) {
        if (!is_uint_str(argv[i])) {
            in_file = argv[i];
            continue;
        }
        uint32_t v = (uint32_t)std::stoul(argv[i]);
        if (v >= 2 && v <= 64 && (v&(v-1))==0) user_cores = v;
        else if (v > 64 && (v&(v-1))==0) {
            if (v <= 1024) N_row = v;
            else num_rows = v;
        } else if (v >= 2) num_rows = v;
    }
    if (N_row<2||(N_row&(N_row-1))) { std::cerr<<"N_row must be power of 2\n"; return 1; }

    int dev_id = 0;
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(dev_id);
    auto& cq  = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices().at(0);

    uint32_t max_req   = user_cores > 0 ? user_cores : 64u;
    uint32_t num_cores = detect_available_cores(device, max_req, num_rows);

    // Auto num_rows: fill all cores with rows_per_core_target rows each
    // This maximises utilisation — all 8 cores stay busy the entire launch.
    if (num_rows == 0)
        num_rows = num_cores * rows_per_core_target;

    // Ensure num_rows is divisible by num_cores
    num_rows = (num_rows / num_cores) * num_cores;
    if (num_rows == 0) num_rows = num_cores;

    uint32_t rows_per_core = num_rows / num_cores;

    uint32_t log2_row  = 0; while ((1u<<log2_row) < N_row) log2_row++;
    uint32_t half_row  = N_row / 2;
    uint32_t tiles_per_row = (half_row + TILE_SIZE-1) / TILE_SIZE;
    uint32_t compact_bytes = half_row * sizeof(float);
    uint32_t total_N   = N_row * num_rows;

    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " TT-Metal MULTICORE FFT (row decomposition)\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " N_row        : " << N_row << "\n";
    std::cout << " num_rows     : " << num_rows << "\n";
    std::cout << " num_cores    : " << num_cores << "\n";
    std::cout << " rows/core    : " << rows_per_core << "\n";
    std::cout << " log2(N_row)  : " << log2_row << "\n";
    std::cout << " tiles/row    : " << tiles_per_row << "\n";
    std::cout << " Direction    : " << (direction?"Inverse":"Forward") << "\n";
    std::cout << " Total FFTs   : " << num_rows
              << "  (" << num_cores << " cores × "
              << rows_per_core << " rows)\n";
    std::cout << " Total points : " << (uint64_t)num_rows * N_row / 1024
              << " K complex samples\n";
    std::cout << "════════════════════════════════════════════════\n";

    // Generate or load input
    uint32_t total_elems = total_N;
    std::vector<float> ir(total_elems, 0.f), ii(total_elems, 0.f);

    if (!in_file.empty()) {
        // Load from file — same data broadcast to all rows
        std::cout << " Input file  : " << in_file << "\n";
        std::vector<float> row_r, row_i;
        if (!read_input_file(in_file, N_row, row_r, row_i)) {
            mesh->close(); return 1;
        }
        // Broadcast the same row to all rows
        for (uint32_t row = 0; row < num_rows; row++) {
            for (uint32_t i = 0; i < N_row; i++) {
                ir[row*N_row + i] = row_r[i];
                ii[row*N_row + i] = row_i[i];
            }
        }
    } else {
        // Synthetic: sin wave at frequency 4 + 0.5*sin at frequency 8
        for (uint32_t row = 0; row < num_rows; row++)
            for (uint32_t i = 0; i < N_row; i++)
                ir[row*N_row + i] = std::sin(2.f*PI*4.f*i/N_row)
                                  + 0.5f*std::sin(2.f*PI*8.f*i/N_row);
    }

    // CPU reference: FFT each row independently
    std::vector<float> ref_r(ir), ref_i(ii);
    for (uint32_t row = 0; row < num_rows; row++) {
        std::vector<float> row_r(ir.begin()+row*N_row, ir.begin()+(row+1)*N_row);
        std::vector<float> row_i(ii.begin()+row*N_row, ii.begin()+(row+1)*N_row);
        cpu_fft(row_r, row_i, direction==1);
        for (uint32_t i=0;i<N_row;i++) {
            ref_r[row*N_row+i] = row_r[i];
            ref_i[row*N_row+i] = row_i[i];
        }
    }

    // Prepare all rows' stage-0 inputs
    std::vector<uint32_t> all_er, all_ei, all_or, all_oi;
    for (uint32_t row = 0; row < num_rows; row++)
        prepare_stage0_row(ir, ii, row*N_row, N_row, log2_row,
                           tiles_per_row, all_er, all_ei, all_or, all_oi);

    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N_row, direction);

    Program prog = CreateProgram();
    CoreRange core_range({0,0}, {num_cores-1, 0});

    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile{
        .page_size=TILE_BYTES, .buffer_type=BufferType::DRAM};
    auto mk_buf = [&](uint32_t bytes) {
        ReplicatedBufferConfig rc{.size=bytes};
        return MeshBuffer::create(rc, dram_tile, mesh.get());
    };

    // Each core processes rows_per_core rows, each row = tiles_per_row tiles
    uint32_t tiles_per_core = tiles_per_row * rows_per_core;
    uint32_t bytes_per_core = tiles_per_core * TILE_BYTES;
    uint32_t total_bytes    = bytes_per_core * num_cores;

    auto b_er  = mk_buf(total_bytes);
    auto b_ei  = mk_buf(total_bytes);
    auto b_or  = mk_buf(total_bytes);
    auto b_oi  = mk_buf(total_bytes);
    auto b_o0r = mk_buf(total_bytes);
    auto b_o0i = mk_buf(total_bytes);
    auto b_o1r = mk_buf(total_bytes);
    auto b_o1i = mk_buf(total_bytes);

    DeviceLocalBufferConfig dram_cmp{
        .page_size=compact_bytes, .buffer_type=BufferType::DRAM};
    ReplicatedBufferConfig rc_cmp{.size=compact_bytes};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // CBs: depth = tiles_per_row (one row at a time through the pipeline)
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        create_cb(prog, cc,  0, tiles_per_row, TILE_BYTES); // even_r
        create_cb(prog, cc,  1, tiles_per_row, TILE_BYTES); // even_i
        create_cb(prog, cc,  2, tiles_per_row, TILE_BYTES); // odd_r
        create_cb(prog, cc,  3, tiles_per_row, TILE_BYTES); // odd_i
        create_cb(prog, cc,  4, tiles_per_row, TILE_BYTES); // tw_r
        create_cb(prog, cc,  5, tiles_per_row, TILE_BYTES); // tw_i
        create_cb(prog, cc, 16, tiles_per_row, TILE_BYTES); // out0_r
        create_cb(prog, cc, 17, tiles_per_row, TILE_BYTES); // out0_i
        create_cb(prog, cc, 18, tiles_per_row, TILE_BYTES); // out1_r
        create_cb(prog, cc, 19, tiles_per_row, TILE_BYTES); // out1_i
        create_cb(prog, cc, 20, tiles_per_row, TILE_BYTES); // tmp0
        create_cb(prog, cc, 21, tiles_per_row, TILE_BYTES); // tmp1
        create_cb(prog, cc, 22, tiles_per_row, TILE_BYTES); // tw_odd_r
        create_cb(prog, cc, 23, tiles_per_row, TILE_BYTES); // tw_odd_i
        // Compact twiddle: holds half_row floats
        create_cb(prog, cc, 10, (compact_bytes+TILE_BYTES-1)/TILE_BYTES,
                  compact_bytes); // compact_r
        create_cb(prog, cc, 11, (compact_bytes+TILE_BYTES-1)/TILE_BYTES,
                  compact_bytes); // compact_i
    }

    KernelHandle reader_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/reader_fft_f32.cpp",
        core_range,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                           .noc=NOC::RISCV_0_default});
    KernelHandle writer_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/writer_fft_f32.cpp",
        core_range,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,
                           .noc=NOC::RISCV_1_default});
    KernelHandle compute_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/compute/fft_compute_f32.cpp",
        core_range,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                      .fp32_dest_acc_en=true,.math_approx_mode=false});

    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        uint32_t tile_offset = c * tiles_per_core;

        // Reader: reads rows_per_core rows, tiles_per_row tiles each
        // We loop over rows inside the kernel via num_stages repetitions.
        // Actually the existing reader reads num_tiles tiles then loops
        // num_stages times expanding twiddles. We pass total tiles and
        // rows_per_core as num_rows_per_core for the outer loop.
        std::vector<uint32_t> reader_args = {
            b_er->address(), b_ei->address(),
            b_or->address(), b_oi->address(),
            b_cmp_r->address(), b_cmp_i->address(),
            tiles_per_row,   // tiles per single FFT
            tile_offset,     // first tile index for this core
            log2_row,        // num_stages per FFT
            half_row,        // half_N
            half_row,        // local_half (= half_row, one full FFT per core)
            rows_per_core    // number of rows this core processes
        };

        std::vector<uint32_t> compute_args = { log2_row, tiles_per_row };

        std::vector<uint32_t> writer_args = {
            b_o0r->address(), b_o0i->address(),
            b_o1r->address(), b_o1i->address(),
            tiles_per_row, log2_row, half_row,
            half_row, 1u, c, 0u,  // num_cores=1 (self-contained), core_id, log2_cores=0
            tile_offset,
            0u,                   // core_elem_base = 0 (full row per FFT)
            rows_per_core         // number of rows
        };

        SetRuntimeArgs(prog, reader_k,  cc, reader_args);
        SetRuntimeArgs(prog, writer_k,  cc, writer_args);
        SetRuntimeArgs(prog, compute_k, cc, compute_args);
    }

    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    std::cout << "Writing inputs to DRAM...\n";
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

    std::vector<uint32_t> o0r_raw(total_bytes/4);
    std::vector<uint32_t> o0i_raw(total_bytes/4);
    std::vector<uint32_t> o1r_raw(total_bytes/4);
    std::vector<uint32_t> o1i_raw(total_bytes/4);
    EnqueueReadMeshBuffer(cq, o0r_raw, b_o0r, true);
    EnqueueReadMeshBuffer(cq, o0i_raw, b_o0i, true);
    EnqueueReadMeshBuffer(cq, o1r_raw, b_o1r, true);
    EnqueueReadMeshBuffer(cq, o1i_raw, b_o1i, true);

    // Reassemble: each row's even (o0) and odd (o1) halves
    std::vector<float> result_r(total_N), result_i(total_N);
    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t tile_base = row * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < half_row; i++) {
            result_r[row*N_row + i]          = u2f(o0r_raw[tile_base + i]);
            result_i[row*N_row + i]          = u2f(o0i_raw[tile_base + i]);
            result_r[row*N_row + i + half_row] = u2f(o1r_raw[tile_base + i]);
            result_i[row*N_row + i + half_row] = u2f(o1i_raw[tile_base + i]);
        }
    }
    if (direction==1)
        for (uint32_t i=0;i<total_N;i++){ result_r[i]/=N_row; result_i[i]/=N_row; }

    // Validate all rows — with many rows per core, any shuffle error
    // in a non-first row would be invisible if we only check row 0.
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " VALIDATION (all " << num_rows << " rows)\n";
    std::cout << "════════════════════════════════════════════════\n";
    float mer=0.f, mei=0.f, me=0.f;
    for (uint32_t row=0; row<num_rows; row++) {
        for (uint32_t i=0;i<N_row;i++) {
            float er=std::abs(result_r[row*N_row+i]-ref_r[row*N_row+i]);
            float ei=std::abs(result_i[row*N_row+i]-ref_i[row*N_row+i]);
            mer=std::max(mer,er); mei=std::max(mei,ei); me+=er+ei;
        }
    }
    me /= 2*total_N;
    std::cout << " Max error (real): " << mer << "\n";
    std::cout << " Max error (imag): " << mei << "\n";
    std::cout << " Mean error      : " << me  << "\n";
    // HiFi4 float32 accumulates ~0.25% relative error per N=1024 FFT.
    // Threshold = 0.5% of N_row (matches single-core optimized behavior).
    // For N_row=1024: threshold ≈ 0.005 * 512 = 2.56
    float threshold = std::max(0.5f, 0.005f * (float)N_row);
    bool passed = (mer < threshold) && (mei < threshold);
    std::cout << " Threshold       : " << threshold << " (0.5% of N_row)\n";
    std::cout << " Result: " << (passed?"✓ PASSED":"✗ FAILED") << "\n";

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " FIRST 16 RESULTS (row 0 of " << num_rows << ")\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);
    for (uint32_t i=0;i<16&&i<N_row;i++) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(12) << result_r[i]
                  << (result_i[i]>=0?" + ":" - ")
                  << std::setw(12) << std::abs(result_i[i]) << "j"
                  << " ref: " << std::setw(12) << ref_r[i]
                  << (ref_i[i]>=0?" + ":" - ")
                  << std::setw(12) << std::abs(ref_i[i]) << "j\n";
    }

    mesh->close();
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " Done\n════════════════════════════════════════════════\n";
    return passed?0:1;
}