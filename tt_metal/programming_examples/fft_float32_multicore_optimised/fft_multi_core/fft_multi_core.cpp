// fft1d_wormhole.cpp — Host program
// Wormhole 1-D Cooley-Tukey FFT (decimation-in-time, radix-2)
// Multi-row, multi-core.  Each core handles a contiguous slice of rows.
// ═══════════════════════════════════════════════════════════════════════
//
//  Usage:
//    ./fft1d_wormhole <direction> [N] [num_rows] [num_cores] [input.txt]
//    direction: 0=forward, 1=inverse
//
//  Input file format (same as previous):
//    Interleaved complex:  re0 im0 re1 im1 ...  (2*N values)
//    Real-only:            re0 re1 re2 ...        (N values, imag=0)
//    Commas optional, one or more values per line.
//
//  CB layout (per core):
//    CB  0  even_r   depth=tiles_per_row   reader/writer → compute
//    CB  1  even_i   depth=tiles_per_row
//    CB  2  odd_r    depth=tiles_per_row
//    CB  3  odd_i    depth=tiles_per_row
//    CB  4  tw_r     depth=tiles_per_row   reader → compute
//    CB  5  tw_i     depth=tiles_per_row
//    CB  6  out_er   depth=tiles_per_row   compute → writer
//    CB  7  out_ei   depth=tiles_per_row
//    CB  8  out_or   depth=tiles_per_row
//    CB  9  out_oi   depth=tiles_per_row
//    CB 10-13        depth=1               compute scratch
//    CB 14-15        depth=1               reader L1 compact twiddle table
//    CB 16-19        depth=1               writer L1 shuffle scratch
//
//  DRAM buffers:
//    in_er, in_ei   bit-reversed even elements (real, imag)
//    in_or, in_oi   bit-reversed odd  elements (real, imag)
//    out_er, out_ei final even output
//    out_or, out_oi final odd  output
//    ctw_r, ctw_i   compact twiddle table (one tile, padded to TILE_SIZE)

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"

using namespace tt;
using namespace tt::tt_metal;

// ── Constants ─────────────────────────────────────────────────────────
constexpr float    PI         = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;   // 32
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;    // 32
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;              // 1024 floats
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);    // 4096 bytes

// ── Bit helpers ───────────────────────────────────────────────────────
static inline uint32_t f2u(float f)    { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float    u2f(uint32_t u) { float f;    memcpy(&f, &u, 4); return f; }

static uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

static uint32_t log2_exact(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) ++l;
    return l;
}

// ── CPU reference FFT ─────────────────────────────────────────────────
static void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const uint32_t N    = static_cast<uint32_t>(re.size());
    const uint32_t logN = log2_exact(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t j = bit_reverse(i, logN);
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (uint32_t s = 0; s < logN; ++s) {
        const uint32_t m  = 1u << (s + 1);
        const float    ab = (inv ? 2.f : -2.f) * PI / static_cast<float>(m);
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; ++j) {
                const float wr = cosf(ab * j), wi = sinf(ab * j);
                const uint32_t e = k + j, o = k + j + m / 2;
                const float tr = wr * re[o] - wi * im[o];
                const float ti = wr * im[o] + wi * re[o];
                re[o] = re[e] - tr;  im[o] = im[e] - ti;
                re[e] = re[e] + tr;  im[e] = im[e] + ti;
            }
        }
    }
    if (inv) {
        const float s = 1.f / N;
        for (auto& v : re) v *= s;
        for (auto& v : im) v *= s;
    }
}

// ── Input file reader ─────────────────────────────────────────────────
static bool read_input(const std::string& path, uint32_t N,
                       std::vector<float>& out_r, std::vector<float>& out_i) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }

    std::vector<float> vals;
    std::string tok;
    while (f >> tok) {
        if (!tok.empty() && tok.back() == ',') tok.pop_back();
        if (tok.empty()) continue;
        try { vals.push_back(std::stof(tok)); }
        catch (...) { std::cerr << "Bad token: '" << tok << "'\n"; return false; }
    }
    if (vals.empty()) { std::cerr << "Empty input file\n"; return false; }

    out_r.assign(N, 0.f);
    out_i.assign(N, 0.f);
    if (vals.size() >= 2 * N) {
        std::cout << " File mode: interleaved complex (" << vals.size()
                  << " values → " << N << " complex)\n";
        for (uint32_t i = 0; i < N; ++i) {
            out_r[i] = vals[2 * i];
            out_i[i] = vals[2 * i + 1];
        }
    } else {
        std::cout << " File mode: real-only (" << vals.size()
                  << " values → " << N << " points)\n";
        for (uint32_t i = 0; i < N && i < vals.size(); ++i)
            out_r[i] = vals[i];
    }
    return true;
}

// ── Twiddle precomputation ────────────────────────────────────────────
// Returns W_N^k = cos(-2πk/N) + j*sin(-2πk/N) for forward FFT,
// or W_N^{-k} for inverse.  k = 0..N/2-1.
static void precompute_twiddles(uint32_t N, bool inv,
                                std::vector<uint32_t>& tw_r,
                                std::vector<uint32_t>& tw_i) {
    const uint32_t half = N / 2;
    const float    sign = inv ? 1.f : -1.f;
    tw_r.resize(TILE_SIZE, f2u(0.f));
    tw_i.resize(TILE_SIZE, f2u(0.f));
    for (uint32_t k = 0; k < half && k < TILE_SIZE; ++k) {
        const float angle = sign * 2.f * PI * k / N;
        tw_r[k] = f2u(cosf(angle));
        tw_i[k] = f2u(sinf(angle));
    }
}

// ── Host-side bit-reversal + even/odd split ───────────────────────────
// Produces the stage-0 input buffers: bit-reversed permutation, then split
// into even-indexed and odd-indexed elements (which is what the first-stage
// butterfly needs).
//
// Standard DIT FFT: input[bit_reverse(i, log2N)] → split into even/odd.
// Even elements: x[0], x[2], x[4], ...  (bit_reversed indices 0,2,4,...)
// Odd  elements: x[1], x[3], x[5], ...  (bit_reversed indices 1,3,5,...)
//
// We store them in tiles of size TILE_SIZE, zero-padding if N/2 < TILE_SIZE.
struct StagedInput {
    std::vector<uint32_t> er, ei, or_, oi;  // even/odd real/imag, tile-packed
};

static StagedInput prepare_stage0(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t row_offset, uint32_t N, uint32_t log2N, uint32_t tiles_per_row)
{
    const uint32_t half = N / 2;
    std::vector<float> _er(half), _ei(half), _or(half), _oi(half);
    for (uint32_t i = 0; i < half; ++i) {
        // Bit-reversed indices
        const uint32_t e_idx = bit_reverse(2 * i,     log2N);
        const uint32_t o_idx = bit_reverse(2 * i + 1, log2N);
        _er[i] = sr[row_offset + e_idx];  _ei[i] = si[row_offset + e_idx];
        _or[i] = sr[row_offset + o_idx];  _oi[i] = si[row_offset + o_idx];
    }
    // Pack into tiles (zero-pad to TILE_SIZE boundary)
    const uint32_t n_elems = tiles_per_row * TILE_SIZE;
    StagedInput out;
    out.er.assign(n_elems, f2u(0.f));
    out.ei.assign(n_elems, f2u(0.f));
    out.or_.assign(n_elems, f2u(0.f));
    out.oi.assign(n_elems, f2u(0.f));
    for (uint32_t i = 0; i < half; ++i) {
        out.er[i]  = f2u(_er[i]);  out.ei[i]  = f2u(_ei[i]);
        out.or_[i] = f2u(_or[i]);  out.oi[i]  = f2u(_oi[i]);
    }
    return out;
}

// ── CB helper ─────────────────────────────────────────────────────────
static CBHandle make_cb(Program& prog, CoreCoord cc,
                        uint32_t id, uint32_t depth) {
    CircularBufferConfig cfg =
        CircularBufferConfig(depth * TILE_BYTES,
                             {{id, tt::DataFormat::Float32}})
            .set_page_size(id, TILE_BYTES);
    return CreateCircularBuffer(prog, cc, cfg);
}

// ── Core detection ────────────────────────────────────────────────────
static uint32_t pick_cores(IDevice* dev, uint32_t max_req, uint32_t num_rows) {
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    std::cout << " Device grid: " << grid.x << "×" << grid.y << " Tensix\n";
    uint32_t avail = 0;
    for (uint32_t x = 0; x < grid.x; ++x) {
        try { (void)dev->worker_core_from_logical_core({x, 0}); ++avail; }
        catch (...) { break; }
    }
    uint32_t cap = std::min(avail, max_req);
    uint32_t n   = 1;
    if (num_rows == 0)
        while (n * 2 <= cap) n *= 2;
    else
        while (n * 2 <= cap && num_rows % (n * 2) == 0) n *= 2;
    std::cout << " Using " << n << " core(s) (of " << avail << " available)\n";
    return n;
}

// ════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <0|1> [N] [rows] [cores] [input.txt]\n"
                  << "  0=forward  1=inverse\n"
                  << "  N must be a power of 2 (default 64)\n";
        return 1;
    }

    const bool   inv        = (std::atoi(argv[1]) == 1);
    uint32_t     N          = 64;
    uint32_t     num_rows   = 0;
    uint32_t     user_cores = 0;
    std::string  in_file;

    if (argc >= 3) N          = static_cast<uint32_t>(std::stoul(argv[2]));
    if (argc >= 4) num_rows   = static_cast<uint32_t>(std::stoul(argv[3]));
    if (argc >= 5) user_cores = static_cast<uint32_t>(std::stoul(argv[4]));
    if (argc >= 6) in_file    = argv[5];

    if (N < 2 || (N & (N - 1))) {
        std::cerr << "N must be a power of 2\n"; return 1; }
    if (argc >= 4 && num_rows == 0) {
        std::cerr << "num_rows must be >= 1\n"; return 1; }
    if (argc >= 5 && user_cores == 0) {
        std::cerr << "num_cores must be >= 1\n"; return 1; }

    // ── Device init ───────────────────────────────────────────────────
    auto mesh   = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto& cq    = mesh->mesh_command_queue();
    IDevice* dev = mesh->get_devices().at(0);

    const uint32_t max_req      = user_cores > 0 ? user_cores : 64u;
    const uint32_t num_cores    = pick_cores(dev, max_req, num_rows);
    if (num_rows == 0) num_rows = num_cores * 4;
    num_rows = (num_rows / num_cores) * num_cores;
    if (num_rows == 0) num_rows = num_cores;
    const uint32_t rows_per_core = num_rows / num_cores;

    const uint32_t log2N        = log2_exact(N);
    const uint32_t half_N       = N / 2;
    const uint32_t tiles_per_row = (half_N + TILE_SIZE - 1) / TILE_SIZE;
    const uint32_t tiles_per_core = tiles_per_row * rows_per_core;
    const uint32_t total_N       = N * num_rows;

    std::cout << "══════════════════════════════════════\n"
              << " Wormhole 1-D FFT\n"
              << "══════════════════════════════════════\n"
              << " N           = " << N          << "\n"
              << " num_rows    = " << num_rows   << "\n"
              << " num_cores   = " << num_cores  << "\n"
              << " rows/core   = " << rows_per_core << "\n"
              << " log2(N)     = " << log2N      << "\n"
              << " tiles/row   = " << tiles_per_row << "\n"
              << " direction   = " << (inv ? "inverse" : "forward") << "\n"
              << "══════════════════════════════════════\n";

    // ── Input data ────────────────────────────────────────────────────
    std::vector<float> in_r(total_N, 0.f), in_i(total_N, 0.f);

    if (!in_file.empty()) {
        std::cout << " Input: " << in_file << "\n";
        std::vector<float> row_r, row_i;
        if (!read_input(in_file, N, row_r, row_i)) { mesh->close(); return 1; }
        for (uint32_t row = 0; row < num_rows; ++row)
            for (uint32_t i = 0; i < N; ++i) {
                in_r[row * N + i] = row_r[i];
                in_i[row * N + i] = row_i[i];
            }
    } else {
        // Default: mix of two sine waves
        for (uint32_t row = 0; row < num_rows; ++row)
            for (uint32_t i = 0; i < N; ++i)
                in_r[row * N + i] =
                    sinf(2.f * PI * 4.f * i / N) +
                    0.5f * sinf(2.f * PI * 8.f * i / N);
    }

    // ── CPU reference ─────────────────────────────────────────────────
    std::vector<float> ref_r(in_r), ref_i(in_i);
    for (uint32_t row = 0; row < num_rows; ++row) {
        std::vector<float> rr(ref_r.begin() + row*N, ref_r.begin() + (row+1)*N);
        std::vector<float> ri(ref_i.begin() + row*N, ref_i.begin() + (row+1)*N);
        cpu_fft(rr, ri, inv);
        for (uint32_t i = 0; i < N; ++i) {
            ref_r[row*N+i] = rr[i]; ref_i[row*N+i] = ri[i];
        }
    }

    // ── Prepare stage-0 buffers (all rows) ────────────────────────────
    const uint32_t tiles_total   = tiles_per_core * num_cores;
    const uint32_t elems_total   = tiles_total * TILE_SIZE;

    std::vector<uint32_t> all_er(elems_total, f2u(0.f));
    std::vector<uint32_t> all_ei(elems_total, f2u(0.f));
    std::vector<uint32_t> all_or(elems_total, f2u(0.f));
    std::vector<uint32_t> all_oi(elems_total, f2u(0.f));

    for (uint32_t row = 0; row < num_rows; ++row) {
        StagedInput si = prepare_stage0(in_r, in_i, row * N, N, log2N, tiles_per_row);
        const uint32_t base = row * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < si.er.size(); ++i) {
            all_er[base+i] = si.er[i];  all_ei[base+i] = si.ei[i];
            all_or[base+i] = si.or_[i]; all_oi[base+i] = si.oi[i];
        }
    }

    // ── Compact twiddle table (one tile) ──────────────────────────────
    std::vector<uint32_t> ctw_r, ctw_i;
    precompute_twiddles(N, inv, ctw_r, ctw_i);

    // ── DRAM buffers ──────────────────────────────────────────────────
    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile_cfg{
        .page_size = TILE_BYTES, .buffer_type = BufferType::DRAM };

    const uint32_t buf_bytes = elems_total * sizeof(float);

    auto mk = [&](uint32_t bytes) {
        ReplicatedBufferConfig rc{.size = bytes};
        return MeshBuffer::create(rc, dram_tile_cfg, mesh.get());
    };

    auto b_er  = mk(buf_bytes);    // stage-0 even real
    auto b_ei  = mk(buf_bytes);    // stage-0 even imag
    auto b_or  = mk(buf_bytes);    // stage-0 odd  real
    auto b_oi  = mk(buf_bytes);    // stage-0 odd  imag
    auto b_oer = mk(buf_bytes);    // final output even real
    auto b_oei = mk(buf_bytes);    // final output even imag
    auto b_oor = mk(buf_bytes);    // final output odd  real
    auto b_ooi = mk(buf_bytes);    // final output odd  imag

    ReplicatedBufferConfig rc_ctw{.size = TILE_BYTES};
    DeviceLocalBufferConfig dram_ctw_cfg{
        .page_size = TILE_BYTES, .buffer_type = BufferType::DRAM };
    auto b_ctw_r = MeshBuffer::create(rc_ctw, dram_ctw_cfg, mesh.get());
    auto b_ctw_i = MeshBuffer::create(rc_ctw, dram_ctw_cfg, mesh.get());

    // ── Program + kernels ─────────────────────────────────────────────
    Program prog = CreateProgram();
    CoreRange core_range({0, 0}, {num_cores - 1, 0});

    constexpr const char* KPATH =
        "tt_metal/programming_examples/fft1d_wormhole/kernels/";

    KernelHandle reader_k = CreateKernel(
        prog,
        std::string(KPATH) + "dataflow/reader.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default });

    KernelHandle writer_k = CreateKernel(
        prog,
        std::string(KPATH) + "dataflow/writer.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default });

    KernelHandle compute_k = CreateKernel(
        prog,
        std::string(KPATH) + "compute/compute.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false });

    // ── CBs + runtime args (per core) ─────────────────────────────────
    for (uint32_t c = 0; c < num_cores; ++c) {
        CoreCoord cc = {c, 0};

        // Stage-0 / feedback input CBs (reader stage 0, writer stages 1+)
        make_cb(prog, cc,  0, tiles_per_row);  // even_r
        make_cb(prog, cc,  1, tiles_per_row);  // even_i
        make_cb(prog, cc,  2, tiles_per_row);  // odd_r
        make_cb(prog, cc,  3, tiles_per_row);  // odd_i
        // Twiddle CBs (reader → compute, all stages)
        make_cb(prog, cc,  4, tiles_per_row);  // tw_r
        make_cb(prog, cc,  5, tiles_per_row);  // tw_i
        // Butterfly output CBs (compute → writer)
        make_cb(prog, cc,  6, tiles_per_row);  // out_even_r
        make_cb(prog, cc,  7, tiles_per_row);  // out_even_i
        make_cb(prog, cc,  8, tiles_per_row);  // out_odd_r
        make_cb(prog, cc,  9, tiles_per_row);  // out_odd_i
        // Compute scratch (depth=1 — must never exceed 1 in-flight)
        make_cb(prog, cc, 10, 1);
        make_cb(prog, cc, 11, 1);
        make_cb(prog, cc, 12, 1);
        make_cb(prog, cc, 13, 1);
        // Reader compact twiddle table (depth=1, held for full kernel lifetime)
        make_cb(prog, cc, 14, 1);
        make_cb(prog, cc, 15, 1);
        // Writer L1 shuffle scratch (depth=1, plain memory — NEVER pushed/popped)
        make_cb(prog, cc, 16, 1);
        make_cb(prog, cc, 17, 1);
        make_cb(prog, cc, 18, 1);
        make_cb(prog, cc, 19, 1);

        const uint32_t tile_offset = c * tiles_per_core;

        // Reader args
        SetRuntimeArgs(prog, reader_k, cc, std::vector<uint32_t>{
            b_er->address(),     // [0]  even_r DRAM
            b_ei->address(),     // [1]  even_i DRAM
            b_or->address(),     // [2]  odd_r  DRAM
            b_oi->address(),     // [3]  odd_i  DRAM
            b_ctw_r->address(),  // [4]  compact twiddle real
            b_ctw_i->address(),  // [5]  compact twiddle imag
            tiles_per_row,       // [6]  tiles per row
            tile_offset,         // [7]  first tile index
            log2N,               // [8]  num_stages
            half_N,              // [9]  N/2
            rows_per_core,       // [10] rows this core handles
        });

        // Compute args
        SetRuntimeArgs(prog, compute_k, cc, std::vector<uint32_t>{
            log2N,               // [0] num_stages
            tiles_per_row,       // [1] tiles_per_stage
            rows_per_core,       // [2] rows_per_core
        });

        // Writer args
        SetRuntimeArgs(prog, writer_k, cc, std::vector<uint32_t>{
            b_oer->address(),    // [0]  output even real
            b_oei->address(),    // [1]  output even imag
            b_oor->address(),    // [2]  output odd  real
            b_ooi->address(),    // [3]  output odd  imag
            tiles_per_row,       // [4]  tiles per row
            log2N,               // [5]  num_stages
            half_N,              // [6]  N/2
            tile_offset,         // [7]  first output tile
            rows_per_core,       // [8]  rows this core handles
        });
    }

    // ── Build and dispatch workload ───────────────────────────────────
    MeshWorkload wl;
    MeshCoordinateRange rng = MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_er,    all_er, false);
    EnqueueWriteMeshBuffer(cq, b_ei,    all_ei, false);
    EnqueueWriteMeshBuffer(cq, b_or,    all_or, false);
    EnqueueWriteMeshBuffer(cq, b_oi,    all_oi, false);
    EnqueueWriteMeshBuffer(cq, b_ctw_r, ctw_r,  false);
    EnqueueWriteMeshBuffer(cq, b_ctw_i, ctw_i,  false);
    Finish(cq);

    std::cout << "Launching FFT on " << num_cores << " core(s), "
              << num_rows << " row(s) of " << N << " points...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

    // ── Read back results ─────────────────────────────────────────────
    std::vector<uint32_t> raw_oer(elems_total), raw_oei(elems_total);
    std::vector<uint32_t> raw_oor(elems_total), raw_ooi(elems_total);
    EnqueueReadMeshBuffer(cq, raw_oer, b_oer, true);
    EnqueueReadMeshBuffer(cq, raw_oei, b_oei, true);
    EnqueueReadMeshBuffer(cq, raw_oor, b_oor, true);
    EnqueueReadMeshBuffer(cq, raw_ooi, b_ooi, true);

    // ── Reconstruct full FFT output ───────────────────────────────────
    // Out layout: first half_N bins in out_even, next half_N in out_odd.
    std::vector<float> res_r(total_N), res_i(total_N);
    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t tile_base = row * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < half_N; ++i) {
            res_r[row*N + i]          = u2f(raw_oer[tile_base + i]);
            res_i[row*N + i]          = u2f(raw_oei[tile_base + i]);
            res_r[row*N + i + half_N] = u2f(raw_oor[tile_base + i]);
            res_i[row*N + i + half_N] = u2f(raw_ooi[tile_base + i]);
        }
    }

    // Inverse scale (host applies 1/N to match CPU reference)
    if (inv) {
        const float s = 1.f / N;
        for (auto& v : res_r) v *= s;
        for (auto& v : res_i) v *= s;
    }

    // ── Validation ────────────────────────────────────────────────────
    float max_err_r = 0.f, max_err_i = 0.f, mean_err = 0.f;
    uint32_t worst_row = 0;
    float    worst_val = 0.f;

    for (uint32_t row = 0; row < num_rows; ++row) {
        float row_worst = 0.f;
        for (uint32_t i = 0; i < N; ++i) {
            const float er = fabsf(res_r[row*N+i] - ref_r[row*N+i]);
            const float ei = fabsf(res_i[row*N+i] - ref_i[row*N+i]);
            max_err_r = std::max(max_err_r, er);
            max_err_i = std::max(max_err_i, ei);
            mean_err += er + ei;
            row_worst = std::max(row_worst, er + ei);
        }
        if (row_worst > worst_val) { worst_val = row_worst; worst_row = row; }
    }
    mean_err /= 2.f * total_N;

    const float threshold = std::max(0.5f, 0.005f * N);
    const bool  passed    = (max_err_r < threshold) && (max_err_i < threshold);

    std::cout << "\n══════════════════════════════════════\n"
              << " VALIDATION (" << num_rows << " row(s))\n"
              << "══════════════════════════════════════\n"
              << " Max error (real) : " << max_err_r  << "\n"
              << " Max error (imag) : " << max_err_i  << "\n"
              << " Mean error       : " << mean_err   << "\n"
              << " Worst row        : " << worst_row  << "\n"
              << " Threshold        : " << threshold  << "\n"
              << " Result           : " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";

    std::cout << "\n══════════════════════════════════════\n"
              << " FIRST 16 BINS (row 0)\n"
              << "══════════════════════════════════════\n"
              << std::fixed << std::setprecision(4);
    for (uint32_t i = 0; i < 16 && i < N; ++i) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(10) << res_r[i]
                  << (res_i[i] >= 0 ? " + " : " - ")
                  << std::setw(10) << fabsf(res_i[i]) << "j"
                  << "  ref: "
                  << std::setw(10) << ref_r[i]
                  << (ref_i[i] >= 0 ? " + " : " - ")
                  << std::setw(10) << fabsf(ref_i[i]) << "j\n";
    }

    mesh->close();
    std::cout << "\n══════════════════════════════════════\n Done\n";
    return passed ? 0 : 1;
}