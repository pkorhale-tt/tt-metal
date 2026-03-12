// fft_single_core_opt.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  WHAT CHANGED VS ORIGINAL HOST
// ═══════════════════════════════════════════════════════════════════
//
//  Original:
//    for stage in 0..log2N:
//      pack butterfly pairs on CPU    ← CPU work every stage
//      compute cos/sin on CPU          ← CPU work every stage
//      write to DRAM (PCIe)           ← PCIe write every stage
//      EnqueueProgram                 ← kernel launch every stage
//      read from DRAM (PCIe)          ← PCIe read every stage
//      unpack results on CPU          ← CPU work every stage
//    = log2(N) × (2 PCIe + CPU pack/unpack)
//    For N=1024: 10 × ~450μs = ~4500μs overhead
//
//  Optimized:
//    precompute ALL twiddles for ALL stages  ← once
//    bit-reverse input                       ← once
//    write input + twiddles to DRAM          ← once
//    EnqueueProgram                          ← ONCE (all stages on chip)
//    read result from DRAM                   ← once
//    = 2 PCIe transfers total
//    For N=1024: ~450μs total overhead  (~10× speedup)
//
//  CB index map (must match compute kernel exactly):
//    c_0/c_1   stage-0 even r/i          depth=2
//    c_2/c_3   stage-0 odd  r/i          depth=2
//    c_4/c_5   twiddle r/i               depth=2
//    c_6/c_7   pong odd  r/i             depth=1  ← inter-stage
//    c_10/c_11 ping even r/i             depth=1  ← inter-stage
//    c_12/c_13 ping odd  r/i             depth=1  ← inter-stage
//    c_14/c_15 pong even r/i             depth=1  ← inter-stage
//    c_16/c_17 output X[k]       r/i     depth=2
//    c_18/c_19 output X[k+N/2]   r/i     depth=2
//    c_20..c_24 scratch                  depth=1
//
// ═══════════════════════════════════════════════════════════════════

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;

// ── Tile constants ───────────────────────────────────────────────────
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;  // 32
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;   // 32
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;             // 1024
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);   // 4096

// ════════════════════════════════════════════════════════════════════
//  UTILITY: float ↔ uint32
// ════════════════════════════════════════════════════════════════════
inline uint32_t f2u(float f) {
    uint32_t u; std::memcpy(&u, &f, 4); return u;
}
inline float u2f(uint32_t u) {
    float f; std::memcpy(&f, &u, 4); return f;
}

// ════════════════════════════════════════════════════════════════════
//  UTILITY: pack floats into uint32 tile vector
// ════════════════════════════════════════════════════════════════════
std::vector<uint32_t> pack_tiles(
    const std::vector<float>& data,
    uint32_t num_tiles)
{
    std::vector<uint32_t> out(num_tiles * TILE_SIZE, 0);
    for (uint32_t i = 0; i < data.size() && i < out.size(); i++)
        out[i] = f2u(data[i]);
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  UTILITY: unpack uint32 tile vector to floats
// ════════════════════════════════════════════════════════════════════
std::vector<float> unpack_tiles(
    const std::vector<uint32_t>& data,
    uint32_t num_elements)
{
    std::vector<float> out(num_elements);
    for (uint32_t i = 0; i < num_elements && i < data.size(); i++)
        out[i] = u2f(data[i]);
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  BIT-REVERSAL PERMUTATION
// ════════════════════════════════════════════════════════════════════
uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

void bit_reverse_permutation(
    std::vector<float>& real,
    std::vector<float>& imag,
    uint32_t N)
{
    uint32_t log2n = 0;
    while ((1u << log2n) < N) log2n++;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2n);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  REFERENCE CPU FFT (validation only)
// ════════════════════════════════════════════════════════════════════
void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inv) {
    uint32_t N = real.size(), log2N = 0;
    while ((1u << log2N) < N) log2N++;
    bit_reverse_permutation(real, imag, N);

    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m   = 1u << (s + 1);
        float    ab  = (inv ? 2.f : -2.f) * PI / m;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float wr = std::cos(ab * j), wi = std::sin(ab * j);
                uint32_t e = k + j, o = k + j + m / 2;
                float tr = wr * real[o] - wi * imag[o];
                float ti = wr * imag[o] + wi * real[o];
                float er = real[e], ei = imag[e];
                real[e] = er + tr; imag[e] = ei + ti;
                real[o] = er - tr; imag[o] = ei - ti;
            }
        }
    }
    if (inv) {
        for (uint32_t i = 0; i < N; i++) {
            real[i] /= N; imag[i] /= N;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  PRECOMPUTE ALL TWIDDLE FACTORS FOR ALL STAGES
//
//  Layout: stage-major flat array.
//    index = stage * tiles_per_stage * TILE_SIZE + tile * TILE_SIZE + elem
//
//  Always stored as FORWARD twiddles: W_k = exp(-2πi·k/m) = (cos, -sin)
//  Device conjugates for IFFT by negating the imaginary part (sfpu_neg).
// ════════════════════════════════════════════════════════════════════
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_all_twiddles(uint32_t N, uint32_t log2N, uint32_t tiles_per_stage)
{
    const uint32_t total_tiles = log2N * tiles_per_stage;
    std::vector<uint32_t> tw_r(total_tiles * TILE_SIZE, 0);
    std::vector<uint32_t> tw_i(total_tiles * TILE_SIZE, 0);

    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m      = 1u << (stage + 1);
        uint32_t half_m = m / 2;

        uint32_t bf_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                float angle = -2.0f * PI * (float)j / (float)m;
                float wr    = std::cos(angle);
                float wi    = std::sin(angle);

                uint32_t flat = stage * tiles_per_stage * TILE_SIZE + bf_idx;
                tw_r[flat] = f2u(wr);
                tw_i[flat] = f2u(wi);

                bf_idx++;
            }
        }
    }

    return {tw_r, tw_i};
}

// ════════════════════════════════════════════════════════════════════
//  PREPARE STAGE 0 INPUT (even/odd split, bit-reversed)
//
//  After bit-reversal, butterfly pairs for stage 0 are:
//    first half  → even inputs  (indices 0 .. N/2-1)
//    second half → odd  inputs  (indices N/2 .. N-1)
//
//  Each is packed into separate tile arrays for separate DRAM buffers.
// ════════════════════════════════════════════════════════════════════
void prepare_stage0_input(
    const std::vector<float>& src_r,
    const std::vector<float>& src_i,
    uint32_t N,
    uint32_t tiles_per_stage,
    std::vector<uint32_t>& even_r_tiles,
    std::vector<uint32_t>& even_i_tiles,
    std::vector<uint32_t>& odd_r_tiles,
    std::vector<uint32_t>& odd_i_tiles)
{
    const uint32_t half_N = N / 2;
    std::vector<float> even_r(half_N), even_i(half_N);
    std::vector<float> odd_r(half_N),  odd_i(half_N);

    for (uint32_t i = 0; i < half_N; i++) {
        even_r[i] = src_r[i];
        even_i[i] = src_i[i];
        odd_r[i]  = src_r[i + half_N];
        odd_i[i]  = src_i[i + half_N];
    }

    even_r_tiles = pack_tiles(even_r, tiles_per_stage);
    even_i_tiles = pack_tiles(even_i, tiles_per_stage);
    odd_r_tiles  = pack_tiles(odd_r,  tiles_per_stage);
    odd_i_tiles  = pack_tiles(odd_i,  tiles_per_stage);
}

// ════════════════════════════════════════════════════════════════════
//  CREATE CIRCULAR BUFFER HELPER
// ════════════════════════════════════════════════════════════════════
void create_cb(
    Program& program,
    CoreCoord core,
    uint32_t cb_id,
    uint32_t num_tiles,
    uint32_t tile_bytes)
{
    CircularBufferConfig cfg =
        CircularBufferConfig(
            num_tiles * tile_bytes,
            {{cb_id, tt::DataFormat::Float32}})
        .set_page_size(cb_id, tile_bytes);
    CreateCircularBuffer(program, core, cfg);
}

// ════════════════════════════════════════════════════════════════════
//  FILE READER
// ════════════════════════════════════════════════════════════════════
bool read_input_file(
    const std::string& path,
    uint32_t& N,
    bool N_from_cmdline,
    std::vector<float>& input_r,
    std::vector<float>& input_i)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open input file: " << path << "\n";
        return false;
    }

    std::vector<float> vals;
    std::string token;
    while (f >> token) {
        if (!token.empty() && token.back() == ',')
            token.pop_back();
        if (token.empty()) continue;
        try {
            vals.push_back(std::stof(token));
        } catch (...) {
            std::cerr << "Bad token in file: '" << token << "'\n";
            return false;
        }
    }

    if (vals.empty()) {
        std::cerr << "Input file is empty.\n";
        return false;
    }

    uint32_t count = (uint32_t)vals.size();
    bool interleaved = false;

    if (N_from_cmdline) {
        if (count == 2 * N) {
            interleaved = true;
            std::cout << " File mode     : interleaved real+imag (" << count << " values → N=" << N << ")\n";
        } else if (count == N) {
            std::cout << " File mode     : real-only (" << count << " values, imag=0)\n";
        } else if (count < N) {
            std::cerr << "File has " << count << " values but N=" << N << " — padding with zeros.\n";
        } else {
            std::cerr << "File has " << count << " values but N=" << N << " — truncating to N.\n";
            count = N;
            vals.resize(N);
        }
    } else {
        N = 1;
        while (N < count) N <<= 1;
        std::cout << " File mode     : real-only (" << count << " values, N inferred as " << N << ")\n";
    }

    input_r.assign(N, 0.f);
    input_i.assign(N, 0.f);

    if (interleaved) {
        for (uint32_t i = 0; i < N && 2*i+1 < (uint32_t)vals.size(); i++) {
            input_r[i] = vals[2*i];
            input_i[i] = vals[2*i+1];
        }
    } else {
        for (uint32_t i = 0; i < N && i < (uint32_t)vals.size(); i++) {
            input_r[i] = vals[i];
        }
    }

    return true;
}

// ════════════════════════════════════════════════════════════════════
//  MAIN
//
//  Usage:
//    ./fft_single_core <direction> [input_file] [N]
//
//    direction : 0=forward FFT, 1=inverse FFT
//    input_file: optional path to text file of floats
//    N         : optional FFT size (power of 2)
// ════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {

    uint32_t    direction      = 0;
    uint32_t    N              = 1024;
    std::string in_file        = "";
    bool        N_from_cmdline = false;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <direction 0|1> [input_file] [N]\n";
        return 1;
    }

    direction = (uint32_t)std::atoi(argv[1]);

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        bool looks_like_file = (a.find('.') != std::string::npos ||
                                a.find('/') != std::string::npos ||
                                a.find('\\') != std::string::npos);
        if (looks_like_file && in_file.empty()) {
            in_file = a;
        } else {
            try {
                N = (uint32_t)std::stol(a);
                N_from_cmdline = true;
            } catch (...) {
                if (in_file.empty()) in_file = a;
            }
        }
    }

    if (N_from_cmdline && (N == 0 || (N & (N-1)) != 0)) {
        std::cerr << "N must be a power of 2, got " << N << "\n";
        return 1;
    }

    uint32_t log2N           = 0;
    while ((1u << log2N) < N) log2N++;
    uint32_t half_N          = N / 2;
    uint32_t tiles_per_stage = (half_N + TILE_SIZE - 1) / TILE_SIZE;

    // ── Input signal ──────────────────────────────────────────────
    std::vector<float> input_r(N, 0.f), input_i(N, 0.f);
    if (!in_file.empty()) {
        if (!read_input_file(in_file, N, N_from_cmdline, input_r, input_i))
            return 1;
        log2N           = 0;
        while ((1u << log2N) < N) log2N++;
        half_N          = N / 2;
        tiles_per_stage = (half_N + TILE_SIZE - 1) / TILE_SIZE;
        input_r.resize(N, 0.f);
        input_i.resize(N, 0.f);
        if (N < 2 || (N & (N-1)) != 0) {
            std::cerr << "Inferred N=" << N << " is not a valid power-of-2 >= 2\n";
            return 1;
        }
    } else {
        for (uint32_t i = 0; i < N; i++) {
            input_r[i] = std::sin(2.f * PI * 4.f * i / N)
                       + 0.5f * std::sin(2.f * PI * 8.f * i / N);
        }
    }

    std::cout << "═══════════════════════════════════════\n";
    std::cout << " TT-Metal FFT (Optimised Single Core)\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " N             : " << N          << "\n";
    std::cout << " log2N         : " << log2N      << "\n";
    std::cout << " Direction     : " << (direction ? "Inverse" : "Forward") << "\n";
    std::cout << " tiles/stage   : " << tiles_per_stage << "\n";
    std::cout << " total twiddle : " << log2N * tiles_per_stage << " tiles\n";
    std::cout << "═══════════════════════════════════════\n";

    // ── Reference FFT for validation ──────────────────────────────
    std::vector<float> ref_r(input_r), ref_i(input_i);
    cpu_fft(ref_r, ref_i, direction == 1);

    // ── Bit-reverse input (once on host) ──────────────────────────
    bit_reverse_permutation(input_r, input_i, N);

    // ── Precompute ALL twiddles for ALL stages (once) ─────────────
    auto [tw_r_tiles, tw_i_tiles] =
        precompute_all_twiddles(N, log2N, tiles_per_stage);

    // ── Prepare stage 0 even/odd split ────────────────────────────
    std::vector<uint32_t> even_r_t, even_i_t, odd_r_t, odd_i_t;
    prepare_stage0_input(
        input_r, input_i, N, tiles_per_stage,
        even_r_t, even_i_t, odd_r_t, odd_i_t);

    // ── Device setup ──────────────────────────────────────────────
    int device_id = 0;
    auto mesh_device =
        tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core  = {0, 0};

    // ── DRAM buffer sizes ─────────────────────────────────────────
    uint32_t input_buf_bytes   = tiles_per_stage * TILE_BYTES;
    uint32_t twiddle_buf_bytes = log2N * tiles_per_stage * TILE_BYTES;
    uint32_t output_buf_bytes  = tiles_per_stage * TILE_BYTES;

    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_cfg{
        .page_size   = TILE_BYTES,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    auto mk_buf = [&](uint32_t bytes) {
        tt::tt_metal::distributed::ReplicatedBufferConfig rcfg{ .size = bytes };
        return tt::tt_metal::distributed::MeshBuffer::create(
            rcfg, dram_cfg, mesh_device.get());
    };

    auto buf_even_r = mk_buf(input_buf_bytes);
    auto buf_even_i = mk_buf(input_buf_bytes);
    auto buf_odd_r  = mk_buf(input_buf_bytes);
    auto buf_odd_i  = mk_buf(input_buf_bytes);
    auto buf_tw_r   = mk_buf(twiddle_buf_bytes);
    auto buf_tw_i   = mk_buf(twiddle_buf_bytes);
    auto buf_out0_r = mk_buf(output_buf_bytes);
    auto buf_out0_i = mk_buf(output_buf_bytes);
    auto buf_out1_r = mk_buf(output_buf_bytes);
    auto buf_out1_i = mk_buf(output_buf_bytes);

    // ── Circular buffers ──────────────────────────────────────────
    //
    //  IO CBs: depth=2 (double-buffered)
    //  Inter-stage ping/pong: depth=tiles_per_stage
    //    Each side holds one full stage worth of data in L1.
    //    Ping and pong alternate so no DRAM access between stages.
    //    Even and odd are SEPARATE CBs — butterfly needs both
    //    simultaneously without one blocking the other.
    //  Intermediates: depth=1 (transient, immediately consumed)

    constexpr uint32_t IO_DEPTH  = 2;
    constexpr uint32_t TMP_DEPTH = 1;

    // Stage-0 input CBs (depth=2, double-buffered from DRAM)
    create_cb(program, core, tt::CBIndex::c_0, IO_DEPTH, TILE_BYTES);  // even_r
    create_cb(program, core, tt::CBIndex::c_1, IO_DEPTH, TILE_BYTES);  // even_i
    create_cb(program, core, tt::CBIndex::c_2, IO_DEPTH, TILE_BYTES);  // odd_r
    create_cb(program, core, tt::CBIndex::c_3, IO_DEPTH, TILE_BYTES);  // odd_i

    // Twiddle CBs (depth=2, double-buffered from DRAM)
    create_cb(program, core, tt::CBIndex::c_4, IO_DEPTH, TILE_BYTES);  // tw_r
    create_cb(program, core, tt::CBIndex::c_5, IO_DEPTH, TILE_BYTES);  // tw_i

    // Pong odd CBs (c_6/c_7 — safe, not used by reader or writer)
    create_cb(program, core, tt::CBIndex::c_6,  tiles_per_stage, TILE_BYTES);  // pong_odd_r
    create_cb(program, core, tt::CBIndex::c_7,  tiles_per_stage, TILE_BYTES);  // pong_odd_i

    // Ping even/odd CBs
    create_cb(program, core, tt::CBIndex::c_10, tiles_per_stage, TILE_BYTES);  // ping_even_r
    create_cb(program, core, tt::CBIndex::c_11, tiles_per_stage, TILE_BYTES);  // ping_even_i
    create_cb(program, core, tt::CBIndex::c_12, tiles_per_stage, TILE_BYTES);  // ping_odd_r
    create_cb(program, core, tt::CBIndex::c_13, tiles_per_stage, TILE_BYTES);  // ping_odd_i

    // Pong even CBs
    create_cb(program, core, tt::CBIndex::c_14, tiles_per_stage, TILE_BYTES);  // pong_even_r
    create_cb(program, core, tt::CBIndex::c_15, tiles_per_stage, TILE_BYTES);  // pong_even_i

    // Output CBs (depth=2, double-buffered to DRAM)
    create_cb(program, core, tt::CBIndex::c_16, IO_DEPTH, TILE_BYTES);  // out0_r  X[k]
    create_cb(program, core, tt::CBIndex::c_17, IO_DEPTH, TILE_BYTES);  // out0_i
    create_cb(program, core, tt::CBIndex::c_18, IO_DEPTH, TILE_BYTES);  // out1_r  X[k+N/2]
    create_cb(program, core, tt::CBIndex::c_19, IO_DEPTH, TILE_BYTES);  // out1_i

    // Scratch intermediates (depth=1)
    create_cb(program, core, tt::CBIndex::c_20, TMP_DEPTH, TILE_BYTES);  // tmp0
    create_cb(program, core, tt::CBIndex::c_21, TMP_DEPTH, TILE_BYTES);  // tmp1
    create_cb(program, core, tt::CBIndex::c_22, TMP_DEPTH, TILE_BYTES);  // tw_odd_r
    create_cb(program, core, tt::CBIndex::c_23, TMP_DEPTH, TILE_BYTES);  // tw_odd_i
    create_cb(program, core, tt::CBIndex::c_24, TMP_DEPTH, TILE_BYTES);  // neg_tw_i

    // ── Kernels ───────────────────────────────────────────────────
    auto reader_k = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default
        });

    auto writer_k = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default
        });

    // UnpackToDestFp32 for all CBs — full FP32 precision throughout.
    // Without this, Tf32 truncation (10-bit mantissa) accumulates
    // across log2(N) stages — significant error for large N.
    std::vector<UnpackToDestMode> unpack_modes(32, UnpackToDestMode::UnpackToDestFp32);

    auto compute_k = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode    = false
        });

    // ── Runtime args ──────────────────────────────────────────────
    std::vector<uint32_t> reader_args = {
        buf_even_r->address(),
        buf_even_i->address(),
        buf_odd_r->address(),
        buf_odd_i->address(),
        buf_tw_r->address(),
        buf_tw_i->address(),
        tiles_per_stage,
        log2N
    };

    std::vector<uint32_t> writer_args = {
        buf_out0_r->address(),
        buf_out0_i->address(),
        buf_out1_r->address(),
        buf_out1_i->address(),
        tiles_per_stage
    };

    std::vector<uint32_t> compute_args = {
        direction,
        log2N,
        tiles_per_stage
    };

    // ── Single workload ───────────────────────────────────────────
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    auto& prog = workload.get_programs().begin()->second;
    SetRuntimeArgs(prog, reader_k,  core, reader_args);
    SetRuntimeArgs(prog, writer_k,  core, writer_args);
    SetRuntimeArgs(prog, compute_k, core, compute_args);

    // ── Write inputs to DRAM (ONCE) ───────────────────────────────
    std::cout << "Writing inputs to DRAM...\n";
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_even_r, even_r_t,   false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_even_i, even_i_t,   false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_odd_r,  odd_r_t,    false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_odd_i,  odd_i_t,    false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_tw_r,   tw_r_tiles, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf_tw_i,   tw_i_tiles, false);
    tt::tt_metal::distributed::Finish(cq);

    // ── Launch kernel ONCE (all stages on device) ─────────────────
    std::cout << "Launching FFT kernel (all " << log2N << " stages on device)...\n";
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);
    std::cout << "Kernel complete.\n";

    // ── Read results (ONCE) ───────────────────────────────────────
    std::vector<uint32_t> out0_r_raw(tiles_per_stage * TILE_SIZE);
    std::vector<uint32_t> out0_i_raw(tiles_per_stage * TILE_SIZE);
    std::vector<uint32_t> out1_r_raw(tiles_per_stage * TILE_SIZE);
    std::vector<uint32_t> out1_i_raw(tiles_per_stage * TILE_SIZE);

    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_r_raw, buf_out0_r, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_i_raw, buf_out0_i, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_r_raw, buf_out1_r, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_i_raw, buf_out1_i, true);

    // ── Reconstruct result in natural order ───────────────────────
    // out0 = X[0..N/2-1], out1 = X[N/2..N-1]
    auto out0_r = unpack_tiles(out0_r_raw, half_N);
    auto out0_i = unpack_tiles(out0_i_raw, half_N);
    auto out1_r = unpack_tiles(out1_r_raw, half_N);
    auto out1_i = unpack_tiles(out1_i_raw, half_N);

    std::vector<float> result_r(N), result_i(N);
    for (uint32_t i = 0; i < half_N; i++) {
        result_r[i]          = out0_r[i];
        result_i[i]          = out0_i[i];
        result_r[i + half_N] = out1_r[i];
        result_i[i + half_N] = out1_i[i];
    }

    // Apply 1/N scaling for IFFT
    if (direction == 1) {
        for (uint32_t i = 0; i < N; i++) {
            result_r[i] /= N;
            result_i[i] /= N;
        }
    }

    // ── Validation ────────────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << " VALIDATION\n";
    std::cout << "═══════════════════════════════════════\n";

    float max_err_r = 0.f, max_err_i = 0.f, mean_err = 0.f;
    for (uint32_t i = 0; i < N; i++) {
        float er = std::abs(result_r[i] - ref_r[i]);
        float ei = std::abs(result_i[i] - ref_i[i]);
        max_err_r = std::max(max_err_r, er);
        max_err_i = std::max(max_err_i, ei);
        mean_err += er + ei;
    }
    mean_err /= 2 * N;

    std::cout << " Max error  (real): " << max_err_r << "\n";
    std::cout << " Max error  (imag): " << max_err_i << "\n";
    std::cout << " Mean error       : " << mean_err  << "\n";

    bool passed = (max_err_r < 1e-3f) && (max_err_i < 1e-3f);
    std::cout << " Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";

    // ── Print first 16 outputs ────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << " FIRST 16 RESULTS\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);
    for (uint32_t i = 0; i < 16 && i < N; i++) {
        std::cout
            << " X[" << std::setw(3) << i << "] = "
            << std::setw(12) << result_r[i]
            << (result_i[i] >= 0 ? " + " : " - ")
            << std::setw(12) << std::abs(result_i[i]) << "j"
            << "   ref: "
            << std::setw(12) << ref_r[i]
            << (ref_i[i] >= 0 ? " + " : " - ")
            << std::setw(12) << std::abs(ref_i[i]) << "j"
            << "\n";
    }

    mesh_device->close();
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << " Done\n";
    std::cout << "═══════════════════════════════════════\n";

    return passed ? 0 : 1;
}