// fft_single_core_opt.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ═══════════════════════════════════════════════════════════════════
//  HOST — uses standard single-device TT-Metal API
//
//  Uses CreateDevice / Buffer / EnqueueProgram instead of the
//  distributed MeshDevice / MeshBuffer / MeshWorkload API.
//  This is the correct API for a single-core single-device kernel.
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

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;

constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float    u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

std::vector<uint32_t> pack_tiles(const std::vector<float>& data, uint32_t num_tiles) {
    std::vector<uint32_t> out(num_tiles * TILE_SIZE, 0);
    for (uint32_t i = 0; i < data.size() && i < out.size(); i++)
        out[i] = f2u(data[i]);
    return out;
}

std::vector<float> unpack_tiles(const std::vector<uint32_t>& data, uint32_t num_elements) {
    std::vector<float> out(num_elements);
    for (uint32_t i = 0; i < num_elements && i < data.size(); i++)
        out[i] = u2f(data[i]);
    return out;
}

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

void bit_reverse_permutation(std::vector<float>& real, std::vector<float>& imag, uint32_t N) {
    uint32_t log2n = 0;
    while ((1u << log2n) < N) log2n++;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2n);
        if (i < j) { std::swap(real[i], real[j]); std::swap(imag[i], imag[j]); }
    }
}

void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inv) {
    uint32_t N = real.size(), log2N = 0;
    while ((1u << log2N) < N) log2N++;
    bit_reverse_permutation(real, imag, N);
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u << (s + 1);
        float ab = (inv ? 2.f : -2.f) * PI / m;
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
    if (inv) for (uint32_t i = 0; i < N; i++) { real[i] /= N; imag[i] /= N; }
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_all_twiddles(uint32_t N, uint32_t log2N, uint32_t tiles_per_stage) {
    const uint32_t total_tiles = log2N * tiles_per_stage;
    std::vector<uint32_t> tw_r(total_tiles * TILE_SIZE, 0);
    std::vector<uint32_t> tw_i(total_tiles * TILE_SIZE, 0);
    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t bf_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float angle = -2.0f * PI * (float)j / (float)m;
                uint32_t flat = stage * tiles_per_stage * TILE_SIZE + bf_idx;
                tw_r[flat] = f2u(std::cos(angle));
                tw_i[flat] = f2u(std::sin(angle));
                bf_idx++;
            }
        }
    }
    return {tw_r, tw_i};
}

void prepare_stage0_input(
    const std::vector<float>& src_r, const std::vector<float>& src_i,
    uint32_t N, uint32_t tiles_per_stage,
    std::vector<uint32_t>& even_r_t, std::vector<uint32_t>& even_i_t,
    std::vector<uint32_t>& odd_r_t,  std::vector<uint32_t>& odd_i_t)
{
    const uint32_t half_N = N / 2;
    std::vector<float> er(half_N), ei(half_N), or_(half_N), oi(half_N);
    for (uint32_t i = 0; i < half_N; i++) {
        er[i] = src_r[i];     ei[i] = src_i[i];
        or_[i] = src_r[i + half_N]; oi[i] = src_i[i + half_N];
    }
    even_r_t = pack_tiles(er,  tiles_per_stage);
    even_i_t = pack_tiles(ei,  tiles_per_stage);
    odd_r_t  = pack_tiles(or_, tiles_per_stage);
    odd_i_t  = pack_tiles(oi,  tiles_per_stage);
}

// ── Create a DRAM buffer using the standard single-device API ──────
std::shared_ptr<Buffer> make_dram_buffer(Device* device, uint32_t bytes) {
    InterleavedBufferConfig cfg{
        .device      = device,
        .size        = bytes,
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::DRAM
    };
    return CreateBuffer(cfg);
}

// ── Create a CB ────────────────────────────────────────────────────
void create_cb(Program& program, CoreCoord core, uint32_t cb_id,
               uint32_t num_tiles, uint32_t tile_bytes)
{
    CircularBufferConfig cfg =
        CircularBufferConfig(num_tiles * tile_bytes, {{cb_id, tt::DataFormat::Float32}})
        .set_page_size(cb_id, tile_bytes);
    CreateCircularBuffer(program, core, cfg);
}

bool read_input_file(const std::string& path, uint32_t& N, bool N_from_cmdline,
                     std::vector<float>& input_r, std::vector<float>& input_i)
{
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << "\n"; return false; }
    std::vector<float> vals;
    std::string token;
    while (f >> token) {
        if (!token.empty() && token.back() == ',') token.pop_back();
        if (token.empty()) continue;
        try { vals.push_back(std::stof(token)); }
        catch (...) { std::cerr << "Bad token: '" << token << "'\n"; return false; }
    }
    if (vals.empty()) { std::cerr << "Empty file\n"; return false; }
    uint32_t count = (uint32_t)vals.size();
    bool interleaved = false;
    if (N_from_cmdline) {
        if (count == 2 * N) interleaved = true;
        else if (count > N) { vals.resize(N); count = N; }
    } else {
        N = 1; while (N < count) N <<= 1;
    }
    input_r.assign(N, 0.f); input_i.assign(N, 0.f);
    if (interleaved) {
        for (uint32_t i = 0; i < N && 2*i+1 < (uint32_t)vals.size(); i++) {
            input_r[i] = vals[2*i]; input_i[i] = vals[2*i+1];
        }
    } else {
        for (uint32_t i = 0; i < N && i < count; i++) input_r[i] = vals[i];
    }
    return true;
}

int main(int argc, char** argv) {

    uint32_t direction = 0, N = 1024;
    std::string in_file;
    bool N_from_cmdline = false;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <direction 0|1> [input_file] [N]\n";
        return 1;
    }
    direction = (uint32_t)std::atoi(argv[1]);
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        bool looks_like_file = (a.find('.') != std::string::npos ||
                                a.find('/') != std::string::npos);
        if (looks_like_file && in_file.empty()) { in_file = a; }
        else { try { N = (uint32_t)std::stol(a); N_from_cmdline = true; } catch (...) {} }
    }
    if (N_from_cmdline && (N == 0 || (N & (N-1)) != 0)) {
        std::cerr << "N must be power of 2\n"; return 1;
    }

    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;
    uint32_t half_N         = N / 2;
    uint32_t tiles_per_stage = (half_N + TILE_SIZE - 1) / TILE_SIZE;

    std::vector<float> input_r(N, 0.f), input_i(N, 0.f);
    if (!in_file.empty()) {
        if (!read_input_file(in_file, N, N_from_cmdline, input_r, input_i)) return 1;
        log2N = 0; while ((1u << log2N) < N) log2N++;
        half_N = N / 2;
        tiles_per_stage = (half_N + TILE_SIZE - 1) / TILE_SIZE;
        input_r.resize(N, 0.f); input_i.resize(N, 0.f);
    } else {
        for (uint32_t i = 0; i < N; i++)
            input_r[i] = std::sin(2.f * PI * 4.f * i / N)
                       + 0.5f * std::sin(2.f * PI * 8.f * i / N);
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

    // Reference FFT
    std::vector<float> ref_r(input_r), ref_i(input_i);
    cpu_fft(ref_r, ref_i, direction == 1);

    // Prepare inputs
    bit_reverse_permutation(input_r, input_i, N);
    auto [tw_r_tiles, tw_i_tiles] = precompute_all_twiddles(N, log2N, tiles_per_stage);
    std::vector<uint32_t> even_r_t, even_i_t, odd_r_t, odd_i_t;
    prepare_stage0_input(input_r, input_i, N, tiles_per_stage,
                         even_r_t, even_i_t, odd_r_t, odd_i_t);

    // ── Device setup — standard single-device API ──────────────────
    Device* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program  = CreateProgram();
    CoreCoord core   = {0, 0};

    // ── DRAM buffers ───────────────────────────────────────────────
    uint32_t input_buf_bytes   = tiles_per_stage * TILE_BYTES;
    uint32_t twiddle_buf_bytes = log2N * tiles_per_stage * TILE_BYTES;
    uint32_t output_buf_bytes  = tiles_per_stage * TILE_BYTES;

    auto buf_even_r = make_dram_buffer(device, input_buf_bytes);
    auto buf_even_i = make_dram_buffer(device, input_buf_bytes);
    auto buf_odd_r  = make_dram_buffer(device, input_buf_bytes);
    auto buf_odd_i  = make_dram_buffer(device, input_buf_bytes);
    auto buf_tw_r   = make_dram_buffer(device, twiddle_buf_bytes);
    auto buf_tw_i   = make_dram_buffer(device, twiddle_buf_bytes);
    auto buf_out0_r = make_dram_buffer(device, output_buf_bytes);
    auto buf_out0_i = make_dram_buffer(device, output_buf_bytes);
    auto buf_out1_r = make_dram_buffer(device, output_buf_bytes);
    auto buf_out1_i = make_dram_buffer(device, output_buf_bytes);

    // ── Circular buffers ───────────────────────────────────────────
    constexpr uint32_t IO_DEPTH  = 2;
    constexpr uint32_t TMP_DEPTH = 1;

    create_cb(program, core, tt::CBIndex::c_0,  IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_1,  IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_2,  IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_3,  IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_4,  IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_5,  IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_6,  tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_7,  tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_10, tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_11, tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_12, tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_13, tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_14, tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_15, tiles_per_stage, TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_16, IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_17, IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_18, IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_19, IO_DEPTH,        TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_20, TMP_DEPTH,       TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_21, TMP_DEPTH,       TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_22, TMP_DEPTH,       TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_23, TMP_DEPTH,       TILE_BYTES);
    create_cb(program, core, tt::CBIndex::c_24, TMP_DEPTH,       TILE_BYTES);

    // ── Kernels ────────────────────────────────────────────────────
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

    auto compute_k = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false
        });

    // ── Runtime args ───────────────────────────────────────────────
    SetRuntimeArgs(program, reader_k, core, {
        buf_even_r->address(), buf_even_i->address(),
        buf_odd_r->address(),  buf_odd_i->address(),
        buf_tw_r->address(),   buf_tw_i->address(),
        tiles_per_stage, log2N
    });

    SetRuntimeArgs(program, writer_k, core, {
        buf_out0_r->address(), buf_out0_i->address(),
        buf_out1_r->address(), buf_out1_i->address(),
        tiles_per_stage
    });

    SetRuntimeArgs(program, compute_k, core, {
        direction, log2N, tiles_per_stage
    });

    // ── Write inputs to DRAM ───────────────────────────────────────
    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteBuffer(cq, buf_even_r, even_r_t,   false);
    EnqueueWriteBuffer(cq, buf_even_i, even_i_t,   false);
    EnqueueWriteBuffer(cq, buf_odd_r,  odd_r_t,    false);
    EnqueueWriteBuffer(cq, buf_odd_i,  odd_i_t,    false);
    EnqueueWriteBuffer(cq, buf_tw_r,   tw_r_tiles, false);
    EnqueueWriteBuffer(cq, buf_tw_i,   tw_i_tiles, false);
    Finish(cq);

    // ── Launch kernel ──────────────────────────────────────────────
    std::cout << "Launching FFT kernel (all " << log2N << " stages on device)...\n";
    EnqueueProgram(cq, program, false);
    Finish(cq);
    std::cout << "Kernel complete.\n";

    // ── Read results ───────────────────────────────────────────────
    std::vector<uint32_t> out0_r_raw(tiles_per_stage * TILE_SIZE);
    std::vector<uint32_t> out0_i_raw(tiles_per_stage * TILE_SIZE);
    std::vector<uint32_t> out1_r_raw(tiles_per_stage * TILE_SIZE);
    std::vector<uint32_t> out1_i_raw(tiles_per_stage * TILE_SIZE);

    EnqueueReadBuffer(cq, buf_out0_r, out0_r_raw, true);
    EnqueueReadBuffer(cq, buf_out0_i, out0_i_raw, true);
    EnqueueReadBuffer(cq, buf_out1_r, out1_r_raw, true);
    EnqueueReadBuffer(cq, buf_out1_i, out1_i_raw, true);

    // ── Reconstruct ────────────────────────────────────────────────
    auto out0_r = unpack_tiles(out0_r_raw, half_N);
    auto out0_i = unpack_tiles(out0_i_raw, half_N);
    auto out1_r = unpack_tiles(out1_r_raw, half_N);
    auto out1_i = unpack_tiles(out1_i_raw, half_N);

    std::vector<float> result_r(N), result_i(N);
    for (uint32_t i = 0; i < half_N; i++) {
        result_r[i]          = out0_r[i]; result_i[i]          = out0_i[i];
        result_r[i + half_N] = out1_r[i]; result_i[i + half_N] = out1_i[i];
    }
    if (direction == 1) for (uint32_t i = 0; i < N; i++) { result_r[i] /= N; result_i[i] /= N; }

    // ── Validate ───────────────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════\n VALIDATION\n";
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
    std::cout << " Max error (real): " << max_err_r << "\n";
    std::cout << " Max error (imag): " << max_err_i << "\n";
    std::cout << " Mean error      : " << mean_err  << "\n";

    bool passed = (max_err_r < 1e-3f) && (max_err_i < 1e-3f);
    std::cout << " Result: " << (passed ? "PASSED" : "FAILED (dry run — expected)") << "\n";

    std::cout << "\n First 8 results vs reference:\n" << std::fixed << std::setprecision(4);
    for (uint32_t i = 0; i < 8 && i < N; i++) {
        std::cout << "  X[" << i << "] = " << result_r[i]
                  << (result_i[i] >= 0 ? " +" : " ") << result_i[i] << "j"
                  << "  ref: " << ref_r[i]
                  << (ref_i[i] >= 0 ? " +" : " ") << ref_i[i] << "j\n";
    }

    CloseDevice(device);
    std::cout << "\n Done\n═══════════════════════════════════════\n";
    return 0;
}