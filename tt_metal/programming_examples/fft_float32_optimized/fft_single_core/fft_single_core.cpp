// fft_single_core_opt.cpp  — OPTIMAL SINGLE CORE FFT
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <fstream>
#include <vector>
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
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f) { 
    uint32_t u; 
    std::memcpy(&u, &f, 4); 
    return u; 
}

inline float u2f(uint32_t u) { 
    float f; 
    std::memcpy(&f, &u, 4); 
    return f; 
}

std::vector<uint32_t> pack_tiles(const std::vector<float>& d, uint32_t n) {
    std::vector<uint32_t> o(n * TILE_SIZE, 0);
    for (uint32_t i = 0; i < d.size() && i < o.size(); i++) 
        o[i] = f2u(d[i]);
    return o;
}

std::vector<float> unpack_tiles(const std::vector<uint32_t>& d, uint32_t n) {
    std::vector<float> o(n);
    for (uint32_t i = 0; i < n && i < d.size(); i++) 
        o[i] = u2f(d[i]);
    return o;
}

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    uint32_t N = re.size(), log2N = 0;
    while ((1u << log2N) < N) log2N++;
    
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u << (s + 1);
        float ab = (inv ? 2.f : -2.f) * PI / m;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float wr = std::cos(ab * j), wi = std::sin(ab * j);
                uint32_t e = k + j, o = k + j + m / 2;
                float tr = wr * re[o] - wi * im[o];
                float ti = wr * im[o] + wi * re[o];
                float er = re[e], ei = im[e];
                re[e] = er + tr; im[e] = ei + ti;
                re[o] = er - tr; im[o] = ei - ti;
            }
        }
    }
    
    if (inv) {
        for (uint32_t i = 0; i < N; i++) {
            re[i] /= N;
            im[i] /= N;
        }
    }
}

void prepare_stage0(const std::vector<float>& sr, const std::vector<float>& si,
                    uint32_t N, uint32_t log2N, uint32_t tiles,
                    std::vector<uint32_t>& er, std::vector<uint32_t>& ei,
                    std::vector<uint32_t>& or_, std::vector<uint32_t>& oi) {
    uint32_t half_N = N / 2;
    std::vector<float> _er(half_N), _ei(half_N), _or(half_N), _oi(half_N);
    
    for (uint32_t i = 0; i < half_N; i++) {
        uint32_t e = bit_reverse(2 * i, log2N);
        uint32_t o = bit_reverse(2 * i + 1, log2N);
        _er[i] = sr[e]; _ei[i] = si[e];
        _or[i] = sr[o]; _oi[i] = si[o];
    }
    
    er = pack_tiles(_er, tiles); 
    ei = pack_tiles(_ei, tiles);
    or_ = pack_tiles(_or, tiles); 
    oi = pack_tiles(_oi, tiles);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_twiddles(uint32_t N, uint32_t log2N, uint32_t tiles, uint32_t direction) {
    uint32_t total = log2N * tiles;
    std::vector<uint32_t> tw_r(total * TILE_SIZE, 0), tw_i(total * TILE_SIZE, 0);
    float sign = (direction == 1) ? 1.f : -1.f;
    
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u << (s + 1), hm = m >> 1, idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < hm; j++) {
                float a = sign * 2.f * PI * (float)j / (float)m;
                uint32_t flat = s * tiles * TILE_SIZE + idx;
                tw_r[flat] = f2u(std::cos(a));
                tw_i[flat] = f2u(std::sin(a));
                idx++;
            }
        }
    }
    return {tw_r, tw_i};
}

void create_cb(Program& p, CoreCoord c, uint32_t id, uint32_t n, uint32_t b) {
    CircularBufferConfig cfg = CircularBufferConfig(n * b, {{id, tt::DataFormat::Float32}})
        .set_page_size(id, b);
    CreateCircularBuffer(p, c, cfg);
}

bool read_file(const std::string& path, uint32_t& N, bool from_cmd,
               std::vector<float>& ir, std::vector<float>& ii) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }
    
    std::vector<float> v;
    std::string t;
    while (f >> t) {
        if (!t.empty() && t.back() == ',') t.pop_back();
        if (t.empty()) continue;
        try {
            v.push_back(std::stof(t));
        } catch (...) {
            std::cerr << "Bad token\n";
            return false;
        }
    }
    
    if (v.empty()) {
        std::cerr << "Empty file\n";
        return false;
    }
    
    uint32_t cnt = (uint32_t)v.size();
    bool interleaved = false;
    
    if (from_cmd) {
        if (cnt == 2 * N) {
            interleaved = true;
            std::cout << " File: interleaved (" << cnt << " values)\n";
        } else if (cnt == N) {
            std::cout << " File: real-only (" << cnt << " values)\n";
        } else if (cnt < N) {
            std::cerr << "File has " << cnt << " values, padding to N=" << N << "\n";
        } else {
            cnt = N;
            v.resize(N);
        }
    } else {
        N = 1;
        while (N < cnt) N <<= 1;
        std::cout << " File: real-only (" << cnt << " values, N inferred=" << N << ")\n";
    }
    
    ir.assign(N, 0.f);
    ii.assign(N, 0.f);
    
    if (interleaved) {
        for (uint32_t i = 0; i < N && 2 * i + 1 < (uint32_t)v.size(); i++) {
            ir[i] = v[2 * i];
            ii[i] = v[2 * i + 1];
        }
    } else {
        for (uint32_t i = 0; i < N && i < (uint32_t)v.size(); i++)
            ir[i] = v[i];
    }
    
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <0|1> [file] [N]\n";
        return 1;
    }
    
    uint32_t direction = (uint32_t)std::atoi(argv[1]);
    uint32_t N = 1024;
    std::string in_file;
    bool from_cmd = false;
    
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        bool is_file = (a.find('.') != std::string::npos || a.find('/') != std::string::npos);
        if (is_file && in_file.empty()) {
            in_file = a;
        } else {
            try {
                N = (uint32_t)std::stol(a);
                from_cmd = true;
            } catch (...) {
                if (in_file.empty()) in_file = a;
            }
        }
    }
    
    if (from_cmd && (N == 0 || (N & (N - 1)))) {
        std::cerr << "N must be power of 2\n";
        return 1;
    }
    
    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;
    uint32_t half_N = N / 2;
    uint32_t tiles = (half_N + TILE_SIZE - 1) / TILE_SIZE;
    
    std::vector<float> ir(N, 0.f), ii(N, 0.f);
    
    if (!in_file.empty()) {
        if (!read_file(in_file, N, from_cmd, ir, ii)) return 1;
        log2N = 0;
        while ((1u << log2N) < N) log2N++;
        half_N = N / 2;
        tiles = (half_N + TILE_SIZE - 1) / TILE_SIZE;
        ir.resize(N, 0.f);
        ii.resize(N, 0.f);
        if (N < 2 || (N & (N - 1))) {
            std::cerr << "Invalid N=" << N << "\n";
            return 1;
        }
    } else {
        for (uint32_t i = 0; i < N; i++)
            ir[i] = std::sin(2.f * PI * 4.f * i / N) + 0.5f * std::sin(2.f * PI * 8.f * i / N);
    }
    
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " TT-Metal FFT  (Optimized Single Core)\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " N           : " << N << "\n";
    std::cout << " log2N       : " << log2N << "\n";
    std::cout << " Direction   : " << (direction ? "Inverse" : "Forward") << "\n";
    std::cout << " tiles/stage : " << tiles << "\n";
    std::cout << " DRAM upload : " << (4 * tiles * TILE_BYTES + log2N * tiles * 2 * TILE_BYTES) / 1024 << " KB\n";
    std::cout << " DRAM dl     : " << (4 * tiles * TILE_BYTES) / 1024 << " KB\n";
    std::cout << "═══════════════════════════════════════\n";
    
    // Reference
    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, direction == 1);
    
    // Prepare inputs
    std::vector<uint32_t> even_r_t, even_i_t, odd_r_t, odd_i_t;
    prepare_stage0(ir, ii, N, log2N, tiles, even_r_t, even_i_t, odd_r_t, odd_i_t);
    auto [tw_r_t, tw_i_t] = precompute_twiddles(N, log2N, tiles, direction);
    
    // Device
    int dev_id = 0;
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(dev_id);
    auto& cq = mesh->mesh_command_queue();
    
    Program prog = CreateProgram();
    CoreCoord core = {0, 0};
    
    uint32_t in_bytes = tiles * TILE_BYTES;
    uint32_t tw_bytes = log2N * tiles * TILE_BYTES;
    uint32_t out_bytes = tiles * TILE_BYTES;
    
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram{
        .page_size = TILE_BYTES, 
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    
    auto mk = [&](uint32_t b) {
        tt::tt_metal::distributed::ReplicatedBufferConfig rc{.size = b};
        return tt::tt_metal::distributed::MeshBuffer::create(rc, dram, mesh.get());
    };
    
    auto b_er = mk(in_bytes), b_ei = mk(in_bytes);
    auto b_or = mk(in_bytes), b_oi = mk(in_bytes);
    auto b_tr = mk(tw_bytes), b_ti = mk(tw_bytes);
    auto b_o0r = mk(out_bytes), b_o0i = mk(out_bytes);
    auto b_o1r = mk(out_bytes), b_o1i = mk(out_bytes);
    
    // ═══════════════════════════════════════════════════════════
    // OPTIMIZED CIRCULAR BUFFERS
    // ═══════════════════════════════════════════════════════════
    create_cb(prog, core, 0, tiles, TILE_BYTES);   // even_r
    create_cb(prog, core, 1, tiles, TILE_BYTES);   // even_i
    create_cb(prog, core, 2, tiles, TILE_BYTES);   // odd_r
    create_cb(prog, core, 3, tiles, TILE_BYTES);   // odd_i
    
    // ✅ OPTIMIZED: Streaming twiddles (depth = tiles, not log2N*tiles)
    create_cb(prog, core, 4, tiles, TILE_BYTES);   // tw_r
    create_cb(prog, core, 5, tiles, TILE_BYTES);   // tw_i
    
    create_cb(prog, core, 16, tiles, TILE_BYTES);  // out0_r
    create_cb(prog, core, 17, tiles, TILE_BYTES);  // out0_i
    create_cb(prog, core, 18, tiles, TILE_BYTES);  // out1_r
    create_cb(prog, core, 19, tiles, TILE_BYTES);  // out1_i
    
    auto reader_k = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    auto writer_k = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );

    auto compute_k = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .math_approx_mode = false}
    );

    std::vector<uint32_t> reader_args = {
        b_er->address(), b_ei->address(),
        b_or->address(), b_oi->address(),
        b_tr->address(), b_ti->address(),
        tiles, log2N
    };
    
    std::vector<uint32_t> writer_args = {
        b_o0r->address(), b_o0i->address(),
        b_o1r->address(), b_o1i->address(),
        tiles, log2N, half_N
    };
    
    std::vector<uint32_t> compute_args = {log2N, tiles};
    
    tt::tt_metal::distributed::MeshWorkload wl;
    tt::tt_metal::distributed::MeshCoordinateRange rng =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));
    auto& p = wl.get_programs().begin()->second;
    
    SetRuntimeArgs(p, reader_k, core, reader_args);
    SetRuntimeArgs(p, writer_k, core, writer_args);
    SetRuntimeArgs(p, compute_k, core, compute_args);
    
    std::cout << "Writing inputs to DRAM...\n";
    using namespace tt::tt_metal::distributed;
    EnqueueWriteMeshBuffer(cq, b_er, even_r_t, false);
    EnqueueWriteMeshBuffer(cq, b_ei, even_i_t, false);
    EnqueueWriteMeshBuffer(cq, b_or, odd_r_t, false);
    EnqueueWriteMeshBuffer(cq, b_oi, odd_i_t, false);
    EnqueueWriteMeshBuffer(cq, b_tr, tw_r_t, false);
    EnqueueWriteMeshBuffer(cq, b_ti, tw_i_t, false);
    Finish(cq);
    
    std::cout << "Launching FFT kernel (" << log2N << " stages on device)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";
    
    std::vector<uint32_t> o0r_raw(tiles * TILE_SIZE), o0i_raw(tiles * TILE_SIZE);
    std::vector<uint32_t> o1r_raw(tiles * TILE_SIZE), o1i_raw(tiles * TILE_SIZE);
    
    EnqueueReadMeshBuffer(cq, o0r_raw, b_o0r, true);
    EnqueueReadMeshBuffer(cq, o0i_raw, b_o0i, true);
    EnqueueReadMeshBuffer(cq, o1r_raw, b_o1r, true);
    EnqueueReadMeshBuffer(cq, o1i_raw, b_o1i, true);
    
    auto o0r = unpack_tiles(o0r_raw, half_N);
    auto o0i = unpack_tiles(o0i_raw, half_N);
    auto o1r = unpack_tiles(o1r_raw, half_N);
    auto o1i = unpack_tiles(o1i_raw, half_N);
    
    std::vector<float> result_r(N), result_i(N);
    for (uint32_t i = 0; i < half_N; i++) {
        result_r[i] = o0r[i];
        result_i[i] = o0i[i];
        result_r[i + half_N] = o1r[i];
        result_i[i + half_N] = o1i[i];
    }
    
    if (direction == 1) {
        for (uint32_t i = 0; i < N; i++) {
            result_r[i] /= N;
            result_i[i] /= N;
        }
    }
    
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << " VALIDATION\n";
    std::cout << "═══════════════════════════════════════\n";
    
    float mer = 0.f, mei = 0.f, me = 0.f;
    for (uint32_t i = 0; i < N; i++) {
        float er = std::abs(result_r[i] - ref_r[i]);
        float ei = std::abs(result_i[i] - ref_i[i]);
        mer = std::max(mer, er);
        mei = std::max(mei, ei);
        me += er + ei;
    }
    me /= 2 * N;
    
    std::cout << " Max error (real): " << mer << "\n";
    std::cout << " Max error (imag): " << mei << "\n";
    std::cout << " Mean error      : " << me << "\n";
    
    bool passed = (mer < 0.5f) && (mei < 0.5f);
    std::cout << " Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << " FIRST 16 RESULTS\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);
    
    for (uint32_t i = 0; i < 16 && i < N; i++) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(12) << result_r[i]
                  << (result_i[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(result_i[i]) << "j"
                  << "   ref: " << std::setw(12) << ref_r[i]
                  << (ref_i[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(ref_i[i]) << "j\n";
    }
    
    mesh->close();
    
    std::cout << "\n═══════════════════════════════════════\n Done\n";
    std::cout << "═══════════════════════════════════════\n";
    
    return passed ? 0 : 1;
}