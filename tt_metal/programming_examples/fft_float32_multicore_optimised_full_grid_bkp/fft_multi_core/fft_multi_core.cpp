// fft_multi_core.cpp - COMPLETE FIXED HOST PROGRAM
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <chrono>
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

inline uint32_t f2u(float f)  { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float    u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    uint32_t N = re.size(), log2N = 0;
    while ((1u << log2N) < N) log2N++;

    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }

    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m     = 1u << (s + 1);
        float    angle = (inv ? 2.f : -2.f) * PI / m;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float wr = std::cos(angle * j);
                float wi = std::sin(angle * j);
                uint32_t e = k + j, o = k + j + m / 2;
                float tr = wr * re[o] - wi * im[o];
                float ti = wr * im[o] + wi * re[o];
                float er = re[e], ei = im[e];
                re[e] = er + tr;  im[e] = ei + ti;
                re[o] = er - tr;  im[o] = ei - ti;
            }
        }
    }

    if (inv) {
        for (uint32_t i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
    }
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, int direction) {
    uint32_t half_N = N / 2;
    float    sign   = (direction > 0) ? 1.f : -1.f;
    std::vector<uint32_t> tw_r(half_N), tw_i(half_N);
    for (uint32_t k = 0; k < half_N; k++) {
        float angle = sign * 2.f * PI * k / N;
        tw_r[k] = f2u(std::cos(angle));
        tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bytes_per_tile) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles * bytes_per_tile, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes_per_tile);
    return CreateCircularBuffer(p, c, cfg);
}

void prepare_multicore_stage0(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t N, uint32_t log2N, uint32_t num_cores,
    std::vector<std::vector<uint32_t>>& even_r,
    std::vector<std::vector<uint32_t>>& even_i,
    std::vector<std::vector<uint32_t>>& odd_r,
    std::vector<std::vector<uint32_t>>& odd_i)
{
    uint32_t half_N        = N / 2;
    uint32_t half_per_core = half_N / num_cores;
    uint32_t tiles_per_core = (half_per_core + TILE_SIZE - 1) / TILE_SIZE;
    even_r.assign(num_cores, std::vector<uint32_t>(tiles_per_core * TILE_SIZE, 0u));
    even_i.assign(num_cores, std::vector<uint32_t>(tiles_per_core * TILE_SIZE, 0u));
    odd_r .assign(num_cores, std::vector<uint32_t>(tiles_per_core * TILE_SIZE, 0u));
    odd_i .assign(num_cores, std::vector<uint32_t>(tiles_per_core * TILE_SIZE, 0u));

    for (uint32_t i = 0; i < half_N; i++) {
        uint32_t e_idx  = bit_reverse(2 * i,     log2N);
        uint32_t o_idx  = bit_reverse(2 * i + 1, log2N);
        uint32_t cid    = i / half_per_core;
        uint32_t local  = i % half_per_core;
        if (cid < num_cores) {
            even_r[cid][local] = f2u(sr[e_idx]);
            even_i[cid][local] = f2u(si[e_idx]);
            odd_r [cid][local] = f2u(sr[o_idx]);
            odd_i [cid][local] = f2u(si[o_idx]);
        }
    }
}

int main(int argc, char** argv) {
    uint32_t N         = 65536;
    uint32_t num_cores = 64;
    if (argc > 1) N         = std::stoul(argv[1]);
    if (argc > 2) num_cores = std::stoul(argv[2]);

    uint32_t log2N = 0;      while ((1u << log2N) < N)         log2N++;
    uint32_t log2c = 0;      while ((1u << log2c) < num_cores) log2c++;
    if ((1u << log2N) != N || (1u << log2c) != num_cores) {
        std::cerr << "N and num_cores must be powers of 2\n"; return 1;
    }

    // FIX: Add validation to prevent divide-by-zero for small N
    if (N < num_cores * 2) {
        std::cerr << "ERROR: FFT size N (" << N << ") is too small for the number of cores ("
                  << num_cores << ").\n";
        std::cerr << "       N must be at least 2 * num_cores.\n";
        return 1;
    }

    const uint32_t elems_per_core  = N / num_cores;
    const uint32_t half_per_core   = elems_per_core / 2;
    const uint32_t tiles_per_core  = (half_per_core + TILE_SIZE - 1) / TILE_SIZE;
    const uint32_t local_stages    = log2N - log2c;
    const uint32_t cross_stages    = log2c;

    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " TT-Metal 64-Core FFT (Fixed)\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " N              : " << N             << "\n";
    std::cout << " Cores          : " << num_cores     << "\n";
    std::cout << " Elems/core     : " << elems_per_core << "\n";
    std::cout << " Local stages   : " << local_stages  << "\n";
    std::cout << " Cross stages   : " << cross_stages  << "\n";
    std::cout << " Tiles/core     : " << tiles_per_core << "\n";
    std::cout << "════════════════════════════════════════════════\n";

    std::vector<float> ir(N), ii(N, 0.0f);
    for (uint32_t i = 0; i < N; i++)
        ir[i] = std::sin(2.f * PI * 4.f * i / N) +
                0.5f * std::sin(2.f * PI * 8.f * i / N);

    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, false);

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto& cq  = mesh->mesh_command_queue();

    std::vector<std::vector<uint32_t>> core_even_r, core_even_i, core_odd_r,  core_odd_i;
    prepare_multicore_stage0(ir, ii, N, log2N, num_cores,
                             core_even_r, core_even_i, core_odd_r, core_odd_i);

    auto flatten = [&](const std::vector<std::vector<uint32_t>>& v) {
        std::vector<uint32_t> out;
        out.reserve(v.size() * v[0].size());
        for (const auto& c : v) out.insert(out.end(), c.begin(), c.end());
        return out;
    };
    auto all_even_r = flatten(core_even_r);
    auto all_even_i = flatten(core_even_i);
    auto all_odd_r  = flatten(core_odd_r);
    auto all_odd_i  = flatten(core_odd_i);

    auto [tw_r, tw_i] = precompute_compact_twiddles(N, -1);
    {
        uint32_t tw_tiles = ((N / 2 * sizeof(float) + TILE_BYTES - 1) / TILE_BYTES);
        tw_r.resize(tw_tiles * TILE_SIZE, 0u);
        tw_i.resize(tw_tiles * TILE_SIZE, 0u);
    }

    Program    prog = CreateProgram();
    CoreRange  core_range({0, 0}, {7, 7});
    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_cfg{ .page_size = TILE_BYTES,
                                      .buffer_type = BufferType::DRAM };
    auto mk_buf = [&](uint32_t size) {
        ReplicatedBufferConfig rc{ .size = size };
        return MeshBuffer::create(rc, dram_cfg, mesh.get());
    };
    const uint32_t total_bytes   = tiles_per_core * num_cores * TILE_BYTES;
    const uint32_t compact_bytes = (((N / 2) * sizeof(float) + TILE_BYTES - 1) / TILE_BYTES) * TILE_BYTES;
    const uint32_t compact_tiles = compact_bytes / TILE_BYTES;

    auto b_even_r = mk_buf(total_bytes);
    auto b_even_i = mk_buf(total_bytes);
    auto b_odd_r  = mk_buf(total_bytes);
    auto b_odd_i  = mk_buf(total_bytes);
    auto b_out0_r = mk_buf(total_bytes);
    auto b_out0_i = mk_buf(total_bytes);
    auto b_out1_r = mk_buf(total_bytes);
    auto b_out1_i = mk_buf(total_bytes);
    auto b_tw_r   = mk_buf(compact_bytes);
    auto b_tw_i   = mk_buf(compact_bytes);

    for (uint32_t cy = 0; cy < 8; cy++) {
        for (uint32_t cx = 0; cx < 8; cx++) {
            if (cy * 8 + cx >= num_cores) continue;
            CoreCoord cc = {cx, cy};
            create_cb(prog, cc,  0, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc,  1, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc,  2, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc,  3, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc,  4, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc,  5, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 10, compact_tiles,  TILE_BYTES);
            create_cb(prog, cc, 11, compact_tiles,  TILE_BYTES);
            create_cb(prog, cc, 16, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 17, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 18, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 19, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 20, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 21, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 22, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 23, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 24, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 25, tiles_per_core, TILE_BYTES);
            create_cb(prog, cc, 28, 1,              32);
        }
    }

    std::string reader_path = "tt_metal/programming_examples/fft_float32_multicore_optimised_full_grid/fft_multi_core/kernels/dataflow/reader.cpp";
    std::string writer_path = "tt_metal/programming_examples/fft_float32_multicore_optimised_full_grid/fft_multi_core/kernels/dataflow/writer.cpp";
    std::string compute_path = "tt_metal/programming_examples/fft_float32_multicore_optimised_full_grid/fft_multi_core/kernels/compute/compute.cpp";

    KernelHandle reader_k = CreateKernel(prog, reader_path, core_range, DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default });
    KernelHandle writer_k = CreateKernel(prog, writer_path, core_range, DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default });
    KernelHandle compute_k = CreateKernel(prog, compute_path, core_range, ComputeConfig{ .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false });

    for (uint32_t cid = 0; cid < num_cores; cid++) {
        CoreCoord cc             = {(cid % 8), (cid / 8)};
        uint32_t  tile_offset    = cid * tiles_per_core;
        uint32_t  core_elem_base = cid * half_per_core;
        SetRuntimeArgs(prog, reader_k, cc, { b_even_r->address(), b_even_i->address(), b_odd_r->address(), b_odd_i->address(), b_tw_r->address(), b_tw_i->address(), tiles_per_core, tile_offset, log2N, N / 2, half_per_core, core_elem_base, cid, num_cores, log2c, local_stages });
        SetRuntimeArgs(prog, compute_k, cc, { log2N, tiles_per_core });
        SetRuntimeArgs(prog, writer_k, cc, { b_out0_r->address(), b_out0_i->address(), b_out1_r->address(), b_out1_i->address(), tiles_per_core, log2N, half_per_core, N / 2, num_cores, cid, log2c, tile_offset, core_elem_base, local_stages });
    }

    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(prog));

    std::cout << "Uploading data to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_even_r, all_even_r, false);
    EnqueueWriteMeshBuffer(cq, b_even_i, all_even_i, false);
    EnqueueWriteMeshBuffer(cq, b_odd_r,  all_odd_r,  false);
    EnqueueWriteMeshBuffer(cq, b_odd_i,  all_odd_i,  false);
    EnqueueWriteMeshBuffer(cq, b_tw_r,   tw_r,       false);
    EnqueueWriteMeshBuffer(cq, b_tw_i,   tw_i,       false);
    Finish(cq);

    std::cout << "Launching " << num_cores << "-core FFT (N=" << N << ")...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    EnqueueMeshWorkload(cq, wl, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us  = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "Execution time: " << us << " µs\n";

    const uint32_t raw_elems = total_bytes / 4;
    std::vector<uint32_t> out0_r_raw(raw_elems), out0_i_raw(raw_elems), out1_r_raw(raw_elems), out1_i_raw(raw_elems);
    EnqueueReadMeshBuffer(cq, out0_r_raw, b_out0_r, true);
    EnqueueReadMeshBuffer(cq, out0_i_raw, b_out0_i, true);
    EnqueueReadMeshBuffer(cq, out1_r_raw, b_out1_r, true);
    EnqueueReadMeshBuffer(cq, out1_i_raw, b_out1_i, true);

    std::vector<float> result_r(N, 0.f), result_i(N, 0.f);
    for (uint32_t cid = 0; cid < num_cores; cid++) {
        uint32_t base_idx  = cid * tiles_per_core * TILE_SIZE;
        uint32_t elem_base = cid * elems_per_core;
        for (uint32_t i = 0; i < half_per_core; i++) {
            result_r[elem_base + i]                = u2f(out0_r_raw[base_idx + i]);
            result_i[elem_base + i]                = u2f(out0_i_raw[base_idx + i]);
            result_r[elem_base + half_per_core + i] = u2f(out1_r_raw[base_idx + i]);
            result_i[elem_base + half_per_core + i] = u2f(out1_i_raw[base_idx + i]);
        }
    }

    float max_err_r = 0.f, max_err_i = 0.f;
    uint32_t worst_r = 0, worst_i = 0;
    for (uint32_t i = 0; i < N; i++) {
        float er = std::abs(result_r[i] - ref_r[i]);
        float ei = std::abs(result_i[i] - ref_i[i]);
        if (er > max_err_r) { max_err_r = er; worst_r = i; }
        if (ei > max_err_i) { max_err_i = ei; worst_i = i; }
    }
    float threshold = 0.01f * std::sqrt(static_cast<float>(N));
    bool  passed    = (max_err_r < threshold) && (max_err_i < threshold);

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " RESULTS\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Max error (real) : " << max_err_r << "  @ index " << worst_r << "\n";
    std::cout << " Max error (imag) : " << max_err_i << "  @ index " << worst_i << "\n";
    std::cout << " Threshold        : " << threshold << "\n";
    std::cout << " Status           : " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    std::cout << "\nFirst 8 output bins:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (uint32_t i = 0; i < 8 && i < N; i++) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(10) << result_r[i]
                  << (result_i[i] >= 0 ? " +" : " -")
                  << std::setw(10) << std::abs(result_i[i]) << "j"
                  << "   ref: "
                  << std::setw(10) << ref_r[i]
                  << (ref_i[i] >= 0 ? " +" : " -")
                  << std::setw(10) << std::abs(ref_i[i]) << "j\n";
    }

    mesh->close();
    return passed ? 0 : 1;
}