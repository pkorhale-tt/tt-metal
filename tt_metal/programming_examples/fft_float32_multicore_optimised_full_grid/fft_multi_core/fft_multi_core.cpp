// fft_multi_core.cpp - FIXED HOST PROGRAM
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Launches a 64-core radix-2 DIT 1D FFT on Tenstorrent Wormhole.
// Currently validates local stages (10 stages within each core).
// Cross-core stages (6 stages) are stubbed in the writer kernel.

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
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;   // 1024 elements
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

// ── Float ↔ uint32 bit-cast helpers ─────────────────────────────────────────
inline uint32_t f2u(float f)  { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float    u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ── Bit-reversal ─────────────────────────────────────────────────────────────
uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// ── Reference CPU FFT (in-place, DIT Cooley-Tukey) ──────────────────────────
void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    uint32_t N = re.size(), log2N = 0;
    while ((1u << log2N) < N) log2N++;

    // Bit-reversal permutation
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

// ── Compact twiddle table: W_N^k = exp(-2πjk/N) for k = 0..N/2-1 ───────────
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, int direction /* -1=forward, +1=inv */) {
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

// ── Create a single-CB circular buffer ──────────────────────────────────────
CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bytes_per_tile) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles * bytes_per_tile, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes_per_tile);
    return CreateCircularBuffer(p, c, cfg);
}

// ── Stage-0 input preparation ────────────────────────────────────────────────
// Bit-reverse the input indices and distribute even/odd pairs across cores.
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

// ═════════════════════════════════════════════════════════════════════════════
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

    const uint32_t elems_per_core  = N / num_cores;
    const uint32_t half_per_core   = elems_per_core / 2;
    const uint32_t tiles_per_core  = (half_per_core + TILE_SIZE - 1) / TILE_SIZE;
    const uint32_t local_stages    = log2N - log2c;  // stages handled inside one core
    const uint32_t cross_stages    = log2c;           // stages requiring NOC exchange

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

    // ── Input signal ─────────────────────────────────────────────
    std::vector<float> ir(N), ii(N, 0.0f);
    for (uint32_t i = 0; i < N; i++)
        ir[i] = std::sin(2.f * PI * 4.f * i / N) +
                0.5f * std::sin(2.f * PI * 8.f * i / N);

    // ── Reference FFT ────────────────────────────────────────────
    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, false);

    // ── Device setup ─────────────────────────────────────────────
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto& cq  = mesh->mesh_command_queue();

    // ── Prepare bit-reversed input buffers ───────────────────────
    std::vector<std::vector<uint32_t>> core_even_r, core_even_i;
    std::vector<std::vector<uint32_t>> core_odd_r,  core_odd_i;
    prepare_multicore_stage0(ir, ii, N, log2N, num_cores,
                             core_even_r, core_even_i, core_odd_r, core_odd_i);

    // Flatten into contiguous arrays (one tile per core, packed)
    auto flatten = [&](const std::vector<std::vector<uint32_t>>& v) {
        std::vector<uint32_t> out;
        out.reserve(v.size() * v[0].size());
        for (auto& c : v) out.insert(out.end(), c.begin(), c.end());
        return out;
    };
    auto all_even_r = flatten(core_even_r);
    auto all_even_i = flatten(core_even_i);
    auto all_odd_r  = flatten(core_odd_r);
    auto all_odd_i  = flatten(core_odd_i);

    // ── Compact twiddle table ─────────────────────────────────────
    auto [tw_r, tw_i] = precompute_compact_twiddles(N, -1 /* forward */);
    {   // Pad to tile boundary
        uint32_t tw_tiles = ((N / 2 * sizeof(float) + TILE_BYTES - 1) / TILE_BYTES);
        tw_r.resize(tw_tiles * TILE_SIZE, 0u);
        tw_i.resize(tw_tiles * TILE_SIZE, 0u);
    }

    // ── Program & CBs ────────────────────────────────────────────
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
    const uint32_t compact_bytes =
        (((N / 2) * sizeof(float) + TILE_BYTES - 1) / TILE_BYTES) * TILE_BYTES;
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

            // Input / twiddle CBs
            create_cb(prog, cc,  0, tiles_per_core, TILE_BYTES);  // cb_even_r
            create_cb(prog, cc,  1, tiles_per_core, TILE_BYTES);  // cb_even_i
            create_cb(prog, cc,  2, tiles_per_core, TILE_BYTES);  // cb_odd_r
            create_cb(prog, cc,  3, tiles_per_core, TILE_BYTES);  // cb_odd_i
            create_cb(prog, cc,  4, tiles_per_core, TILE_BYTES);  // cb_tw_r
            create_cb(prog, cc,  5, tiles_per_core, TILE_BYTES);  // cb_tw_i

            // Compact twiddle scratch
            create_cb(prog, cc, 10, compact_tiles,  TILE_BYTES);  // cb_compact_r
            create_cb(prog, cc, 11, compact_tiles,  TILE_BYTES);  // cb_compact_i

            // Compute outputs
            create_cb(prog, cc, 16, tiles_per_core, TILE_BYTES);  // cb_out0_r
            create_cb(prog, cc, 17, tiles_per_core, TILE_BYTES);  // cb_out0_i
            create_cb(prog, cc, 18, tiles_per_core, TILE_BYTES);  // cb_out1_r
            create_cb(prog, cc, 19, tiles_per_core, TILE_BYTES);  // cb_out1_i

            // Compute temporaries
            create_cb(prog, cc, 20, tiles_per_core, TILE_BYTES);  // cb_tmp0
            create_cb(prog, cc, 21, tiles_per_core, TILE_BYTES);  // cb_tmp1
            create_cb(prog, cc, 22, tiles_per_core, TILE_BYTES);  // cb_tw_odd_r
            create_cb(prog, cc, 23, tiles_per_core, TILE_BYTES);  // cb_tw_odd_i

            // Writer scratch (recv, sync)
            create_cb(prog, cc, 24, tiles_per_core, TILE_BYTES);  // cb_recv_r
            create_cb(prog, cc, 25, tiles_per_core, TILE_BYTES);  // cb_recv_i
            create_cb(prog, cc, 28, 1,              32);           // cb_sync (semaphore)
        }
    }

    // ── Kernels ───────────────────────────────────────────────────
    KernelHandle reader_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore_optimised_full_grid/"
        "fft_multi_core/kernels/dataflow/reader.cpp",
        core_range,
        DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0,
                            .noc       = NOC::RISCV_0_default });

    KernelHandle writer_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore_optimised_full_grid/"
        "fft_multi_core/kernels/dataflow/writer.cpp",
        core_range,
        DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1,
                            .noc       = NOC::RISCV_1_default });

    KernelHandle compute_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore_optimised_full_grid/"
        "fft_multi_core/kernels/compute/compute.cpp",
        core_range,
        ComputeConfig{ .math_fidelity    = MathFidelity::HiFi4,
                       .fp32_dest_acc_en = true,
                       .math_approx_mode = false });

    // ── Per-core runtime args ─────────────────────────────────────
    for (uint32_t cid = 0; cid < num_cores; cid++) {
        uint32_t  cx             = cid % 8;
        uint32_t  cy             = cid / 8;
        CoreCoord cc             = {cx, cy};
        uint32_t  tile_offset    = cid * tiles_per_core;
        uint32_t  core_elem_base = cid * half_per_core;

        SetRuntimeArgs(prog, reader_k, cc, {
            b_even_r->address(), b_even_i->address(),
            b_odd_r->address(),  b_odd_i->address(),
            b_tw_r->address(),   b_tw_i->address(),
            tiles_per_core,      // arg 6
            tile_offset,         // arg 7
            log2N,               // arg 8: num_stages
            N / 2,               // arg 9: half_N
            half_per_core,       // arg 10: local_half
            core_elem_base,      // arg 11
            cid,                 // arg 12: core_id
            num_cores,           // arg 13
            log2c,               // arg 14: log2_cores
            local_stages         // arg 15
        });

        SetRuntimeArgs(prog, compute_k, cc, {
            log2N,          // arg 0: num_stages
            tiles_per_core  // arg 1: tiles_per_stage
        });

        SetRuntimeArgs(prog, writer_k, cc, {
            b_out0_r->address(), b_out0_i->address(),
            b_out1_r->address(), b_out1_i->address(),
            tiles_per_core,   // arg 4
            log2N,            // arg 5: num_stages
            half_per_core,    // arg 6: local_half
            N / 2,            // arg 7: half_N
            num_cores,        // arg 8
            cid,              // arg 9: core_id
            log2c,            // arg 10: log2_cores
            tile_offset,      // arg 11
            core_elem_base,   // arg 12
            local_stages      // arg 13
        });
    }

    // ── Workload ──────────────────────────────────────────────────
    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    // ── Upload inputs ─────────────────────────────────────────────
    std::cout << "Uploading data to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_even_r, all_even_r, false);
    EnqueueWriteMeshBuffer(cq, b_even_i, all_even_i, false);
    EnqueueWriteMeshBuffer(cq, b_odd_r,  all_odd_r,  false);
    EnqueueWriteMeshBuffer(cq, b_odd_i,  all_odd_i,  false);
    EnqueueWriteMeshBuffer(cq, b_tw_r,   tw_r,        false);
    EnqueueWriteMeshBuffer(cq, b_tw_i,   tw_i,        false);
    Finish(cq);

    // ── Launch ────────────────────────────────────────────────────
    std::cout << "Launching " << num_cores << "-core FFT (N=" << N << ")...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    EnqueueMeshWorkload(cq, wl, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto us  = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "Execution time: " << us << " µs\n";

    // ── Read back ─────────────────────────────────────────────────
    const uint32_t raw_elems = total_bytes / 4;
    std::vector<uint32_t> out0_r_raw(raw_elems), out0_i_raw(raw_elems);
    std::vector<uint32_t> out1_r_raw(raw_elems), out1_i_raw(raw_elems);

    EnqueueReadMeshBuffer(cq, out0_r_raw, b_out0_r, true);
    EnqueueReadMeshBuffer(cq, out0_i_raw, b_out0_i, true);
    EnqueueReadMeshBuffer(cq, out1_r_raw, b_out1_r, true);
    EnqueueReadMeshBuffer(cq, out1_i_raw, b_out1_i, true);

    // ── Reconstruct output ────────────────────────────────────────
    // out0 holds the first half_per_core elements of each core's output,
    // out1 holds the second half_per_core elements.
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

    // ── Accuracy check ────────────────────────────────────────────
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
    std::cout << " Max error (real) : " << max_err_r
              << "  @ index " << worst_r << "\n";
    std::cout << " Max error (imag) : " << max_err_i
              << "  @ index " << worst_i << "\n";
    std::cout << " Threshold        : " << threshold << "\n";
    std::cout << " Status           : " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";

    std::cout << "\nFirst 8 output bins:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (uint32_t i = 0; i < 8; i++) {
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