// fft_multi_core.cpp — MULTICORE FFT host driver (updated for current tt-metal)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>

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

constexpr uint32_t TILE_H    = tt::constants::TILE_HEIGHT;  // 32
constexpr uint32_t TILE_W    = tt::constants::TILE_WIDTH;   // 32
constexpr uint32_t TILE_SIZE = TILE_H * TILE_W;             // 1024
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);  // 4096 bytes

// ── Float/uint bit-cast helpers ───────────────────────────────────────
inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float   u2f(uint32_t u){ float f; std::memcpy(&f,&u,4); return f; }

// ── Tile packing / unpacking ──────────────────────────────────────────
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

// ── Bit-reverse ────────────────────────────────────────────────────────
uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r<<1)|(x&1); x >>= 1; }
    return r;
}

// ── Reference CPU FFT (for validation) ───────────────────────────────
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

// ── Stage-0 input preparation (bit-reversed, stride-2 split) ─────────
void prepare_stage0(
    const std::vector<float>& sr,
    const std::vector<float>& si,
    uint32_t N,
    uint32_t log2N,
    uint32_t total_tiles,
    std::vector<uint32_t>& er,
    std::vector<uint32_t>& ei,
    std::vector<uint32_t>& or_,
    std::vector<uint32_t>& oi
) {
    uint32_t half_N = N/2;
    std::vector<float> _er(half_N),_ei(half_N),_or(half_N),_oi(half_N);
    for (uint32_t i = 0; i < half_N; i++) {
        uint32_t e = bit_reverse(2*i,   log2N);
        uint32_t o = bit_reverse(2*i+1, log2N);
        _er[i]=sr[e]; _ei[i]=si[e]; _or[i]=sr[o]; _oi[i]=si[o];
    }
    er  = pack_tiles(_er, total_tiles);
    ei  = pack_tiles(_ei, total_tiles);
    or_ = pack_tiles(_or, total_tiles);
    oi  = pack_tiles(_oi, total_tiles);
}

// ── Compact twiddle table ─────────────────────────────────────────────
std::pair<std::vector<uint32_t>,std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, uint32_t direction) {
    uint32_t half_N = N/2;
    float sign = (direction==1) ? 1.f : -1.f;
    std::vector<uint32_t> tw_r(TILE_SIZE, 0u), tw_i(TILE_SIZE, 0u);
    for (uint32_t k = 0; k < half_N; k++) {
        float angle = sign * 2.f*PI*(float)k/(float)N;
        tw_r[k] = f2u(std::cos(angle));
        tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

// ── CB creation helper ────────────────────────────────────────────────
CBHandle create_cb(Program& p, CoreCoord c, uint32_t id, uint32_t ntiles, uint32_t bytes) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles*bytes, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes);
    return CreateCircularBuffer(p, c, cfg);
}

// ── Core auto-detection ───────────────────────────────────────────────
uint32_t detect_available_cores(IDevice* device, uint32_t max_requested, uint32_t N) {
    CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t max_cols = grid.x;
    uint32_t max_rows = grid.y;

    std::cout << " Device grid : " << max_cols << " x " << max_rows
              << " Tensix cores\n";

    uint32_t usable = 0;
    for (uint32_t col = 0; col < max_cols; col++) {
        CoreCoord logical = {col, 0};
        try {
            CoreCoord physical = device->worker_core_from_logical_core(logical);
            (void)physical;
            usable++;
        } catch (...) {
            std::cout << " Core {" << col << ",0} is harvested — stopping scan\n";
            break;
        }
    }

    std::cout << " Usable row-0 cores: " << usable << "\n";

    uint32_t cap = std::min({usable, max_requested, N / 2});

    uint32_t result = 1;
    while (result * 2 <= cap) result *= 2;

    return result;
}

// ── Compute per-core CB base address analytically ─────────────────────
// Assumptions:
//  - Program-local CBs start at `l1_unreserved_base` in L1 on every core.
//  - CBs are created in the same order on every core.
//  - Each CB created here has size TILE_BYTES and depth=1 (one tile).
//  - This FFT example creates a fixed sequence of CB IDs per core:
//      0,1,2,3,4,5,16,17,18,19,20,21,22,23,10,11

uint32_t cb_slot_index_from_id(uint32_t cb_id) {
    switch (cb_id) {
        case 0:  return 0;
        case 1:  return 1;
        case 2:  return 2;
        case 3:  return 3;
        case 4:  return 4;
        case 5:  return 5;
        case 16: return 6;
        case 17: return 7;
        case 18: return 8;
        case 19: return 9;
        case 20: return 10;
        case 21: return 11;
        case 22: return 12;
        case 23: return 13;
        case 10: return 14;
        case 11: return 15;
        default:
            TT_FATAL(false, "Unexpected CB ID {} in FFT example", cb_id);
            return 0;
    }
}

uint32_t cb_base_addr_for_core_and_id(
    uint32_t l1_unreserved_base,
    uint32_t cb_id
) {
    uint32_t slot  = cb_slot_index_from_id(cb_id);
    uint32_t bytes = TILE_BYTES; // each CB is depth=1 tile
    return l1_unreserved_base + slot * bytes;
}

// ── Simple numeric check for argv parsing ─────────────────────────────
bool is_uint_str(const char* s) {
    if (!s || !*s) return false;
    for (const char* p = s; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <direction:0|1> [N] [num_cores] [other args...]\n"
                  << " num_cores: optional override (must be power of 2).\n"
                  << " If omitted, auto-detected from device.\n";
        return 1;
    }
    uint32_t direction = (uint32_t)std::atoi(argv[1]);
    uint32_t N = 1024;
    uint32_t user_cores_request = 0;

    // Safely parse only numeric extra args as N / num_cores
    for (int i = 2; i < argc; i++) {
        if (!is_uint_str(argv[i])) {
            // Non-numeric arg (e.g., a file path) – ignore for FFT params
            continue;
        }
        uint32_t v = (uint32_t)std::stoul(argv[i]);
        // Heuristic: small power-of-2 values (<=64) treated as core count
        if (v >= 2 && v <= 64 && (v & (v-1)) == 0)
            user_cores_request = v;
        else if (v >= 2 && (v & (v-1)) == 0)
            N = v;
        else
            std::cerr << "Warning: ignoring numeric argument " << v
                      << " (not a power of 2)\n";
    }

    if (N < 2 || (N & (N-1))) {
        std::cerr << "N must be power of 2\n";
        return 1;
    }

    // ── Open mesh device ─────────────────────────────────────────────
    int dev_id = 0;
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(dev_id);
    auto& cq  = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices().at(0);

    // ── Auto-detect num_cores ───────────────────────────────────────
    uint32_t max_request = (user_cores_request > 0) ? user_cores_request : 64u;
    uint32_t num_cores   = detect_available_cores(device, max_request, N);

    if (user_cores_request > 0 && num_cores < user_cores_request) {
        std::cout << " WARNING: requested " << user_cores_request
                  << " cores but only " << num_cores
                  << " are available/usable. Continuing with " << num_cores << ".\n";
    }
    if (num_cores < 1) {
        std::cerr << "No usable cores found on device.\n";
        return 1;
    }

    uint32_t log2N      = 0; while ((1u<<log2N)      < N)         log2N++;
    uint32_t log2_cores = 0; while ((1u<<log2_cores) < num_cores) log2_cores++;
    uint32_t half_N     = N / 2;
    uint32_t local_half = half_N / num_cores;
    uint32_t local_tiles= (local_half + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t total_tiles= local_tiles * num_cores;
    uint32_t compact_bytes = half_N * sizeof(float);

    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " TT-Metal MULTICORE FFT (compact twiddles)\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " N           : " << N << "\n";
    std::cout << " log2N       : " << log2N << "\n";
    std::cout << " Direction   : " << (direction?"Inverse":"Forward") << "\n";
    std::cout << " num_cores   : " << num_cores << "\n";
    std::cout << " log2_cores  : " << log2_cores << "\n";
    std::cout << " local_half  : " << local_half << " elements/core\n";
    std::cout << " local_tiles : " << local_tiles << " tiles/core\n";
    std::cout << " Cross stages: 0 .. " << (log2_cores > 0 ? log2_cores-1 : 0) << "\n";
    std::cout << " Local stages: " << log2_cores << " .. " << log2N-1 << "\n";
    std::cout << " DRAM upload : "
              << (4*total_tiles*TILE_BYTES + 2*compact_bytes) / 1024 << " KB\n";
    std::cout << " DRAM dl     : "
              << (4*total_tiles*TILE_BYTES) / 1024 << " KB\n";
    std::cout << "════════════════════════════════════════════════\n";

    // ── Generate input signal (synthetic) ───────────────────────────
    std::vector<float> ir(N,0.f), ii(N,0.f);
    for (uint32_t i = 0; i < N; i++)
        ir[i] = std::sin(2.f*PI*4.f*i/N) + 0.5f*std::sin(2.f*PI*8.f*i/N);

    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, direction==1);

    // ── Prepare device inputs ───────────────────────────────────────
    std::vector<uint32_t> even_r_t, even_i_t, odd_r_t, odd_i_t;
    prepare_stage0(ir, ii, N, log2N, total_tiles,
                   even_r_t, even_i_t, odd_r_t, odd_i_t);

    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N, direction);

    // ── Create program ──────────────────────────────────────────────
    Program prog = CreateProgram();

    CoreRange core_range({0,0}, {num_cores-1, 0});

    // ── Shared DRAM buffers via MeshBuffer ─────────────────────────
    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile{
        .page_size = TILE_BYTES,
        .buffer_type = BufferType::DRAM
    };
    auto mk_tile = [&](uint32_t bytes) {
        ReplicatedBufferConfig rc{.size=bytes};
        return MeshBuffer::create(rc, dram_tile, mesh.get());
    };
    uint32_t input_bytes = total_tiles * TILE_BYTES;
    auto b_er  = mk_tile(input_bytes);
    auto b_ei  = mk_tile(input_bytes);
    auto b_or  = mk_tile(input_bytes);
    auto b_oi  = mk_tile(input_bytes);
    auto b_o0r = mk_tile(input_bytes);
    auto b_o0i = mk_tile(input_bytes);
    auto b_o1r = mk_tile(input_bytes);
    auto b_o1i = mk_tile(input_bytes);

    DeviceLocalBufferConfig dram_cmp{
        .page_size=compact_bytes,
        .buffer_type=BufferType::DRAM
    };
    ReplicatedBufferConfig rc_cmp{.size=compact_bytes};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // ── Circular buffers on each core ──────────────────────────────
    std::vector<std::array<uint32_t, 4>> cb_ids(num_cores);

    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        cb_ids[c][0] = 0;
        cb_ids[c][1] = 1;
        cb_ids[c][2] = 2;
        cb_ids[c][3] = 3;

        create_cb(prog, cc, 0,  1, TILE_BYTES); // even_r
        create_cb(prog, cc, 1,  1, TILE_BYTES); // even_i
        create_cb(prog, cc, 2,  1, TILE_BYTES); // odd_r
        create_cb(prog, cc, 3,  1, TILE_BYTES); // odd_i
        create_cb(prog, cc, 4,  1, TILE_BYTES); // tw_r (expanded)
        create_cb(prog, cc, 5,  1, TILE_BYTES); // tw_i
        create_cb(prog, cc, 16, 1, TILE_BYTES); // out0_r
        create_cb(prog, cc, 17, 1, TILE_BYTES); // out0_i
        create_cb(prog, cc, 18, 1, TILE_BYTES); // out1_r
        create_cb(prog, cc, 19, 1, TILE_BYTES); // out1_i
        create_cb(prog, cc, 20, 1, TILE_BYTES); // tmp0
        create_cb(prog, cc, 21, 1, TILE_BYTES); // tmp1
        create_cb(prog, cc, 22, 1, TILE_BYTES); // tw_odd_r
        create_cb(prog, cc, 23, 1, TILE_BYTES); // tw_odd_i
        create_cb(prog, cc, 10, 1, TILE_BYTES); // compact_r
        create_cb(prog, cc, 11, 1, TILE_BYTES); // compact_i
    }

    // ── Kernels (paths match your tree) ─────────────────────────────
    KernelHandle reader_k = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/reader_fft_f32.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default
        });

    KernelHandle writer_k = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/writer_fft_f32.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default
        });

    KernelHandle compute_k = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/compute/fft_compute_f32.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false
        });

    // ── L1 base for program-local CBs ──────────────────────────────
    uint32_t l1_unreserved_base =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);

    // ── Per-core runtime args ──────────────────────────────────────
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        uint32_t tile_offset = c * local_tiles;

        std::vector<uint32_t> reader_args = {
            b_er->address(), b_ei->address(),
            b_or->address(), b_oi->address(),
            b_cmp_r->address(), b_cmp_i->address(),
            local_tiles, tile_offset,
            log2N, half_N, local_half
        };

        std::vector<uint32_t> compute_args = { log2N, local_tiles };

        std::vector<uint32_t> writer_args = {
            b_o0r->address(), b_o0i->address(),
            b_o1r->address(), b_o1i->address(),
            local_tiles, log2N, local_half,
            half_N, num_cores, c, log2_cores,
            tile_offset
        };

        std::vector<uint32_t> cx_noc_x(log2_cores), cx_noc_y(log2_cores);
        std::vector<uint32_t> cx_er(log2_cores),   cx_ei(log2_cores);
        std::vector<uint32_t> cx_or(log2_cores),   cx_oi(log2_cores);

        for (uint32_t s = 0; s < log2_cores; s++) {
            uint32_t partner_id = c ^ (num_cores >> (s + 1));

            CoreCoord partner_logical  = {partner_id, 0};
            CoreCoord partner_physical =
                device->worker_core_from_logical_core(partner_logical);

            cx_noc_x[s] = partner_physical.x;
            cx_noc_y[s] = partner_physical.y;

            // Analytically compute partner's CB base addresses in L1
            cx_er[s] = cb_base_addr_for_core_and_id(
                l1_unreserved_base,
                cb_ids[partner_id][0]); // cb 0: even_r
            cx_ei[s] = cb_base_addr_for_core_and_id(
                l1_unreserved_base,
                cb_ids[partner_id][1]); // cb 1: even_i
            cx_or[s] = cb_base_addr_for_core_and_id(
                l1_unreserved_base,
                cb_ids[partner_id][2]); // cb 2: odd_r
            cx_oi[s] = cb_base_addr_for_core_and_id(
                l1_unreserved_base,
                cb_ids[partner_id][3]); // cb 3: odd_i
        }

        for (auto v : cx_noc_x) writer_args.push_back(v);
        for (auto v : cx_noc_y) writer_args.push_back(v);
        for (auto v : cx_er)    writer_args.push_back(v);
        for (auto v : cx_ei)    writer_args.push_back(v);
        for (auto v : cx_or)    writer_args.push_back(v);
        for (auto v : cx_oi)    writer_args.push_back(v);

        SetRuntimeArgs(prog, reader_k,  cc, reader_args);
        SetRuntimeArgs(prog, writer_k,  cc, writer_args);
        SetRuntimeArgs(prog, compute_k, cc, compute_args);
    }

    // ── Mesh workload ──────────────────────────────────────────────
    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng = distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    // ── Upload inputs ──────────────────────────────────────────────
    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_er,    even_r_t, false);
    EnqueueWriteMeshBuffer(cq, b_ei,    even_i_t, false);
    EnqueueWriteMeshBuffer(cq, b_or,    odd_r_t,  false);
    EnqueueWriteMeshBuffer(cq, b_oi,    odd_i_t,  false);
    EnqueueWriteMeshBuffer(cq, b_cmp_r, cmp_r_t,  false);
    EnqueueWriteMeshBuffer(cq, b_cmp_i, cmp_i_t,  false);
    Finish(cq);

    // ── Run ────────────────────────────────────────────────────────
    std::cout << "Launching multicore FFT ("
              << num_cores << " cores, " << log2N << " stages)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

    // ── Read back results ─────────────────────────────────────────
    std::vector<uint32_t> o0r_raw(total_tiles*TILE_SIZE);
    std::vector<uint32_t> o0i_raw(total_tiles*TILE_SIZE);
    std::vector<uint32_t> o1r_raw(total_tiles*TILE_SIZE);
    std::vector<uint32_t> o1i_raw(total_tiles*TILE_SIZE);
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
        result_r[i]         = o0r[i];
        result_i[i]         = o0i[i];
        result_r[i+half_N]  = o1r[i];
        result_i[i+half_N]  = o1i[i];
    }
    if (direction==1)
        for (uint32_t i=0;i<N;i++){ result_r[i]/=N; result_i[i]/=N; }

    // ── Validate ───────────────────────────────────────────────────
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " VALIDATION\n";
    std::cout << "════════════════════════════════════════════════\n";
    float mer=0.f, mei=0.f, me=0.f;
    for (uint32_t i=0;i<N;i++) {
        float er = std::abs(result_r[i]-ref_r[i]);
        float ei = std::abs(result_i[i]-ref_i[i]);
        mer = std::max(mer,er);
        mei = std::max(mei,ei);
        me  += er+ei;
    }
    me /= 2*N;
    std::cout << " Max error (real): " << mer << "\n";
    std::cout << " Max error (imag): " << mei << "\n";
    std::cout << " Mean error       : " << me  << "\n";
    bool passed = (mer < 0.5f) && (mei < 0.5f);
    std::cout << " Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " FIRST 16 RESULTS\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);
    for (uint32_t i=0;i<16&&i<N;i++) {
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
    std::cout << " Done\n";
    std::cout << "════════════════════════════════════════════════\n";
    return passed ? 0 : 1;
}