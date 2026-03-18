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

constexpr uint32_t TILE_H    = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W    = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float   u2f(uint32_t u){ float f; std::memcpy(&f,&u,4); return f; }

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
    for (uint32_t i = 0; i < log2n; i++) { r = (r<<1)|(x&1); x >>= 1; }
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

void prepare_stage0(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t N, uint32_t log2N, uint32_t total_tiles,
    std::vector<uint32_t>& er, std::vector<uint32_t>& ei,
    std::vector<uint32_t>& or_, std::vector<uint32_t>& oi
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

std::pair<std::vector<uint32_t>,std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, uint32_t direction) {
    uint32_t half_N = N/2;
    float sign = (direction==1) ? 1.f : -1.f;
    // BUG FIX: vector must be half_N entries, not TILE_SIZE.
    // TILE_SIZE=1024 but half_N can be 8192 for N=16384.
    // Writing beyond TILE_SIZE corrupted the heap → malloc crash.
    // The DRAM buffer for compact twiddles is also sized to half_N*sizeof(float),
    // so the vector must match that exactly.
    std::vector<uint32_t> tw_r(half_N, 0u), tw_i(half_N, 0u);
    for (uint32_t k = 0; k < half_N; k++) {
        float angle = sign * 2.f*PI*(float)k/(float)N;
        tw_r[k] = f2u(std::cos(angle));
        tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

CBHandle create_cb(Program& p, CoreCoord c, uint32_t id, uint32_t ntiles, uint32_t bytes) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles*bytes, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes);
    return CreateCircularBuffer(p, c, cfg);
}

uint32_t detect_available_cores(IDevice* device, uint32_t max_requested, uint32_t N) {
    CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t max_cols = grid.x;
    uint32_t max_rows = grid.y;
    std::cout << " Device grid : " << max_cols << " x " << max_rows << " Tensix cores\n";

    uint32_t usable = 0;
    for (uint32_t col = 0; col < max_cols; col++) {
        try {
            CoreCoord physical = device->worker_core_from_logical_core({col, 0});
            (void)physical;
            usable++;
        } catch (...) {
            std::cout << " Core {" << col << ",0} harvested — stopping\n";
            break;
        }
    }
    std::cout << " Usable row-0 cores: " << usable << "\n";

    // Constraint: local_half = N/(2*num_cores) must be >= TILE_SIZE (1024).
    // So num_cores <= N / (2 * TILE_SIZE).
    // This is the hard architectural limit — the FPU butterfly operates on
    // full tiles, so each core must own at least one full tile of elements.
    uint32_t max_by_tile = N / (2 * TILE_SIZE);
    if (max_by_tile == 0) max_by_tile = 1;

    uint32_t cap = std::min({usable, max_requested, N / 2, max_by_tile});
    uint32_t result = 1;
    while (result * 2 <= cap) result *= 2;

    std::cout << " Max cores for N=" << N << " : " << max_by_tile
              << "  (N must be >= 2*cores*" << TILE_SIZE << ")\n";
    std::cout << " Selected cores  : " << result << "\n";
    return result;
}

// ── CB address layout ──────────────────────────────────────────────────
// CBs are allocated in L1 starting at l1_unreserved_base, one TILE_BYTES
// slot each, in the order they are created per core.
// Creation order in this file: 0,1,2,3,4,5,16,17,18,19,20,21,22,23,10,11
// Slot index maps CB id → its position in that sequence.
uint32_t cb_slot_index_from_id(uint32_t cb_id) {
    switch (cb_id) {
        case  0: return  0;  case  1: return  1;
        case  2: return  2;  case  3: return  3;
        case  4: return  4;  case  5: return  5;
        case 16: return  6;  case 17: return  7;
        case 18: return  8;  case 19: return  9;
        case 20: return 10;  case 21: return 11;
        case 22: return 12;  case 23: return 13;
        case 10: return 14;  case 11: return 15;
        default: TT_FATAL(false, "Unexpected CB ID {}", cb_id); return 0;
    }
}

uint32_t cb_l1_addr(uint32_t l1_base, uint32_t cb_id) {
    return l1_base + cb_slot_index_from_id(cb_id) * TILE_BYTES;
}

bool is_uint_str(const char* s) {
    if (!s || !*s) return false;
    for (const char* p = s; *p; ++p)
        if (*p < '0' || *p > '9') return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <direction:0|1> [N] [num_cores]\n";
        return 1;
    }
    uint32_t direction = (uint32_t)std::atoi(argv[1]);
    // Default N=16384 so that with 8 cores: local_half = 16384/(2*8) = 1024 = TILE_SIZE.
    // Rule: N must be >= 2 * num_cores * TILE_SIZE (2 * cores * 1024).
    // N=1024 only works with num_cores=1 giving local_half=512 which is STILL < TILE_SIZE.
    // Smallest valid N for any multicore run is 2*1*1024=2048 (1 core).
    uint32_t N = 16384;
    uint32_t user_cores_request = 0;

    for (int i = 2; i < argc; i++) {
        if (!is_uint_str(argv[i])) continue;
        uint32_t v = (uint32_t)std::stoul(argv[i]);
        if (v >= 2 && v <= 64 && (v & (v-1)) == 0)
            user_cores_request = v;
        else if (v >= 2 && (v & (v-1)) == 0)
            N = v;
        else
            std::cerr << "Warning: ignoring " << v << " (not power of 2)\n";
    }
    if (N < 2 || (N & (N-1))) { std::cerr << "N must be power of 2\n"; return 1; }

    int dev_id = 0;
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(dev_id);
    auto& cq  = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices().at(0);

    uint32_t max_request = (user_cores_request > 0) ? user_cores_request : 64u;
    uint32_t num_cores   = detect_available_cores(device, max_request, N);

    if (user_cores_request > 0 && num_cores < user_cores_request)
        std::cout << " WARNING: only " << num_cores << " cores available.\n";
    if (num_cores < 1) { std::cerr << "No usable cores.\n"; return 1; }

    uint32_t log2N      = 0; while ((1u<<log2N)      < N)         log2N++;
    uint32_t log2_cores = 0; while ((1u<<log2_cores) < num_cores) log2_cores++;
    uint32_t half_N     = N / 2;
    uint32_t local_half = half_N / num_cores;
    uint32_t local_tiles= (local_half + TILE_SIZE - 1) / TILE_SIZE;

    // detect_available_cores() already enforces local_half >= TILE_SIZE
    // by capping num_cores at N/(2*TILE_SIZE). No further check needed.
    uint32_t total_tiles= local_tiles * num_cores;
    uint32_t compact_bytes = half_N * sizeof(float);

    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " TT-Metal MULTICORE FFT (compact twiddles)\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << " N           : " << N           << "\n";
    std::cout << " log2N       : " << log2N       << "\n";
    std::cout << " Direction   : " << (direction?"Inverse":"Forward") << "\n";
    std::cout << " num_cores   : " << num_cores   << "\n";
    std::cout << " log2_cores  : " << log2_cores  << "\n";
    std::cout << " local_half  : " << local_half  << " elements/core\n";
    std::cout << " local_tiles : " << local_tiles << " tiles/core\n";
    std::cout << " Cross stages: 0 .. " << (log2_cores > 0 ? log2_cores-1 : 0) << "\n";
    std::cout << " Local stages: " << log2_cores << " .. " << log2N-1 << "\n";
    std::cout << " DRAM upload : "
              << (4*total_tiles*TILE_BYTES + 2*compact_bytes) / 1024 << " KB\n";
    std::cout << " DRAM dl     : "
              << (4*total_tiles*TILE_BYTES) / 1024 << " KB\n";
    std::cout << "════════════════════════════════════════════════\n";

    std::vector<float> ir(N,0.f), ii(N,0.f);
    for (uint32_t i = 0; i < N; i++)
        ir[i] = std::sin(2.f*PI*4.f*i/N) + 0.5f*std::sin(2.f*PI*8.f*i/N);

    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, direction==1);

    std::vector<uint32_t> even_r_t, even_i_t, odd_r_t, odd_i_t;
    prepare_stage0(ir, ii, N, log2N, total_tiles,
                   even_r_t, even_i_t, odd_r_t, odd_i_t);

    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N, direction);

    Program prog = CreateProgram();
    CoreRange core_range({0,0}, {num_cores-1, 0});

    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile{
        .page_size = TILE_BYTES, .buffer_type = BufferType::DRAM };
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
        .page_size=compact_bytes, .buffer_type=BufferType::DRAM };
    ReplicatedBufferConfig rc_cmp{.size=compact_bytes};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // ── Create CBs — order determines L1 layout ───────────────────────
    // Must match cb_slot_index_from_id() above.
    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        create_cb(prog, cc,  0, 1, TILE_BYTES); // slot 0  even_r
        create_cb(prog, cc,  1, 1, TILE_BYTES); // slot 1  even_i
        create_cb(prog, cc,  2, 1, TILE_BYTES); // slot 2  odd_r
        create_cb(prog, cc,  3, 1, TILE_BYTES); // slot 3  odd_i
        create_cb(prog, cc,  4, 1, TILE_BYTES); // slot 4  tw_r
        create_cb(prog, cc,  5, 1, TILE_BYTES); // slot 5  tw_i
        create_cb(prog, cc, 16, 1, TILE_BYTES); // slot 6  out0_r
        create_cb(prog, cc, 17, 1, TILE_BYTES); // slot 7  out0_i
        create_cb(prog, cc, 18, 1, TILE_BYTES); // slot 8  out1_r
        create_cb(prog, cc, 19, 1, TILE_BYTES); // slot 9  out1_i
        create_cb(prog, cc, 20, 1, TILE_BYTES); // slot 10 tmp0
        create_cb(prog, cc, 21, 1, TILE_BYTES); // slot 11 tmp1
        create_cb(prog, cc, 22, 1, TILE_BYTES); // slot 12 tw_odd_r
        create_cb(prog, cc, 23, 1, TILE_BYTES); // slot 13 tw_odd_i
        create_cb(prog, cc, 10, 1, TILE_BYTES); // slot 14 compact_r
        create_cb(prog, cc, 11, 1, TILE_BYTES); // slot 15 compact_i
    }

    KernelHandle reader_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/reader_fft_f32.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default });

    KernelHandle writer_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/writer_fft_f32.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default });

    KernelHandle compute_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/compute/fft_compute_f32.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false });

    // ── L1 base address for program-local CBs ─────────────────────────
    // All cores share the same L1 layout because they have identical CB
    // creation sequences.  The base is the same on every Tensix core.
    uint32_t l1_base =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Sync flag region: sits just after the 16 CBs in L1.
    // Each stage s uses a 4-byte flag at l1_sync_base + s*4.
    // noc_semaphore_inc / noc_semaphore_wait operate on this address.
    constexpr uint32_t NUM_CBS      = 16;
    const     uint32_t l1_sync_base = l1_base + NUM_CBS * TILE_BYTES;

    std::cout << " l1_base = 0x" << std::hex << l1_base << std::dec << "\n";
    // Debug: print CB 0-3 addresses so you can verify they match
    // get_write_ptr() values printed from inside the kernels.
    for (uint32_t id : {0u, 1u, 2u, 3u})
        std::cout << " CB " << id << " addr = 0x"
                  << std::hex << cb_l1_addr(l1_base, id) << std::dec << "\n";

    // ── Per-core runtime args ─────────────────────────────────────────
    const uint32_t tile_bytes = TILE_BYTES;
    constexpr uint32_t ELEM  = sizeof(float);

    for (uint32_t c = 0; c < num_cores; c++) {
        CoreCoord cc = {c, 0};
        uint32_t tile_offset = c * local_tiles;

        std::vector<uint32_t> reader_args = {
            b_er->address(), b_ei->address(),
            b_or->address(), b_oi->address(),
            b_cmp_r->address(), b_cmp_i->address(),
            local_tiles, tile_offset, log2N, half_N, local_half
        };

        std::vector<uint32_t> compute_args = { log2N, local_tiles };

        const uint32_t core_elem_base = tile_offset * (tile_bytes / ELEM);
        std::vector<uint32_t> writer_args = {
            b_o0r->address(), b_o0i->address(),
            b_o1r->address(), b_o1i->address(),
            local_tiles, log2N, local_half,
            half_N, num_cores, c, log2_cores, tile_offset,
            core_elem_base   // arg 12: global element base for local shuffle
        };

        std::vector<uint32_t> cx_noc_x(log2_cores), cx_noc_y(log2_cores);
        std::vector<uint32_t> cx_er(log2_cores), cx_ei(log2_cores);
        std::vector<uint32_t> cx_or(log2_cores), cx_oi(log2_cores);

        std::vector<uint32_t> cx_p_sem(log2_cores);  // partner semaphore L1 addr
        std::vector<uint32_t> cx_my_sem(log2_cores); // my semaphore L1 addr

        for (uint32_t s = 0; s < log2_cores; s++) {
            uint32_t partner_id = c ^ (num_cores >> (s + 1));

            CoreCoord partner_physical =
                device->worker_core_from_logical_core({partner_id, 0});
            cx_noc_x[s] = partner_physical.x;
            cx_noc_y[s] = partner_physical.y;

            // CB addresses — same on all cores (identical layout)
            cx_er[s] = cb_l1_addr(l1_base, 0);
            cx_ei[s] = cb_l1_addr(l1_base, 1);
            cx_or[s] = cb_l1_addr(l1_base, 2);
            cx_oi[s] = cb_l1_addr(l1_base, 3);

            // Sync flag addresses in L1 scratch region.
            // Same l1_sync_base on every core (identical L1 layout).
            // Flag for stage s is at l1_sync_base + s * sizeof(uint32_t).
            cx_p_sem[s]  = l1_sync_base + s * sizeof(uint32_t);
            cx_my_sem[s] = l1_sync_base + s * sizeof(uint32_t);
        }

        for (auto v : cx_noc_x)  writer_args.push_back(v);
        for (auto v : cx_noc_y)  writer_args.push_back(v);
        for (auto v : cx_er)     writer_args.push_back(v);
        for (auto v : cx_ei)     writer_args.push_back(v);
        for (auto v : cx_or)     writer_args.push_back(v);
        for (auto v : cx_oi)     writer_args.push_back(v);
        for (auto v : cx_p_sem)  writer_args.push_back(v);
        for (auto v : cx_my_sem) writer_args.push_back(v);

        SetRuntimeArgs(prog, reader_k,  cc, reader_args);
        SetRuntimeArgs(prog, writer_k,  cc, writer_args);
        SetRuntimeArgs(prog, compute_k, cc, compute_args);
    }

    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_er,    even_r_t, false);
    EnqueueWriteMeshBuffer(cq, b_ei,    even_i_t, false);
    EnqueueWriteMeshBuffer(cq, b_or,    odd_r_t,  false);
    EnqueueWriteMeshBuffer(cq, b_oi,    odd_i_t,  false);
    EnqueueWriteMeshBuffer(cq, b_cmp_r, cmp_r_t,  false);
    EnqueueWriteMeshBuffer(cq, b_cmp_i, cmp_i_t,  false);
    Finish(cq);

    std::cout << "Launching multicore FFT ("
              << num_cores << " cores, " << log2N << " stages)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

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
        result_r[i]        = o0r[i]; result_i[i]        = o0i[i];
        result_r[i+half_N] = o1r[i]; result_i[i+half_N] = o1i[i];
    }
    if (direction==1)
        for (uint32_t i=0;i<N;i++){ result_r[i]/=N; result_i[i]/=N; }

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << " VALIDATION\n";
    std::cout << "════════════════════════════════════════════════\n";
    float mer=0.f, mei=0.f, me=0.f;
    for (uint32_t i=0;i<N;i++) {
        float er = std::abs(result_r[i]-ref_r[i]);
        float ei = std::abs(result_i[i]-ref_i[i]);
        mer=std::max(mer,er); mei=std::max(mei,ei); me+=er+ei;
    }
    me /= 2*N;
    std::cout << " Max error (real): " << mer << "\n";
    std::cout << " Max error (imag): " << mei << "\n";
    std::cout << " Mean error      : " << me  << "\n";
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