// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_host.cpp — multi-core radix-2 DIT FFT on Wormhole.
//
// Constraints:
//   * N is a power of two, 2 <= N <= 65536.
//   * FFT only (no IFFT).
//   * 2D row-major core grid; P = max(1, N/1024) cores.
//       N <= 1024  => P=1  (1x1)   pure single-core path, as before.
//       N = 2048   => P=2  (2x1)
//       N = 4096   => P=4  (4x1)
//       N = 8192   => P=8  (8x1)
//       N = 16384  => P=16 (8x2)
//       N = 32768  => P=32 (8x4)
//       N = 65536  => P=64 (8x8)   full Wormhole worker grid.
//     Logical core index c (0..P-1) is row-major: x=c%cols, y=c/cols.
//     The kernels only use `c` + a NoC-coord lookup table, so the grid
//     shape is purely a host-side decision.
//
// Flow (see kernel/fft_common.h for details):
//   1. pack_input does a global N-bit bit-reversal and packs the result into
//      P contiguous 1024-element tiles (real+imag split). Tile c holds the
//      chunk of bit-reversed input that core c owns as initial state.
//   2. precompute_twiddles builds LOG2N * P tiles of twiddle factors:
//        - For local stages (s < 10 or s < LOG2N if P==1) all P tiles at
//          stage s are identical — the twiddles depend only on position
//          within the tile.
//        - For cross-core stages (s >= 10) each core's tile differs; the
//          twiddle index at slot j is
//            k = ((c & ~(1<<(s-10))) & ((1<<(s-9))-1)) * 1024 + j
//          where c is the core index; both cores in a pair use the same k
//          (the "lower" core of the pair).
//   3. run_fft uploads input+twiddles, launches P-core program, reads back.

#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include <cmath>
#include <vector>
#include <cassert>
#include <complex>
#include <utility>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace fft_example {

constexpr uint32_t kTileHW       = 32;
constexpr uint32_t kTileElems    = kTileHW * kTileHW;            // 1024
constexpr uint32_t kTileSizeFp32 = kTileElems * sizeof(float);   // 4096 bytes

// CB layout — must match kernel/fft_common.h
constexpr uint32_t CB_EVEN_R   = 0;
constexpr uint32_t CB_EVEN_I   = 1;
constexpr uint32_t CB_ODD_R    = 2;
constexpr uint32_t CB_ODD_I    = 3;
constexpr uint32_t CB_TW_R     = 4;
constexpr uint32_t CB_TW_I     = 5;
constexpr uint32_t CB_OUT0_R   = 6;
constexpr uint32_t CB_OUT0_I   = 7;
constexpr uint32_t CB_OUT1_R   = 8;
constexpr uint32_t CB_OUT1_I   = 9;
constexpr uint32_t CB_TMP_R    = 10;
constexpr uint32_t CB_TMP_I    = 11;
constexpr uint32_t CB_TW_ODD_R = 12;
constexpr uint32_t CB_TW_ODD_I = 13;
constexpr uint32_t CB_STATE_R  = 14;
constexpr uint32_t CB_STATE_I  = 15;
constexpr uint32_t CB_SYNC     = 16;
constexpr uint32_t CB_RECV_R   = 17;
constexpr uint32_t CB_RECV_I   = 18;
constexpr uint32_t NUM_CBS     = 19;

struct FFTConfig { uint32_t N; };

// ── Helpers ───────────────────────────────────────────────────────────────

inline std::shared_ptr<MeshBuffer> make_mesh_buf(
    std::shared_ptr<MeshDevice> md, uint32_t size, uint32_t page_size)
{
    ReplicatedBufferConfig rep{.size = size};
    DeviceLocalBufferConfig dev{.page_size = page_size, .buffer_type = BufferType::DRAM};
    return MeshBuffer::create(rep, dev, md.get());
}

inline uint32_t buf_addr(const std::shared_ptr<MeshBuffer>& mb) {
    return mb->get_device_buffer(MeshCoordinate(0, 0))->address();
}

inline uint32_t log2u(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) ++r;
    return r;
}

inline uint32_t bit_rev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) { r = (r << 1) | (x & 1u); x >>= 1u; }
    return r;
}

// ── Per-FFT sizing ────────────────────────────────────────────────────────

struct Sizing {
    uint32_t N;
    uint32_t log2N;
    uint32_t P;              // num cores
    uint32_t log2P;
    uint32_t log2N_local;    // local butterfly stages per core (<=10)
    uint32_t total_tiles;    // tiles across all cores (== P)
    uint32_t grid_cols;      // 2D grid dimensions, row-major
    uint32_t grid_rows;
};

inline Sizing compute_sizing(uint32_t N) {
    Sizing z{};
    z.N     = N;
    z.log2N = log2u(N);
    z.P     = (N <= kTileElems) ? 1u : (N / kTileElems);
    z.log2P = log2u(z.P);
    z.log2N_local = z.log2N - z.log2P;
    z.total_tiles = z.P;

    // 2D row-major grid. Up to 8 wide (Wormhole worker row), then stack rows.
    // All P values are powers of two so the grid is always rectangular.
    z.grid_cols = (z.P < 8u) ? z.P : 8u;
    z.grid_rows = z.P / z.grid_cols;
    return z;
}

// Map logical core index c (0..P-1) to its logical CoreCoord in the 2D grid.
inline CoreCoord logical_core_for(uint32_t c, const Sizing& z) {
    return CoreCoord{c % z.grid_cols, c / z.grid_cols};
}

// Pack an N-point complex signal into P contiguous bit-reversed tiles.
// Returns (real_all, imag_all) each of size P*TILE_ELEMS.
// When N < TILE_ELEMS (P=1), the tile is zero-padded past slot N.
inline std::pair<std::vector<float>, std::vector<float>> pack_input(
    const std::vector<std::complex<float>>& x, const Sizing& z)
{
    assert(static_cast<uint32_t>(x.size()) == z.N);
    std::vector<float> r(z.P * kTileElems, 0.0f), i(z.P * kTileElems, 0.0f);
    for (uint32_t k = 0; k < z.N; ++k) {
        const uint32_t src  = bit_rev(k, z.log2N);
        const uint32_t tile = k / kTileElems;
        const uint32_t slot = k % kTileElems;
        r[tile * kTileElems + slot] = x[src].real();
        i[tile * kTileElems + slot] = x[src].imag();
    }
    return {std::move(r), std::move(i)};
}

inline std::vector<std::complex<float>> unpack_output(
    const std::vector<float>& r, const std::vector<float>& i, const Sizing& z)
{
    std::vector<std::complex<float>> out(z.N);
    for (uint32_t k = 0; k < z.N; ++k) {
        const uint32_t tile = k / kTileElems;
        const uint32_t slot = k % kTileElems;
        out[k] = {r[tile * kTileElems + slot], i[tile * kTileElems + slot]};
    }
    return out;
}

// Build LOG2N * P twiddle tiles (one per (stage, core)).
// Layout: flat buffer of log2N * P * TILE_ELEMS floats; the tile for
// (stage s, core c) starts at offset (s * P + c) * TILE_ELEMS.
inline std::pair<std::vector<float>, std::vector<float>> precompute_twiddles(
    const Sizing& z)
{
    const uint32_t n_stages = z.log2N;
    const size_t   tiles    = static_cast<size_t>(n_stages) * z.P;
    std::vector<float> r(tiles * kTileElems, 0.0f), i(tiles * kTileElems, 0.0f);

    for (uint32_t s = 0; s < n_stages; ++s) {
        const double M = static_cast<double>(1u << (s + 1));
        const uint32_t stride_mask = (1u << s) - 1u;  // for local stages

        for (uint32_t c = 0; c < z.P; ++c) {
            float* tile_r = r.data() + (static_cast<size_t>(s) * z.P + c) * kTileElems;
            float* tile_i = i.data() + (static_cast<size_t>(s) * z.P + c) * kTileElems;

            if (s < z.log2N_local) {
                // Local stage: twiddle index depends only on slot position
                // within the tile (pair index p). All cores get the same
                // tile at this stage.
                const uint32_t num_pairs = (z.P == 1) ? (z.N / 2) : (kTileElems / 2);
                for (uint32_t p = 0; p < num_pairs; ++p) {
                    const uint32_t k     = p & stride_mask;
                    const double   angle = -2.0 * M_PI * static_cast<double>(k) / M;
                    tile_r[p] = static_cast<float>(std::cos(angle));
                    tile_i[p] = static_cast<float>(std::sin(angle));
                }
            } else {
                // Cross-core stage: twiddle index per slot j depends on
                // the lower core of the pair at this stage.
                const uint32_t kshift   = s - z.log2N_local;     // s - 10 if P>1
                const uint32_t low_mask = ~(1u << kshift);
                const uint32_t grp_mask = (1u << (kshift + 1)) - 1u;
                const uint32_t c_low    = c & low_mask;
                const uint32_t c_in_grp = c_low & grp_mask;
                const uint32_t k_base   = c_in_grp * kTileElems;
                for (uint32_t j = 0; j < kTileElems; ++j) {
                    const uint32_t k     = k_base + j;
                    const double   angle = -2.0 * M_PI * static_cast<double>(k) / M;
                    tile_r[j] = static_cast<float>(std::cos(angle));
                    tile_i[j] = static_cast<float>(std::sin(angle));
                }
            }
        }
    }
    return {std::move(r), std::move(i)};
}

// ── Launch ────────────────────────────────────────────────────────────────

inline void run_fft(
    std::shared_ptr<MeshDevice> md,
    const FFTConfig& cfg,
    std::shared_ptr<MeshBuffer> in_r_buf,     // P tiles
    std::shared_ptr<MeshBuffer> in_i_buf,     // P tiles
    std::shared_ptr<MeshBuffer> out_r_buf,    // P tiles
    std::shared_ptr<MeshBuffer> out_i_buf)    // P tiles
{
    assert(cfg.N >= 2);
    assert((cfg.N & (cfg.N - 1)) == 0);

    const Sizing z = compute_sizing(cfg.N);
    assert(z.P <= 64 && "This example is capped at P=64 (8x8 Wormhole grid). "
                        "For larger N you need multi-tile state per core or a "
                        "multi-device mesh.");

    std::printf(
        "[run_fft] N=%u  P=%u  grid=%ux%u  local_stages=%u  cross_core_stages=%u\n",
        z.N, z.P, z.grid_cols, z.grid_rows, z.log2N_local, z.log2P);

    MeshCommandQueue& cq = md->mesh_command_queue();

    // --- twiddle tables in DRAM --------------------------------------------
    auto [tw_r_data, tw_i_data] = precompute_twiddles(z);
    const uint32_t tw_total_bytes =
        static_cast<uint32_t>(tw_r_data.size() * sizeof(float));
    auto tw_r_buf = make_mesh_buf(md, tw_total_bytes, kTileSizeFp32);
    auto tw_i_buf = make_mesh_buf(md, tw_total_bytes, kTileSizeFp32);
    WriteShard(cq, tw_r_buf, tw_r_data, MeshCoordinate(0, 0), false);
    WriteShard(cq, tw_i_buf, tw_i_data, MeshCoordinate(0, 0), false);

    // --- program & CBs -----------------------------------------------------
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{z.grid_cols - 1, z.grid_rows - 1};
    const CoreRange cr(first, last);

    // Pipelined CBs: EVEN/ODD/TW/OUT get 2 tiles, scratch/state/sync/recv 1.
    // Indices:                        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
    constexpr uint32_t kCbTiles[19] = {2,2,2,2,2,2,2,2,2,2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static_assert(sizeof(kCbTiles) / sizeof(kCbTiles[0]) == NUM_CBS);

    for (uint32_t id = 0; id < NUM_CBS; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    // --- semaphore for cross-core exchange ---------------------------------
    // One semaphore on the whole range. Each core uses it to signal its
    // partner that a state tile has landed. Initial value 0.
    const uint32_t sem_id = CreateSemaphore(prog, cr, 0);

    // --- kernels -----------------------------------------------------------
    auto rk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/fft_reader.cpp",
        cr,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = {z.N, z.log2N, z.P, z.log2P}});

    auto wk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/fft_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    // Unpack every CB directly to DEST in fp32 (see compute kernel notes).
    constexpr uint32_t kNumCbSlots = 32;
    std::vector<UnpackToDestMode> u2d(kNumCbSlots, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < NUM_CBS; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/fft_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
            .compile_args        = {z.log2N}});

    // --- runtime args per core --------------------------------------------
    // Build the NoC-coord lookup table once; all cores get the same table,
    // just parameterised by their own `my_core` index c = 0..P-1 (row-major).
    std::vector<uint32_t> noc_xy;
    noc_xy.reserve(2 * z.P);
    for (uint32_t c = 0; c < z.P; ++c) {
        const CoreCoord logical  = logical_core_for(c, z);
        const CoreCoord physical = md->worker_core_from_logical_core(logical);
        noc_xy.push_back(physical.x);
        noc_xy.push_back(physical.y);
    }

    for (uint32_t c = 0; c < z.P; ++c) {
        const CoreCoord core = logical_core_for(c, z);

        std::vector<uint32_t> reader_args = {
            buf_addr(in_r_buf),  buf_addr(in_i_buf),
            buf_addr(tw_r_buf),  buf_addr(tw_i_buf),
            c,                    // my_core (logical index)
            sem_id,               // semaphore id
        };
        reader_args.insert(reader_args.end(), noc_xy.begin(), noc_xy.end());
        SetRuntimeArgs(prog, rk, core, reader_args);

        SetRuntimeArgs(prog, wk, core, {
            buf_addr(out_r_buf), buf_addr(out_i_buf), c,
        });
    }

    MeshWorkload workload;
    workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    cq.finish();
}

// Public helper for allocating the input/output DRAM buffers at the right
// size for a given N — keeps the test code oblivious to the P sharding.
inline std::shared_ptr<MeshBuffer> make_io_buf(
    std::shared_ptr<MeshDevice> md, uint32_t N)
{
    const Sizing z = compute_sizing(N);
    return make_mesh_buf(md, z.P * kTileSizeFp32, kTileSizeFp32);
}

// ── Persistent plan: build once, execute many times ───────────────────────
//
// A plan captures every device-side resource that depends only on N:
//   * twiddle buffers (precomputed, uploaded once)
//   * input / output DRAM buffers (addresses are stable, so runtime args
//     stay valid across calls)
//   * the compiled Program wrapped in a MeshWorkload (runtime args bound
//     once to the persistent buffer addresses)
//
// On repeated calls with the same N:
//   per-call cost  = WriteShard(input) + EnqueueMeshWorkload + ReadShard
// instead of
//   per-call cost  = above + program build + CBs + kernels + SetRuntimeArgs
// which at N=65536 is the difference between ~25 ms and ~2–3 ms.

struct FFTPlan {
    uint32_t N = 0;
    Sizing   z{};
    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<MeshBuffer> in_r_buf, in_i_buf;
    std::shared_ptr<MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<MeshBuffer> tw_r_buf, tw_i_buf;
    MeshWorkload workload;
    bool initialized = false;
};

inline std::shared_ptr<FFTPlan> make_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N)
{
    auto plan = std::make_shared<FFTPlan>();
    plan->md  = md;
    plan->N   = N;
    plan->z   = compute_sizing(N);
    const Sizing& z = plan->z;

    assert(z.P <= 64);

    MeshCommandQueue& cq = md->mesh_command_queue();

    // Twiddles — precompute once, upload once. Stay resident for plan's life.
    auto [tw_r_data, tw_i_data] = precompute_twiddles(z);
    const uint32_t tw_total_bytes =
        static_cast<uint32_t>(tw_r_data.size() * sizeof(float));
    plan->tw_r_buf = make_mesh_buf(md, tw_total_bytes, kTileSizeFp32);
    plan->tw_i_buf = make_mesh_buf(md, tw_total_bytes, kTileSizeFp32);
    WriteShard(cq, plan->tw_r_buf, tw_r_data, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan->tw_i_buf, tw_i_data, MeshCoordinate(0, 0), false);

    // Persistent I/O buffers. Their DRAM addresses stay the same across
    // calls, so runtime args baked into the program remain valid.
    plan->in_r_buf  = make_io_buf(md, N);
    plan->in_i_buf  = make_io_buf(md, N);
    plan->out_r_buf = make_io_buf(md, N);
    plan->out_i_buf = make_io_buf(md, N);

    // Build program.
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{z.grid_cols - 1, z.grid_rows - 1};
    const CoreRange cr(first, last);

    constexpr uint32_t kCbTiles[19] = {2,2,2,2,2,2,2,2,2,2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static_assert(sizeof(kCbTiles) / sizeof(kCbTiles[0]) == NUM_CBS);

    for (uint32_t id = 0; id < NUM_CBS; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    const uint32_t sem_id = CreateSemaphore(prog, cr, 0);

    auto rk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/fft_reader.cpp",
        cr,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = {z.N, z.log2N, z.P, z.log2P}});

    auto wk = CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/fft_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    constexpr uint32_t kNumCbSlots = 32;
    std::vector<UnpackToDestMode> u2d(kNumCbSlots, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < NUM_CBS; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    CreateKernel(
        prog,
        "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/fft_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
            .compile_args        = {z.log2N}});

    std::vector<uint32_t> noc_xy;
    noc_xy.reserve(2 * z.P);
    for (uint32_t c = 0; c < z.P; ++c) {
        const CoreCoord logical  = logical_core_for(c, z);
        const CoreCoord physical = md->worker_core_from_logical_core(logical);
        noc_xy.push_back(physical.x);
        noc_xy.push_back(physical.y);
    }

    for (uint32_t c = 0; c < z.P; ++c) {
        const CoreCoord core = logical_core_for(c, z);

        std::vector<uint32_t> reader_args = {
            buf_addr(plan->in_r_buf),  buf_addr(plan->in_i_buf),
            buf_addr(plan->tw_r_buf),  buf_addr(plan->tw_i_buf),
            c, sem_id,
        };
        reader_args.insert(reader_args.end(), noc_xy.begin(), noc_xy.end());
        SetRuntimeArgs(prog, rk, core, reader_args);

        SetRuntimeArgs(prog, wk, core, {
            buf_addr(plan->out_r_buf), buf_addr(plan->out_i_buf), c,
        });
    }

    plan->workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));

    plan->initialized = true;
    return plan;
}

// Fast-path: execute a built plan against a new signal.
inline std::vector<std::complex<float>> execute(
    FFTPlan& plan,
    const std::vector<std::complex<float>>& signal)
{
    assert(plan.initialized);
    assert(static_cast<uint32_t>(signal.size()) == plan.N);

    MeshCommandQueue& cq = plan.md->mesh_command_queue();

    auto [in_r, in_i] = pack_input(signal, plan.z);

    WriteShard(cq, plan.in_r_buf, in_r, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan.in_i_buf, in_i, MeshCoordinate(0, 0), false);

    EnqueueMeshWorkload(cq, plan.workload, false);

    std::vector<float> out_r, out_i;
    ReadShard(cq, out_r, plan.out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i, plan.out_i_buf, MeshCoordinate(0, 0), true);

    return unpack_output(out_r, out_i, plan.z);
}

// Internal per-device plan cache keyed by N. Keeps plans alive for the life
// of the process; if you need to reclaim memory call `clear_plan_cache()`.
namespace detail {
inline std::unordered_map<uint64_t, std::shared_ptr<FFTPlan>>& plan_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<FFTPlan>> c;
    return c;
}
inline uint64_t plan_key(MeshDevice* md, uint32_t N) {
    return (reinterpret_cast<uint64_t>(md) ^ (uint64_t{N} * 0x9E3779B97F4A7C15ull));
}
}  // namespace detail

inline std::shared_ptr<FFTPlan> get_cached_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N)
{
    const uint64_t key = detail::plan_key(md.get(), N);
    auto& cache = detail::plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto plan = make_plan(md, N);
    cache.emplace(key, plan);
    return plan;
}

inline void clear_plan_cache() { detail::plan_cache().clear(); }

// ── PyTorch-style one-shot FFT API (now backed by the plan cache) ────────
//
// First call for a given N builds the plan (~10–25 ms depending on N).
// Subsequent calls with the same N skip all device-side setup and only pay
// for input upload + enqueue + output download (~1–3 ms typically).
//
// Usage (C++ equivalent of `torch.fft.fft(signal)`):
//
//     std::vector<std::complex<float>> signal = { {10,0}, {20,0}, {30,0}, {40,0} };
//     auto spectrum = fft_example::fft(md, signal);

inline std::vector<std::complex<float>> fft(
    std::shared_ptr<MeshDevice> md,
    const std::vector<std::complex<float>>& signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 2 && "FFT requires N >= 2");
    assert((N & (N - 1)) == 0 && "FFT requires N to be a power of two");

    auto plan = get_cached_plan(md, N);
    return execute(*plan, signal);
}

// Overload for real-valued input. Imaginary part is treated as 0.
inline std::vector<std::complex<float>> fft(
    std::shared_ptr<MeshDevice> md,
    const std::vector<float>& signal)
{
    std::vector<std::complex<float>> cx(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) cx[i] = {signal[i], 0.0f};
    return fft(md, cx);
}

}  // namespace fft_example
