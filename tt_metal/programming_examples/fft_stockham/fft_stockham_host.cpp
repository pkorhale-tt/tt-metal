// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_stockham_host.cpp — Multi-pass Stockham (six-step / Bailey 4-step) FFT
//                        orchestrator that lifts our radix-2 single-shot FFT
//                        from N <= 65,536 to N up to ~1M points (and beyond).
//
// Strategy:
//   * Factor N as N1 * N2 with both ≤ 65,536 (and both powers of two).
//     For N ≤ 1M (our current Stockham regime) this also gives both ≤ 1024,
//     so every sub-FFT fits in one Tensix tile.
//   * Reshape input as (N1, N2) row-major.
//   * Pass 1: row-FFT of length N2  (N1 sub-FFTs, all of length N2).
//             Dispatched in ONE batched kernel launch via `batch_fft`:
//             64 cores each run N1/64 sub-FFTs in parallel (Optimisation 1).
//   * Pass 2: per-element twiddle multiply  W_N^(i*j)  +  transpose to (N2, N1).
//             Done on the HOST today — kept on host for now; a future
//             optimisation is a dedicated BRISC-only device kernel that
//             collapses the ~50–80 ms host work into <5 ms on-chip.
//   * Pass 3: row-FFT of length N1  (N2 sub-FFTs of length N1) — also
//             one batched dispatch.
//   * Final reorder on host: X[k] = D[k % N2, k / N2].
//
// Total DRAM round-trips: 2 (one per pass), not one per stage. Each sub-FFT
// stays fully L1-resident inside the batch kernel.
//
// Public API (mirrors fft_example::fft):
//
//     auto X = fft_stockham::fft(md, signal);   // 1D power-of-two of any size
//                                                // (capped by host memory).
//
// For N <= 65,536 we transparently fall back to the inner radix-2 path so
// callers never need to know which algorithm ran.

#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include "../fft/fft_host.cpp"   // reuse the inner radix-2 kernel & plan cache

#include <cmath>
#include <complex>
#include <vector>
#include <utility>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <memory>
#include <unordered_map>

// fft/fft_host.cpp does `using namespace tt::tt_metal::distributed;` at file
// scope, so MeshDevice and friends are visible here without further qualification.

namespace fft_stockham {

using Complex = std::complex<float>;
using fft_example::log2u;
using fft_example::bit_rev;
using fft_example::kTileElems;
using fft_example::kTileSizeFp32;
using fft_example::make_mesh_buf;
using fft_example::buf_addr;
using tt::tt_metal::distributed::MeshDevice;

// ── Sizing & factorisation ────────────────────────────────────────────────

// Maximum N a single inner radix-2 dispatch can handle.
constexpr uint32_t kInnerMaxN = 65536u;

// Power-of-two check.
inline bool is_pow2(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

struct StockhamPlan {
    uint32_t N        = 0;
    uint32_t N1       = 0;     // outer (column-FFT) dimension
    uint32_t N2       = 0;     // inner (row-FFT)    dimension
    bool     stockham = false; // false => fall through to inner radix-2
};

// Choose a balanced factorisation N = N1 * N2 such that both halves fit in
// the inner radix-2 kernel. Strategy: pick N2 = sqrt(N) rounded to the next
// power of two; clamp to kInnerMaxN. This keeps both passes well L1-resident.
inline StockhamPlan plan(uint32_t N) {
    StockhamPlan p{};
    p.N = N;

    if (N <= kInnerMaxN) { p.stockham = false; p.N1 = N; p.N2 = 1; return p; }

    assert(is_pow2(N) && "Stockham path requires N to be a power of two.");

    // log2N is the total number of butterfly stages.
    const uint32_t log2N = log2u(N);

    // Split log2N as evenly as possible, then clamp each half to fit the
    // inner kernel (at most log2(kInnerMaxN) = 16 bits per pass).
    uint32_t log2N2 = log2N / 2;             // inner / row-FFT length
    uint32_t log2N1 = log2N - log2N2;        // outer / column-FFT length
    const uint32_t log2_inner_max = log2u(kInnerMaxN);
    if (log2N1 > log2_inner_max) {
        const uint32_t spill = log2N1 - log2_inner_max;
        log2N1 -= spill;
        log2N2 += spill;
    }
    if (log2N2 > log2_inner_max) {
        const uint32_t spill = log2N2 - log2_inner_max;
        log2N2 -= spill;
        log2N1 += spill;
    }

    p.N1 = 1u << log2N1;
    p.N2 = 1u << log2N2;
    p.stockham = true;

    assert(p.N1 <= kInnerMaxN && p.N2 <= kInnerMaxN);
    assert(static_cast<uint64_t>(p.N1) * static_cast<uint64_t>(p.N2) ==
           static_cast<uint64_t>(p.N));
    return p;
}

// ╔════════════════════════════════════════════════════════════════════════╗
// ║                Optimisation 1 — device-side BATCH FFT                  ║
// ╚════════════════════════════════════════════════════════════════════════╝
//
// One device dispatch executes `batch` independent FFTs of length sub_N
// (sub_N <= 1024, so each sub-FFT fits in one Tensix tile = no cross-core
// stages). We use 64 cores and assign batch_per_core = batch / 64 sub-FFTs
// to each core. This collapses the per-sub-FFT host overhead (program
// build, runtime args, enqueue, finish) that today's host loop pays N1 (or
// N2) times.
//
// Coverage. Stockham's kInnerMaxN=65536 split keeps both N1 and N2 ≤ 1024
// for every N up to 1,048,576 (the regime our orchestrator handles), so
// every sub-FFT in pass-1 and pass-3 is single-tile and goes through the
// batch path. Larger N is unchanged (still asserts in plan()).
//
// DRAM layout (one tile = 1024 fp32 = 4096 bytes per side, real and imag
// in separate buffers):
//   in_r_buf[t]   = bit-reversed real of sub-FFT t   (t in [0, batch))
//   in_i_buf[t]   = bit-reversed imag of sub-FFT t
//   out_r_buf[t]  = real spectrum of sub-FFT t       (natural order)
//   out_i_buf[t]  = imag spectrum of sub-FFT t
// Every core c handles tiles [c*batch_per_core, (c+1)*batch_per_core).
//
// Twiddles (LOG2_SUB_N tiles per side, shared across cores — local stages
// only depend on stage index s):
//   tw_r_buf[s]   = cos(-2*pi * (p mod 2^s) / 2^(s+1))   p = 0..sub_N/2-1
//   tw_i_buf[s]   = sin(-2*pi * (p mod 2^s) / 2^(s+1))

struct BatchFFTPlan {
    uint32_t sub_N          = 0;
    uint32_t log2_sub_N     = 0;
    uint32_t batch          = 0;     // total number of sub-FFTs
    uint32_t num_cores      = 0;     // <= 64
    uint32_t batch_per_core = 0;     // batch / num_cores  (must divide cleanly)
    uint32_t grid_cols      = 0;
    uint32_t grid_rows      = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<MeshBuffer> in_r_buf,  in_i_buf;
    std::shared_ptr<MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<MeshBuffer> tw_r_buf,  tw_i_buf;
    tt::tt_metal::distributed::MeshWorkload workload;
    bool initialized = false;
};

inline std::pair<uint32_t, uint32_t> pick_batch_grid(uint32_t num_cores) {
    const uint32_t cols = (num_cores < 8u) ? num_cores : 8u;
    const uint32_t rows = num_cores / cols;
    return {cols, rows};
}

inline tt::tt_metal::CoreCoord batch_logical_core(
    uint32_t c, uint32_t grid_cols)
{
    return tt::tt_metal::CoreCoord{c % grid_cols, c / grid_cols};
}

// LOG2_SUB_N tiles per side; tile s holds the stage-s twiddles for a
// single-tile (P=1) radix-2 sub-FFT of length sub_N. Identical to the
// inner kernel's local-stage twiddle layout.
inline std::pair<std::vector<float>, std::vector<float>> batch_twiddles(
    uint32_t sub_N, uint32_t log2_sub_N)
{
    const size_t total = static_cast<size_t>(log2_sub_N) * kTileElems;
    std::vector<float> r(total, 0.0f), i(total, 0.0f);
    const uint32_t num_pairs = sub_N / 2u;

    for (uint32_t s = 0; s < log2_sub_N; ++s) {
        const double M = static_cast<double>(1u << (s + 1));
        const uint32_t stride_mask = (1u << s) - 1u;
        float* tile_r = r.data() + static_cast<size_t>(s) * kTileElems;
        float* tile_i = i.data() + static_cast<size_t>(s) * kTileElems;
        for (uint32_t p = 0; p < num_pairs; ++p) {
            const uint32_t k     = p & stride_mask;
            const double   angle = -2.0 * M_PI * static_cast<double>(k) / M;
            tile_r[p] = static_cast<float>(std::cos(angle));
            tile_i[p] = static_cast<float>(std::sin(angle));
        }
    }
    return {std::move(r), std::move(i)};
}

inline std::shared_ptr<BatchFFTPlan> make_batch_plan(
    std::shared_ptr<MeshDevice> md, uint32_t sub_N, uint32_t batch)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto bp = std::make_shared<BatchFFTPlan>();
    bp->md          = md;
    bp->sub_N       = sub_N;
    bp->log2_sub_N  = log2u(sub_N);
    bp->batch       = batch;

    assert(sub_N <= kTileElems   && "batch path requires sub_N <= 1024 (single tile per sub-FFT)");
    assert(is_pow2(sub_N) && sub_N >= 2);
    assert(is_pow2(batch) && batch >= 1);

    bp->num_cores      = (batch < 64u) ? batch : 64u;
    bp->batch_per_core = batch / bp->num_cores;
    assert(bp->num_cores * bp->batch_per_core == batch);
    std::tie(bp->grid_cols, bp->grid_rows) = pick_batch_grid(bp->num_cores);

    MeshCommandQueue& cq = md->mesh_command_queue();

    // ── DRAM buffers ────────────────────────────────────────────────────
    const uint32_t io_bytes = batch * kTileSizeFp32;
    bp->in_r_buf  = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    bp->in_i_buf  = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    bp->out_r_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);
    bp->out_i_buf = make_mesh_buf(md, io_bytes, kTileSizeFp32);

    auto [tw_r_data, tw_i_data] = batch_twiddles(sub_N, bp->log2_sub_N);
    const uint32_t tw_bytes = static_cast<uint32_t>(tw_r_data.size() * sizeof(float));
    bp->tw_r_buf = make_mesh_buf(md, tw_bytes, kTileSizeFp32);
    bp->tw_i_buf = make_mesh_buf(md, tw_bytes, kTileSizeFp32);
    WriteShard(cq, bp->tw_r_buf, tw_r_data, MeshCoordinate(0, 0), false);
    WriteShard(cq, bp->tw_i_buf, tw_i_data, MeshCoordinate(0, 0), false);

    // ── Program ─────────────────────────────────────────────────────────
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{bp->grid_cols - 1, bp->grid_rows - 1};
    const CoreRange cr(first, last);

    // CB indices match batch_fft_common.h (17 CBs, no RECV used).
    constexpr uint32_t kBatchNumCbs = 17;
    constexpr uint32_t kCbTiles[kBatchNumCbs] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   // EVEN/ODD/TW/OUT — 2-tile pipelined
        1, 1, 1, 1,                     // TMP, TW_ODD
        1, 1,                           // STATE_R, STATE_I
        1                               // SYNC
    };

    for (uint32_t id = 0; id < kBatchNumCbs; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_stockham/kernel/batch_fft_reader.cpp",
        cr,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = {sub_N, bp->log2_sub_N}});

    auto wk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_stockham/kernel/batch_fft_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    constexpr uint32_t kNumCbSlots = 32;
    std::vector<UnpackToDestMode> u2d(kNumCbSlots, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kBatchNumCbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto ck = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_stockham/kernel/batch_fft_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
            .compile_args        = {bp->log2_sub_N}});

    for (uint32_t c = 0; c < bp->num_cores; ++c) {
        const CoreCoord  logical  = batch_logical_core(c, bp->grid_cols);
        const CoreCoord  physical = md->worker_core_from_logical_core(logical);
        const uint32_t   base     = c * bp->batch_per_core;

        SetRuntimeArgs(prog, rk, logical, {
            buf_addr(bp->in_r_buf), buf_addr(bp->in_i_buf),
            buf_addr(bp->tw_r_buf), buf_addr(bp->tw_i_buf),
            base, bp->batch_per_core,
            static_cast<uint32_t>(physical.x),
            static_cast<uint32_t>(physical.y),
        });

        SetRuntimeArgs(prog, wk, logical, {
            buf_addr(bp->out_r_buf), buf_addr(bp->out_i_buf),
            base, bp->batch_per_core,
        });

        SetRuntimeArgs(prog, ck, logical, {bp->batch_per_core});
    }

    bp->workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    bp->initialized = true;
    return bp;
}

namespace detail {
inline std::unordered_map<uint64_t, std::shared_ptr<BatchFFTPlan>>& batch_plan_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<BatchFFTPlan>> c;
    return c;
}
inline uint64_t batch_plan_key(MeshDevice* md, uint32_t sub_N, uint32_t batch) {
    return reinterpret_cast<uint64_t>(md)
         ^ (uint64_t{sub_N} * 0x9E3779B97F4A7C15ull)
         ^ (uint64_t{batch} * 0xBF58476D1CE4E5B9ull);
}
}  // namespace detail

inline std::shared_ptr<BatchFFTPlan> get_cached_batch_plan(
    std::shared_ptr<MeshDevice> md, uint32_t sub_N, uint32_t batch)
{
    const uint64_t key = detail::batch_plan_key(md.get(), sub_N, batch);
    auto& cache = detail::batch_plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto bp = make_batch_plan(md, sub_N, batch);
    cache.emplace(key, bp);
    return bp;
}

// in_r / in_i are BATCH * kTileElems floats: batch contiguous tiles, each
// tile holding bit-reversed input slots [0, sub_N) and zeros after.
// Returns out_r / out_i in the same layout, natural-order spectrum slots
// [0, sub_N) and undefined slots after.
//
// NOTE: `WriteShard` takes the data vector by non-const reference (same
// signature as the inner kernel uses), so in_r / in_i must be non-const.
inline void execute_batch(
    BatchFFTPlan&            plan,
    std::vector<float>&      in_r,
    std::vector<float>&      in_i,
    std::vector<float>&      out_r,
    std::vector<float>&      out_i)
{
    using namespace tt::tt_metal::distributed;
    assert(plan.initialized);
    assert(in_r.size() == static_cast<size_t>(plan.batch) * kTileElems);
    assert(in_i.size() == in_r.size());

    MeshCommandQueue& cq = plan.md->mesh_command_queue();

    WriteShard(cq, plan.in_r_buf, in_r, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan.in_i_buf, in_i, MeshCoordinate(0, 0), false);

    EnqueueMeshWorkload(cq, plan.workload, false);

    ReadShard(cq, out_r, plan.out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i, plan.out_i_buf, MeshCoordinate(0, 0), true);
}

// Convenience: run `batch` length-`sub_N` FFTs given a flat (batch * sub_N)
// natural-order input. Handles the bit-reversal pack and the natural-order
// unpack so callers can think purely in terms of "sub-FFT i, slot j".
inline void batch_fft(
    std::shared_ptr<MeshDevice>  md,
    uint32_t                     sub_N,
    uint32_t                     batch,
    const std::vector<Complex>&  in_natural,    // size batch * sub_N
    std::vector<Complex>&        out_natural)   // resized to batch * sub_N
{
    assert(in_natural.size() == static_cast<size_t>(sub_N) * batch);

    auto plan = get_cached_batch_plan(md, sub_N, batch);
    const uint32_t log2_sub_N = plan->log2_sub_N;

    const size_t tile_floats   = kTileElems;
    const size_t total_floats  = static_cast<size_t>(batch) * tile_floats;
    std::vector<float> in_r(total_floats, 0.0f);
    std::vector<float> in_i(total_floats, 0.0f);
    std::vector<float> out_r, out_i;

    // Bit-reverse pack. Tile t holds sub-FFT t.
    for (uint32_t t = 0; t < batch; ++t) {
        const Complex* src = in_natural.data() + static_cast<size_t>(t) * sub_N;
        float* tr = in_r.data() + static_cast<size_t>(t) * tile_floats;
        float* ti = in_i.data() + static_cast<size_t>(t) * tile_floats;
        for (uint32_t k = 0; k < sub_N; ++k) {
            const uint32_t s = bit_rev(k, log2_sub_N);
            tr[k] = src[s].real();
            ti[k] = src[s].imag();
        }
    }

    execute_batch(*plan, in_r, in_i, out_r, out_i);

    out_natural.resize(static_cast<size_t>(batch) * sub_N);
    for (uint32_t t = 0; t < batch; ++t) {
        const float* tr = out_r.data() + static_cast<size_t>(t) * tile_floats;
        const float* ti = out_i.data() + static_cast<size_t>(t) * tile_floats;
        Complex* dst = out_natural.data() + static_cast<size_t>(t) * sub_N;
        for (uint32_t k = 0; k < sub_N; ++k) dst[k] = {tr[k], ti[k]};
    }
}

// ── Pass 1: N1 row-FFTs of length N2 ──────────────────────────────────────
//
// Cooley-Tukey decomposition for N = N1 * N2 requires the FIRST FFT pass to
// run over the slow-varying axis of the natural row-major reshape (i.e. it
// is intrinsically a "column-FFT" of length N2 with stride N1 through the
// 1D input). To turn it into a row-FFT we hand to our existing radix-2
// kernel, we transpose-on-pack here:
//
//     packed[i, j]  =  x[j*N1 + i]    for i in [0, N1), j in [0, N2)
//
// Every sub-FFT is a single tile (N2 <= 1024 across the entire Stockham
// regime up to N=1M), so we route through the device-side BATCH kernel:
// one dispatch instead of N1 dispatches.

inline std::vector<Complex> pass1_row_ffts(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  x,
    const StockhamPlan&          p)
{
    assert(static_cast<uint32_t>(x.size()) == p.N);

    // Build a flat (N1 * N2) buffer where row i is the transposed slice
    // {x[0*N1 + i], x[1*N1 + i], ..., x[(N2-1)*N1 + i]}.
    std::vector<Complex> in_natural(static_cast<size_t>(p.N1) * p.N2);
    for (uint32_t i = 0; i < p.N1; ++i) {
        Complex* dst = in_natural.data() + static_cast<size_t>(i) * p.N2;
        for (uint32_t j = 0; j < p.N2; ++j) {
            dst[j] = x[static_cast<size_t>(j) * p.N1 + i];
        }
    }

    std::vector<Complex> out_natural;
    batch_fft(md, /*sub_N=*/p.N2, /*batch=*/p.N1, in_natural, out_natural);
    return out_natural;   // row-major (N1, N2) — exactly what pass 2 expects.
}

// ── Pass 2: per-element twiddle  +  transpose to (N2, N1) ────────────────
//
//   B[i, j] = A[i, j] * exp(-2*pi*i*i*j / N)         (twiddle)
//   C[j, i] = B[i, j]                                (transpose)
//
// On host this is a tight O(N) loop with cos/sin per element. At N=1M it
// completes in ~50–100 ms on a normal x86. A future optimisation is a
// BRISC-only device kernel that fuses both steps and runs in <5 ms.

inline std::vector<Complex> pass2_twiddle_transpose(
    const std::vector<Complex>& A,
    const StockhamPlan&         p)
{
    std::vector<Complex> C(p.N);
    const double tau_over_N = -2.0 * M_PI / static_cast<double>(p.N);

    for (uint32_t i = 0; i < p.N1; ++i) {
        const Complex* src = A.data() + static_cast<size_t>(i) * p.N2;
        for (uint32_t j = 0; j < p.N2; ++j) {
            const double angle = tau_over_N *
                                 static_cast<double>(i) *
                                 static_cast<double>(j);
            const float  cw = static_cast<float>(std::cos(angle));
            const float  sw = static_cast<float>(std::sin(angle));
            const Complex w(cw, sw);
            C[static_cast<size_t>(j) * p.N1 + i] = src[j] * w;
        }
    }
    return C;
}

// ── Pass 3: N2 row-FFTs of length N1 ──────────────────────────────────────
//
// Pass 2 already laid C out as (N2, N1) row-major. Each row j is a length-N1
// natural-order signal we FFT in place. With N1 <= 1024 across our regime,
// the batch path runs all N2 sub-FFTs in one device dispatch.

inline std::vector<Complex> pass3_row_ffts(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  C,
    const StockhamPlan&          p)
{
    assert(static_cast<uint32_t>(C.size()) == p.N);
    // C is already (N2, N1) row-major: rows are sub-FFT inputs in natural
    // order. Hand the buffer straight to the batch dispatcher.
    std::vector<Complex> D;
    batch_fft(md, /*sub_N=*/p.N1, /*batch=*/p.N2, C, D);
    return D;
}

// ── Final reorder: D is (N2, N1) row-major; natural 1D output:
//     X[k] = D[k % N2, k / N2] = D_flat[(k % N2) * N1 + (k / N2)]
inline std::vector<Complex> final_reorder(
    const std::vector<Complex>& D,
    const StockhamPlan&         p)
{
    std::vector<Complex> X(p.N);
    for (uint32_t k = 0; k < p.N; ++k) {
        const uint32_t j  = k % p.N2;
        const uint32_t ip = k / p.N2;
        X[k] = D[static_cast<size_t>(j) * p.N1 + ip];
    }
    return X;
}

// ── Public API ────────────────────────────────────────────────────────────
//
// fft_stockham::fft(md, signal) — drop-in equivalent of
// fft_example::fft(md, signal) that supports any N (power of two).
//
// For N <= 65,536 we just call the inner radix-2 directly (zero overhead).
// For N >  65,536 we run the four-pass Stockham orchestrator.

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 2 && "FFT requires N >= 2");
    assert(is_pow2(N) && "FFT requires N to be a power of two");

    const StockhamPlan p = plan(N);

    if (!p.stockham) {
        return fft_example::fft(md, signal);
    }

    std::printf(
        "[fft_stockham] N=%u  =>  N1=%u  x  N2=%u   (batch FFT: pass-1 = "
        "%u x len %u, pass-3 = %u x len %u; 64 cores per dispatch)\n",
        p.N, p.N1, p.N2, p.N1, p.N2, p.N2, p.N1);

    const auto A = pass1_row_ffts        (md, signal, p);
    const auto C = pass2_twiddle_transpose(    A,      p);
    const auto D = pass3_row_ffts        (md, C,      p);
    return final_reorder(D, p);
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    signal)
{
    std::vector<Complex> cx(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) cx[i] = {signal[i], 0.0f};
    return fft(md, cx);
}

}  // namespace fft_stockham
