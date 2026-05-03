// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_host.cpp — Host-side FFT that accepts ANY N >= 2.
//
// Reuses fft_stockham::fft (which itself routes to fft_example::fft for
// N <= 65,536 and runs the 4-pass Stockham for larger powers of two), and
// adds ONE new device kernel (Opt #5, under ./kernel/) for the tile-packed
// direct DFT that handles every sub-FFT with length <= kPackedMaxN.
//
// Dispatch tree (every compute path ends on Wormhole):
//   * N == 1                -> identity.
//   * N is a power of two   -> fft_stockham::fft (direct pass-through, device).
//   * N factors as 2^k * q  -> Cooley-Tukey split on (2^k, q), then recurse.
//   * N is odd composite    -> Cooley-Tukey split on (smallest-prime, rest),
//                              then recurse.
//   * N is prime (>=3)      -> Bluestein (chirp-z): one forward + one inverse
//                              pow2 FFT on the device (M = next pow2 >= 2N-1).
//
// Leaf path (batched_siblings_fft):
//   * len <= 32   -> PACKED DIRECT-DFT kernel (Opt #5): 32 sub-FFTs per
//                    tile, 4 real 32x32 matmuls per tile on the FPU. One
//                    dispatch, tile efficiency = len/32 instead of
//                    len/1024 — eliminates the ~99% PCIe padding waste of
//                    the old small-leaf paths.
//
// Batched-recursion architecture (Opt #2 + #2b, complete):
//
//   Every recursion level operates on `count` sibling signals laid out
//   back-to-back in row-major order. A single sub-FFT pass at depth d always
//   becomes ONE of:
//     * one fft_stockham::batch_fft dispatch (pow2 length <= 1024, any count),
//     * one batched Bluestein → 2 pow2 batch_fft dispatches (prime length),
//     * one batched Cooley-Tukey split → two recursive calls with count *= Nk.
//
//   Because composite splits MULTIPLY `count` by N1 or N2, batching width
//   grows as we descend; by the time we reach the leaves, we're typically
//   running a single pow2 batch_fft dispatch over thousands of sibling
//   sub-FFTs — regardless of the original N's structure.
//
// Cost summary:
//   * pow2 N                -> same as fft_stockham (no extra overhead).
//   * prime N (single call) -> 2 batch_fft dispatches + O(N) host multiplies.
//   * composite N           -> O(log N) batch_fft dispatches + O(N) host
//                              reshape/twiddle/transpose per level. Dispatch
//                              count is independent of how many sibling
//                              sub-FFTs live at any level.
//
// Caches (globals; single-threaded use):
//   * bluestein_cache keyed on N keeps the chirp table and B_fft so the second
//     call for the same N skips all host pre-work.
//   * ct_plan_cache keyed on (N1, N2) keeps the N1*N2 twiddle table so the
//     Cooley-Tukey hot path is plain complex-multiply (no cos/sin per iter).
//   * fft_stockham::fft carries its own program-build cache; we piggy-back.

#pragma once

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tilize_utils.hpp"

#include "../fft_stockham/fft_stockham_host.cpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fft_universal {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

// ─── Tunables ────────────────────────────────────────────────────────────────
// Largest power-of-two that fft_stockham::fft currently accepts. Bluestein
// requires M = next_pow2(2N - 1) <= this ceiling, i.e. prime N <= 524,288.
constexpr uint32_t kStockhamMaxPow2 = 1048576u;

// fft_stockham::batch_fft requires each sub-FFT to fit in a single Tensix
// tile (1024 complex elements). Above this we cannot batch and fall back to
// serial recursion.
constexpr uint32_t kBatchMaxSubN = 1024u;

// Maximum sub-FFT length handled by the PACKED DIRECT-DFT kernel (Opt #5).
// 32 is the natural ceiling: a 32x32 Tensix tile fits 32 sub-FFTs (one per
// row) and the whole N×N twiddle matrix in a single tile. For len <= this
// we always route through the packed kernel — tile efficiency jumps from
// ~1-3% (batch_fft with sub_N << 1024) to (len / 32) ≈ 6-100%, which is
// the dominant PCIe saving.
constexpr uint32_t kPackedMaxN = 32u;

// ─── Small helpers ───────────────────────────────────────────────────────────
inline bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

inline uint32_t next_pow2(uint32_t n) {
    uint32_t p = 1u;
    while (p < n) p <<= 1;
    return p;
}

inline uint32_t smallest_prime_factor(uint32_t n) {
    if (n < 2u) return n;
    if ((n & 1u) == 0u) return 2u;
    for (uint32_t p = 3u; static_cast<uint64_t>(p) * p <= n; p += 2u) {
        if (n % p == 0u) return p;
    }
    return n;   // n itself is prime
}

inline bool is_prime(uint32_t n) {
    if (n < 2u) return false;
    return smallest_prime_factor(n) == n;
}

// Pick (N1, N2) with N = N1 * N2. Prefer the largest pow2 factor as N1 so
// pass-1 sub-FFTs are pow2 and hit the optimised device path directly.
// Fall back to (smallest-prime-factor, rest) for odd composites.
inline std::pair<uint32_t, uint32_t> pick_factors(uint32_t N) {
    uint32_t pow2 = 1u, odd = N;
    while ((odd & 1u) == 0u) { odd >>= 1; pow2 <<= 1; }
    if (pow2 > 1u && odd > 1u) {
        return {pow2, odd};
    }
    // N is odd; peel off its smallest prime factor.
    const uint32_t p = smallest_prime_factor(N);
    return {p, N / p};
}

// ─── Bluestein (chirp-z) plan ────────────────────────────────────────────────
//
// Identity used:
//   X[k] = sum_{n=0}^{N-1} x[n] exp(-2πi k n / N)
//        = w[k] * sum_n (x[n] * w[n]) * conj(w)[k - n]
//   where w[n] = exp(-i π n² / N).
//
// The inner sum is the linear convolution of (x * w) with conj(w), computed
// as a length-M cyclic convolution with M = next_pow2(2N - 1):
//   A = FFT_M(a), B = FFT_M(b_ext), c = IFFT_M(A * B), X[k] = w[k] * c[k]
// where a is (x * w) zero-padded and b_ext is conj(w) symmetrically extended.
//
// B_fft, chirp_fwd, and M depend only on N — cache them.
struct BluesteinPlan {
    uint32_t             N = 0u;
    uint32_t             M = 0u;
    std::vector<Complex> chirp_fwd;   // w[n] = exp(-i π n² / N), length N
    std::vector<Complex> B_fft;       // FFT_M(b_ext), length M
};

inline std::unordered_map<uint32_t, std::shared_ptr<BluesteinPlan>>&
bluestein_cache() {
    static std::unordered_map<uint32_t, std::shared_ptr<BluesteinPlan>> m;
    return m;
}

inline std::shared_ptr<BluesteinPlan> get_bluestein_plan(
    std::shared_ptr<MeshDevice> md,
    uint32_t                    N)
{
    auto& cache = bluestein_cache();
    if (auto it = cache.find(N); it != cache.end()) return it->second;

    auto plan = std::make_shared<BluesteinPlan>();
    plan->N = N;
    plan->M = next_pow2(2u * N - 1u);
    const uint32_t M = plan->M;

    // chirp_fwd[n] = exp(-i π n² / N). Reduce (n²) mod 2N to keep the trig
    // argument bounded and preserve precision for large n.
    plan->chirp_fwd.resize(N);
    const double   pi_over_N = M_PI / static_cast<double>(N);
    const uint64_t mod2N     = 2ull * static_cast<uint64_t>(N);
    for (uint32_t n = 0; n < N; ++n) {
        const uint64_t nn = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
        const double   a  = pi_over_N * static_cast<double>(nn % mod2N);
        plan->chirp_fwd[n] = Complex(static_cast<float>( std::cos(a)),
                                     static_cast<float>(-std::sin(a)));
    }

    // b_ext: length-M symmetric extension of conj(w).
    //   b_ext[0]   = 1
    //   b_ext[n]   = conj(w[n])    for n = 1..N-1
    //   b_ext[M-n] = conj(w[n])    for n = 1..N-1   (negative-index mirror)
    //   b_ext[n]   = 0             otherwise
    std::vector<Complex> b_ext(M, Complex(0.0f, 0.0f));
    b_ext[0] = Complex(1.0f, 0.0f);
    for (uint32_t n = 1; n < N; ++n) {
        const Complex g = std::conj(plan->chirp_fwd[n]);
        b_ext[n]     = g;
        b_ext[M - n] = g;
    }

    // M is a power of two by construction, so Stockham handles it directly.
    plan->B_fft = fft_stockham::fft(md, b_ext);

    cache[N] = plan;
    return plan;
}

// ─── Cooley-Tukey twiddle plan ───────────────────────────────────────────────
//
// For a mixed-radix split N = N1·N2 the twiddle factor
//     T[n1, k2] = exp(-2πi · n1 · k2 / N)
// is the same for every call and every sibling — cache it per (N1, N2) so
// the per-call hot path is a plain complex-multiply instead of 2 transcendental
// evaluations per element. Cache key packs both dims into 64 bits.
struct CooleyTukeyPlan {
    uint32_t             N1 = 0u;
    uint32_t             N2 = 0u;
    std::vector<Complex> twiddle;  // twiddle[n1 * N2 + k2], size N1 * N2
};

inline std::unordered_map<uint64_t, std::shared_ptr<CooleyTukeyPlan>>&
ct_plan_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<CooleyTukeyPlan>> m;
    return m;
}

inline std::shared_ptr<CooleyTukeyPlan> get_ct_plan(uint32_t N1, uint32_t N2) {
    const uint64_t key =
        (static_cast<uint64_t>(N1) << 32) | static_cast<uint64_t>(N2);
    auto& cache = ct_plan_cache();
    if (auto it = cache.find(key); it != cache.end()) return it->second;

    auto plan = std::make_shared<CooleyTukeyPlan>();
    plan->N1 = N1;
    plan->N2 = N2;
    plan->twiddle.resize(static_cast<size_t>(N1) * N2);
    const double tau_over_N =
        -2.0 * M_PI / static_cast<double>(N1) / static_cast<double>(N2);
    for (uint32_t n1 = 0; n1 < N1; ++n1) {
        for (uint32_t k2 = 0; k2 < N2; ++k2) {
            const double a = tau_over_N
                           * static_cast<double>(n1)
                           * static_cast<double>(k2);
            plan->twiddle[static_cast<size_t>(n1) * N2 + k2] =
                Complex(static_cast<float>(std::cos(a)),
                        static_cast<float>(std::sin(a)));
        }
    }
    cache[key] = plan;
    return plan;
}

// ╔════════════════════════════════════════════════════════════════════════╗
// ║  Optimisation 5 — PACKED DIRECT-DFT kernel for small sub-FFTs (N<=32)  ║
// ╚════════════════════════════════════════════════════════════════════════╝
//
// Motivation
// ----------
// The existing fft_stockham::batch_fft kernel stores ONE sub-FFT per tile.
// For small sub_N the tile is padded with (1024 - sub_N) zeros, wasting
// ~99% of every PCIe/DRAM byte for sub_N=8 and ~97% for sub_N=32. Since
// fft_universal decomposes composite N via Cooley-Tukey down to small-prime
// leaves, these tiny sub_N leaves dominate the dispatch-bytes budget.
//
// This kernel packs 32 sub-FFTs per tile (one sub-FFT per tile row) and
// computes the direct DFT as a 32x32 complex matmul against a cached
// twiddle matrix T[n, k] = exp(-2πi · k · n / N).  The 4 real 32x32 matmuls
// that make up the complex matmul are offered to the Tensix FPU via
// matmul_tiles(), so compute throughput is O(N²) per sub-FFT but N ≤ 32 so
// it's trivially fast — the dispatch is bandwidth-bound, not compute-bound.
//
// Applicability
// -------------
//   * Routed for EVERY len ∈ [2, kPackedMaxN=32] at the batched_siblings_fft
//     dispatcher, replacing both the batch_fft path (pow2 lengths 2..32)
//     AND the batched_bluestein path (prime/composite 3..32).
//   * Above 32, batch_fft already runs at >6% tile efficiency and its
//     Stockham butterflies beat an O(N²) DFT; we keep those paths.
//
// Layout details (keyed with packed_dft_common.h)
// -----------------------------------------------
//   Per tile:
//     row i ∈ [0, 32)   →   one sub-FFT (or zero-padding for i ≥ rows_used)
//     col k ∈ [0, N)    →   spectrum slot k of that sub-FFT
//     col k ∈ [N, 32)   →   padded zero
//
//   Host scratch uses ROW-MAJOR 32*num_tiles × 32 layout; we call
//   tilize_nfaces before WriteShard and untilize_nfaces after ReadShard to
//   convert to / from the Tensix face-interleaved tile format required by
//   the matmul FPU. tilize_nfaces supports float, so no precision loss.
//
//   Twiddle matrix T (single tile per cached N): stored tilized too, with
//   a separate T_I_neg tile so every on-device matmul adds into DST and no
//   SFPU subtract is needed.

struct PackedDFTPlan {
    uint32_t N              = 0;    // DFT length   (<= kPackedMaxN)
    uint32_t count          = 0;    // number of sub-FFTs as requested by caller
    uint32_t num_tiles      = 0;    // ceil(count / 32), then padded to divide num_cores
    uint32_t num_cores      = 0;    // <= 64
    uint32_t tiles_per_core = 0;    // num_tiles / num_cores
    uint32_t grid_cols      = 0;
    uint32_t grid_rows      = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_r_buf,  in_i_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> tw_r_buf, tw_i_buf, tw_i_neg_buf;
    tt::tt_metal::distributed::MeshWorkload workload;

    // Reused host scratch (same pattern as BatchFFTPlan / Pass2Plan): a
    // 32*num_tiles × 32 row-major float buffer plus its tilized mirror.
    // Zero-initialised once at plan-construction time; per-call repack
    // overwrites only the valid [0, N) slots of each active row, so the
    // padding zeros stay put across calls.
    std::vector<float> in_r_rm,  in_i_rm;     // row-major (host layout)
    std::vector<float> in_r_til, in_i_til;    // tilized   (device layout)
    std::vector<float> out_r_til, out_i_til;
    std::vector<float> out_r_rm,  out_i_rm;

    bool initialized = false;
};

// Build the N×N twiddle matrix in ROW-MAJOR 32x32 layout (one tile). The
// matmul on device computes out[i, k] = Σ_n in[i, n] · T[n, k], so T is
// indexed T[n, k] = exp(-2πi · k · n / N). Positions outside [0, N)² are
// zero so padded input slots contribute nothing.
inline std::pair<std::vector<float>, std::vector<float>> packed_dft_twiddle_rm(uint32_t N) {
    std::vector<float> tr(32u * 32u, 0.0f), ti(32u * 32u, 0.0f);
    const double tau_over_N = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < N; ++k) {
            const double a = tau_over_N * static_cast<double>(n) * static_cast<double>(k);
            tr[n * 32u + k] = static_cast<float>(std::cos(a));
            ti[n * 32u + k] = static_cast<float>(std::sin(a));
        }
    }
    return {std::move(tr), std::move(ti)};
}

inline std::shared_ptr<PackedDFTPlan> make_packed_dft_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t count)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto pp = std::make_shared<PackedDFTPlan>();
    pp->md    = md;
    pp->N     = N;
    pp->count = count;

    assert(N >= 2u && N <= kPackedMaxN);
    assert(count >= 1u);

    constexpr uint32_t kRowsPerTile = 32u;
    const uint32_t raw_num_tiles = (count + kRowsPerTile - 1u) / kRowsPerTile;

    // Distribute tiles across up to 64 cores. fft_stockham::pick_batch_grid
    // only produces a valid (cols, rows) rectangle when num_cores is either
    // <= 7 (cols=n, rows=1) OR a multiple of 8 (cols=8, rows=n/8). If we
    // handed it raw_num_tiles directly it would silently drop cores (e.g.
    // n=38 → grid=8x4=32, and cores 32..37 receive no kernel placement →
    // SetRuntimeArgs asserts). Round num_cores up to the next valid shape,
    // clamp to 64, then pad num_tiles out to num_cores * tiles_per_core.
    // Extra tiles get zero input (pre-zeroed scratch) → zero output that
    // we discard at unpack time, same strategy BatchFFTPlan uses for its
    // pow2 padding.
    uint32_t num_cores;
    if (raw_num_tiles <= 7u) {
        num_cores = raw_num_tiles;
    } else {
        num_cores = ((raw_num_tiles + 7u) / 8u) * 8u;   // round up to multiple of 8
        if (num_cores > 64u) num_cores = 64u;
    }
    uint32_t tiles_per_core = (raw_num_tiles + num_cores - 1u) / num_cores;
    uint32_t num_tiles      = num_cores * tiles_per_core;

    pp->num_cores      = num_cores;
    pp->tiles_per_core = tiles_per_core;
    pp->num_tiles      = num_tiles;
    std::tie(pp->grid_cols, pp->grid_rows) = fft_stockham::pick_batch_grid(num_cores);

    std::printf(
        "[packed_dft] N=%u  count=%u  =>  num_tiles=%u  cores=%u  grid=%ux%u  "
        "tiles/core=%u  tile-eff=%.1f%%\n",
        N, count, num_tiles, num_cores, pp->grid_cols, pp->grid_rows,
        tiles_per_core,
        100.0 * static_cast<double>(N * kRowsPerTile) / 1024.0);

    MeshCommandQueue& cq = md->mesh_command_queue();

    // ── DRAM buffers ────────────────────────────────────────────────────
    const uint32_t io_bytes = num_tiles * fft_stockham::kTileSizeFp32;
    pp->in_r_buf  = fft_stockham::make_mesh_buf(md, io_bytes, fft_stockham::kTileSizeFp32);
    pp->in_i_buf  = fft_stockham::make_mesh_buf(md, io_bytes, fft_stockham::kTileSizeFp32);
    pp->out_r_buf = fft_stockham::make_mesh_buf(md, io_bytes, fft_stockham::kTileSizeFp32);
    pp->out_i_buf = fft_stockham::make_mesh_buf(md, io_bytes, fft_stockham::kTileSizeFp32);

    // Single-tile twiddle buffers (one each for T_R, T_I, T_I_neg).
    const uint32_t tw_bytes = fft_stockham::kTileSizeFp32;
    pp->tw_r_buf     = fft_stockham::make_mesh_buf(md, tw_bytes, fft_stockham::kTileSizeFp32);
    pp->tw_i_buf     = fft_stockham::make_mesh_buf(md, tw_bytes, fft_stockham::kTileSizeFp32);
    pp->tw_i_neg_buf = fft_stockham::make_mesh_buf(md, tw_bytes, fft_stockham::kTileSizeFp32);

    // Build and upload twiddle tiles (row-major → tilize → WriteShard).
    auto [tr_rm, ti_rm] = packed_dft_twiddle_rm(N);
    std::vector<float> ti_neg_rm(ti_rm.size());
    for (size_t i = 0; i < ti_rm.size(); ++i) ti_neg_rm[i] = -ti_rm[i];

    std::vector<float> tr_til     = tilize_nfaces(tr_rm,     32u, 32u);
    std::vector<float> ti_til     = tilize_nfaces(ti_rm,     32u, 32u);
    std::vector<float> ti_neg_til = tilize_nfaces(ti_neg_rm, 32u, 32u);

    WriteShard(cq, pp->tw_r_buf,     tr_til,     MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_buf,     ti_til,     MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_neg_buf, ti_neg_til, MeshCoordinate(0, 0), false);

    // ── Host scratch (reused across calls — see PackedDFTPlan comments) ─
    const size_t rm_floats  = static_cast<size_t>(num_tiles) * 32u * 32u;
    const size_t til_floats = static_cast<size_t>(num_tiles) * fft_stockham::kTileElems;
    pp->in_r_rm .assign(rm_floats,  0.0f);
    pp->in_i_rm .assign(rm_floats,  0.0f);
    pp->in_r_til.assign(til_floats, 0.0f);
    pp->in_i_til.assign(til_floats, 0.0f);
    pp->out_r_til.assign(til_floats, 0.0f);
    pp->out_i_til.assign(til_floats, 0.0f);
    pp->out_r_rm .assign(rm_floats,  0.0f);
    pp->out_i_rm .assign(rm_floats,  0.0f);

    // ── Program ─────────────────────────────────────────────────────────
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{pp->grid_cols - 1, pp->grid_rows - 1};
    const CoreRange cr(first, last);

    // CBs: CB_A, CB_B (depth 4 so the reader can queue all 4 matmul pairs
    // upfront), CB_OUT_R, CB_OUT_I (depth 2, ordinary double-buffer).
    constexpr uint32_t kCbCount = 4u;
    constexpr uint32_t kCbTiles[kCbCount] = {4u, 4u, 2u, 2u};
    for (uint32_t id = 0; id < kCbCount; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * fft_stockham::kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, fft_stockham::kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_universal/kernel/packed_dft_reader.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    auto wk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_universal/kernel/packed_dft_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    // IMPORTANT: do NOT set UnpackToDestFp32 for a matmul kernel. That
    // mode routes tiles straight into DST (used by some SFPU paths), which
    // leaves srcA / srcB uninitialised when matmul_tiles tries to read
    // them → FPU multiplies garbage → 1e28 / inf outputs. The standard
    // tt-metal matmul_single_core example omits unpack_to_dest_mode for
    // exactly this reason; keep matmul defaults here and rely on
    // fp32_dest_acc_en to carry fp32 accumulation.
    auto ck = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_universal/kernel/packed_dft_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args     = {pp->tiles_per_core}});
    (void)ck;  // compute kernel needs no per-core runtime args

    for (uint32_t c = 0; c < pp->num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, pp->grid_cols);
        const uint32_t  base    = c * pp->tiles_per_core;

        SetRuntimeArgs(prog, rk, logical, {
            fft_stockham::buf_addr(pp->in_r_buf),
            fft_stockham::buf_addr(pp->in_i_buf),
            fft_stockham::buf_addr(pp->tw_r_buf),
            fft_stockham::buf_addr(pp->tw_i_buf),
            fft_stockham::buf_addr(pp->tw_i_neg_buf),
            base,
            pp->tiles_per_core,
        });

        SetRuntimeArgs(prog, wk, logical, {
            fft_stockham::buf_addr(pp->out_r_buf),
            fft_stockham::buf_addr(pp->out_i_buf),
            base,
            pp->tiles_per_core,
        });
    }

    pp->workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    pp->initialized = true;
    return pp;
}

namespace detail_packed {
inline std::unordered_map<uint64_t, std::shared_ptr<PackedDFTPlan>>& packed_dft_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<PackedDFTPlan>> c;
    return c;
}
inline uint64_t packed_dft_key(MeshDevice* md, uint32_t N, uint32_t count) {
    return reinterpret_cast<uint64_t>(md)
         ^ (uint64_t{N}     * 0x9E3779B97F4A7C15ull)
         ^ (uint64_t{count} * 0xBF58476D1CE4E5B9ull);
}
}  // namespace detail_packed

inline std::shared_ptr<PackedDFTPlan> get_cached_packed_dft_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t count)
{
    const uint64_t key = detail_packed::packed_dft_key(md.get(), N, count);
    auto& cache = detail_packed::packed_dft_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto pp = make_packed_dft_plan(md, N, count);
    cache.emplace(key, pp);
    return pp;
}

// Host wrapper. Computes `count` independent length-N DFTs via the packed
// direct-DFT kernel. in/out layout is row-major (count * N complex values).
inline void packed_direct_dft_batched(
    std::shared_ptr<MeshDevice>   md,
    uint32_t                      N,
    uint32_t                      count,
    const std::vector<Complex>&   in_natural,
    std::vector<Complex>&         out_natural)
{
    using namespace tt::tt_metal::distributed;
    assert(in_natural.size() == static_cast<size_t>(count) * N);

    auto plan = get_cached_packed_dft_plan(md, N, count);
    constexpr uint32_t kRowsPerTile = 32u;

    std::vector<float>& in_r_rm  = plan->in_r_rm;
    std::vector<float>& in_i_rm  = plan->in_i_rm;
    std::vector<float>& in_r_til = plan->in_r_til;
    std::vector<float>& in_i_til = plan->in_i_til;
    std::vector<float>& out_r_til = plan->out_r_til;
    std::vector<float>& out_i_til = plan->out_i_til;

    // Pack natural-order input into row-major (32 * num_tiles) × 32. Each
    // sub-FFT r → row r, positions [0, N); positions [N, 32) stay zero.
    // Extra rows r ∈ [count, 32 * num_tiles) stay zero from plan init.
    for (uint32_t r = 0; r < count; ++r) {
        const Complex* src = in_natural.data() + static_cast<size_t>(r) * N;
        float* tr = in_r_rm.data() + static_cast<size_t>(r) * 32u;
        float* ti = in_i_rm.data() + static_cast<size_t>(r) * 32u;
        for (uint32_t k = 0; k < N; ++k) {
            tr[k] = src[k].real();
            ti[k] = src[k].imag();
        }
    }
    // If a previous call used a larger `count` with the SAME plan (rare —
    // only hits on plan reuse across counts), those rows would still have
    // stale data in cols [0, N). For our dispatcher, plan keys include
    // count so this cannot happen — but zero the just-vacated tail anyway
    // for safety when count hit the plan exactly.
    // (Left as-is: PackedDFTPlan is keyed on (N, count), so no cross-call
    // leakage is possible.)

    // Row-major → tilized (matmul expects Tensix face layout).
    const uint32_t total_rows = plan->num_tiles * kRowsPerTile;
    in_r_til = tilize_nfaces(in_r_rm, total_rows, 32u);
    in_i_til = tilize_nfaces(in_i_rm, total_rows, 32u);

    MeshCommandQueue& cq = plan->md->mesh_command_queue();
    WriteShard(cq, plan->in_r_buf, in_r_til, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan->in_i_buf, in_i_til, MeshCoordinate(0, 0), false);

    EnqueueMeshWorkload(cq, plan->workload, false);

    ReadShard(cq, out_r_til, plan->out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i_til, plan->out_i_buf, MeshCoordinate(0, 0), true);

    std::vector<float>& out_r_rm = plan->out_r_rm;
    std::vector<float>& out_i_rm = plan->out_i_rm;
    out_r_rm = untilize_nfaces(out_r_til, total_rows, 32u);
    out_i_rm = untilize_nfaces(out_i_til, total_rows, 32u);

    // Unpack: sub-FFT r output = row r positions [0, N).
    out_natural.resize(static_cast<size_t>(count) * N);
    for (uint32_t r = 0; r < count; ++r) {
        const float* tr = out_r_rm.data() + static_cast<size_t>(r) * 32u;
        const float* ti = out_i_rm.data() + static_cast<size_t>(r) * 32u;
        Complex*     dst = out_natural.data() + static_cast<size_t>(r) * N;
        for (uint32_t k = 0; k < N; ++k) dst[k] = {tr[k], ti[k]};
    }
}

// ─── Forward declarations ────────────────────────────────────────────────────
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal);

inline void batched_siblings_fft(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          len,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out);

inline void batched_bluestein(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out);

inline void cooley_tukey_split_batched(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N1,
    uint32_t                          N2,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out);

// ─── batched_siblings_fft: the universal dispatcher ──────────────────────────
//
// Computes `count` independent FFTs of length `len`, stored back-to-back in
// row-major order (in[r * len + k] is element k of sibling r). Every recursive
// level of the engine passes through here.
//
// Dispatch (in order of priority):
//   * len == 1 or count == 0         -> trivial copy.
//   * len pow2 and <= 1024 (tile)    -> ONE fft_stockham::batch_fft dispatch,
//                                        padding `count` up to next_pow2 with
//                                        zero-signal sibling rows (whose DFTs
//                                        are also zero — harmless, discarded).
//   * len pow2 but > 1024            -> serial fft_stockham::fft per row (the
//                                        kernel's multi-pass Stockham handles
//                                        up to N=1,048,576 for a single row).
//   * len prime (>= 3)               -> batched_bluestein: ONE pre-mul, TWO
//                                        batched length-M FFTs (M=next_pow2(2N-1)),
//                                        ONE pointwise mul, ONE post-mul.
//   * len composite non-pow2         -> cooley_tukey_split_batched, which does
//                                        two recursive calls with count *= Nk.
//                                        Batching width grows with depth.
inline void batched_siblings_fft(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          len,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out)
{
    assert(in.size() == static_cast<size_t>(count) * len);

    if (len == 1u || count == 0u) {
        out = in;
        return;
    }

    // Path 0 (Opt #5): any len ∈ [2, 32] goes through the packed direct-DFT
    // kernel. For these small lengths this is strictly a win vs both the
    // pow2 batch_fft path (which wastes ~97-99% of every tile on padding
    // zeros) AND the prime-length Bluestein path (2 dispatches at M up to
    // 64, each with similarly poor tile efficiency). Pack 32 sub-FFTs per
    // tile, compute as a complex 32x32 matmul → one dispatch, (len/32)
    // tile efficiency. Covers every small prime/composite leaf the CT
    // recursion would otherwise bounce into Bluestein or serial batch_fft.
    if (len >= 2u && len <= kPackedMaxN) {
        packed_direct_dft_batched(md, len, count, in, out);
        return;
    }

    // Path A: pow2 length fitting in a tile → single batched dispatch.
    if (is_pow2(len) && len <= kBatchMaxSubN) {
        const uint32_t padded = next_pow2(count);
        if (padded == count) {
            fft_stockham::batch_fft(md, len, count, in, out);
            return;
        }
        std::vector<Complex> in_padded(static_cast<size_t>(padded) * len,
                                       Complex{0.0f, 0.0f});
        std::copy(in.begin(), in.end(), in_padded.begin());
        std::vector<Complex> out_padded;
        fft_stockham::batch_fft(md, len, padded, in_padded, out_padded);
        out.assign(out_padded.begin(),
                   out_padded.begin() + static_cast<size_t>(count) * len);
        return;
    }

    // Path B: pow2 length too big for a tile — batch_fft can't help, so
    // serialize across the `count` rows. Each row uses multi-pass Stockham.
    if (is_pow2(len)) {
        out.resize(static_cast<size_t>(count) * len);
        std::vector<Complex> row(len);
        for (uint32_t r = 0; r < count; ++r) {
            const size_t base = static_cast<size_t>(r) * len;
            std::copy(in.begin() + base, in.begin() + base + len, row.begin());
            const std::vector<Complex> Yr = fft_stockham::fft(md, row);
            std::copy(Yr.begin(), Yr.end(), out.begin() + base);
        }
        return;
    }

    // Path C: prime length → batched Bluestein (fuses all `count` siblings
    // into exactly 2 batched pow2 FFT dispatches of length M).
    if (is_prime(len)) {
        batched_bluestein(md, count, len, in, out);
        return;
    }

    // Path D: composite non-pow2 → batched Cooley-Tukey split. The two
    // recursive calls inside propagate `count` multiplied by N1 / N2, so
    // batching width GROWS with recursion depth.
    const auto [N1, N2] = pick_factors(len);
    cooley_tukey_split_batched(md, count, N1, N2, in, out);
}

// ─── batched_bluestein ───────────────────────────────────────────────────────
//
// Compute `count` sibling length-N Bluestein FFTs. Instead of `count`
// independent Bluestein chains (2 × count length-M dispatches), we run:
//   1. batched pre-multiply by chirp w   (host, O(count·N))
//   2. ONE batched length-M forward FFT  (device, via batched_siblings_fft)
//   3. batched pointwise × B_fft          (host, O(count·M))
//   4. ONE batched length-M inverse FFT  (device, via conjugate trick)
//   5. batched post-multiply by chirp w  (host, O(count·N))
// Total device dispatches: 2 — independent of `count`.
inline void batched_bluestein(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out)
{
    assert(in.size() == static_cast<size_t>(count) * N);
    auto           plan = get_bluestein_plan(md, N);
    const uint32_t M    = plan->M;
    const auto&    w    = plan->chirp_fwd;   // chirp, length N
    const auto&    B    = plan->B_fft;       // FFT of mirrored conj(w), length M

    // Step 1: pre-multiply each sibling by w, zero-pad to length M.
    std::vector<Complex> A(static_cast<size_t>(count) * M, Complex{0.0f, 0.0f});
    for (uint32_t r = 0; r < count; ++r) {
        const size_t in_base  = static_cast<size_t>(r) * N;
        const size_t out_base = static_cast<size_t>(r) * M;
        for (uint32_t n = 0; n < N; ++n) {
            A[out_base + n] = in[in_base + n] * w[n];
        }
    }

    // Step 2: batched forward FFT of length M (pow2 — falls into Path A/B).
    std::vector<Complex> A_fft;
    batched_siblings_fft(md, count, M, A, A_fft);

    // Step 3: elementwise A_fft[r, k] *= B[k] (same B for every sibling).
    for (uint32_t r = 0; r < count; ++r) {
        const size_t base = static_cast<size_t>(r) * M;
        for (uint32_t k = 0; k < M; ++k) A_fft[base + k] *= B[k];
    }

    // Step 4: batched inverse FFT via conjugate trick —
    //         IFFT(X) = conj(FFT(conj(X))) / M.
    for (auto& z : A_fft) z = std::conj(z);
    std::vector<Complex> c;
    batched_siblings_fft(md, count, M, A_fft, c);
    const float inv_M = 1.0f / static_cast<float>(M);
    for (auto& z : c) z = std::conj(z) * inv_M;

    // Step 5: post-multiply first N samples of each row by w, drop padding.
    out.resize(static_cast<size_t>(count) * N);
    for (uint32_t r = 0; r < count; ++r) {
        const size_t in_base  = static_cast<size_t>(r) * M;
        const size_t out_base = static_cast<size_t>(r) * N;
        for (uint32_t k = 0; k < N; ++k) {
            out[out_base + k] = c[in_base + k] * w[k];
        }
    }
}

// ─── cooley_tukey_split_batched ──────────────────────────────────────────────
//
// Compute `count` sibling length-(N1·N2) FFTs via one mixed-radix split.
// Every host-side reshape/twiddle/transpose is loop-extended over `count`,
// and every sub-FFT call scales `count` by N1 or N2 — so batching width
// accumulates as we recurse deeper.
//
// Index schemes (matching fft_stockham, scheme (c)):
//   n  = n1 + N1 * n2     (n1 ∈ [0,N1), n2 ∈ [0,N2))
//   k  = N2 * k1 + k2     (k1 ∈ [0,N1), k2 ∈ [0,N2))
//
//   1. transposed reshape:  A[r, n1, n2] = in[r, n1 + N1 * n2]
//   2. pass-1  len N2:      batched_siblings_fft(count * N1, N2)
//   3. twiddle:             A[r, n1, k2] *= exp(-2πi · n1 · k2 / N)
//   4. transpose:           C[r, k2, n1] = A[r, n1, k2]
//   5. pass-2  len N1:      batched_siblings_fft(count * N2, N1)
//   6. output permute:      out[r, N2·k1 + k2] = D[r, k2, k1]
inline void cooley_tukey_split_batched(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N1,
    uint32_t                          N2,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out)
{
    const uint32_t N     = N1 * N2;
    const size_t   total = static_cast<size_t>(count) * N;
    assert(in.size() == total);

    // Step 1: per-row transposed reshape.
    //   pass1_in[(r * N1 + n1) * N2 + n2] = in[r * N + n2 * N1 + n1]
    std::vector<Complex> pass1_in(total);
    for (uint32_t r = 0; r < count; ++r) {
        const size_t in_base = static_cast<size_t>(r) * N;
        for (uint32_t n1 = 0; n1 < N1; ++n1) {
            const size_t out_base =
                (static_cast<size_t>(r) * N1 + n1) * N2;
            for (uint32_t n2 = 0; n2 < N2; ++n2) {
                pass1_in[out_base + n2] = in[in_base + n2 * N1 + n1];
            }
        }
    }

    // Step 2: (count * N1) sibling sub-FFTs of length N2.
    std::vector<Complex> A;
    batched_siblings_fft(md, count * N1, N2, pass1_in, A);

    // Steps 3+4 fused: twiddle-multiply and transpose in a single pass over A.
    //   C[r, k2, n1] = A[r, n1, k2] * twiddle[n1, k2]
    // Twiddle is cached per (N1, N2) — no cos/sin on the hot path.
    auto ct_plan = get_ct_plan(N1, N2);
    const Complex* __restrict__ twid = ct_plan->twiddle.data();
    std::vector<Complex> C(total);
    for (uint32_t r = 0; r < count; ++r) {
        for (uint32_t n1 = 0; n1 < N1; ++n1) {
            const size_t A_base    = (static_cast<size_t>(r) * N1 + n1) * N2;
            const size_t twid_base = static_cast<size_t>(n1) * N2;
            const size_t C_row     = static_cast<size_t>(r) * N2;
            for (uint32_t k2 = 0; k2 < N2; ++k2) {
                const Complex v = A[A_base + k2] * twid[twid_base + k2];
                C[(C_row + k2) * N1 + n1] = v;
            }
        }
    }

    // Step 5: (count * N2) sibling sub-FFTs of length N1.
    std::vector<Complex> D;
    batched_siblings_fft(md, count * N2, N1, C, D);

    // Step 6: per-row output permute  out[r, N2·k1 + k2] = D[r, k2, k1].
    out.resize(total);
    for (uint32_t r = 0; r < count; ++r) {
        for (uint32_t k1 = 0; k1 < N1; ++k1) {
            for (uint32_t k2 = 0; k2 < N2; ++k2) {
                const size_t out_idx =
                    static_cast<size_t>(r) * N + k1 * N2 + k2;
                const size_t D_idx =
                    (static_cast<size_t>(r) * N2 + k2) * N1 + k1;
                out[out_idx] = D[D_idx];
            }
        }
    }
}

// ─── Top-level single-signal dispatch ────────────────────────────────────────
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 1u && "FFT requires N >= 1");

    if (N == 1u)    return signal;
    // Large pow2 Ns go directly to fft_stockham so we keep its optimised
    // multi-pass path (up to N = 1M). Everything else — including small pow2
    // — funnels through the batched engine with count=1, which still uses
    // batch_fft internally for tile-sized pow2s.
    if (is_pow2(N)) return fft_stockham::fft(md, signal);

    if (is_prime(N)) {
        const uint32_t M = next_pow2(2u * N - 1u);
        assert(M <= kStockhamMaxPow2 &&
               "Bluestein M exceeds fft_stockham's max pow2. Raise the "
               "Stockham ceiling (multi-pass) before using larger prime N.");
        (void)M;
    }

    std::vector<Complex> out;
    batched_siblings_fft(md, /*count=*/1u, /*len=*/N, signal, out);
    return out;
}

// Convenience overload for purely real input.
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    real_signal)
{
    std::vector<Complex> cx(real_signal.size());
    for (size_t i = 0; i < real_signal.size(); ++i) {
        cx[i] = Complex(real_signal[i], 0.0f);
    }
    return fft(md, cx);
}

// ─── Inverse FFT via the conjugate trick ─────────────────────────────────────
//
//   IFFT(X) = conj( FFT( conj(X) ) ) / N
//
// Reuses the entire forward dispatch tree (Stockham pow2, packed_dft,
// Bluestein, mixed-radix CT) without writing a single new device kernel.
// All plan caches (PackedDFTPlan, BluesteinPlan, CooleyTukeyPlan,
// BatchFFTPlan, Pass2Plan) are direction-agnostic — they key on N (and
// counts), not on FFT vs IFFT — so steady-state IFFT performance is
// identical to forward FFT performance.
//
// Total host overhead vs. a forward FFT: 3 O(N) loops over fp32 complex
// elements (one conj-in, one conj-out, one /N scale). Microseconds even
// at N = 1M.
//
// The same identity is already used internally by batched_bluestein for
// its IFFT-of-M step; this just exposes the pattern as a public API.
inline std::vector<Complex> ifft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  spectrum)
{
    const uint32_t N = static_cast<uint32_t>(spectrum.size());
    assert(N >= 1u && "IFFT requires N >= 1");
    if (N == 1u) return spectrum;

    std::vector<Complex> conj_in(N);
    for (uint32_t k = 0; k < N; ++k) conj_in[k] = std::conj(spectrum[k]);

    const std::vector<Complex> fwd = fft(md, conj_in);

    const float inv_N = 1.0f / static_cast<float>(N);
    std::vector<Complex> out(N);
    for (uint32_t k = 0; k < N; ++k) out[k] = std::conj(fwd[k]) * inv_N;
    return out;
}

// Convenience overload for purely real input (e.g. a power spectrum that
// is being inverted as a real-valued signal — caller embeds it as zero-
// imaginary complex).
inline std::vector<Complex> ifft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    real_spectrum)
{
    std::vector<Complex> cx(real_spectrum.size());
    for (size_t i = 0; i < real_spectrum.size(); ++i) {
        cx[i] = Complex(real_spectrum[i], 0.0f);
    }
    return ifft(md, cx);
}

}  // namespace fft_universal
