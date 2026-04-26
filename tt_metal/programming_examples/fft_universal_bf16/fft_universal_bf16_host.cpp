// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_host.cpp — TRUE-bf16 variant of fft_universal.
//
// Goal
// ----
// Every on-device multiply is done by the Tensix FPU in bf16 × bf16 with
// fp32 accumulation. No "bf16 storage + fp32 SFPU compute" tricks — on
// Wormhole the SFPU has no native bf16 math path, so anything not routed
// through matmul_tiles implicitly upcasts to fp32 at the unpacker. We call
// that out explicitly in the dispatcher: if a path can't be expressed as
// FPU matmul, we error rather than silently demoting to fp32.
//
// Phase 1 status (this file)
// --------------------------
// Implemented:
//   * Host scaffolding: fp32 ↔ bf16 conversion, plan cache, dispatch tree.
//   * TRUE-bf16 packed direct-DFT kernel for N in [2, 32] via FPU matmul.
//     This is genuine bf16 compute: Float16_b CBs, fp32_dest_acc_en=true,
//     mm_init + matmul_tiles (bf16 srcA × bf16 srcB → fp32 DST → bf16 CB).
//
// Pending (Phase 2 — NOT in this file yet):
//   * Radix-32 Stockham bf16 matmul kernel for pow2 N > 32.
//   * Once Phase 2 lands, Bluestein (prime N) and Cooley-Tukey (composite N)
//     automatically become bf16 because they delegate their inner pow2 FFT
//     to the new radix-32 kernel.
//
// For any N > 32 the top-level fft() currently throws with a clear
// "Phase 2 not yet implemented" message. We refuse to silently fall back
// to the fp32 path because that would defeat the point of this binary.
//
// API
// ---
//   fft_universal_bf16::fft(md, vector<complex<float>>) -> vector<complex<float>>
//
// Input and output are fp32 complex for drop-in compatibility with the
// test / benchmark / demo harnesses. The fp32 → bf16 conversion happens
// once at the host boundary (same tile pack as the device consumes), and
// the bf16 → fp32 conversion happens once at read-back. Everything in
// between — DRAM, CBs, FPU — is bf16.
//
// Precision expectations
// ----------------------
//   Random complex input, |x| ≤ 1 :
//     * fp32 reference            : baseline
//     * fft_universal (fp32)      : rel err ~ 1e-6 on N ≤ 32
//     * THIS (true bf16)          : rel err ~ 2-4e-3 on N ≤ 32
//                                    (~42-45 dB SNR, bf16 accuracy floor)
// The ~3e-3 bound comes from:
//   * 1 bf16 rounding on input pack
//   * N ≤ 32 bf16 × bf16 products accumulated in fp32 per output element
//   * 1 bf16 rounding on output pack
// Matmul-based reduction keeps rounding depth O(log N) instead of O(N), so
// this is the best bf16 SNR achievable for a length-32 DFT on this hardware.

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
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fft_universal_bf16 {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

// ─── Tunables ────────────────────────────────────────────────────────────────
// Maximum sub-FFT length the Phase 1 packed direct-DFT bf16 kernel handles.
// Same 32 ceiling as the fp32 variant — a 32×32 tile fits 32 sub-FFTs and
// the N×N twiddle matrix in a single tile.
constexpr uint32_t kPackedMaxN = 32u;

// ─── Small helpers ───────────────────────────────────────────────────────────
inline bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

// ─── bf16 conversion (round-to-nearest-even, IEEE compliant) ────────────────
inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        // NaN: preserve payload bit so result is still NaN after truncation.
        return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    const uint32_t lsb  = (bits >> 16) & 1u;
    const uint32_t bias = 0x7FFFu + lsb;
    bits += bias;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert a vector of fp32 values to bf16 (stored as uint16_t). No layout
// change — this is purely elementwise.
inline std::vector<uint16_t> fp32_to_bf16_vec(const std::vector<float>& src) {
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = fp32_to_bf16(src[i]);
    return dst;
}

inline std::vector<float> bf16_to_fp32_vec(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = bf16_to_fp32(src[i]);
    return dst;
}

// ╔════════════════════════════════════════════════════════════════════════╗
// ║  TRUE-bf16 packed direct-DFT plan for small N (N <= kPackedMaxN=32)    ║
// ╚════════════════════════════════════════════════════════════════════════╝
//
// Dispatch shape is a 1-to-1 copy of fft_universal::PackedDFTPlan; only
// the tile data format (Float16_b instead of Float32) and the host-side
// tile size (2048 B vs 4096 B) change. Compare with
// ../fft_universal/fft_universal_host.cpp for the detailed layout commentary.
struct PackedDFTBf16Plan {
    uint32_t N              = 0;
    uint32_t count          = 0;
    uint32_t num_tiles      = 0;
    uint32_t num_cores      = 0;
    uint32_t tiles_per_core = 0;
    uint32_t grid_cols      = 0;
    uint32_t grid_rows      = 0;

    std::shared_ptr<MeshDevice> md;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_r_buf,  in_i_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out_r_buf, out_i_buf;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> tw_r_buf, tw_i_buf, tw_i_neg_buf;
    tt::tt_metal::distributed::MeshWorkload workload;

    // Host scratch. Row-major fp32 mirrors are used for packing / twiddle
    // prep; tilized fp32 is the layout fed to tilize_nfaces; uint16_t
    // scratch holds the bf16 tiles we actually ship to DRAM.
    std::vector<float>    in_r_rm,   in_i_rm;      // row-major host layout (fp32)
    std::vector<float>    in_r_til_f32, in_i_til_f32;  // tilized (fp32 mirror)
    std::vector<uint16_t> in_r_til,  in_i_til;     // bf16 tile bytes → DRAM
    std::vector<uint16_t> out_r_til, out_i_til;    // bf16 tile bytes ← DRAM
    std::vector<float>    out_r_rm,  out_i_rm;     // row-major host (fp32 after convert)

    bool initialized = false;
};

inline uint32_t bf16_tile_bytes() {
    return 32u * 32u * sizeof(uint16_t);   // 2048
}

// Build the 32×32 complex twiddle matrix T[n, k] = exp(-2πi · k · n / N)
// in ROW-MAJOR layout (cos in tr_rm, sin in ti_rm). Entries outside
// [0, N)² stay zero so padding rows/cols contribute nothing.
inline std::pair<std::vector<float>, std::vector<float>>
packed_dft_twiddle_rm(uint32_t N) {
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

inline std::shared_ptr<PackedDFTBf16Plan> make_packed_dft_bf16_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t count)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto pp = std::make_shared<PackedDFTBf16Plan>();
    pp->md    = md;
    pp->N     = N;
    pp->count = count;

    assert(N >= 2u && N <= kPackedMaxN);
    assert(count >= 1u);

    constexpr uint32_t kRowsPerTile = 32u;
    const uint32_t raw_num_tiles = (count + kRowsPerTile - 1u) / kRowsPerTile;

    // Same core-count rounding rule as the fp32 plan: fft_stockham's
    // pick_batch_grid only yields a valid (cols, rows) when num_cores is
    // <= 7 (cols=n, rows=1) OR a multiple of 8 (cols=8, rows=n/8). Round
    // up, clamp to 64, pad the tile count to match. Extra tiles receive
    // zero input (scratch is zero-initialised) → zero output, discarded.
    uint32_t num_cores;
    if (raw_num_tiles <= 7u) {
        num_cores = raw_num_tiles;
    } else {
        num_cores = ((raw_num_tiles + 7u) / 8u) * 8u;
        if (num_cores > 64u) num_cores = 64u;
    }
    const uint32_t tiles_per_core = (raw_num_tiles + num_cores - 1u) / num_cores;
    const uint32_t num_tiles      = num_cores * tiles_per_core;

    pp->num_cores      = num_cores;
    pp->tiles_per_core = tiles_per_core;
    pp->num_tiles      = num_tiles;
    std::tie(pp->grid_cols, pp->grid_rows) = fft_stockham::pick_batch_grid(num_cores);

    std::printf(
        "[packed_dft_bf16] N=%u  count=%u  =>  num_tiles=%u  cores=%u  grid=%ux%u  "
        "tiles/core=%u  tile-eff=%.1f%%\n",
        N, count, num_tiles, num_cores, pp->grid_cols, pp->grid_rows,
        tiles_per_core,
        100.0 * static_cast<double>(N * kRowsPerTile) / 1024.0);

    MeshCommandQueue& cq = md->mesh_command_queue();

    // ── DRAM buffers (bf16 tile = 2048 B) ───────────────────────────────
    const uint32_t ts_bf16  = bf16_tile_bytes();
    const uint32_t io_bytes = num_tiles * ts_bf16;
    pp->in_r_buf  = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);
    pp->in_i_buf  = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);
    pp->out_r_buf = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);
    pp->out_i_buf = fft_stockham::make_mesh_buf(md, io_bytes, ts_bf16);

    // Single-tile twiddle buffers (T_R, T_I, T_I_neg).
    pp->tw_r_buf     = fft_stockham::make_mesh_buf(md, ts_bf16, ts_bf16);
    pp->tw_i_buf     = fft_stockham::make_mesh_buf(md, ts_bf16, ts_bf16);
    pp->tw_i_neg_buf = fft_stockham::make_mesh_buf(md, ts_bf16, ts_bf16);

    // Build twiddles in fp32, tilize, convert to bf16, ship to DRAM.
    // The fp32 → bf16 rounding on the twiddles happens once per plan;
    // subsequent calls reuse the already-tilized bf16 tiles.
    auto [tr_rm, ti_rm] = packed_dft_twiddle_rm(N);
    std::vector<float> ti_neg_rm(ti_rm.size());
    for (size_t i = 0; i < ti_rm.size(); ++i) ti_neg_rm[i] = -ti_rm[i];

    const std::vector<float> tr_til_f32     = tilize_nfaces(tr_rm,     32u, 32u);
    const std::vector<float> ti_til_f32     = tilize_nfaces(ti_rm,     32u, 32u);
    const std::vector<float> ti_neg_til_f32 = tilize_nfaces(ti_neg_rm, 32u, 32u);

    const std::vector<uint16_t> tr_til     = fp32_to_bf16_vec(tr_til_f32);
    const std::vector<uint16_t> ti_til     = fp32_to_bf16_vec(ti_til_f32);
    const std::vector<uint16_t> ti_neg_til = fp32_to_bf16_vec(ti_neg_til_f32);

    WriteShard(cq, pp->tw_r_buf,     tr_til,     MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_buf,     ti_til,     MeshCoordinate(0, 0), false);
    WriteShard(cq, pp->tw_i_neg_buf, ti_neg_til, MeshCoordinate(0, 0), false);

    // ── Host scratch ────────────────────────────────────────────────────
    const size_t rm_floats  = static_cast<size_t>(num_tiles) * 32u * 32u;
    const size_t til_elems  = static_cast<size_t>(num_tiles) * fft_stockham::kTileElems;
    pp->in_r_rm .assign(rm_floats, 0.0f);
    pp->in_i_rm .assign(rm_floats, 0.0f);
    pp->in_r_til_f32.assign(til_elems, 0.0f);
    pp->in_i_til_f32.assign(til_elems, 0.0f);
    pp->in_r_til .assign(til_elems, 0u);
    pp->in_i_til .assign(til_elems, 0u);
    pp->out_r_til.assign(til_elems, 0u);
    pp->out_i_til.assign(til_elems, 0u);
    pp->out_r_rm .assign(rm_floats, 0.0f);
    pp->out_i_rm .assign(rm_floats, 0.0f);

    // ── Program ─────────────────────────────────────────────────────────
    Program prog = CreateProgram();

    const CoreCoord first{0, 0};
    const CoreCoord last{pp->grid_cols - 1, pp->grid_rows - 1};
    const CoreRange cr(first, last);

    // CBs: CB_A / CB_B depth 4 so the reader can queue all 4 matmul pairs
    // upfront. CB_OUT_R / CB_OUT_I depth 2, ordinary double-buffer.
    constexpr uint32_t kCbCount = 4u;
    constexpr uint32_t kCbTiles[kCbCount] = {4u, 4u, 2u, 2u};
    for (uint32_t id = 0; id < kCbCount; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * ts_bf16,
            {{id, tt::DataFormat::Float16_b}});
        c.set_page_size(id, ts_bf16);
        CreateCircularBuffer(prog, cr, c);
    }

    auto rk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_universal_bf16/kernel/packed_dft_bf16_reader.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    auto wk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_universal_bf16/kernel/packed_dft_bf16_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    // IMPORTANT (same trap the fp32 version documents): do NOT set
    // unpack_to_dest_mode = UnpackToDestFp32 on a matmul kernel. That
    // mode routes unpacker output straight into DST and leaves srcA/srcB
    // uninitialised, so matmul_tiles reads garbage (1e28 / inf outputs).
    // fp32_dest_acc_en=true keeps DST fp32 for the accumulating matmul,
    // the operand path stays bf16 (Float16_b CBs) — that's the TRUE-bf16
    // compute mode we want.
    auto ck = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft_universal_bf16/kernel/packed_dft_bf16_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args     = {pp->tiles_per_core}});
    (void)ck;

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
inline std::unordered_map<uint64_t, std::shared_ptr<PackedDFTBf16Plan>>&
packed_dft_bf16_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<PackedDFTBf16Plan>> c;
    return c;
}
inline uint64_t packed_dft_bf16_key(MeshDevice* md, uint32_t N, uint32_t count) {
    return reinterpret_cast<uint64_t>(md)
         ^ (uint64_t{N}     * 0x9E3779B97F4A7C15ull)
         ^ (uint64_t{count} * 0xBF58476D1CE4E5B9ull);
}
}  // namespace detail_packed

inline std::shared_ptr<PackedDFTBf16Plan> get_cached_packed_dft_bf16_plan(
    std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t count)
{
    const uint64_t key = detail_packed::packed_dft_bf16_key(md.get(), N, count);
    auto& cache = detail_packed::packed_dft_bf16_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto pp = make_packed_dft_bf16_plan(md, N, count);
    cache.emplace(key, pp);
    return pp;
}

// Host wrapper. Computes `count` independent length-N DFTs via the TRUE-bf16
// packed direct-DFT kernel. The fp32 input is converted to bf16 at pack
// time (once per call), ships as bf16 through DRAM/CBs/FPU, and the fp32
// output is reconstructed from the bf16 result at unpack time.
inline void packed_direct_dft_bf16_batched(
    std::shared_ptr<MeshDevice>   md,
    uint32_t                      N,
    uint32_t                      count,
    const std::vector<Complex>&   in_natural,
    std::vector<Complex>&         out_natural)
{
    using namespace tt::tt_metal::distributed;
    assert(in_natural.size() == static_cast<size_t>(count) * N);

    auto plan = get_cached_packed_dft_bf16_plan(md, N, count);
    constexpr uint32_t kRowsPerTile = 32u;

    std::vector<float>&    in_r_rm      = plan->in_r_rm;
    std::vector<float>&    in_i_rm      = plan->in_i_rm;
    std::vector<float>&    in_r_til_f32 = plan->in_r_til_f32;
    std::vector<float>&    in_i_til_f32 = plan->in_i_til_f32;
    std::vector<uint16_t>& in_r_til     = plan->in_r_til;
    std::vector<uint16_t>& in_i_til     = plan->in_i_til;
    std::vector<uint16_t>& out_r_til    = plan->out_r_til;
    std::vector<uint16_t>& out_i_til    = plan->out_i_til;

    // Pack natural-order fp32 input into row-major (32 * num_tiles) × 32.
    // Slot [r * 32 + k] for k ∈ [0, N) is sub-FFT r's sample k. Slots
    // [N, 32) stay zero. Rows r ∈ [count, 32 * num_tiles) stay zero.
    for (uint32_t r = 0; r < count; ++r) {
        const Complex* src = in_natural.data() + static_cast<size_t>(r) * N;
        float* tr = in_r_rm.data() + static_cast<size_t>(r) * 32u;
        float* ti = in_i_rm.data() + static_cast<size_t>(r) * 32u;
        for (uint32_t k = 0; k < N; ++k) {
            tr[k] = src[k].real();
            ti[k] = src[k].imag();
        }
    }

    // Row-major → tilized (still fp32) → bf16 tile bytes.
    const uint32_t total_rows = plan->num_tiles * kRowsPerTile;
    in_r_til_f32 = tilize_nfaces(in_r_rm, total_rows, 32u);
    in_i_til_f32 = tilize_nfaces(in_i_rm, total_rows, 32u);
    for (size_t i = 0; i < in_r_til_f32.size(); ++i) {
        in_r_til[i] = fp32_to_bf16(in_r_til_f32[i]);
        in_i_til[i] = fp32_to_bf16(in_i_til_f32[i]);
    }

    MeshCommandQueue& cq = plan->md->mesh_command_queue();
    WriteShard(cq, plan->in_r_buf, in_r_til, MeshCoordinate(0, 0), false);
    WriteShard(cq, plan->in_i_buf, in_i_til, MeshCoordinate(0, 0), false);

    EnqueueMeshWorkload(cq, plan->workload, false);

    ReadShard(cq, out_r_til, plan->out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i_til, plan->out_i_buf, MeshCoordinate(0, 0), true);

    // bf16 → fp32 → untilize → per-row unpack.
    const std::vector<float> out_r_til_f32 = bf16_to_fp32_vec(out_r_til);
    const std::vector<float> out_i_til_f32 = bf16_to_fp32_vec(out_i_til);
    std::vector<float>& out_r_rm = plan->out_r_rm;
    std::vector<float>& out_i_rm = plan->out_i_rm;
    out_r_rm = untilize_nfaces(out_r_til_f32, total_rows, 32u);
    out_i_rm = untilize_nfaces(out_i_til_f32, total_rows, 32u);

    out_natural.resize(static_cast<size_t>(count) * N);
    for (uint32_t r = 0; r < count; ++r) {
        const float* tr = out_r_rm.data() + static_cast<size_t>(r) * 32u;
        const float* ti = out_i_rm.data() + static_cast<size_t>(r) * 32u;
        Complex*     dst = out_natural.data() + static_cast<size_t>(r) * N;
        for (uint32_t k = 0; k < N; ++k) dst[k] = {tr[k], ti[k]};
    }
}

// ─── Top-level dispatcher ────────────────────────────────────────────────────
//
// Phase 1: only len in [2, 32] is handled (true-bf16 packed direct-DFT).
// Every other path hard-errors with an explanatory message. We intentionally
// refuse to delegate to fft_universal::fft on N > 32, because that would
// run the fp32 kernels and silently defeat the "true bf16" contract of
// this binary.
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    if (N == 1u) return signal;
    assert(N >= 1u && "FFT requires N >= 1");

    if (N >= 2u && N <= kPackedMaxN) {
        std::vector<Complex> out;
        packed_direct_dft_bf16_batched(md, N, /*count=*/1u, signal, out);
        return out;
    }

    // Phase 2 stubs — all paths still to be implemented as TRUE bf16 FPU
    // matmul kernels. Do not fall back to fp32; that would silently break
    // the precision contract of this binary.
    throw std::runtime_error(
        "fft_universal_bf16::fft: N=" + std::to_string(N) +
        " not yet implemented (Phase 2). Phase 1 covers N in [2, 32] via the "
        "true-bf16 packed direct-DFT kernel. Phase 2 will add:\n"
        "  * radix-32 Stockham bf16 matmul kernel for pow2 N > 32\n"
        "  * Bluestein (prime N) and Cooley-Tukey (composite N) via the "
        "pow2 kernel above\n"
        "Use fft_universal::fft() in the sibling fp32 binary for these sizes "
        "today.");
}

// Convenience overload for real input.
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

}  // namespace fft_universal_bf16
