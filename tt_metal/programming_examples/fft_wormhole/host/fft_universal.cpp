// =============================================================================
// fft_universal.cpp  —  Universal FFT host driver for Tenstorrent Wormhole
// Uses TT-Metalium (tt-Metal) SDK
//
// Build:
//   g++ -std=c++17 -O2 -I$TT_METAL_HOME \
//       fft_universal.cpp -ltt_metal -o fft_universal
//
// Tenstorrent Wormhole n300 specifics assumed:
//   • 8×8 Tensix grid  (64 usable compute cores)
//   • 1.5 MB L1 SRAM per Tensix core
//   • Two NOC fabrics (NOC-0 and NOC-1) per core
//   • SFPU: 32-element float32 vector unit
//   • No implicit synchronisation — all barriers are explicit
// =============================================================================

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "common/core_coord.h"

#include <cstdint>
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tt;
using namespace tt::tt_metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t TILE_HW       = 32;           // Wormhole SFPU tile width
static constexpr uint32_t L1_BUDGET     = 1024 * 1024;  // 1 MB safe L1 budget per core
static constexpr uint32_t FLOAT_BYTES   = 4;
// Each FFT point = complex = 2 floats = 8 bytes
static constexpr uint32_t COMPLEX_BYTES = 2 * FLOAT_BYTES;

static constexpr uint32_t SMALL_THRESH  =  4 * 1024;   // ≤ 4K  → Tier 1
static constexpr uint32_t MEDIUM_THRESH = 32 * 1024;   // ≤ 32K → Tier 2
                                                        //  > 32K → Tier 3

// ---------------------------------------------------------------------------
// Strategy tag
// ---------------------------------------------------------------------------
enum class FFTStrategy { SMALL, MEDIUM, LARGE };

static FFTStrategy select_strategy(uint32_t size, uint32_t batch) {
    if (size <= SMALL_THRESH)  return FFTStrategy::SMALL;
    if (size <= MEDIUM_THRESH) return FFTStrategy::MEDIUM;
    return FFTStrategy::LARGE;
}

static const char* strategy_name(FFTStrategy s) {
    switch (s) {
        case FFTStrategy::SMALL:  return "Tier1-Small";
        case FFTStrategy::MEDIUM: return "Tier2-Medium";
        case FFTStrategy::LARGE:  return "Tier3-Large";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Twiddle-factor precomputation  (host side, uploaded once at init time)
// W[k] = exp(-2πi·k/N)  stored as interleaved [re, im, re, im, ...]
// ---------------------------------------------------------------------------
static std::vector<float> precompute_twiddles(uint32_t N) {
    std::vector<float> tw(N * 2);
    for (uint32_t k = 0; k < N; ++k) {
        double angle = -2.0 * M_PI * k / N;
        tw[2*k]   = static_cast<float>(std::cos(angle));
        tw[2*k+1] = static_cast<float>(std::sin(angle));
    }
    return tw;
}

// ---------------------------------------------------------------------------
// Circular-buffer descriptor helper
// ---------------------------------------------------------------------------
static uint32_t cb_page_size_aligned(uint32_t bytes) {
    // Metalium requires CB pages to be 32-byte aligned
    return (bytes + 31) & ~31u;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  TIER 1 — Small FFT  (size ≤ 4K)
//  Strategy: pack multiple FFTs per core, pure SRAM, NO NOC during butterfly.
//  Mapping:  min(batch, 64) cores active; each core handles ⌈batch/active⌉ FFTs.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
static void run_small_fft(
    Device*        device,
    CommandQueue&  cq,
    Buffer&        src_buf,      // interleaved complex float32 on DRAM
    Buffer&        dst_buf,      // interleaved complex float32 on DRAM
    Buffer&        tw_buf,       // twiddle factors on DRAM (size * 2 floats)
    uint32_t       size,
    uint32_t       batch,
    bool           inverse)
{
    const uint32_t log2n         = static_cast<uint32_t>(std::log2(size));
    const uint32_t active_cores  = std::min(batch, 64u);
    const uint32_t ffts_per_core = (batch + active_cores - 1) / active_cores;
    const uint32_t bytes_per_fft = size * COMPLEX_BYTES;

    // ---- Build core range (row-major, top-left of the 8×8 grid) -----------
    uint32_t rows = (active_cores + 7) / 8;
    uint32_t cols = std::min(active_cores, 8u);
    CoreRange core_range(CoreCoord{0, 0}, CoreCoord{cols - 1, rows - 1});

    // ---- Program -----------------------------------------------------------
    Program program = CreateProgram();

    // Circular buffers per core:
    //   CB0 — input  data  (ffts_per_core pages, each = bytes_per_fft)
    //   CB1 — twiddle table (1 page = size * COMPLEX_BYTES)
    //   CB2 — output data  (ffts_per_core pages)
    uint32_t cb0_page = cb_page_size_aligned(bytes_per_fft);
    uint32_t cb1_page = cb_page_size_aligned(size * COMPLEX_BYTES);
    uint32_t cb2_page = cb_page_size_aligned(bytes_per_fft);

    CircularBufferConfig cb0_cfg(ffts_per_core * cb0_page, {{0, tt::DataFormat::Float32}});
    cb0_cfg.set_page_size(0, cb0_page);
    auto cb0 = CreateCircularBuffer(program, core_range, cb0_cfg);

    CircularBufferConfig cb1_cfg(cb1_page, {{1, tt::DataFormat::Float32}});
    cb1_cfg.set_page_size(1, cb1_page);
    auto cb1 = CreateCircularBuffer(program, core_range, cb1_cfg);

    CircularBufferConfig cb2_cfg(ffts_per_core * cb2_page, {{2, tt::DataFormat::Float32}});
    cb2_cfg.set_page_size(2, cb2_page);
    auto cb2 = CreateCircularBuffer(program, core_range, cb2_cfg);

    // ---- Reader kernel (RISC0 / data-movement-0) ---------------------------
    auto reader = CreateKernel(
        program,
        "kernels/fft_small_reader.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default});

    // ---- Compute kernel  ---------------------------------------------------
    auto compute = CreateKernel(
        program,
        "kernels/fft_small_compute.cpp",
        core_range,
        ComputeConfig{.math_fidelity          = MathFidelity::HiFi4,
                      .fp32_dest_acc_en        = true,
                      .math_approx_mode        = false,
                      .compile_args            = {log2n, (uint32_t)inverse}});

    // ---- Writer kernel (RISC1 / data-movement-1) ---------------------------
    auto writer = CreateKernel(
        program,
        "kernels/fft_small_writer.cpp",
        core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default});

    // ---- Runtime args per core -------------------------------------------
    uint32_t core_idx = 0;
    for (uint32_t r = 0; r < rows && core_idx < active_cores; ++r) {
        for (uint32_t c = 0; c < cols && core_idx < active_cores; ++c) {
            CoreCoord coord{c, r};
            uint32_t fft_offset = core_idx * ffts_per_core;
            uint32_t my_batch   = std::min(ffts_per_core, batch - fft_offset);

            // Reader args: src_buf addr, twiddle addr, fft_offset, my_batch, size
            SetRuntimeArgs(program, reader, coord, {
                src_buf.address(),
                tw_buf.address(),
                fft_offset,
                my_batch,
                size
            });

            // Compute args: my_batch, size, log2n, inverse
            SetRuntimeArgs(program, compute, coord, {
                my_batch, size, log2n, (uint32_t)inverse
            });

            // Writer args: dst_buf addr, fft_offset, my_batch, size
            SetRuntimeArgs(program, writer, coord, {
                dst_buf.address(),
                fft_offset,
                my_batch,
                size
            });

            ++core_idx;
        }
    }

    EnqueueProgram(cq, program, /*blocking=*/false);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  TIER 2 — Medium FFT  (4K < size ≤ 32K)
//  Strategy: 1 complete FFT per core, L1 NOC sync between butterfly stages.
//  Mapping:  each core independently computes its assigned FFT.
//            Batch distributed across up to 64 cores.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
static void run_medium_fft(
    Device*        device,
    CommandQueue&  cq,
    Buffer&        src_buf,
    Buffer&        dst_buf,
    Buffer&        tw_buf,
    uint32_t       size,
    uint32_t       batch,
    bool           inverse)
{
    const uint32_t log2n        = static_cast<uint32_t>(std::log2(size));
    const uint32_t active_cores = std::min(batch, 64u);

    uint32_t rows = (active_cores + 7) / 8;
    uint32_t cols = std::min(active_cores, 8u);
    CoreRange core_range(CoreCoord{0, 0}, CoreCoord{cols - 1, rows - 1});

    Program program = CreateProgram();

    // Each core holds one full FFT in L1 + twiddle table
    uint32_t fft_bytes = size * COMPLEX_BYTES;
    uint32_t tw_bytes  = size * COMPLEX_BYTES;

    // Verify L1 budget: data + twiddles + scratch must fit
    uint32_t l1_needed = fft_bytes * 2 + tw_bytes; // ping-pong + twiddles
    if (l1_needed > L1_BUDGET) {
        throw std::runtime_error(
            "Medium FFT of size " + std::to_string(size) +
            " exceeds L1 budget (" + std::to_string(l1_needed) + " > " +
            std::to_string(L1_BUDGET) + " bytes). Use Tier 3.");
    }

    // CB layout per core: CB0=input(ping), CB1=scratch(pong), CB2=twiddles, CB3=output
    auto make_cb = [&](uint32_t idx, uint32_t bytes) {
        uint32_t pgsz = cb_page_size_aligned(bytes);
        CircularBufferConfig cfg(pgsz, {{idx, tt::DataFormat::Float32}});
        cfg.set_page_size(idx, pgsz);
        return CreateCircularBuffer(program, core_range, cfg);
    };

    auto cb_in  = make_cb(0, fft_bytes);
    auto cb_scr = make_cb(1, fft_bytes);
    auto cb_tw  = make_cb(2, tw_bytes);
    auto cb_out = make_cb(3, fft_bytes);

    auto reader = CreateKernel(program, "kernels/fft_medium_reader.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default});

    auto compute = CreateKernel(program, "kernels/fft_medium_compute.cpp", core_range,
        ComputeConfig{.math_fidelity   = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true,
                      .compile_args    = {log2n, (uint32_t)inverse}});

    auto writer = CreateKernel(program, "kernels/fft_medium_writer.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default});

    for (uint32_t i = 0; i < active_cores; ++i) {
        CoreCoord coord{i % cols, i / cols};
        // Each core processes ceil(batch/active_cores) FFTs sequentially
        uint32_t fft_offset = i * ((batch + active_cores - 1) / active_cores);
        uint32_t my_batch   = std::min((batch + active_cores - 1) / active_cores,
                                       batch - fft_offset);

        SetRuntimeArgs(program, reader, coord,
            {src_buf.address(), tw_buf.address(), fft_offset, my_batch, size});
        SetRuntimeArgs(program, compute, coord,
            {my_batch, size, log2n, (uint32_t)inverse});
        SetRuntimeArgs(program, writer, coord,
            {dst_buf.address(), fft_offset, my_batch, size});
    }

    EnqueueProgram(cq, program, /*blocking=*/false);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  TIER 3 — Large FFT  (size > 32K)
//  Strategy: ALL 64 cores cooperate on a SINGLE FFT.
//  Algorithm: 2D Cooley–Tukey decomposition
//    • Decompose N into (R × S) where R = sqrt(N), S = N/R
//    • Phase 1: Each core computes S-point FFTs on R rows  (in-SRAM)
//    • Transpose: NOC multicast row data to column-owning cores
//    • Phase 2: Each core computes R-point FFTs on S columns (in-SRAM)
//  For batched large FFTs: split cores into groups, each group handles 1 FFT.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
static void run_large_fft(
    Device*        device,
    CommandQueue&  cq,
    Buffer&        src_buf,
    Buffer&        dst_buf,
    Buffer&        tw_buf,
    uint32_t       size,
    uint32_t       batch,
    bool           inverse)
{
    const uint32_t log2n = static_cast<uint32_t>(std::log2(size));

    // Decompose N = R * S  (prefer square-ish split)
    uint32_t R = 1u << (log2n / 2);          // row-FFT length
    uint32_t S = size / R;                   // column-FFT length
    uint32_t log2R = log2n / 2;
    uint32_t log2S = log2n - log2R;

    // For batched large: group cores.  Each group = 64 / batch cores (clamped).
    uint32_t cores_per_fft = std::max(1u, 64u / batch);
    uint32_t active_groups = std::min(batch, 64u);

    // ---- Per-group core ranges --------------------------------------------
    std::vector<CoreRange> group_ranges;
    for (uint32_t g = 0; g < active_groups; ++g) {
        uint32_t c0 = (g * cores_per_fft) % 8;
        uint32_t r0 = (g * cores_per_fft) / 8;
        uint32_t c1 = ((g * cores_per_fft + cores_per_fft - 1)) % 8;
        uint32_t r1 = ((g * cores_per_fft + cores_per_fft - 1)) / 8;
        group_ranges.push_back(CoreRange(CoreCoord{c0, r0}, CoreCoord{c1, r1}));
    }

    // ---- Program -----------------------------------------------------------
    Program program = CreateProgram();

    for (uint32_t g = 0; g < active_groups; ++g) {
        auto& cr = group_ranges[g];

        // Each core in the group owns (R / cores_per_fft) rows of the matrix
        uint32_t rows_per_core = std::max(1u, R / cores_per_fft);
        uint32_t row_bytes     = S * COMPLEX_BYTES;       // one row of the N-point DFT matrix
        uint32_t chunk_bytes   = rows_per_core * row_bytes;

        // CB layout: CB0=input chunk, CB1=twiddle-R, CB2=twiddle-S, CB3=output chunk
        auto make_cb_g = [&](uint32_t idx, uint32_t bytes) {
            uint32_t pgsz = cb_page_size_aligned(bytes);
            CircularBufferConfig cfg(pgsz, {{idx, tt::DataFormat::Float32}});
            cfg.set_page_size(idx, pgsz);
            return CreateCircularBuffer(program, cr, cfg);
        };

        make_cb_g(0, chunk_bytes);             // input rows
        make_cb_g(1, R * COMPLEX_BYTES);       // twiddle for R-point FFT
        make_cb_g(2, S * COMPLEX_BYTES);       // twiddle for S-point FFT
        make_cb_g(3, chunk_bytes);             // output rows

        // Phase-1: row FFTs
        auto reader1 = CreateKernel(program, "kernels/fft_large_reader1.cpp", cr,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc       = NOC::RISCV_0_default});

        // Phase-2 after NOC transpose: column FFTs
        auto compute = CreateKernel(program, "kernels/fft_large_compute.cpp", cr,
            ComputeConfig{.math_fidelity    = MathFidelity::HiFi4,
                          .fp32_dest_acc_en  = true,
                          .compile_args     = {log2R, log2S, (uint32_t)inverse,
                                               cores_per_fft, rows_per_core}});

        auto writer = CreateKernel(program, "kernels/fft_large_writer.cpp", cr,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc       = NOC::RISCV_1_default});

        // Runtime args for every core in this group
        uint32_t local_core = 0;
        for (uint32_t r = cr.start_coord.y; r <= cr.end_coord.y; ++r) {
            for (uint32_t c = cr.start_coord.x; c <= cr.end_coord.x; ++c) {
                CoreCoord coord{c, r};
                uint32_t row_start = local_core * rows_per_core;

                SetRuntimeArgs(program, reader1, coord, {
                    src_buf.address(), tw_buf.address(),
                    g,               // batch index
                    row_start,
                    rows_per_core,
                    R, S,
                    (uint32_t)inverse
                });
                SetRuntimeArgs(program, compute, coord, {
                    row_start, rows_per_core, R, S, local_core, g
                });
                SetRuntimeArgs(program, writer, coord, {
                    dst_buf.address(), g, row_start, rows_per_core, R, S
                });
                ++local_core;
            }
        }
    }

    EnqueueProgram(cq, program, /*blocking=*/false);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  PUBLIC API
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// Context holds device + buffers across calls (init once, call many times)
struct FFTContext {
    Device*      device   = nullptr;
    CommandQueue cq;
    uint32_t     max_size = 0;

    FFTContext() = default;
    explicit FFTContext(int device_id) {
        device = CreateDevice(device_id);
        cq     = CommandQueue(device, /*id=*/0);
    }
    ~FFTContext() {
        if (device) CloseDevice(device);
    }
};

// ---------------------------------------------------------------------------
// fft_universal — the single entry point described in your architecture doc
//
//   data_host : interleaved float32 complex array  [batch * size * 2]
//               format: [re0, im0, re1, im1, ...]  per FFT, FFTs back-to-back
//   out_host  : same layout, will hold results
//   size      : number of complex points per FFT  (must be power-of-2)
//   batch     : number of independent FFTs
//   inverse   : true = IFFT  (output is NOT normalised — divide by N yourself)
// ---------------------------------------------------------------------------
void fft_universal(
    FFTContext&  ctx,
    const float* data_host,
    float*       out_host,
    uint32_t     size,
    uint32_t     batch,
    bool         inverse = false)
{
    // --- Validate -----------------------------------------------------------
    assert(size >= 2 && (size & (size - 1)) == 0 && "size must be power of 2");
    assert(batch >= 1);

    Device*       device = ctx.device;
    CommandQueue& cq     = ctx.cq;

    const uint32_t total_floats = batch * size * 2;
    const uint32_t total_bytes  = total_floats * FLOAT_BYTES;

    // --- Pick strategy -------------------------------------------------------
    FFTStrategy strategy = select_strategy(size, batch);
    std::cout << "[fft_universal] size=" << size << " batch=" << batch
              << " strategy=" << strategy_name(strategy) << "\n";

    // --- Precompute twiddles ------------------------------------------------
    auto twiddles = precompute_twiddles(size);

    // --- Allocate DRAM buffers ----------------------------------------------
    InterleavedBufferConfig src_cfg{
        .device         = device,
        .size           = total_bytes,
        .page_size      = COMPLEX_BYTES,           // one complex sample per page
        .buffer_type    = BufferType::DRAM
    };
    InterleavedBufferConfig tw_cfg{
        .device         = device,
        .size           = (uint32_t)(twiddles.size() * FLOAT_BYTES),
        .page_size      = COMPLEX_BYTES,
        .buffer_type    = BufferType::DRAM
    };

    auto src_buf = CreateBuffer(src_cfg);
    auto dst_buf = CreateBuffer(src_cfg);   // same layout as src
    auto tw_buf  = CreateBuffer(tw_cfg);

    // --- Upload host → DRAM -------------------------------------------------
    EnqueueWriteBuffer(cq, src_buf, data_host,     /*blocking=*/false);
    EnqueueWriteBuffer(cq, tw_buf,  twiddles.data(), /*blocking=*/false);

    // --- Dispatch to the right tier -----------------------------------------
    switch (strategy) {
        case FFTStrategy::SMALL:
            run_small_fft (device, cq, *src_buf, *dst_buf, *tw_buf, size, batch, inverse);
            break;
        case FFTStrategy::MEDIUM:
            run_medium_fft(device, cq, *src_buf, *dst_buf, *tw_buf, size, batch, inverse);
            break;
        case FFTStrategy::LARGE:
            run_large_fft (device, cq, *src_buf, *dst_buf, *tw_buf, size, batch, inverse);
            break;
    }

    // --- Read back DRAM → host  (this call implicitly flushes the queue) ----
    EnqueueReadBuffer(cq, dst_buf, out_host, /*blocking=*/true);
}

// ---------------------------------------------------------------------------
// Quick smoke test  (replace with your proper benchmark harness)
// ---------------------------------------------------------------------------
int main() {
    FFTContext ctx(/*device_id=*/0);

    // --- Test 1: 256 × 1K  (Tier 1 — small, batched ML workload) -----------
    {
        const uint32_t N = 1024, B = 256;
        std::vector<float> in(B * N * 2, 0.f), out(B * N * 2);
        // Simple impulse at index 0 of each FFT → flat spectrum
        for (uint32_t b = 0; b < B; ++b) in[b * N * 2] = 1.f;

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Tier1 smoke: out[0]=" << out[0]
                  << " out[2]=" << out[2] << " (both should be ~1.0)\n";
    }

    // --- Test 2: 1 × 64K  (Tier 3 — large single FFT) ----------------------
    {
        const uint32_t N = 65536, B = 1;
        std::vector<float> in(N * 2, 0.f), out(N * 2);
        in[0] = 1.f;

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Tier3 smoke: out[0]=" << out[0] << " (should be ~1.0)\n";
    }

    // --- Test 3: round-trip  IFFT(FFT(x)) ≈ x  (normalised) ----------------
    {
        const uint32_t N = 8192, B = 4;
        std::vector<float> in(B * N * 2), out_fwd(B * N * 2), out_inv(B * N * 2);
        for (auto& v : in) v = static_cast<float>(rand()) / RAND_MAX;

        fft_universal(ctx, in.data(),     out_fwd.data(), N, B, false);
        fft_universal(ctx, out_fwd.data(), out_inv.data(), N, B, true);

        // Normalise by N and check
        float max_err = 0.f;
        for (uint32_t i = 0; i < B * N * 2; ++i) {
            float reconstructed = out_inv[i] / N;
            max_err = std::max(max_err, std::abs(reconstructed - in[i]));
        }
        std::cout << "Round-trip max error: " << max_err
                  << " (should be < 1e-4)\n";
    }

    return 0;
}