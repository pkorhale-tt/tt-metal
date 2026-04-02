// =============================================================================
// fft_universal.cpp  —  Universal FFT host driver for Tenstorrent Wormhole
// Uses TT-Metalium (tt-Metal) SDK
//
// Tenstorrent Wormhole n300 specifics assumed:
//   • 8×8 Tensix grid  (64 usable compute cores)
//   • 1.5 MB L1 SRAM per Tensix core
//   • Two NOC fabrics (NOC-0 and NOC-1) per core
//   • SFPU: 32-element float32 vector unit
//   • No implicit synchronisation — all barriers are explicit
// =============================================================================

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/bfloat16.hpp"
#include "tt_metal/api/tt-metalium/device.hpp"
#include "tt_metal/api/tt-metalium/core_coord.hpp"

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
static constexpr uint32_t L1_BUDGET     = 1024 * 1024;  // 1 MB safe L1 budget per core
static constexpr uint32_t FLOAT_BYTES   = 4;
static constexpr uint32_t COMPLEX_BYTES = 2 * FLOAT_BYTES; // re + im = 8 bytes

static constexpr uint32_t SMALL_THRESH  =  4 * 1024;   // ≤ 4K  → Tier 1
static constexpr uint32_t MEDIUM_THRESH = 32 * 1024;   // ≤ 32K → Tier 2
                                                        //  > 32K → Tier 3

// ---------------------------------------------------------------------------
// Strategy tag
// ---------------------------------------------------------------------------
enum class FFTStrategy { SMALL, MEDIUM, LARGE };

// batch is unused in selection — strategy is purely size-driven.
// Suppress the unused-parameter warning with the unnamed parameter idiom.
static FFTStrategy select_strategy(uint32_t size, uint32_t /*batch*/) {
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
// Twiddle-factor precomputation (host side, uploaded once per call)
// W[k] = exp(-2πi·k/N), stored as interleaved [re, im, re, im, ...]
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
// Helper: create and register a circular buffer, discarding the handle.
// The handle is only needed if you later call UpdateCircularBufferConfig;
// discarding it avoids the unused-variable warning.
// ---------------------------------------------------------------------------
static void make_cb(
    Program&         program,
    const CoreRange& cr,
    uint32_t         idx,
    uint32_t         total_bytes,
    uint32_t         page_bytes)
{
    // Metalium requires CB pages to be 32-byte aligned
    auto aligned = [](uint32_t b){ return (b + 31) & ~31u; };
    CircularBufferConfig cfg(aligned(total_bytes), {{idx, tt::DataFormat::Float32}});
    cfg.set_page_size(idx, aligned(page_bytes));
    CreateCircularBuffer(program, cr, cfg);  // handle discarded intentionally
}

// ===========================================================================
//  TIER 1 — Small FFT  (size ≤ 4K)
//  Pack multiple FFTs per core. All data stays in L1 — zero DRAM in butterfly.
// ===========================================================================
static void run_small_fft(
    IDevice*       device,
    CommandQueue&  cq,
    Buffer&        src_buf,
    Buffer&        dst_buf,
    Buffer&        tw_buf,
    uint32_t       size,
    uint32_t       batch,
    bool           inverse)
{
    const uint32_t log2n         = static_cast<uint32_t>(std::log2(size));
    const uint32_t active_cores  = std::min(batch, 64u);
    const uint32_t ffts_per_core = (batch + active_cores - 1) / active_cores;
    const uint32_t bytes_per_fft = size * COMPLEX_BYTES;

    const uint32_t rows = (active_cores + 7) / 8;
    const uint32_t cols = std::min(active_cores, 8u);
    CoreRange core_range(CoreCoord{0, 0}, CoreCoord{cols - 1, rows - 1});

    Program program = CreateProgram();

    // CB0 — input  (ffts_per_core pages, each bytes_per_fft)
    // CB1 — twiddle table (1 page, stays resident for entire kernel lifetime)
    // CB2 — output (ffts_per_core pages)
    make_cb(program, core_range, 0, ffts_per_core * bytes_per_fft, bytes_per_fft);
    make_cb(program, core_range, 1, size * COMPLEX_BYTES,           size * COMPLEX_BYTES);
    make_cb(program, core_range, 2, ffts_per_core * bytes_per_fft, bytes_per_fft);

    auto reader = CreateKernel(
        program, "kernels/fft_small_reader.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default});

    auto compute = CreateKernel(
        program, "kernels/fft_small_compute.cpp", core_range,
        ComputeConfig{.math_fidelity   = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true,
                      .math_approx_mode = false,
                      .compile_args     = {log2n, (uint32_t)inverse}});

    auto writer = CreateKernel(
        program, "kernels/fft_small_writer.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default});

    uint32_t core_idx = 0;
    for (uint32_t r = 0; r < rows && core_idx < active_cores; ++r) {
        for (uint32_t c = 0; c < cols && core_idx < active_cores; ++c) {
            CoreCoord coord{c, r};
            uint32_t fft_offset = core_idx * ffts_per_core;
            uint32_t my_batch   = std::min(ffts_per_core, batch - fft_offset);

            SetRuntimeArgs(program, reader,  coord,
                {src_buf.address(), tw_buf.address(), fft_offset, my_batch, size});
            SetRuntimeArgs(program, compute, coord,
                {my_batch, size, log2n, (uint32_t)inverse});
            SetRuntimeArgs(program, writer,  coord,
                {dst_buf.address(), fft_offset, my_batch, size});
            ++core_idx;
        }
    }

    EnqueueProgram(cq, program, /*blocking=*/false);
}

// ===========================================================================
//  TIER 2 — Medium FFT  (4K < size ≤ 32K)
//  1 complete FFT per core. Entire FFT fits in L1 — no DRAM during butterfly.
// ===========================================================================
static void run_medium_fft(
    IDevice*       device,
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

    const uint32_t rows = (active_cores + 7) / 8;
    const uint32_t cols = std::min(active_cores, 8u);
    CoreRange core_range(CoreCoord{0, 0}, CoreCoord{cols - 1, rows - 1});

    const uint32_t fft_bytes = size * COMPLEX_BYTES;
    const uint32_t tw_bytes  = size * COMPLEX_BYTES;

    // Verify L1 budget: input + scratch + twiddles must fit
    if (fft_bytes * 2 + tw_bytes > L1_BUDGET) {
        throw std::runtime_error(
            "Medium FFT size=" + std::to_string(size) +
            " needs " + std::to_string(fft_bytes*2 + tw_bytes) +
            " bytes, L1 budget=" + std::to_string(L1_BUDGET));
    }

    Program program = CreateProgram();

    // CB0=input, CB1=scratch(ping-pong), CB2=twiddles, CB3=output
    make_cb(program, core_range, 0, fft_bytes, fft_bytes);
    make_cb(program, core_range, 1, fft_bytes, fft_bytes);
    make_cb(program, core_range, 2, tw_bytes,  tw_bytes);
    make_cb(program, core_range, 3, fft_bytes, fft_bytes);

    auto reader = CreateKernel(
        program, "kernels/fft_medium_reader.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc       = NOC::RISCV_0_default});

    auto compute = CreateKernel(
        program, "kernels/fft_medium_compute.cpp", core_range,
        ComputeConfig{.math_fidelity   = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true,
                      .compile_args    = {log2n, (uint32_t)inverse}});

    auto writer = CreateKernel(
        program, "kernels/fft_medium_writer.cpp", core_range,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc       = NOC::RISCV_1_default});

    const uint32_t ffts_per_core = (batch + active_cores - 1) / active_cores;
    for (uint32_t i = 0; i < active_cores; ++i) {
        CoreCoord coord{i % cols, i / cols};
        uint32_t fft_offset = i * ffts_per_core;
        uint32_t my_batch   = std::min(ffts_per_core, batch - fft_offset);

        SetRuntimeArgs(program, reader,  coord,
            {src_buf.address(), tw_buf.address(), fft_offset, my_batch, size});
        SetRuntimeArgs(program, compute, coord,
            {my_batch, size, log2n, (uint32_t)inverse});
        SetRuntimeArgs(program, writer,  coord,
            {dst_buf.address(), fft_offset, my_batch, size});
    }

    EnqueueProgram(cq, program, /*blocking=*/false);
}

// ===========================================================================
//  TIER 3 — Large FFT  (size > 32K)
//  All 64 cores cooperate via 2D Cooley–Tukey (N = R × S).
//  Batched large: cores split into groups, one group per FFT.
// ===========================================================================
static void run_large_fft(
    IDevice*       device,
    CommandQueue&  cq,
    Buffer&        src_buf,
    Buffer&        dst_buf,
    Buffer&        tw_buf,
    uint32_t       size,
    uint32_t       batch,
    bool           inverse)
{
    const uint32_t log2n = static_cast<uint32_t>(std::log2(size));
    const uint32_t log2R = log2n / 2;
    const uint32_t log2S = log2n - log2R;
    const uint32_t R     = 1u << log2R;
    const uint32_t S     = size / R;

    const uint32_t cores_per_fft = std::max(1u, 64u / batch);
    const uint32_t active_groups = std::min(batch, 64u);

    std::vector<CoreRange> group_ranges;
    group_ranges.reserve(active_groups);
    for (uint32_t g = 0; g < active_groups; ++g) {
        uint32_t first = g * cores_per_fft;
        uint32_t last  = first + cores_per_fft - 1;
        group_ranges.push_back(CoreRange(
            CoreCoord{first % 8, first / 8},
            CoreCoord{last  % 8, last  / 8}));
    }

    Program program = CreateProgram();

    for (uint32_t g = 0; g < active_groups; ++g) {
        auto& cr = group_ranges[g];

        const uint32_t rows_per_core = std::max(1u, R / cores_per_fft);
        const uint32_t chunk_bytes   = rows_per_core * S * COMPLEX_BYTES;

        // CB0=input chunk, CB1=twiddle-R, CB2=twiddle-S, CB3=output chunk
        make_cb(program, cr, 0, chunk_bytes,       chunk_bytes);
        make_cb(program, cr, 1, R * COMPLEX_BYTES, R * COMPLEX_BYTES);
        make_cb(program, cr, 2, S * COMPLEX_BYTES, S * COMPLEX_BYTES);
        make_cb(program, cr, 3, chunk_bytes,       chunk_bytes);

        auto reader = CreateKernel(
            program, "kernels/fft_large_reader1.cpp", cr,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc       = NOC::RISCV_0_default});

        auto compute = CreateKernel(
            program, "kernels/fft_large_compute.cpp", cr,
            ComputeConfig{.math_fidelity   = MathFidelity::HiFi4,
                          .fp32_dest_acc_en = true,
                          .compile_args    = {log2R, log2S, (uint32_t)inverse,
                                              cores_per_fft, rows_per_core}});

        auto writer = CreateKernel(
            program, "kernels/fft_large_writer.cpp", cr,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc       = NOC::RISCV_1_default});

        uint32_t local_core = 0;
        for (uint32_t r = cr.start_coord.y; r <= cr.end_coord.y; ++r) {
            for (uint32_t c = cr.start_coord.x; c <= cr.end_coord.x; ++c) {
                CoreCoord coord{c, r};
                uint32_t row_start = local_core * rows_per_core;

                SetRuntimeArgs(program, reader,  coord,
                    {src_buf.address(), tw_buf.address(),
                     g, row_start, rows_per_core, R, S, (uint32_t)inverse});
                SetRuntimeArgs(program, compute, coord,
                    {row_start, rows_per_core, R, S, local_core, g});
                SetRuntimeArgs(program, writer,  coord,
                    {dst_buf.address(), g, row_start, rows_per_core, R, S});
                ++local_core;
            }
        }
    }

    EnqueueProgram(cq, program, /*blocking=*/false);
}

// ===========================================================================
//  PUBLIC API
// ===========================================================================

// FFTContext — open device once, reuse across many fft_universal() calls.
struct FFTContext {
    IDevice*      device = nullptr;
    CommandQueue* cq     = nullptr;

    FFTContext() = default;

    explicit FFTContext(int device_id) {
        device = CreateDevice(device_id);
        cq     = &device->command_queue(0);  // queue 0, lifetime tied to device
    }

    ~FFTContext() {
        if (device) CloseDevice(device);
    }

    FFTContext(const FFTContext&)            = delete;
    FFTContext& operator=(const FFTContext&) = delete;
};

// ---------------------------------------------------------------------------
// fft_universal — single entry point for all sizes and batch counts.
//
//   data_host : float32 interleaved complex [batch * size * 2]
//               layout per FFT: [re0, im0, re1, im1, ...]
//   out_host  : same layout, pre-allocated by caller
//   size      : complex points per FFT (must be power-of-2)
//   batch     : number of independent FFTs
//   inverse   : false=FFT, true=IFFT (result NOT normalised; divide by size)
// ---------------------------------------------------------------------------
void fft_universal(
    FFTContext&  ctx,
    const float* data_host,
    float*       out_host,
    uint32_t     size,
    uint32_t     batch,
    bool         inverse = false)
{
    assert(size >= 2 && (size & (size - 1)) == 0 && "size must be power of 2");
    assert(batch >= 1);

    IDevice*      device = ctx.device;
    CommandQueue& cq     = *ctx.cq;

    const uint32_t total_bytes = batch * size * COMPLEX_BYTES;

    FFTStrategy strategy = select_strategy(size, batch);
    std::cout << "[fft_universal] size=" << size
              << " batch=" << batch
              << " strategy=" << strategy_name(strategy) << "\n";

    auto twiddles = precompute_twiddles(size);

    InterleavedBufferConfig data_cfg{
        .device      = device,
        .size        = total_bytes,
        .page_size   = COMPLEX_BYTES,
        .buffer_type = BufferType::DRAM
    };
    InterleavedBufferConfig tw_cfg{
        .device      = device,
        .size        = static_cast<uint32_t>(twiddles.size() * FLOAT_BYTES),
        .page_size   = COMPLEX_BYTES,
        .buffer_type = BufferType::DRAM
    };

    auto src_buf = CreateBuffer(data_cfg);
    auto dst_buf = CreateBuffer(data_cfg);
    auto tw_buf  = CreateBuffer(tw_cfg);

    EnqueueWriteBuffer(cq, src_buf, data_host,       /*blocking=*/false);
    EnqueueWriteBuffer(cq, tw_buf,  twiddles.data(),  /*blocking=*/false);

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

    EnqueueReadBuffer(cq, dst_buf, out_host, /*blocking=*/true);
}

// ===========================================================================
//  Smoke tests
// ===========================================================================
int main() {
    FFTContext ctx(/*device_id=*/0);

    // Test 1: 256 × 1K — Tier 1
    {
        const uint32_t N = 1024, B = 256;
        std::vector<float> in(B * N * 2, 0.f), out(B * N * 2);
        for (uint32_t b = 0; b < B; ++b) in[b * N * 2] = 1.f;  // unit impulse

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Tier1 smoke: out[0]=" << out[0]
                  << " out[2]=" << out[2] << "  (both ~1.0)\n";
    }

    // Test 2: 1 × 64K — Tier 3
    {
        const uint32_t N = 65536, B = 1;
        std::vector<float> in(N * 2, 0.f), out(N * 2);
        in[0] = 1.f;

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Tier3 smoke: out[0]=" << out[0] << "  (~1.0)\n";
    }

    // Test 3: round-trip IFFT(FFT(x)) / N ≈ x
    {
        const uint32_t N = 8192, B = 4;
        std::vector<float> in(B*N*2), fwd(B*N*2), inv_out(B*N*2);
        for (auto& v : in) v = static_cast<float>(rand()) / RAND_MAX;

        fft_universal(ctx, in.data(),  fwd.data(),     N, B, false);
        fft_universal(ctx, fwd.data(), inv_out.data(), N, B, true);

        float max_err = 0.f;
        for (uint32_t i = 0; i < B*N*2; ++i)
            max_err = std::max(max_err, std::abs(inv_out[i] / N - in[i]));
        std::cout << "Round-trip max error: " << max_err << "  (< 1e-4)\n";
    }

    return 0;
}