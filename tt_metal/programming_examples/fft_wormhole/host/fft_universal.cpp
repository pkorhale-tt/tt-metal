// =============================================================================
// fft_universal.cpp  —  Universal FFT host driver for Tenstorrent Wormhole
// Fixed version: targets 64×64 = 4096-point 1D FFT batch across cores.
// =============================================================================

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/bfloat16.hpp"
#include "tt_metal/api/tt-metalium/device.hpp"
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/tensor_accessor_args.hpp"

#include <cstdint>
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <unordered_map>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t L1_BUDGET     = 1024 * 1024;   // 1 MB (Wormhole has 1.3 MB; keep headroom)
static constexpr uint32_t FLOAT_BYTES   = 4;
static constexpr uint32_t COMPLEX_BYTES = 2 * FLOAT_BYTES;

// FIX 1: Medium threshold raised so 4096-point (64×64) fits in Medium tier.
// 4096 × 8 bytes × 4 CBs = 131072 bytes < 1 MB.  32 K was fine too, keeping it.
static constexpr uint32_t MEDIUM_THRESH = 32 * 1024;

// ---------------------------------------------------------------------------
// Strategy tag
// ---------------------------------------------------------------------------
enum class FFTStrategy { MEDIUM, LARGE };

static FFTStrategy select_strategy(uint32_t size, uint32_t /*batch*/) {
    if (size <= MEDIUM_THRESH) return FFTStrategy::MEDIUM;
    return FFTStrategy::LARGE;
}

static const char* strategy_name(FFTStrategy s) {
    switch (s) {
        case FFTStrategy::MEDIUM: return "Tier2-Medium";
        case FFTStrategy::LARGE:  return "Tier3-Large";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Twiddle cache
// ---------------------------------------------------------------------------
static std::unordered_map<uint32_t, std::vector<float>> g_twiddle_cache;

static const std::vector<float>& get_twiddles(uint32_t N) {
    auto it = g_twiddle_cache.find(N);
    if (it != g_twiddle_cache.end()) return it->second;

    std::vector<float> tw(N * 2);
    for (uint32_t k = 0; k < N; ++k) {
        double angle = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(N);
        tw[2 * k]     = static_cast<float>(std::cos(angle));
        tw[2 * k + 1] = static_cast<float>(std::sin(angle));
    }
    g_twiddle_cache[N] = std::move(tw);
    return g_twiddle_cache[N];
}

// Tier 3 split twiddle buffer: [R twiddles | S twiddles]
static std::vector<float> make_large_twiddles(uint32_t R, uint32_t S) {
    const auto& tw_r = get_twiddles(R);
    const auto& tw_s = get_twiddles(S);

    std::vector<float> combined;
    combined.reserve(tw_r.size() + tw_s.size());
    combined.insert(combined.end(), tw_r.begin(), tw_r.end());
    combined.insert(combined.end(), tw_s.begin(), tw_s.end());
    return combined;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// FIX 2: Align to 32 bytes (TT-Metal CB requirement).
static uint32_t align32(uint32_t x) {
    return (x + 31u) & ~31u;
}

static void make_cb(
    Program&         program,
    const CoreRange& cr,
    uint32_t         idx,
    uint32_t         total_bytes,
    uint32_t         page_bytes)
{
    CircularBufferConfig cfg(
        align32(total_bytes),
        {{idx, tt::DataFormat::Float32}}
    );
    cfg.set_page_size(idx, align32(page_bytes));
    CreateCircularBuffer(program, cr, cfg);
}

// FIX 3: make_accessor_args helpers — use raw buffer address directly as a
// single uint32 runtime arg.  TensorAccessorArgs is for tile-layout buffers;
// our buffers are raw interleaved DRAM with page_size=8 (one complex float).
// Using TensorAccessorArgs on a raw buffer causes the reader kernels to receive
// incorrect bank addresses and is the primary cause of hangs/stalls.
static uint32_t buf_addr(const std::shared_ptr<distributed::MeshBuffer>& b) {
    return b->address();
}

static CoreCoord get_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* dev = mesh_device->get_devices()[0];
    return dev->compute_with_storage_grid_size();
}

// ===========================================================================
//  TIER 2 — Medium FFT
//  Distributes a batch of N-point FFTs across the full worker grid.
//  Each core gets ceil(batch / num_cores) FFTs.
//  All data fits in L1 — no inter-core NOC during butterfly computation.
// ===========================================================================
static void run_medium_fft(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue&                  cq,
    const std::shared_ptr<distributed::MeshBuffer>& src_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    const std::shared_ptr<distributed::MeshBuffer>& tw_buf,
    uint32_t                                        size,
    uint32_t                                        batch,
    bool                                            inverse)
{
    const uint32_t log2n     = static_cast<uint32_t>(std::round(std::log2(size)));
    const uint32_t fft_bytes = size * COMPLEX_BYTES;
    const uint32_t tw_bytes  = size * COMPLEX_BYTES;

    // 4 CBs: CB_IN, CB_SCR, CB_TW, CB_OUT — each one page of fft_bytes
    const uint32_t l1_needed = fft_bytes * 4;
    if (l1_needed > L1_BUDGET) {
        throw std::runtime_error(
            "Medium FFT size=" + std::to_string(size) +
            " needs " + std::to_string(l1_needed) +
            " bytes across 4 CBs, L1 budget=" + std::to_string(L1_BUDGET));
    }

    const CoreCoord grid            = get_worker_grid(mesh_device);
    const uint32_t  num_cores_x     = grid.x;
    const uint32_t  num_cores_y     = grid.y;
    const uint32_t  total_grid_cores = num_cores_x * num_cores_y;

    std::cout << "[medium] worker_grid=" << num_cores_x << "x" << num_cores_y
              << " total_grid_cores=" << total_grid_cores
              << " batch=" << batch << "\n";

    // FIX 4: Don't pass accessor args as compile-time args to data movement kernels.
    // Reader and writer kernels receive buffer addresses as *runtime* args (arg[0], arg[1]).
    // Compile-time args for data movement kernels on Wormhole must be empty or
    // contain only truly compile-time constants (not buffer addresses which change).
    CoreRange core_range(CoreCoord{0, 0}, CoreCoord{num_cores_x - 1, num_cores_y - 1});
    Program program = CreateProgram();

    make_cb(program, core_range, 0, fft_bytes, fft_bytes);   // CB_IN
    make_cb(program, core_range, 1, fft_bytes, fft_bytes);   // CB_SCR (scratch)
    make_cb(program, core_range, 2, tw_bytes,  tw_bytes);    // CB_TW
    make_cb(program, core_range, 3, fft_bytes, fft_bytes);   // CB_OUT

    auto reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_medium_reader.cpp",
        core_range,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = {}   // FIX: no accessor compile args
        });

    auto compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_medium_compute.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args     = {log2n, static_cast<uint32_t>(inverse)}
        });

    auto writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_medium_writer.cpp",
        core_range,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_1,
            .noc          = NOC::RISCV_1_default,
            .compile_args = {}   // FIX: no accessor compile args
        });

    const uint32_t base_batch = batch / total_grid_cores;
    const uint32_t extra      = batch % total_grid_cores;

    uint32_t fft_offset = 0;
    uint32_t core_idx   = 0;
    for (uint32_t ry = 0; ry < num_cores_y; ++ry) {
        for (uint32_t cx = 0; cx < num_cores_x; ++cx, ++core_idx) {
            CoreCoord coord{cx, ry};
            const uint32_t my_batch = base_batch + (core_idx < extra ? 1u : 0u);

            // FIX 5: Runtime args match exactly what the kernel reads:
            //   reader: [src_buf_addr, tw_buf_addr, fft_offset, my_batch, size]
            SetRuntimeArgs(program, reader, coord,
                {buf_addr(src_buf), buf_addr(tw_buf), fft_offset, my_batch, size});

            //   compute: [my_batch, size, log2n, inverse]
            SetRuntimeArgs(program, compute, coord,
                {my_batch, size, log2n, static_cast<uint32_t>(inverse)});

            //   writer: [dst_buf_addr, fft_offset, my_batch, size]
            SetRuntimeArgs(program, writer, coord,
                {buf_addr(dst_buf), fft_offset, my_batch, size});

            fft_offset += my_batch;
        }
    }

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::cout << "[medium done] total_fft_assigned=" << fft_offset << "\n";
}

// ===========================================================================
//  TIER 3 — Large FFT
//  2D Cooley–Tukey decomposition, N = R × S, distributed across cores.
//  Phase 1: each core computes S-point row FFTs on its row slice.
//  Transpose: all-to-all NOC scatter.
//  Phase 2: each core computes R-point column FFTs + twiddle multiply.
// ===========================================================================
static void run_large_fft(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue&                  cq,
    const std::shared_ptr<distributed::MeshBuffer>& src_buf,
    const std::shared_ptr<distributed::MeshBuffer>& dst_buf,
    const std::shared_ptr<distributed::MeshBuffer>& tw_buf,
    uint32_t                                        size,
    uint32_t                                        batch,
    bool                                            inverse)
{
    const uint32_t log2n = static_cast<uint32_t>(std::round(std::log2(size)));

    // FIX 6: Support odd log2 sizes by allowing log2R = floor(log2n/2),
    // log2S = log2n - log2R.  This handles e.g. 8192 (log2=13 → R=64, S=128).
    const uint32_t log2R = log2n / 2;
    const uint32_t log2S = log2n - log2R;
    const uint32_t R     = 1u << log2R;
    const uint32_t S     = 1u << log2S;

    // FIX 7: Remove the R==S assertion that rejected non-square splits.

    const CoreCoord grid         = get_worker_grid(mesh_device);
    const uint32_t total_cores   = grid.x * grid.y;

    // FIX 8: cores_per_fft must divide R evenly. Find the largest divisor of R
    // that is ≤ total_cores.
    uint32_t cores_per_fft = std::min(R, total_cores);
    while (R % cores_per_fft != 0) {
        --cores_per_fft;
    }
    const uint32_t rows_per_core = R / cores_per_fft;

    std::cout << "[large] size=" << size
              << " R=" << R << " S=" << S
              << " cores_per_fft=" << cores_per_fft
              << " rows_per_core=" << rows_per_core << "\n";

    // L1 sanity check
    const uint32_t fft_row_bytes = rows_per_core * S * COMPLEX_BYTES;
    const uint32_t tw_r_bytes    = R * COMPLEX_BYTES;
    const uint32_t tw_s_bytes    = S * COMPLEX_BYTES;
    const uint32_t col_bytes     = R * rows_per_core * COMPLEX_BYTES;
    const uint32_t cb0_bytes     = std::max(fft_row_bytes, col_bytes);
    const uint32_t l1_needed     = cb0_bytes + tw_r_bytes + tw_s_bytes + fft_row_bytes;

    if (l1_needed > L1_BUDGET) {
        throw std::runtime_error(
            "Large FFT size=" + std::to_string(size) +
            " needs ~" + std::to_string(l1_needed) +
            " bytes, L1 budget=" + std::to_string(L1_BUDGET));
    }

    // Lay out cores_per_fft cores in a grid (fill x first, then y)
    const uint32_t num_cols = std::min(cores_per_fft, static_cast<uint32_t>(grid.x));
    const uint32_t num_rows = (cores_per_fft + num_cols - 1) / num_cols;
    CoreRange cr(CoreCoord{0, 0}, CoreCoord{num_cols - 1, num_rows - 1});

    // FIX 9: NOC coordinates on Wormhole are *not* the same as logical core
    // coords — there are harvested/Ethernet rows offset.  Use
    // device->worker_core_from_logical_core() to convert.
    auto* dev = mesh_device->get_devices()[0];

    // Build packed NOC coords for the writer scatter: one entry per logical core.
    std::vector<uint32_t> noc_coords_packed;
    noc_coords_packed.reserve(cores_per_fft);
    {
        uint32_t idx = 0;
        for (uint32_t ry = 0; ry < num_rows && idx < cores_per_fft; ++ry) {
            for (uint32_t cx = 0; cx < num_cols && idx < cores_per_fft; ++cx) {
                CoreCoord noc = dev->worker_core_from_logical_core(CoreCoord{cx, ry});
                // FIX 10: Pack as (noc_y << 16) | noc_x to match writer kernel unpacking.
                noc_coords_packed.push_back((static_cast<uint32_t>(noc.y) << 16) |
                                             static_cast<uint32_t>(noc.x));
                ++idx;
            }
        }
    }

    for (uint32_t g = 0; g < batch; ++g) {
        Program program = CreateProgram();

        // CB0: holds row slice (phase 1 input) OR transposed col data (phase 2 input)
        // CB1: R-point twiddle table
        // CB2: S-point twiddle table
        // CB3: phase-1 row FFT output / phase-2 col FFT output
        make_cb(program, cr, 0, cb0_bytes,     cb0_bytes);
        make_cb(program, cr, 1, tw_r_bytes,    tw_r_bytes);
        make_cb(program, cr, 2, tw_s_bytes,    tw_s_bytes);
        // FIX 11: CB3 must be large enough for both phase-1 rows and phase-2 cols.
        make_cb(program, cr, 3, std::max(fft_row_bytes, col_bytes),
                               std::max(fft_row_bytes, col_bytes));

        // Semaphore 0 per core: reader spins until all writers have incremented it.
        CreateSemaphore(program, cr, 0);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_large_reader1.cpp",
            cr,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_0,
                .noc          = NOC::RISCV_0_default,
                .compile_args = {}   // FIX: no accessor compile args
            });

        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_large_compute.cpp",
            cr,
            ComputeConfig{
                .math_fidelity    = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .compile_args     = {
                    log2R,
                    log2S,
                    static_cast<uint32_t>(inverse),
                    cores_per_fft,
                    rows_per_core
                }
            });

        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_large_writer.cpp",
            cr,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_1,
                .noc          = NOC::RISCV_1_default,
                .compile_args = {}   // FIX: no accessor compile args
            });

        uint32_t local_core = 0;
        for (uint32_t ry = 0; ry < num_rows; ++ry) {
            for (uint32_t cx = 0; cx < num_cols; ++cx) {
                if (local_core >= cores_per_fft) break;
                CoreCoord coord{cx, ry};
                const uint32_t row_start = local_core * rows_per_core;

                // FIX 12: Reader runtime args match fft_large_reader1.cpp exactly:
                //   [0]=src_addr [1]=tw_addr [2]=batch_idx [3]=row_start
                //   [4]=rows_per_core [5]=R [6]=S [7]=inverse
                SetRuntimeArgs(program, reader, coord,
                    {
                        buf_addr(src_buf),
                        buf_addr(tw_buf),
                        g,
                        row_start,
                        rows_per_core,
                        R,
                        S,
                        static_cast<uint32_t>(inverse)
                    });

                // FIX 13: Compute runtime args match fft_large_compute.cpp exactly:
                //   [0]=row_start [1]=rows_per_core [2]=R [3]=S
                //   [4]=local_core_id [5]=batch_idx
                SetRuntimeArgs(program, compute, coord,
                    {
                        row_start,
                        rows_per_core,
                        R,
                        S,
                        local_core,
                        g
                    });

                // FIX 14: Writer runtime args: [0-5] base args, then per-core NOC coords.
                //   [0]=dst_addr [1]=batch_idx [2]=row_start [3]=rows_per_core [4]=R [5]=S
                //   [6..6+cores_per_fft-1] = packed NOC coords for scatter
                std::vector<uint32_t> writer_args = {
                    buf_addr(dst_buf),
                    g,
                    row_start,
                    rows_per_core,
                    R,
                    S
                };
                writer_args.insert(
                    writer_args.end(),
                    noc_coords_packed.begin(),
                    noc_coords_packed.end()
                );
                SetRuntimeArgs(program, writer, coord, writer_args);

                ++local_core;
            }
        }

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::cout << "[large wave done] batch_idx=" << g << "\n";
    }

    std::cout << "[large FFT done] batch=" << batch << "\n";
}

// ===========================================================================
//  PUBLIC API
// ===========================================================================
struct FFTContext {
    std::shared_ptr<distributed::MeshDevice> mesh_device;

    FFTContext() = default;

    explicit FFTContext(int device_id) {
        mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        if (!mesh_device) {
            throw std::runtime_error("Failed to create mesh device");
        }
    }

    ~FFTContext() {
        if (mesh_device) {
            mesh_device->close();
        }
    }

    FFTContext(const FFTContext&) = delete;
    FFTContext& operator=(const FFTContext&) = delete;
};

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

    auto mesh_device = ctx.mesh_device;
    auto& cq = mesh_device->mesh_command_queue();

    const uint32_t total_bytes = batch * size * COMPLEX_BYTES;

    FFTStrategy strategy = select_strategy(size, batch);
    std::cout << "[fft_universal] size=" << size
              << " batch=" << batch
              << " strategy=" << strategy_name(strategy) << "\n";

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size   = COMPLEX_BYTES,    // 8 bytes = 1 complex float
        .buffer_type = BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig data_buffer_config{
        .size = total_bytes
    };

    auto src_buf = distributed::MeshBuffer::create(data_buffer_config, dram_config, mesh_device.get());
    auto dst_buf = distributed::MeshBuffer::create(data_buffer_config, dram_config, mesh_device.get());

    std::shared_ptr<distributed::MeshBuffer> tw_buf;
    {
        std::vector<float> input_vec(data_host, data_host + (static_cast<size_t>(batch) * size * 2));
        distributed::EnqueueWriteMeshBuffer(cq, src_buf, input_vec, false);

        if (strategy == FFTStrategy::LARGE) {
            const uint32_t log2n = static_cast<uint32_t>(std::round(std::log2(size)));
            const uint32_t log2R = log2n / 2;
            const uint32_t R     = 1u << log2R;
            const uint32_t S     = size / R;

            auto combined_twiddles = make_large_twiddles(R, S);
            distributed::ReplicatedBufferConfig tw_config{
                .size = static_cast<uint32_t>(combined_twiddles.size() * FLOAT_BYTES)
            };
            tw_buf = distributed::MeshBuffer::create(tw_config, dram_config, mesh_device.get());
            distributed::EnqueueWriteMeshBuffer(cq, tw_buf, combined_twiddles, false);
        } else {
            const auto& twiddles = get_twiddles(size);
            distributed::ReplicatedBufferConfig tw_config{
                .size = static_cast<uint32_t>(twiddles.size() * FLOAT_BYTES)
            };
            tw_buf = distributed::MeshBuffer::create(tw_config, dram_config, mesh_device.get());
            distributed::EnqueueWriteMeshBuffer(cq, tw_buf, twiddles, false);
        }
    }

    // FIX 15: Flush all writes before launching kernels.
    distributed::Finish(cq);

    switch (strategy) {
        case FFTStrategy::MEDIUM:
            run_medium_fft(mesh_device, cq, src_buf, dst_buf, tw_buf, size, batch, inverse);
            break;
        case FFTStrategy::LARGE:
            run_large_fft(mesh_device, cq, src_buf, dst_buf, tw_buf, size, batch, inverse);
            break;
    }

    // run_*_fft already calls Finish(); one more doesn't hurt but avoids
    // stale state if future code removes the inner Finish.
    distributed::Finish(cq);

    std::vector<float> output_vec;
    distributed::EnqueueReadMeshBuffer(cq, output_vec, dst_buf, true);

    const size_t expected = static_cast<size_t>(batch) * size * 2;
    if (output_vec.size() != expected) {
        throw std::runtime_error(
            "Output size mismatch: expected=" + std::to_string(expected) +
            " got=" + std::to_string(output_vec.size()));
    }

    std::copy(output_vec.begin(), output_vec.end(), out_host);
}

// ===========================================================================
//  Smoke tests
//  Primary target: 64×64 = 4096-point 1D FFT batch (the "wormhole 64×64" case)
// ===========================================================================
int main() {
    FFTContext ctx(/*device_id=*/0);

    // -----------------------------------------------------------------------
    // Test 1: 4096-point FFT, batch=64  (64 × 64 use-case, Medium tier)
    // Each of the 64 cores handles 1 FFT of 4096 complex points.
    // Expected: impulse input → all-ones output.
    // -----------------------------------------------------------------------
    {
        const uint32_t N = 4096;
        const uint32_t B = 64;

        std::vector<float> in(B * N * 2, 0.0f);
        std::vector<float> out(B * N * 2, 0.0f);

        for (uint32_t b = 0; b < B; ++b) {
            in[b * N * 2] = 1.0f;   // impulse at index 0 of each FFT
        }

        fft_universal(ctx, in.data(), out.data(), N, B);

        // All frequency bins should be ~1.0 (real part)
        float max_err = 0.0f;
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t k = 0; k < N; ++k) {
                max_err = std::max(max_err, std::abs(out[b * N * 2 + k * 2] - 1.0f));
                max_err = std::max(max_err, std::abs(out[b * N * 2 + k * 2 + 1]));
            }
        }
        std::cout << "64x64 smoke (4096-pt x64): max_err=" << max_err
                  << "  (should be < 1e-4)\n";
    }

    // -----------------------------------------------------------------------
    // Test 2: 1024-point FFT, batch=256  (Medium tier, multi-core)
    // -----------------------------------------------------------------------
    {
        const uint32_t N = 1024;
        const uint32_t B = 256;

        std::vector<float> in(B * N * 2, 0.0f);
        std::vector<float> out(B * N * 2, 0.0f);

        for (uint32_t b = 0; b < B; ++b) {
            in[b * N * 2] = 1.0f;
        }

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Medium smoke (1K x256): out[0]=" << out[0]
                  << " out[2]=" << out[2] << "  (both ~1.0)\n";
    }

    // -----------------------------------------------------------------------
    // Test 3: 65536-point FFT, batch=1  (Large tier, multi-core transpose)
    // -----------------------------------------------------------------------
    {
        const uint32_t N = 65536;
        const uint32_t B = 1;

        std::vector<float> in(N * 2, 0.0f);
        std::vector<float> out(N * 2, 0.0f);

        in[0] = 1.0f;

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Large smoke (64K x1): out[0]=" << out[0]
                  << " out[2]=" << out[2] << "  (both ~1.0)\n";
    }

    // -----------------------------------------------------------------------
    // Test 4: Round-trip FFT → IFFT correctness check
    // -----------------------------------------------------------------------
    {
        const uint32_t N = 8192;
        const uint32_t B = 4;

        std::vector<float> in(B * N * 2);
        std::vector<float> fwd(B * N * 2);
        std::vector<float> inv_out(B * N * 2);

        for (auto& v : in) {
            v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        }

        fft_universal(ctx, in.data(),  fwd.data(),     N, B, false);
        fft_universal(ctx, fwd.data(), inv_out.data(), N, B, true);

        float max_err = 0.0f;
        for (uint32_t i = 0; i < B * N * 2; ++i) {
            max_err = std::max(max_err,
                std::abs(inv_out[i] / static_cast<float>(N) - in[i]));
        }
        std::cout << "Round-trip (8K x4) max_err=" << max_err
                  << "  (should be < 1e-4)\n";
    }

    return 0;
}