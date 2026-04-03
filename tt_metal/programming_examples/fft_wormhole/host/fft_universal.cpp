// =============================================================================
// fft_universal.cpp  —  Universal FFT host driver for Tenstorrent Wormhole
// Mesh-device version for newer TT-Metal branches
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
static constexpr uint32_t L1_BUDGET     = 1024 * 1024;
static constexpr uint32_t FLOAT_BYTES   = 4;
static constexpr uint32_t COMPLEX_BYTES = 2 * FLOAT_BYTES;

static constexpr uint32_t MEDIUM_THRESH = 32 * 1024;

// ---------------------------------------------------------------------------
// Strategy tag
// ---------------------------------------------------------------------------
enum class FFTStrategy { MEDIUM, LARGE };

// Route everything up to 32K through MEDIUM.
// Tier1-Small is intentionally disabled because that path is the one hanging.
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

// Tier 3 split twiddle buffer [R twiddles | S twiddles]
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

static std::vector<uint32_t> make_accessor_args_2(
    const std::shared_ptr<distributed::MeshBuffer>& a,
    const std::shared_ptr<distributed::MeshBuffer>& b)
{
    std::vector<uint32_t> args;
    TensorAccessorArgs(*a->get_backing_buffer()).append_to(args);
    TensorAccessorArgs(*b->get_backing_buffer()).append_to(args);
    return args;
}

static std::vector<uint32_t> make_accessor_args_1(
    const std::shared_ptr<distributed::MeshBuffer>& a)
{
    std::vector<uint32_t> args;
    TensorAccessorArgs(*a->get_backing_buffer()).append_to(args);
    return args;
}

static CoreCoord get_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* dev = mesh_device->get_devices()[0];
    return dev->compute_with_storage_grid_size();
}

// ===========================================================================
//  TIER 2 — Medium FFT
//  One FFT/core, wave scheduler, actual worker grid.
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
    const uint32_t log2n     = static_cast<uint32_t>(std::log2(size));
    const uint32_t fft_bytes = size * COMPLEX_BYTES;
    const uint32_t tw_bytes  = size * COMPLEX_BYTES;

    const uint32_t l1_needed = fft_bytes * 3 + tw_bytes;
    if (l1_needed > L1_BUDGET) {
        throw std::runtime_error(
            "Medium FFT size=" + std::to_string(size) +
            " needs " + std::to_string(l1_needed) +
            " bytes across 4 CBs, L1 budget=" + std::to_string(L1_BUDGET));
    }

    const CoreCoord grid = get_worker_grid(mesh_device);
    const uint32_t num_cores_x = grid.x;
    const uint32_t num_cores_y = grid.y;
    const uint32_t max_active_cores = num_cores_x * num_cores_y;

    std::cout << "[medium] worker_grid=" << num_cores_x << "x" << num_cores_y
              << " max_active_cores=" << max_active_cores << "\n";

    auto reader_ct_args = make_accessor_args_2(src_buf, tw_buf);
    auto writer_ct_args = make_accessor_args_1(dst_buf);

    for (uint32_t wave_base = 0; wave_base < batch; wave_base += max_active_cores) {
        const uint32_t wave_batch   = std::min(max_active_cores, batch - wave_base);
        const uint32_t active_cores = wave_batch;
        const uint32_t rows         = (active_cores + num_cores_x - 1) / num_cores_x;
        const uint32_t cols         = std::min(active_cores, num_cores_x);

        std::cout << "[medium wave] base=" << wave_base
                  << " count=" << wave_batch
                  << " rows=" << rows
                  << " cols=" << cols << "\n";

        CoreRange core_range(CoreCoord{0, 0}, CoreCoord{cols - 1, rows - 1});
        Program program = CreateProgram();

        make_cb(program, core_range, 0, fft_bytes, fft_bytes);
        make_cb(program, core_range, 1, fft_bytes, fft_bytes);
        make_cb(program, core_range, 2, tw_bytes,  tw_bytes);
        make_cb(program, core_range, 3, fft_bytes, fft_bytes);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_medium_reader.cpp",
            core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
                .compile_args = reader_ct_args
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
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default,
                .compile_args = writer_ct_args
            });

        uint32_t core_idx = 0;
        for (uint32_t r = 0; r < rows && core_idx < active_cores; ++r) {
            for (uint32_t c = 0; c < cols && core_idx < active_cores; ++c) {
                CoreCoord coord{c, r};

                const uint32_t fft_offset = wave_base + core_idx;
                const uint32_t my_batch   = 1;

                SetRuntimeArgs(
                    program, reader, coord,
                    {src_buf->address(), tw_buf->address(), fft_offset, my_batch, size});

                SetRuntimeArgs(
                    program, compute, coord,
                    {my_batch, size, log2n, static_cast<uint32_t>(inverse)});

                SetRuntimeArgs(
                    program, writer, coord,
                    {dst_buf->address(), fft_offset, my_batch, size});

                ++core_idx;
            }
        }

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::cout << "[medium wave done] base=" << wave_base << "\n";
    }
}

// ===========================================================================
//  TIER 3 — Large FFT
//  2D Cooley–Tukey decomposition, N = R × S, distributed across cores.
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
    const uint32_t log2n        = static_cast<uint32_t>(std::log2(size));
    const uint32_t log2R        = log2n / 2;
    const uint32_t log2S        = log2n - log2R;
    const uint32_t R            = 1u << log2R;
    const uint32_t S            = 1u << log2S;

    const CoreCoord grid        = get_worker_grid(mesh_device);
    const uint32_t cores_per_fft = std::min(R, grid.x * grid.y);
    const uint32_t rows_per_core = R / cores_per_fft;

    std::cout << "[large] size=" << size
              << " R=" << R << " S=" << S
              << " cores_per_fft=" << cores_per_fft
              << " rows_per_core=" << rows_per_core << "\n";

    auto reader_ct_args = make_accessor_args_2(src_buf, tw_buf);
    auto writer_ct_args = make_accessor_args_1(dst_buf);

    const uint32_t fft_row_bytes = rows_per_core * S * COMPLEX_BYTES;
    const uint32_t tw_r_bytes    = R * COMPLEX_BYTES;
    const uint32_t tw_s_bytes    = S * COMPLEX_BYTES;
    const uint32_t col_bytes     = R * rows_per_core * COMPLEX_BYTES;

    for (uint32_t g = 0; g < batch; ++g) {
        const uint32_t num_cols = std::min(cores_per_fft, grid.x);
        const uint32_t num_rows = (cores_per_fft + num_cols - 1) / num_cols;
        CoreRange cr(CoreCoord{0, 0}, CoreCoord{num_cols - 1, num_rows - 1});

        // Pack target NOC coordinates for the writer scatter
        std::vector<uint32_t> noc_coords_packed;
        noc_coords_packed.reserve(cores_per_fft);
        {
            uint32_t idx = 0;
            for (uint32_t row = 0; row < num_rows && idx < cores_per_fft; ++row) {
                for (uint32_t col = 0; col < num_cols && idx < cores_per_fft; ++col) {
                    // packed = (noc_row << 16) | noc_col  — logical coords == NOC coords on Wormhole
                    noc_coords_packed.push_back((row << 16) | col);
                    ++idx;
                }
            }
        }

        Program program = CreateProgram();

        // CB layout per core:
        //   CB0 (CB_DATA)  — row slice + later transposed column data
        //   CB1 (CB_TW_R)  — R-point twiddles
        //   CB2 (CB_TW_S)  — S-point twiddles
        //   CB3 (CB_OUT)   — phase-1 output (rows after FFT) + phase-2 output (cols)
        make_cb(program, cr, 0, std::max(fft_row_bytes, col_bytes), std::max(fft_row_bytes, col_bytes));
        make_cb(program, cr, 1, tw_r_bytes, tw_r_bytes);
        make_cb(program, cr, 2, tw_s_bytes, tw_s_bytes);
        make_cb(program, cr, 3, fft_row_bytes, fft_row_bytes);

        CreateSemaphore(program, cr, 0);
        
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_wormhole/kernels/fft_large_reader1.cpp",
            cr,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default,
                .compile_args = reader_ct_args
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
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default,
                .compile_args = writer_ct_args
            });
        
        uint32_t local_core = 0;
        for (uint32_t r = cr.start_coord.y; r <= cr.end_coord.y; ++r) {
            for (uint32_t c = cr.start_coord.x; c <= cr.end_coord.x; ++c) {
                CoreCoord coord{c, r};
                uint32_t row_start = local_core * rows_per_core;
                
                SetRuntimeArgs(
                    program, reader, coord,
                    {
                        src_buf->address(),
                        tw_buf->address(),
                        g,
                        row_start,
                        rows_per_core,
                        R,
                        S,
                        static_cast<uint32_t>(inverse)
                    });
                
                SetRuntimeArgs(
                    program, compute, coord,
                    {
                        row_start,
                        rows_per_core,
                        R,
                        S,
                        local_core,
                        g
                    });
                
                std::vector<uint32_t> writer_args = {
                    dst_buf->address(),
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
        .page_size   = COMPLEX_BYTES,
        .buffer_type = BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig data_buffer_config{
        .size = total_bytes
    };

    auto src_buf = distributed::MeshBuffer::create(data_buffer_config, dram_config, mesh_device.get());
    auto dst_buf = distributed::MeshBuffer::create(data_buffer_config, dram_config, mesh_device.get());

    std::shared_ptr<distributed::MeshBuffer> tw_buf;
    if (strategy == FFTStrategy::LARGE) {
        const uint32_t log2n = static_cast<uint32_t>(std::log2(size));
        const uint32_t log2R = log2n / 2;
        const uint32_t R     = 1u << log2R;
        const uint32_t S     = size / R;

        auto combined_twiddles = make_large_twiddles(R, S);
        distributed::ReplicatedBufferConfig tw_config{
            .size = static_cast<uint32_t>(combined_twiddles.size() * FLOAT_BYTES)
        };
        tw_buf = distributed::MeshBuffer::create(tw_config, dram_config, mesh_device.get());

        std::vector<float> input_vec(data_host, data_host + (batch * size * 2));
        distributed::EnqueueWriteMeshBuffer(cq, src_buf, input_vec, false);
        distributed::EnqueueWriteMeshBuffer(cq, tw_buf, combined_twiddles, false);
    } else {
        const auto& twiddles = get_twiddles(size);
        distributed::ReplicatedBufferConfig tw_config{
            .size = static_cast<uint32_t>(twiddles.size() * FLOAT_BYTES)
        };
        tw_buf = distributed::MeshBuffer::create(tw_config, dram_config, mesh_device.get());

        std::vector<float> input_vec(data_host, data_host + (batch * size * 2));
        distributed::EnqueueWriteMeshBuffer(cq, src_buf, input_vec, false);
        distributed::EnqueueWriteMeshBuffer(cq, tw_buf, twiddles, false);
    }

    switch (strategy) {
        case FFTStrategy::MEDIUM:
            run_medium_fft(mesh_device, cq, src_buf, dst_buf, tw_buf, size, batch, inverse);
            break;
        case FFTStrategy::LARGE:
            run_large_fft(mesh_device, cq, src_buf, dst_buf, tw_buf, size, batch, inverse);
            break;
    }

    distributed::Finish(cq);

    std::vector<float> output_vec;
    distributed::EnqueueReadMeshBuffer(cq, output_vec, dst_buf, true);

    const size_t expected = static_cast<size_t>(batch) * size * 2;
    if (output_vec.size() != expected) {
        throw std::runtime_error(
            "Output size mismatch. Expected " + std::to_string(expected) +
            " got " + std::to_string(output_vec.size()));
    }

    std::copy(output_vec.begin(), output_vec.end(), out_host);
}

// ===========================================================================
//  Smoke tests
// ===========================================================================
int main() {
    FFTContext ctx(/*device_id=*/0);

    {
        const uint32_t N = 1024;
        const uint32_t B = 256;

        std::vector<float> in(B * N * 2, 0.0f);
        std::vector<float> out(B * N * 2, 0.0f);

        for (uint32_t b = 0; b < B; ++b) {
            in[b * N * 2] = 1.0f;
        }

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Tier1/2 smoke: out[0]=" << out[0]
                  << " out[2]=" << out[2] << "  (both ~1.0)\n";
    }

    {
        const uint32_t N = 65536;
        const uint32_t B = 1;

        std::vector<float> in(N * 2, 0.0f);
        std::vector<float> out(N * 2, 0.0f);

        in[0] = 1.0f;

        fft_universal(ctx, in.data(), out.data(), N, B);
        std::cout << "Tier3 smoke: out[0]=" << out[0] << "  (~1.0)\n";
    }

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
            max_err = std::max(max_err, std::abs(inv_out[i] / static_cast<float>(N) - in[i]));
        }

        std::cout << "Round-trip max error: " << max_err << "  (< 1e-4)\n";
    }

    return 0;
}