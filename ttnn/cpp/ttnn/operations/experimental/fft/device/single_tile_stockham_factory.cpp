// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SingleTileStockhamFactory implementation.
//
// Builds a ProgramDescriptor that runs ONE radix-2 batched Stockham FFT
// of length N (2 <= N <= 1024, pow-2, fp32) on a single Tensix core, with
// input/output buffers taken directly from device tensors (no PCIe round
// trip per call, no host scratch buffers, no WriteShard/ReadShard on the
// hot path).
//
// Commit 1C: piggyback on fft_stockham::get_cached_batch_plan(N, 1) to
// reuse its already-allocated twiddle MeshBuffers. Allocate a separate
// persistent zero-imag scratch buffer (one tile of zeros, cached per
// device) for the imag input. All host-side buffer setup is one-time
// (Category B in the refactor inventory) — the per-call op path touches
// zero tensor data on host.
//
// Wire-compatible with the existing batch_fft_{reader,writer,compute}.cpp
// kernels (see stockham_host.hpp::make_batch_plan for the legacy build).

#include "single_tile_stockham_factory.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

#include "stockham_host.hpp"   // fft_stockham::get_cached_batch_plan, buf_addr, etc.

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kTileHW    = 32u;
constexpr uint32_t kTileElems = kTileHW * kTileHW;   // 1024
constexpr uint32_t kTileBytesFp32 = kTileElems * sizeof(float);  // 4096

constexpr uint32_t log2u(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) ++r;
    return r;
}

// Renamed `is_pow2_st` (single-tile prefix) so the Unity build can merge
// this TU with fft_device_operation.cpp — which has its own anonymous-
// namespace `is_pow2` — without ODR collision.
constexpr bool is_pow2_st(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// ── Zero-imag scratch buffer cache ─────────────────────────────────────
// For forward FFT of REAL input, the kernel still expects an imag input
// buffer. We allocate ONE tile of zeros per device and reuse it forever.
// Category B work: one-time setup, no per-call host arithmetic.
using MeshBufferPtr = std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>;

struct ZeroScratch {
    MeshBufferPtr buf;
};

inline std::unordered_map<uintptr_t, std::shared_ptr<ZeroScratch>>&
zero_scratch_cache() {
    static std::unordered_map<uintptr_t, std::shared_ptr<ZeroScratch>> c;
    return c;
}

std::shared_ptr<ZeroScratch> get_or_create_zero_scratch(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md)
{
    using namespace tt::tt_metal::distributed;
    const uintptr_t key = reinterpret_cast<uintptr_t>(md.get());
    auto& cache = zero_scratch_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto z = std::make_shared<ZeroScratch>();
    z->buf = fft_example::make_mesh_buf(md, kTileBytesFp32, kTileBytesFp32);

    // One-time zero-fill via WriteShard (Category B: setup, cached forever).
    std::vector<float> zeros(kTileElems, 0.0f);
    MeshCommandQueue& cq = md->mesh_command_queue();
    WriteShard(cq, z->buf, zeros, MeshCoordinate(0, 0), /*blocking=*/true);

    cache.emplace(key, z);
    return z;
}

}  // namespace

tt::tt_metal::ProgramDescriptor SingleTileStockhamFactory::create_descriptor(
    [[maybe_unused]] const FFTParams& operation_attributes,
    const FFTTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    // ── Resolve sizes from the input tensor ────────────────────────────
    const auto& in_real = tensor_args.input_real;
    const uint32_t N    = static_cast<uint32_t>(in_real.padded_shape()[-1]);
    const uint32_t log2N = log2u(N);

    TT_FATAL(is_pow2_st(N) && N >= 2u && N <= kTileElems,
        "SingleTileStockhamFactory: requires pow-2 N in [2, 1024] (got N={}).", N);
    TT_FATAL(in_real.dtype() == DataType::FLOAT32,
        "SingleTileStockhamFactory: fp32 only (got dtype {}).",
        static_cast<int>(in_real.dtype()));

    // ── Output tensor buffer addresses (already created by framework) ──
    const auto& out_r_tensor = std::get<0>(tensor_return_value);
    const auto& out_i_tensor = std::get<1>(tensor_return_value);

    auto* const in_r_buf  = in_real.buffer();
    auto* const out_r_buf = out_r_tensor.buffer();
    auto* const out_i_buf = out_i_tensor.buffer();

    TT_FATAL(in_r_buf != nullptr && out_r_buf != nullptr && out_i_buf != nullptr,
        "SingleTileStockhamFactory: input/output tensors must be on device.");

    // ── Resolve MeshDevice (mirror fft_program_factory.cpp pattern) ────
    auto* device_raw = in_real.device();
    auto md = std::shared_ptr<MeshDevice>(device_raw, [](MeshDevice*){});

    // ── Reuse legacy plan's twiddle MeshBuffers (Category B, cached) ──
    auto plan = fft_stockham::get_cached_batch_plan(md, /*sub_N=*/N, /*batch=*/1u);

    // ── Zero-imag scratch buffer (Category B, one-time per device) ────
    auto zscratch = get_or_create_zero_scratch(md);

    ProgramDescriptor desc;

    // ── Single Tensix core for now (batch=1, single sub-FFT) ───────────
    const CoreCoord core{0, 0};
    const CoreRange core_range(core, core);
    const CoreRangeSet crs({core_range});

    // Get physical core coords (the reader kernel needs them for NoC).
    const CoreCoord phys = md->worker_core_from_logical_core(core);

    // ── Circular Buffers ───────────────────────────────────────────────
    // CB layout mirrors batch_fft_compute.cpp / batch_fft_common.h:
    //   c_0..c_9  : EVEN/ODD/TW/OUT real+imag, 2-tile pipelined
    //   c_10..c_13: TMP, TW_ODD scratch (1 tile)
    //   c_14..c_15: STATE_R, STATE_I (1 tile)
    //   c_16      : SYNC (1 tile)
    constexpr uint32_t kNumCbs = 17;
    constexpr uint32_t kCbTiles[kNumCbs] = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   // EVEN_R/I, ODD_R/I, TW_R/I, OUT0_R/I, OUT1_R/I
        1, 1, 1, 1,                     // TMP_R/I, TW_ODD_R/I
        1, 1,                           // STATE_R, STATE_I
        1                               // SYNC
    };

    for (uint32_t id = 0; id < kNumCbs; ++id) {
        const uint32_t total = kCbTiles[id] * kTileBytesFp32;
        desc.cbs.push_back(CBDescriptor{
            .total_size = total,
            .core_ranges = crs,
            .format_descriptors = {
                CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(id),
                    .data_format  = tt::DataFormat::Float32,
                    .page_size    = kTileBytesFp32,
                }
            },
        });
    }

    // ── Kernels ────────────────────────────────────────────────────────
    // Reader (BRISC0): pulls input + twiddle tiles, bit-reverses, fills CBs.
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_reader.cpp",
        .core_ranges = crs,
        .compile_time_args = {N, log2N},
        .runtime_args = {
            // {in_r, in_i, tw_r, tw_i, base, batch_per_core, phys_x, phys_y}
            std::pair<CoreCoord, std::vector<uint32_t>>{
                core,
                std::vector<uint32_t>{
                    in_r_buf->address(),                           // input real (our tensor)
                    fft_stockham::buf_addr(zscratch->buf),         // zero-imag scratch
                    fft_stockham::buf_addr(plan->tw_r_buf),        // twiddles (legacy cache)
                    fft_stockham::buf_addr(plan->tw_i_buf),        // twiddles (legacy cache)
                    /*base=*/0u,
                    /*batch_per_core=*/1u,
                    static_cast<uint32_t>(phys.x),
                    static_cast<uint32_t>(phys.y),
                },
            },
        },
        .config = ReaderConfigDescriptor{},
    });

    // Writer (BRISC1): drains CB_OUT0/OUT1 into output tensor buffers.
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_writer.cpp",
        .core_ranges = crs,
        .compile_time_args = {},
        .runtime_args = {
            std::pair<CoreCoord, std::vector<uint32_t>>{
                core,
                std::vector<uint32_t>{
                    out_r_buf->address(),
                    out_i_buf->address(),
                    /*base=*/0u,
                    /*batch_per_core=*/1u,
                },
            },
        },
        .config = WriterConfigDescriptor{},
    });

    // Compute (TRISC): the actual Stockham butterfly chain.
    std::vector<UnpackToDestMode> u2d(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t id = 0; id < kNumCbs; ++id) {
        u2d[id] = UnpackToDestMode::UnpackToDestFp32;
    }

    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/compute/batch_fft_compute.cpp",
        .core_ranges = crs,
        .compile_time_args = {log2N},
        .runtime_args = {
            std::pair<CoreCoord, std::vector<uint32_t>>{
                core,
                std::vector<uint32_t>{ /*batch_per_core=*/1u },
            },
        },
        .config = ComputeConfigDescriptor{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
        },
    });

    return desc;
}

}  // namespace ttnn::experimental::prim
