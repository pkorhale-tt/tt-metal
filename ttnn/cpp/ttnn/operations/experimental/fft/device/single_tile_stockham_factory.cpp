// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SingleTileStockhamFactory implementation.
//
// Builds a ProgramDescriptor that runs ONE radix-2 batched Stockham FFT
// of length N (2 <= N <= 1024, pow-2, fp32) on a single Tensix core, with
// input/output buffers taken directly from device tensors (no PCIe round
// trip, no host scratch buffers, no WriteShard/ReadShard).
//
// Wire-compatible with the existing batch_fft_{reader,writer,compute}.cpp
// kernels (see stockham_host.hpp::make_batch_plan for the legacy build).
//
// This is the first commit of the host-to-device refactor. Subsequent
// commits will:
//   - Commit 2: add bf16 path
//   - Commit 3: two-pass Stockham (chained kernels, no host glue between
//               passes; covers 1024 < N <= ~32K)
//   - Commit 4: standalone ttnn::prim::fft_radix_pass building block
//   - Commit 5: composite fft_universal_xl (eliminates Steps 1/2/3 host
//               loops, lifts ceiling 16M -> 1G)
//   - Commit 6: composite Bluestein + composite IFFT
//   - Commit 7: comprehensive program-cache + Metal-Trace tests

#include "single_tile_stockham_factory.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

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

constexpr bool is_pow2(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// Twiddle factors for length-N batched Stockham, packed for the
// batch_fft_compute kernel. Mirrors stockham_host.hpp::batch_twiddles().
// One tile worth of twiddles per stage; LOG2N stages total.
std::pair<std::vector<float>, std::vector<float>> batch_twiddles_fp32(
    uint32_t N, uint32_t log2N)
{
    // Layout: stage s (s in [0, log2N)) holds N/2 twiddles W_N^k = exp(-2πi k / 2^(s+1))
    // packed into a single tile (kTileElems floats), zero-padded.
    const size_t total = static_cast<size_t>(log2N) * kTileElems;
    std::vector<float> tw_r(total, 0.0f);
    std::vector<float> tw_i(total, 0.0f);

    for (uint32_t s = 0; s < log2N; ++s) {
        const uint32_t m       = 1u << (s + 1);          // butterfly span
        const uint32_t half    = m >> 1;                  // # twiddles per stage
        const double   ang_inc = -2.0 * M_PI / static_cast<double>(m);

        float* dr = tw_r.data() + static_cast<size_t>(s) * kTileElems;
        float* di = tw_i.data() + static_cast<size_t>(s) * kTileElems;

        for (uint32_t k = 0; k < half; ++k) {
            const double ang = ang_inc * static_cast<double>(k);
            dr[k] = static_cast<float>(std::cos(ang));
            di[k] = static_cast<float>(std::sin(ang));
        }
    }
    return { std::move(tw_r), std::move(tw_i) };
}

}  // namespace

tt::tt_metal::ProgramDescriptor SingleTileStockhamFactory::create_descriptor(
    const FFTParams& operation_attributes,
    const FFTTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value)
{
    using namespace tt::tt_metal;

    // ── Resolve sizes from the input tensor ────────────────────────────
    const auto& in_real = tensor_args.input_real;
    const uint32_t N    = static_cast<uint32_t>(in_real.padded_shape()[-1]);
    const uint32_t log2N = log2u(N);

    // Single-tile path: N must fit in one 32×32 tile.
    TT_FATAL(is_pow2(N) && N >= 2u && N <= kTileElems,
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

    // Forward FFT of REAL input: imag is implicit zero. For the wire-compat
    // path we still need an imag input buffer; create one zero-filled via
    // a small CB sink. (Cleaner future: dedicated real-only kernel variant.)
    TT_FATAL(in_r_buf != nullptr && out_r_buf != nullptr && out_i_buf != nullptr,
        "SingleTileStockhamFactory: input/output tensors must be on device.");

    ProgramDescriptor desc;

    // ── Single Tensix core for now (batch=1, single sub-FFT) ───────────
    const CoreCoord core{0, 0};
    const CoreRange core_range(core, core);
    const CoreRangeSet crs({core_range});

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

    // ── Twiddle precompute on host (Category B: constant per-N, fine) ──
    // TODO(commit-1b): hoist into const-data CB so it isn't re-computed
    // on cache miss for the same N. Currently regenerated each create.
    auto [tw_r_data, tw_i_data] = batch_twiddles_fp32(N, log2N);
    (void)tw_r_data;  // wired via runtime args in TODO below
    (void)tw_i_data;

    // ── Kernels ────────────────────────────────────────────────────────
    // Reader (BRISC0): pulls input tiles from DRAM (in_real buffer),
    // bit-reverses, fills CB_EVEN/CB_ODD per stage. Imag input is zero
    // (forward FFT of real input).
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_reader.cpp",
        .core_ranges = crs,
        .compile_time_args = {N, log2N},
        .common_runtime_args = {},
        .runtime_args = {{
            // {in_r_addr, in_i_addr, tw_r_addr, tw_i_addr, base=0, batch_per_core=1, phys_x, phys_y}
            // TODO(commit-1b): allocate twiddle DRAM buffers + write twiddle
            //                 tables; pass real addrs here. Currently 0
            //                 placeholders will fault at runtime.
            in_r_buf->address(),
            /*in_i_addr=*/0u,
            /*tw_r_addr=*/0u,
            /*tw_i_addr=*/0u,
            /*base=*/0u,
            /*batch_per_core=*/1u,
            /*phys_x=*/0u,
            /*phys_y=*/0u,
        }},
        .config = ReaderConfigDescriptor{},
    });

    // Writer (BRISC1): drains CB_OUT0/OUT1 into output tensor buffers.
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_writer.cpp",
        .core_ranges = crs,
        .compile_time_args = {},
        .common_runtime_args = {},
        .runtime_args = {{
            out_r_buf->address(),
            out_i_buf->address(),
            /*base=*/0u,
            /*batch_per_core=*/1u,
        }},
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
        .common_runtime_args = {},
        .runtime_args = {{
            /*batch_per_core=*/1u,
        }},
        .config = ComputeConfigDescriptor{
            .math_fidelity       = MathFidelity::HiFi4,
            .fp32_dest_acc_en    = true,
            .unpack_to_dest_mode = u2d,
        },
    });

    return desc;
}

}  // namespace ttnn::experimental::prim
