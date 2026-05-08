// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FFT program factory — Phase 1 (host-pass-through).
//
// Status (Phase 1, "Path A — host pass-through"):
// =====================================================================
// The FFT itself runs on the HOST CPU using a textbook iterative
// radix-2 Cooley–Tukey kernel (forward and inverse). The on-device
// program returned by `create()` is empty; we use the device_operation
// framework purely as a Tensor-plumbing layer so that the public
// `ttnn.experimental.fft` API behaves like any other ttnn op
// (validation, output-tensor allocation, program-cache hash, etc.) and
// the user gets back tensors populated with correct spectrum data.
//
// Why host-only for Phase 1:
//   * Zero new device-build artifacts → no kernel-path / installation
//     issues, no build-system surface area to land in this PR.
//   * The on-device kernels already exist and are validated end-to-end
//     in tt_metal/programming_examples/fft_stockham/ and friends, but
//     they are organised as standalone hosts (program-per-pass with
//     their own EnqueueWriteShard / EnqueueReadShard), not as building
//     blocks of a single fused Program. Wiring them into a single
//     ttnn-style Program is Phase 2 (see TODO block at the end of this
//     file).
//   * Numerically correct FFT on the host is what unblocks downstream
//     model authors who just want `ttnn.experimental.fft(x)` to work.
//
// Phase 2 (later PR) will replace `run_host_fft()` with a true
// on-device program build that reuses the Stockham kernels.

#include "fft_program_factory.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace ttnn::experimental::prim {

namespace {

using Complex = std::complex<float>;

// In-place iterative Cooley–Tukey radix-2 DIT FFT.
// `inverse=false` → e^{-i 2π k n / N};  `inverse=true` → e^{+i 2π k n / N}.
// Caller is responsible for the 1/N scale on the inverse path.
//
// O(N log N) work, O(1) extra memory beyond the in-place buffer.
void radix2_inplace(Complex* a, uint32_t N, bool inverse) {
    // Bit-reversal permutation.
    for (uint32_t i = 1u, j = 0u; i < N; ++i) {
        uint32_t bit = N >> 1u;
        for (; (j & bit) != 0u; bit >>= 1u) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    const double sign = inverse ? +1.0 : -1.0;
    for (uint32_t len = 2u; len <= N; len <<= 1u) {
        const double  theta = sign * 2.0 * M_PI / static_cast<double>(len);
        const Complex wlen  = {static_cast<float>(std::cos(theta)),
                               static_cast<float>(std::sin(theta))};
        const uint32_t half = len >> 1u;
        for (uint32_t i = 0u; i < N; i += len) {
            Complex w{1.0f, 0.0f};
            for (uint32_t k = 0u; k < half; ++k) {
                const Complex u = a[i + k];
                const Complex v = a[i + k + half] * w;
                a[i + k]        = u + v;
                a[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }
}

// Validates that N is a power of two — required by `radix2_inplace`.
// (compute_output_specs / validate_on_program_cache_miss already enforce
// this for the Stockham backend; we re-check here so the host fallback
// fails loudly if anyone widens the dispatch table without updating
// this kernel.)
//
// Renamed `is_pow2_local` (rather than `is_pow2`) so the Unity build
// can pull both this TU and fft_device_operation.cpp — which has its
// own anonymous-namespace `is_pow2` — into the same compilation unit
// without ODR collision.
constexpr bool is_pow2_local(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// Reads the input tensor(s) to host, runs an N-point FFT (or IFFT) on
// every length-N row, and replaces `tensor_return_value` with two new
// device tensors holding the real and imaginary halves of the spectrum.
//
// We replace the existing tensors (rather than writing into their
// existing buffers) to keep this function self-contained — `from_vector`
// allocates a fresh DRAM buffer with the correct spec and uploads in
// one go via the standard ttnn enqueue path. The blank tensors
// originally allocated by `FFTDeviceOperation::create_output_tensors`
// become unreferenced and are freed by RAII.
//
// Phase 1 limitation: this drops the per-shard TensorTopology that the
// device_operation framework imputes onto the blank output tensors
// before calling us. For Phase 1 we only support single-device
// dispatch, so the loss is invisible to callers; Phase 2 (real
// on-device program) will preserve the original tensor objects.
void run_host_fft(
    const FFTParams&            attrs,
    const FFTTensorArgs&        tensor_args,
    std::pair<Tensor, Tensor>&  tensor_return_value) {

    const auto& in_re_tensor = tensor_args.input_real;
    const auto& shape        = in_re_tensor.logical_shape();
    TT_FATAL(shape.size() >= 1u, "fft host fallback: tensor has rank 0");

    const uint32_t N = shape[-1];
    TT_FATAL(is_pow2_local(N),
             "fft host fallback: only power-of-two N supported (got {}).", N);

    const uint64_t total = in_re_tensor.logical_volume();
    TT_FATAL(total % N == 0u,
             "fft host fallback: total volume {} not divisible by N {}.",
             total, N);
    const uint64_t batches = total / N;

    // 1. Bring inputs to host as plain float buffers. `to_vector<float>`
    //    handles device→host transfer + layout/dtype materialisation.
    const std::vector<float> in_re = in_re_tensor.to_vector<float>();
    std::vector<float>       in_im;
    if (attrs.inverse) {
        TT_FATAL(tensor_args.input_imag.has_value(),
                 "fft host fallback: inverse path requires input_imag.");
        in_im = tensor_args.input_imag->to_vector<float>();
        TT_FATAL(in_im.size() == in_re.size(),
                 "fft host fallback: inverse input_real/input_imag size mismatch "
                 "({} vs {}).",
                 in_re.size(), in_im.size());
    } else {
        in_im.assign(in_re.size(), 0.0f);
    }
    TT_FATAL(in_re.size() == total,
             "fft host fallback: to_vector returned {} elements, expected {}.",
             in_re.size(), total);

    // 2. Run radix-2 FFT row-by-row.
    std::vector<float>   out_re(total);
    std::vector<float>   out_im(total);
    std::vector<Complex> work(N);

    const float scale = attrs.inverse ? (1.0f / static_cast<float>(N)) : 1.0f;
    for (uint64_t b = 0u; b < batches; ++b) {
        const uint64_t off = b * static_cast<uint64_t>(N);
        for (uint32_t i = 0u; i < N; ++i) {
            work[i] = Complex{in_re[off + i], in_im[off + i]};
        }
        radix2_inplace(work.data(), N, /*inverse=*/attrs.inverse);
        for (uint32_t i = 0u; i < N; ++i) {
            out_re[off + i] = work[i].real() * scale;
            out_im[off + i] = work[i].imag() * scale;
        }
    }

    // 3. Replace output tensors with fresh device tensors holding the
    //    spectrum. The spec mirrors the input spec (compute_output_specs
    //    enforces this contract), so output dtype/shape/layout/memory-
    //    config all match what the user expects.
    auto*        device = in_re_tensor.device();
    const auto&  spec   = in_re_tensor.tensor_spec();
    tensor_return_value.first  = Tensor::from_vector(std::move(out_re), spec, device);
    tensor_return_value.second = Tensor::from_vector(std::move(out_im), spec, device);
}

}  // namespace

FFTProgramFactory::cached_program_t FFTProgramFactory::create(
    const FFTParams&            operation_attributes,
    const FFTTensorArgs&        tensor_args,
    std::pair<Tensor, Tensor>&  tensor_return_value) {
    // Phase 1: do the FFT on the host. The same call also runs in
    // override_runtime_arguments() so cache hits stay correct.
    run_host_fft(operation_attributes, tensor_args, tensor_return_value);

    // Empty program — nothing to enqueue on device for Phase 1.
    tt::tt_metal::Program program{};
    const uint32_t N = tensor_args.input_real.logical_shape()[-1];

    return cached_program_t{
        std::move(program),
        FFTSharedVariables{
            .kernel_ids = {},
            .cores      = {},
            .N          = N,
        }};
}

void FFTProgramFactory::override_runtime_arguments(
    cached_program_t&           /*cached_program*/,
    const FFTParams&            operation_attributes,
    const FFTTensorArgs&        tensor_args,
    std::pair<Tensor, Tensor>&  tensor_return_value) {
    // Phase 1: cache-hit path — re-run the host FFT. The cached
    // "program" itself is empty, so there are no kernel runtime args
    // to update.
    run_host_fft(operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim

// =====================================================================
// PHASE 2 TODO — true on-device program
// =====================================================================
// Replace `run_host_fft()` with a Program that runs the four-pass
// Stockham pipeline (pass1 / pass2 / pass3 / batch_fft) on the input
// real tensor and writes the (real, imag) spectrum into the two output
// tensors directly in DRAM.
//
// Phase 2-A (DONE): kernel sources are now in-tree at
//   ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/
//     dataflow/{fft,batch_fft,pass2}_{reader,writer}.cpp + *_common.h
//     compute/{fft,batch_fft,pass2}_compute.cpp
// They are installed alongside the ttnn library (see CMakeLists.txt)
// so Phase 2-B can call CreateKernel(...) on them directly.
//
// 1. Refactor tt_metal/programming_examples/fft_stockham/fft_stockham_host.cpp
//    so the program-build code is callable as a free function:
//      build_stockham_program(MeshDevice*, uint32_t N,
//                             Buffer* in_real,
//                             Buffer* out_real, Buffer* out_imag)
//        -> { Program, std::vector<KernelHandle>, std::vector<CoreCoord> }
//    rather than baked into a top-level fft() that owns its own buffers
//    and runs its own enqueues. Place the result at
//      ttnn/cpp/ttnn/operations/experimental/fft/device/stockham_host.hpp
//    (header-only inline functions, mirrors the programming_examples
//    structure).
//
// 2. The current pass1_reader expects a complex-interleaved (real+imag)
//    DRAM layout. ttnn input is real-only. Either:
//      (a) zero-fill the imaginary half on host before kernel launch
//          (one extra DRAM write — simplest), or
//      (b) modify pass1_reader to synthesize imag=0 (faster, kernel
//          change).
//    Phase-2-A: ship (a). Phase-2-B: optimize to (b).
//
// 3. override_runtime_arguments must SetRuntimeArgs(kernel_id, core,
//    {in_buf->address(), out_re->address(), out_im->address(), N})
//    on every call, since the buffer addresses change per dispatch.
//
// 4. Inverse path: y = conj(fft(conj(X))) / N. Run the forward Stockham
//    pipeline on conj(X), then conjugate + scale via a single SFPU pass
//    appended to the writer (or as a tiny dedicated unary kernel).
//
// 5. Batching: ttnn tensors are typically [B, ..., N]. The current
//    fft_stockham handles ONE FFT of length N. Either loop over batches
//    at the program-factory level (B device enqueues — simple, slow for
//    large B), or extend the existing batch_fft kernel — it already does
//    sub_N <= 1024 batched FFTs and could be wrapped for larger N.
//
// All of the above is mechanical; the kernels themselves are correct
// (verified end-to-end in the programming_examples).
