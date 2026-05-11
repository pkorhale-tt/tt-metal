// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::fft — 1-D Fast Fourier Transform.
//
// API design
// ----------
// Inputs  : real-valued tensor (Float32 or BFloat16). Treated as a real
//           signal of length N = input_real.shape()[-1]. Batched along the
//           remaining (leading) dimensions.
// Outputs : pair {real, imag} tensors, same shape as input, holding the
//           full complex spectrum in natural-order bins (X[0], X[1], ...,
//           X[N-1]).
//
// Why two tensors instead of one complex tensor: ttnn does not currently
// have a native complex dtype. Returning {real, imag} matches existing
// ttnn conventions and lets callers reconstruct a torch.complex64 tensor
// trivially: torch.complex(out_real.to_torch(), out_imag.to_torch()).
//
// Backend dispatch (see device/fft_program_factory.cpp):
//   * dtype == Float32, N pow2, N <= 1M  -> fft_stockham (fastest path)
//   * dtype == Float32, N pow2, 1M < N <= 16M -> fft_universal_xl
//   * dtype == Float32, N not pow2       -> fft_universal (Bluestein)
//   * dtype == BFloat16, any N           -> fft_universal_bf16
//
// Phase 1 of this op only ships the fft_stockham backend dispatch
// (Float32, pow2, N <= 1M). The other three branches are present in the
// switch with clear NotImplementedError messages so it is obvious where
// to add them.

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "device/fft_device_operation_types.hpp"

#include <utility>

namespace ttnn {
namespace operations::experimental {

// Re-export of the device-op precision selector at the public op layer
// so callers don't have to reach into the prim:: namespace.
using FFTPrecision = ttnn::experimental::prim::FFTPrecision;

struct FFTOperation {
    // Forward FFT — real input.
    //   input_real : real-valued signal, last dim = N (the FFT length).
    //                Batched along leading dims.
    //   precision  : Precise (default, true fp32) or Fast (FPU bf16-mantissa
    //                matmul). See FFTPrecision for the trade-off; only matters
    //                for Float32 + non-pow2 N. Pow2 fp32 is always Stockham
    //                (already true fp32); bf16 always uses bf16 matmul.
    //   returns    : {real, imag} of the spectrum. Same shape as input.
    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_real,
        FFTPrecision  precision = FFTPrecision::Precise);

    // Forward FFT — complex input (two-tensor form).
    //   input_real / input_imag : real and imaginary halves of the input
    //                             signal. Must match in dtype/shape/layout.
    //   returns                 : {real, imag} of the spectrum, same shape.
    // Equivalent to the standard ``X = fft(input_real + i * input_imag)``.
    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_real,
        const Tensor& input_imag,
        FFTPrecision  precision = FFTPrecision::Precise);
};

// Inverse FFT. Separate registered_operation_t so it appears as
// `ttnn.experimental.ifft(spec_re, spec_im)` in Python alongside
// `ttnn.experimental.fft(x)`.
struct IFFTOperation {
    //   spectrum_real / spectrum_imag : real / imag parts of the spectrum.
    //   precision                     : same selector as FFT (forwarded to
    //                                   the underlying forward dispatch via
    //                                   the conjugate trick).
    //   returns                       : {real, imag} of the time-domain
    //                                   signal divided by N.
    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& spectrum_real,
        const Tensor& spectrum_imag,
        FFTPrecision  precision = FFTPrecision::Precise);
};

}  // namespace operations::experimental

namespace experimental {
// Registered under the experimental namespace per Metal team convention —
// new ops graduate to ttnn:: top-level only after the API stabilises.
// Python users call ``ttnn.experimental.fft(x)`` / ``ttnn.experimental.ifft(re, im)``.
constexpr auto fft =
    ttnn::register_operation<"ttnn::experimental::fft", ttnn::operations::experimental::FFTOperation>();

constexpr auto ifft =
    ttnn::register_operation<"ttnn::experimental::ifft", ttnn::operations::experimental::IFFTOperation>();
}  // namespace experimental

}  // namespace ttnn
