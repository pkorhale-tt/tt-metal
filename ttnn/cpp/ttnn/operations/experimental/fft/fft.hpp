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

#include <utility>

namespace ttnn {
namespace operations::experimental {

struct FFTOperation {
    // Forward FFT.
    //   input_real : real-valued signal, last dim = N (the FFT length).
    //                Batched along leading dims.
    //   returns    : {real, imag} of the spectrum. Same shape as input.
    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke(const Tensor& input_real);

    // Inverse FFT (conjugate-trick wrapper around forward FFT).
    //   spectrum_real / spectrum_imag : real / imag parts of the spectrum.
    //   returns                       : {real, imag} of the time-domain
    //                                   signal divided by N.
    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke_ifft(
        const Tensor& spectrum_real,
        const Tensor& spectrum_imag);
};

}  // namespace operations::experimental

constexpr auto fft =
    ttnn::register_operation<"ttnn::fft", ttnn::operations::experimental::FFTOperation>();

}  // namespace ttnn
