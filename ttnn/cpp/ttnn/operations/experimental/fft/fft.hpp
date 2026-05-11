// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::fft — 1-D Fast Fourier Transform.

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
    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_real,
        FFTPrecision  precision = FFTPrecision::Precise);

    static std::pair<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& input_real,
        const Tensor& input_imag,
        FFTPrecision  precision = FFTPrecision::Precise);
};

// Inverse FFT. Separate registered_operation_t so it appears as
// `ttnn.experimental.ifft(spec_re, spec_im)` in Python alongside
// `ttnn.experimental.fft(x)`.
struct IFFTOperation {
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
