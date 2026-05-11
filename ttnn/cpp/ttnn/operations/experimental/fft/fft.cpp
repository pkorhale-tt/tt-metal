// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/fft.hpp"
#include "device/fft_device_operation.hpp"

namespace ttnn::operations::experimental {

std::pair<ttnn::Tensor, ttnn::Tensor> FFTOperation::invoke(
    const Tensor& input_real, FFTPrecision precision) {
    return ttnn::prim::fft(input_real, /*inverse=*/false,
                           /*input_imag=*/std::nullopt, precision);
}

std::pair<ttnn::Tensor, ttnn::Tensor> FFTOperation::invoke(
    const Tensor& input_real, const Tensor& input_imag, FFTPrecision precision) {
    return ttnn::prim::fft(input_real, /*inverse=*/false, input_imag, precision);
}

std::pair<ttnn::Tensor, ttnn::Tensor> IFFTOperation::invoke(
    const Tensor& spectrum_real, const Tensor& spectrum_imag,
    FFTPrecision precision) {
    return ttnn::prim::fft(spectrum_real, /*inverse=*/true,
                           spectrum_imag, precision);
}

}  // namespace ttnn::operations::experimental
