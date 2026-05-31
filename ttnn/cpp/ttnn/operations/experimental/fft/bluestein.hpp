// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::bluestein_fft — arbitrary-N (not just pow-2) DFT
// via Bluestein's chirp-Z transform.  Composes existing primitives:
//     complex_mul + pad + fft + complex_mul + ifft + slice + complex_mul
//
// See device/bluestein_host.hpp for the algorithm and caching scheme.
//
// ─ Supported sizes ─
//   N ≥ 2 with M := next_pow2(2*N - 1) ≤ 2^20 (1M).
//     → N up to ~ 524_288.
//     N > ~500K (M > 1M) deferred to 6e-2 once the inner FFT can route
//     through fft_three_pass with the pre-shaped-input rebank trick.
//   B (batch) ≥ 1 — chirp / B tensors are replicated to (B, N) / (B, M)
//     and cached per (device, N, dtype, B) on first call.

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"   // FFTPrecision

namespace ttnn::operations::experimental {

// Forward Bluestein FFT.  `input_real` is shape (B, N); `input_imag`, if
// supplied, must match.  Returns (out_real, out_imag) of shape (B, N).
//
// `B` can be any positive integer.  Chirp / B_fft tensors are replicated
// to (B, ·) on cache miss; subsequent calls with the same (N, dtype, B)
// hit the cache.
std::tuple<ttnn::Tensor, ttnn::Tensor> bluestein_fft(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    uint32_t N,
    FFTPrecision precision = FFTPrecision::Precise);

}  // namespace ttnn::operations::experimental
