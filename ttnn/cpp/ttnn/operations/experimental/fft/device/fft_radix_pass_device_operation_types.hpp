// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::fft_radix_pass — the
// fused [batched length-P FFT  +  optional post-twiddle cmul] kernel
// that is the building block for the K-pass composite (commit 5,
// fft_universal_xl for N up to 1G).
//
// Semantics, for input of shape (..., M, P):
//   For each row r ∈ [0, M):
//     y[r, :] = FFT_P(in[r, :])
//     if apply_post_twiddle:
//       y[r, k] *= exp(-2πi * (r % twiddle_N2) * k / (P * twiddle_N2))
//
// twiddle_N1 is implicit and always equals P.

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct FftRadixPassParams {
    // FFT length per row.  Pow-2 in [2, 1024].
    uint32_t P = 0;
    // 0 → no post-twiddle (pure batched FFT, same observable behaviour
    //     as a BatchedStockhamFactory call).
    // >0 → multiply each row's FFT output by twiddle row
    //     (r % twiddle_N2).  Pow-2 in [1, 1024] and must divide the
    //     product of leading dims of the input.
    uint32_t twiddle_N2 = 0;
};

// input_imag is optional: for a Pass-1 (real input) radix pass we leave
// it empty and the factory wires up a cached zero scratch.  For Pass-2
// (complex input) the caller passes the imag tensor.
struct FftRadixPassTensorArgs {
    Tensor                input_real;
    std::optional<Tensor> input_imag;
};

}  // namespace ttnn::experimental::prim
