// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Numerical-precision selector for the small-N (N <= 32) Float32 path.
//
// The packed direct-DFT kernel (``packed_dft_compute``) uses the Tensix FPU
// matmul for throughput. The FPU multiplier is bf16-mantissa even when the
// destination accumulator is set to fp32, so round-trip error at small N is
// ~1e-3 (good enough for most signal-processing/ML use). The Stockham/
// Bluestein/Cooley-Tukey paths use SFPU ``*_binary_tile`` ops which are true
// IEEE fp32 throughout — round-trip error ~1e-7 to match torch.fft.
//
// Only meaningful for ``Float32, !is_pow2(N)``. Pow2 fp32 already routes via
// Stockham (true fp32). bfloat16 paths always use the bf16 FPU matmul.
enum class FFTPrecision : uint8_t {
    Precise = 0,    // SFPU true-fp32 (default; matches torch precision)
    Fast    = 1,    // FPU bf16-mantissa matmul (faster, ~1e-3 round-trip)
};

// Operation-level attributes (kernel-affecting only — see compute_program_hash).
struct FFTParams {
    bool         inverse   = false;
    FFTPrecision precision = FFTPrecision::Precise;
};

// Tensor inputs to the device op. Forward FFT uses input_real only; IFFT
// also requires input_imag (the imaginary half of the spectrum). Carrying
// an optional through the device-op layer keeps the dispatch single-path.
struct FFTTensorArgs {
    Tensor                input_real;
    std::optional<Tensor> input_imag;
};

// Backend selected at validate time, used by the program factory to pick
// which kernel pipeline to instantiate.
enum class FFTBackend : uint8_t {
    Stockham,        // Float32, pow2 N, N <= 1M
    UniversalXL,     // Float32, pow2 N, 1M < N <= 16M
    Universal,       // Float32, non-pow2 N (mixed-radix / Bluestein)
    UniversalBf16,   // BFloat16, any N (true-bf16 FPU matmul)
};

}  // namespace ttnn::experimental::prim
