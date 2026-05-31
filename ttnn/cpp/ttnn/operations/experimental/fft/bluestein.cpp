// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bluestein.cpp — arbitrary-N DFT via Bluestein's chirp-Z transform.
//
// Per-call device dispatch chain (B = 1, length N → length N):
//
//   1.  complex_mul(x, chirp_n)              — pre-twiddle, shape (1, N)
//   2.  pad to (1, M),  trailing zeros       — ttnn::pad
//   3.  fft (forward, length M)              — ttnn::experimental::fft
//   4.  complex_mul(A, B)                    — convolution multiply, (1, M)
//   5.  ifft (length M)                      — ttnn::experimental::ifft
//   6.  slice [:, :N]                        — ttnn::slice
//   7.  complex_mul(c, chirp_k)              — post-twiddle, shape (1, N)
//
// Step 3 / 5 each lower to either the SingleTileStockham factory
// (M ≤ 1024) or fft_two_pass (1024 < M ≤ 1M).  Chirp_n, chirp_k, and
// B = FFT(b_cyc) are pre-computed and cached per (device, N, dtype) —
// see device/bluestein_host.hpp.

#include "ttnn/operations/experimental/fft/bluestein.hpp"

#include "ttnn/operations/experimental/fft/complex_mul.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"
#include "ttnn/operations/experimental/fft/device/bluestein_host.hpp"

#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/creation/creation.hpp"  // ttnn::zeros for the imag input fallback

#include "ttnn/distributed/types.hpp"             // MeshDevice
#include "ttnn/types.hpp"

#include <array>

namespace ttnn::operations::experimental {

namespace {

// Build a (1, N) zeros tensor matching `like` for the implicit zero-imag
// case (Bluestein needs an explicit imag input because the pipeline does
// a complex_mul as its very first step).
ttnn::Tensor make_zeros_like(const ttnn::Tensor& like) {
    auto* dev = like.device();
    TT_FATAL(dev != nullptr, "bluestein_fft: input tensor has no device.");
    return ttnn::zeros(
        like.logical_shape(),
        like.dtype(),
        like.layout(),
        std::ref(*dev),
        like.memory_config());
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor> bluestein_fft(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    uint32_t N,
    FFTPrecision precision)
{
    using namespace ttnn::experimental::prim::bluestein_host;

    // ── Validation ──────────────────────────────────────────────────────
    const auto& in_shape = input_real.padded_shape();
    TT_FATAL(in_shape.size() == 2u,
        "bluestein_fft: input must be 2-D (1, N).  Got rank {}.",
        in_shape.size());
    TT_FATAL(static_cast<uint32_t>(in_shape[0]) == 1u,
        "bluestein_fft: batched input (B > 1) is not yet supported (commit "
        "6d only).  Got B = {}.",
        static_cast<uint32_t>(in_shape[0]));
    TT_FATAL(static_cast<uint32_t>(in_shape[1]) == N,
        "bluestein_fft: input last-dim must equal N = {} (got {}).",
        N, static_cast<uint32_t>(in_shape[1]));
    TT_FATAL(input_real.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "bluestein_fft: input must be ROW_MAJOR.");
    TT_FATAL(input_real.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             input_real.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "bluestein_fft: only Float32 / BFloat16 supported.");

    const uint32_t M = bluestein_M(N);
    TT_FATAL(M <= (1u << 20),
        "bluestein_fft: padded length M = {} exceeds commit-6d cap of 2^20 = "
        "1M.  N must satisfy 2*N - 1 ≤ 2^20, i.e. N ≤ 524_288. (got N = {}).",
        M, N);

    if (input_imag.has_value()) {
        TT_FATAL(input_imag->padded_shape() == in_shape &&
                 input_imag->dtype()        == input_real.dtype() &&
                 input_imag->layout()       == input_real.layout(),
            "bluestein_fft: input_imag must match input_real in "
            "shape/dtype/layout.");
    }

    // ── Get plan (chirp_n, chirp_k, B) ─────────────────────────────────
    auto plan = get_or_create(
        input_real.device(),
        N,
        input_real.dtype(),
        precision);

    // ── Materialise an explicit zero imag input when omitted ───────────
    //   The first step (complex_mul with chirp_n) requires both halves;
    //   our complex_mul is shape-strict and doesn't broadcast a missing
    //   imag, so we synthesise one here.
    const ttnn::Tensor x_imag = input_imag.has_value()
        ? *input_imag
        : make_zeros_like(input_real);

    // ── Step 1: pre-multiply by chirp_n  (1, N) × (1, N) → (1, N).
    auto [a_re, a_im] = complex_mul(
        input_real, x_imag, plan->chirp_n_re, plan->chirp_n_im);

    // ── Step 2: zero-pad last dim from N to M  (1, N) → (1, M).
    //   ttnn::pad takes {before, after} per dim; we only append zeros.
    ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
        {{0u, 0u}},
        {{0u, M - N}},
    };
    auto a_pad_re = ttnn::pad(a_re, padding, /*value=*/0.0f,
                              /*use_multicore=*/true, a_re.memory_config());
    auto a_pad_im = ttnn::pad(a_im, padding, /*value=*/0.0f,
                              /*use_multicore=*/true, a_im.memory_config());

    // ── Step 3: forward FFT_M  ─────────────────────────────────────────
    //   Routes through SingleTileStockham (M ≤ 1024) or fft_two_pass
    //   (1024 < M ≤ 1M).
    auto [A_re, A_im] = fft(a_pad_re, a_pad_im, precision);

    // ── Step 4: convolution multiply  A ⊙ B   (1, M) × (1, M).
    auto [C_re, C_im] = complex_mul(A_re, A_im, plan->B_re, plan->B_im);

    // ── Step 5: inverse FFT_M  ─────────────────────────────────────────
    auto [c_re, c_im] = ifft(C_re, C_im, precision);

    // ── Step 6: slice first N elements (linear-conv result lives in
    //   the first N indices — the trailing M-N indices are cyclic-
    //   wrap-around garbage).
    ttnn::SmallVector<uint32_t> begins = {0u, 0u};
    ttnn::SmallVector<uint32_t> ends   = {1u, N};
    ttnn::SmallVector<uint32_t> step   = {1u, 1u};
    auto c_re_n = ttnn::slice(c_re, begins, ends, step, c_re.memory_config());
    auto c_im_n = ttnn::slice(c_im, begins, ends, step, c_im.memory_config());

    // ── Step 7: post-multiply by chirp_k → final DFT output X[k].
    auto [X_re, X_im] = complex_mul(
        c_re_n, c_im_n, plan->chirp_k_re, plan->chirp_k_im);

    return {std::move(X_re), std::move(X_im)};
}

}  // namespace ttnn::operations::experimental
