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
#include "ttnn/operations/experimental/fft/device/rebank_rm_device_operation.hpp"

#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/creation/creation.hpp"  // ttnn::zeros for the imag input fallback

#include "ttnn/distributed/types.hpp"             // MeshDevice
#include "ttnn/types.hpp"

#include <array>

namespace ttnn::operations::experimental {

namespace {

// Source-page threshold above which ttnn::reshape allocates a CB equal to the
// full source row (e.g. (1,131072)→(128,1024) uses 4×1 MB CB → L1 overflow).
// Matches kRebankThresholdBytes in fft.cpp (kept separate to avoid Unity collision).
constexpr uint32_t kBluesteinRebankThreshold = 64u * 1024u;

// Page-shrinking reshape helper: (rows_in, cols_in) → (rows_out, cols_out).
// Uses rebank_rm (DRAM-to-DRAM, tiny CB ≤ 4 KB) when the source page
// exceeds the L1-safe threshold, otherwise falls through to ttnn::reshape.
static ttnn::Tensor shrink_reshape(
    const ttnn::Tensor& t, uint32_t new_cols)
{
    const auto& s = t.padded_shape();
    const uint32_t src_cols = static_cast<uint32_t>(s[-1]);
    const uint32_t elem_bytes =
        (t.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    if (src_cols * elem_bytes > kBluesteinRebankThreshold) {
        return ttnn::prim::rebank_rm(t, new_cols);
    }
    uint32_t total = 1u;
    for (int d = 0; d < static_cast<int>(s.size()); ++d)
        total *= static_cast<uint32_t>(s[d]);
    const uint32_t new_rows = total / new_cols;
    return ttnn::reshape(t,
        ttnn::Shape{ttnn::SmallVector<uint32_t>{new_rows, new_cols}});
}

// complex_mul_safe: element-wise complex multiply for any last-dim size.
//
// The complex_mul kernel has a hard cap of P ≤ 1024 (one tile row).
// For P > 1024 we split into 1024-element chunks:
//
//   Case A — P divisible by 1024 (always true for Bluestein M, a pow-2):
//     Rebank  (B, P) → (B·P/1024, 1024) via rebank_rm when P is large
//     (avoids the 4 MB CB that ttnn::reshape would require for M=131072).
//     Multiply, then reshape back (page-growing, CB ≤ 1 MB, always safe).
//
//   Case B — P NOT divisible by 1024 (e.g. steps 1 & 7 with N=1997):
//     Pad each input from (B, P) to (B, P_pad) where P_pad is the next
//     multiple of 1024, apply Case A logic, then slice back to (B, P).
//     Cost: 4 extra pad dispatches + 1 slice dispatch — acceptable for the
//     rare non-pow-2 large-N Bluestein case.
static std::tuple<ttnn::Tensor, ttnn::Tensor> complex_mul_safe(
    const ttnn::Tensor& ar, const ttnn::Tensor& ai,
    const ttnn::Tensor& br, const ttnn::Tensor& bi)
{
    const auto& sh = ar.padded_shape();
    const uint32_t P = static_cast<uint32_t>(sh[-1]);
    if (P <= 1024u)
        return complex_mul(ar, ai, br, bi);

    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(sh.size()) - 1; ++d)
        B *= static_cast<uint32_t>(sh[d]);

    if (P % 1024u == 0u) {
        // Case A: P a multiple of 1024.
        // shrink_reshape uses rebank_rm when source page (P × elem_bytes) > 64 KB,
        // so (1,131072)→(128,1024) goes DRAM-to-DRAM without a large L1 CB.
        const auto orig = ttnn::Shape{ttnn::SmallVector<uint32_t>{B, P}};
        auto [cr, ci] = complex_mul(
            shrink_reshape(ar, 1024u), shrink_reshape(ai, 1024u),
            shrink_reshape(br, 1024u), shrink_reshape(bi, 1024u));
        // Reshape back: page grows from 1024 to P — CB ≤ 1 MB, always within L1.
        return {ttnn::reshape(cr, orig), ttnn::reshape(ci, orig)};
    }

    // Case B: P is not a multiple of 1024 (e.g. N=1997 chirp mul).
    // Round up to the next multiple of 1024 via zero-padding, multiply, slice.
    const uint32_t P_pad = (P + 1023u) & ~1023u;   // next multiple of 1024
    const uint32_t pad_len = P_pad - P;

    const ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
        {{0u, 0u}},          // batch dim: no padding
        {{0u, pad_len}},     // last dim: append zeros
    };
    const auto mc = ar.memory_config();
    auto ar_p = ttnn::pad(ar, padding, 0.0f, /*use_multicore=*/true, mc);
    auto ai_p = ttnn::pad(ai, padding, 0.0f, /*use_multicore=*/true, mc);
    auto br_p = ttnn::pad(br, padding, 0.0f, /*use_multicore=*/true, mc);
    auto bi_p = ttnn::pad(bi, padding, 0.0f, /*use_multicore=*/true, mc);

    const auto padded = ttnn::Shape{ttnn::SmallVector<uint32_t>{B, P_pad}};

    auto [cr_p, ci_p] = complex_mul(
        shrink_reshape(ar_p, 1024u), shrink_reshape(ai_p, 1024u),
        shrink_reshape(br_p, 1024u), shrink_reshape(bi_p, 1024u));

    // Slice result back to (B, P) — padded positions are 0 * x = 0.
    const ttnn::SmallVector<uint32_t> begins = {0u, 0u};
    const ttnn::SmallVector<uint32_t> ends   = {B,  P};
    const ttnn::SmallVector<uint32_t> step   = {1u, 1u};
    auto cr = ttnn::slice(ttnn::reshape(cr_p, padded), begins, ends, step, mc);
    auto ci = ttnn::slice(ttnn::reshape(ci_p, padded), begins, ends, step, mc);
    return {std::move(cr), std::move(ci)};
}

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
    FFTPrecision precision,
    bool inverse)
{
    using namespace ttnn::experimental::prim::bluestein_host;

    // ── Validation ──────────────────────────────────────────────────────
    const auto& in_shape = input_real.padded_shape();
    TT_FATAL(in_shape.size() == 2u,
        "bluestein_fft: input must be 2-D (B, N).  Got rank {}.",
        in_shape.size());
    const uint32_t B = static_cast<uint32_t>(in_shape[0]);
    TT_FATAL(B >= 1u,
        "bluestein_fft: batch dim must be ≥ 1 (got {}).", B);
    TT_FATAL(static_cast<uint32_t>(in_shape[1]) == N,
        "bluestein_fft: input last-dim must equal N = {} (got {}).",
        N, static_cast<uint32_t>(in_shape[1]));
    TT_FATAL(input_real.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "bluestein_fft: input must be ROW_MAJOR.");
    TT_FATAL(input_real.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             input_real.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "bluestein_fft: only Float32 / BFloat16 supported.");

    const uint32_t M = bluestein_M(N);
    // M is capped by fft_three_pass limit (2^30).  Inner fft() / ifft() calls
    // route automatically through fft_two_pass (M ≤ 2^20) or fft_three_pass
    // (2^20 < M ≤ 2^30) via the unified router in fft.cpp.
    TT_FATAL(M <= (1u << 30),
        "bluestein_fft: padded M = {} > 2^30 is not yet supported (N = {}).",
        M, N);

    if (input_imag.has_value()) {
        TT_FATAL(input_imag->padded_shape() == in_shape &&
                 input_imag->dtype()        == input_real.dtype() &&
                 input_imag->layout()       == input_real.layout(),
            "bluestein_fft: input_imag must match input_real in "
            "shape/dtype/layout.");
    }

    // ── Get plan (chirp_n, chirp_k, B_fft) — cached per (device, N, dtype, B, inverse).
    auto plan = get_or_create(
        input_real.device(),
        N,
        input_real.dtype(),
        B,
        precision,
        inverse);

    // ── Materialise an explicit zero imag input when omitted ───────────
    //   The first step (complex_mul with chirp_n) requires both halves;
    //   our complex_mul is shape-strict and doesn't broadcast a missing
    //   imag, so we synthesise one here.
    const ttnn::Tensor x_imag = input_imag.has_value()
        ? *input_imag
        : make_zeros_like(input_real);

    // ── Step 1: pre-multiply by chirp_n  (1, N) × (1, N) → (1, N).
    auto [a_re, a_im] = complex_mul_safe(
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

    // ── Step 4: convolution multiply  A ⊙ B   (B, M) × (B, M).
    //   Uses complex_mul_safe because M may exceed the 1024-element kernel cap
    //   (e.g. N=997 → M=2048).
    auto [C_re, C_im] = complex_mul_safe(A_re, A_im, plan->B_re, plan->B_im);

    // ── Step 5: inverse FFT_M  ─────────────────────────────────────────
    auto [c_re, c_im] = ifft(C_re, C_im, precision);

    // ── Step 6: slice first N elements (linear-conv result lives in
    //   the first N indices — the trailing M-N indices are cyclic-
    //   wrap-around garbage).
    ttnn::SmallVector<uint32_t> begins = {0u, 0u};
    ttnn::SmallVector<uint32_t> ends   = {B,  N};
    ttnn::SmallVector<uint32_t> step   = {1u, 1u};
    auto c_re_n = ttnn::slice(c_re, begins, ends, step, c_re.memory_config());
    auto c_im_n = ttnn::slice(c_im, begins, ends, step, c_im.memory_config());

    // ── Step 7: post-multiply by chirp_k → final DFT output X[k].
    auto [X_re, X_im] = complex_mul_safe(
        c_re_n, c_im_n, plan->chirp_k_re, plan->chirp_k_im);

    return {std::move(X_re), std::move(X_im)};
}

}  // namespace ttnn::operations::experimental
