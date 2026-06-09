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

#include "ttnn/operations/data_movement/concat/concat.hpp"
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
// The complex_mul kernel has a hard cap of P ≤ 1024 (one tile row), and its
// CB scales as 4 inputs × 2 (dbl-buf) × B_eff × 1024 × elem_bytes.  For
// large B_eff (= B × P/1024) that exceeds the 1.5 MB L1 limit.
//
// Strategy for P > 1024:
//   1. Rebank (B, P) → (B·P/1024, 1024) via rebank_rm when the source page
//      is large (avoids the multi-MB reshape CB).
//   2. If B_eff = B·P/1024 is within the safe limit (b_safe), one complex_mul
//      call suffices.  Otherwise, slice the rebankd tensor into b_safe-row
//      blocks, call complex_mul once per block, and concat the results.
//   3. Reshape the (B·P/1024, 1024) result back to (B, P) — page-growing
//      reshape, CB ≤ 1 MB, always within L1.
//
// b_safe derivation (CB ≤ 1 MB):
//   fp32: b_safe = 1 MB / (4 × 2 × 1024 × 4 B) = 32 rows
//   bf16: b_safe = 1 MB / (4 × 2 × 1024 × 2 B) = 64 rows
//
// Case B (P not divisible by 1024): pad to next multiple of 1024 first,
// apply the above, then slice back to (B, P).
static std::tuple<ttnn::Tensor, ttnn::Tensor> complex_mul_chunked(
    const ttnn::Tensor& ar, const ttnn::Tensor& ai,
    const ttnn::Tensor& br, const ttnn::Tensor& bi,
    uint32_t B, uint32_t P_col)
{
    // P_col must be a multiple of 1024 on entry.
    const uint32_t nchunks   = P_col / 1024u;
    const uint32_t total_rows = B * nchunks;

    const uint32_t elem_bytes =
        (ar.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    // Max rows per complex_mul to keep CB ≤ 1 MB.
    const uint32_t b_safe = (1u << 20u) / (8u * 1024u * elem_bytes); // 32/64

    // Rebank (B, P_col) → (total_rows, 1024) with tiny CB.
    auto ar_f = shrink_reshape(ar, 1024u);
    auto ai_f = shrink_reshape(ai, 1024u);
    auto br_f = shrink_reshape(br, 1024u);
    auto bi_f = shrink_reshape(bi, 1024u);

    const auto mc = ar.memory_config();

    if (total_rows <= b_safe) {
        // Small enough: one complex_mul call.
        auto [cr, ci] = complex_mul(ar_f, ai_f, br_f, bi_f);
        const auto orig = ttnn::Shape{ttnn::SmallVector<uint32_t>{B, P_col}};
        return {ttnn::reshape(cr, orig), ttnn::reshape(ci, orig)};
    }

    // Large: loop in b_safe-row blocks, collect results, concat.
    std::vector<ttnn::Tensor> cr_vec, ci_vec;
    cr_vec.reserve((total_rows + b_safe - 1u) / b_safe);
    ci_vec.reserve(cr_vec.capacity());

    for (uint32_t start = 0u; start < total_rows; start += b_safe) {
        const uint32_t end_r = std::min(start + b_safe, total_rows);
        const ttnn::SmallVector<uint32_t> beg_idx  = {start, 0u};
        const ttnn::SmallVector<uint32_t> end_idx  = {end_r, 1024u};
        const ttnn::SmallVector<uint32_t> step_idx = {1u, 1u};
        auto arc = ttnn::slice(ar_f, beg_idx, end_idx, step_idx, mc);
        auto aic = ttnn::slice(ai_f, beg_idx, end_idx, step_idx, mc);
        auto brc = ttnn::slice(br_f, beg_idx, end_idx, step_idx, mc);
        auto bic = ttnn::slice(bi_f, beg_idx, end_idx, step_idx, mc);
        auto [crc, cic] = complex_mul(arc, aic, brc, bic);
        cr_vec.push_back(std::move(crc));
        ci_vec.push_back(std::move(cic));
    }

    // Concat along dim 0: (k × b_safe, 1024) → (total_rows, 1024).
    auto cr_f = ttnn::concat(cr_vec, /*dim=*/0);
    auto ci_f = ttnn::concat(ci_vec, /*dim=*/0);

    // Reshape back (page-growing, CB ≤ 1 MB).
    const auto orig = ttnn::Shape{ttnn::SmallVector<uint32_t>{B, P_col}};
    return {ttnn::reshape(cr_f, orig), ttnn::reshape(ci_f, orig)};
}

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

    const auto mc = ar.memory_config();

    if (P % 1024u == 0u) {
        return complex_mul_chunked(ar, ai, br, bi, B, P);
    }

    // Case B: P not divisible by 1024 — pad to P_pad, multiply, slice back.
    const uint32_t P_pad = (P + 1023u) & ~1023u;
    const uint32_t pad_len = P_pad - P;

    const ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
        {{0u, 0u}},
        {{0u, pad_len}},
    };
    auto ar_p = ttnn::pad(ar, padding, 0.0f, /*use_multicore=*/true, mc);
    auto ai_p = ttnn::pad(ai, padding, 0.0f, /*use_multicore=*/true, mc);
    auto br_p = ttnn::pad(br, padding, 0.0f, /*use_multicore=*/true, mc);
    auto bi_p = ttnn::pad(bi, padding, 0.0f, /*use_multicore=*/true, mc);

    auto [cr_p, ci_p] = complex_mul_chunked(ar_p, ai_p, br_p, bi_p, B, P_pad);

    // Slice result back to (B, P) — zero-padded positions are 0 × anything = 0.
    const ttnn::SmallVector<uint32_t> begins = {0u, 0u};
    const ttnn::SmallVector<uint32_t> ends   = {B,  P};
    const ttnn::SmallVector<uint32_t> step   = {1u, 1u};
    auto cr = ttnn::slice(cr_p, begins, ends, step, mc);
    auto ci = ttnn::slice(ci_p, begins, ends, step, mc);
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
