// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/fft.hpp"

#include <cstdlib>
#include <optional>
#include <tuple>
#include <utility>

#include "device/fft_device_operation.hpp"
#include "device/fft_radix_pass_device_operation.hpp"
#include "device/transpose_rm_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"  // ttnn::Shape, ttnn::SmallVector

namespace ttnn::operations::experimental {

namespace {

// ───────────────────────────────────────────────────────────────────────
// Two-pass Cooley–Tukey composite (commit 3c)
//
// For pow-2 N with 1024 < N ≤ 1M, factor N = N1 * N2 (both pow-2, both
// in [32, 1024]) and decompose the length-N DFT as:
//
//   X[k1·N2 + k2] = Σ_{n1} W_N1^(n1·k1) · ω^(n1·k2) · ( Σ_{n2} x[n1,n2]·W_N2^(n2·k2) )
//                    ╰── Pass-2 ──╯  ╰─ twiddle ─╯   ╰─────── Pass-1 ────────╯
//
// where ω = exp(-2πi / N).
//
// Implementation as a chain of FIVE device ops (commit 4 fused Pass-1
// + between-pass twiddle into ONE dispatch via prim::fft_radix_pass):
//   1. reshape (B, N) -> (B*N1, N2)               [metadata-only, free]
//   2. fft_radix_pass(P=N2, twiddle_N2=N1)        → (R2, I2) shape (B*N1, N2)
//                                                  fused [batched FFT
//                                                  + post-twiddle cmul]
//   3. reshape + transpose_rm + reshape           → (R3, I3) shape (B*N2, N1)
//   4. Pass-2 batched length-N1 complex FFT       → (R4, I4) shape (B*N2, N1)
//   5. reshape + transpose_rm + reshape           → final (B, N) tensors
//
// Reduces the per-call dispatch count from 6 to 5 in the host path
// (Step 2 used to be prim::fft followed by prim::apply_twiddles).
//
// Gated by TT_FFT_NATIVE=1 like the rest of the new path.  Falls back to
// the legacy CachedProgram path otherwise.
// ───────────────────────────────────────────────────────────────────────

bool native_path_enabled() {
    const char* v = std::getenv("TT_FFT_NATIVE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

constexpr bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

// Balanced pow-2 factorization: N2 = 2^(log2N/2), N1 = N/N2.
// For our gated range (1024 < N ≤ 1M pow-2) both factors land in [32, 1024].
std::pair<uint32_t, uint32_t> pick_factorization(uint32_t N) {
    uint32_t log2N = 0u;
    while ((1u << log2N) < N) ++log2N;
    const uint32_t log2N2 = log2N / 2u;
    const uint32_t log2N1 = log2N - log2N2;
    return {1u << log2N1, 1u << log2N2};
}

ttnn::Shape make_shape(std::initializer_list<uint32_t> dims) {
    ttnn::SmallVector<uint32_t> v;
    v.reserve(dims.size());
    for (auto d : dims) v.push_back(d);
    return ttnn::Shape{v};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> fft_two_pass(
    const ttnn::Tensor& input_real, FFTPrecision precision) {
    const auto& in_shape = input_real.padded_shape();
    const uint32_t N = static_cast<uint32_t>(in_shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(in_shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(in_shape[d]);
    }

    const auto [N1, N2] = pick_factorization(N);

    // ── Step 1: reshape input  (B, N)  →  (B*N1, N2)  (metadata-only)
    auto x_p1 = ttnn::reshape(input_real, make_shape({B * N1, N2}));

    // ── Step 2: Pass-1 batched length-N2 real FFT + between-pass
    //   twiddle multiply, FUSED into a single device dispatch.
    //
    //   fft_radix_pass treats input as (M, P) and outputs
    //       y[r, k] = FFT_P(x[r, :])[k]
    //               * exp(-2πi · (r % twiddle_N2) · k / (P · twiddle_N2))
    //
    //   We want the row-index r to range over [0, B·N1) and the post-
    //   twiddle modulus to be N1, so the twiddle T[n1, k2] =
    //   exp(-2πi·n1·k2 / (N1·N2)) is broadcast correctly across
    //   the B replicas.  Hence: P = N2, twiddle_N2 = N1.
    //
    //   Replaces the old (prim::fft → prim::apply_twiddles) pair —
    //   1 dispatch instead of 2, no intermediate L1↔DRAM round-trip.
    auto [r2, i2] = ttnn::prim::fft_radix_pass(
        x_p1, /*input_imag=*/std::nullopt,
        /*P=*/N2, /*twiddle_N2=*/N1);

    // ── Step 3: transpose (B*N1, N2) → (B*N2, N1) via (B, N1, N2) view.
    auto r2_3d = ttnn::reshape(r2, make_shape({B, N1, N2}));
    auto i2_3d = ttnn::reshape(i2, make_shape({B, N1, N2}));
    auto r3_3d = ttnn::prim::transpose_rm(r2_3d);
    auto i3_3d = ttnn::prim::transpose_rm(i2_3d);
    auto r3 = ttnn::reshape(r3_3d, make_shape({B * N2, N1}));
    auto i3 = ttnn::reshape(i3_3d, make_shape({B * N2, N1}));

    // ── Step 4: Pass-2 batched length-N1 complex FFT.
    auto [r4, i4] = ttnn::prim::fft(
        r3, /*inverse=*/false, /*input_imag=*/i3, precision);

    // ── Step 5: undo the row/col flip to restore natural ordering.
    auto r4_3d = ttnn::reshape(r4, make_shape({B, N2, N1}));
    auto i4_3d = ttnn::reshape(i4, make_shape({B, N2, N1}));
    auto r5_3d = ttnn::prim::transpose_rm(r4_3d);
    auto i5_3d = ttnn::prim::transpose_rm(i4_3d);

    // ── Final reshape back to the caller-visible (..., N) shape.
    auto out_r = ttnn::reshape(r5_3d, in_shape);
    auto out_i = ttnn::reshape(i5_3d, in_shape);
    return {std::move(out_r), std::move(out_i)};
}

bool two_pass_eligible(const ttnn::Tensor& input_real) {
    if (!native_path_enabled()) return false;
    const auto& shape = input_real.padded_shape();
    if (shape.size() < 1) return false;
    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }
    const auto dt = input_real.dtype();
    const bool dtype_ok =
        dt == tt::tt_metal::DataType::FLOAT32 ||
        dt == tt::tt_metal::DataType::BFLOAT16;
    const bool layout_ok =
        input_real.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    return dtype_ok && layout_ok &&
           is_pow2(N) && N > 1024u && N <= (1u << 20) &&
           is_pow2(B) && B >= 1u;
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real, FFTPrecision precision) {
    if (two_pass_eligible(input_real)) {
        return fft_two_pass(input_real, precision);
    }
    return ttnn::prim::fft(input_real, /*inverse=*/false,
                           /*input_imag=*/std::nullopt, precision);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    FFTPrecision precision) {
    return ttnn::prim::fft(input_real, /*inverse=*/false, input_imag, precision);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ifft(
    const ttnn::Tensor& spectrum_real,
    const ttnn::Tensor& spectrum_imag,
    FFTPrecision precision) {
    return ttnn::prim::fft(spectrum_real, /*inverse=*/true,
                           spectrum_imag, precision);
}

}  // namespace ttnn::operations::experimental
