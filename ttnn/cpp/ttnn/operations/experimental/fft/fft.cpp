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
#include "device/apply_twiddles_xl_device_operation.hpp"
#include "device/transpose_rm_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"  // ttnn::Shape, ttnn::SmallVector

namespace ttnn::operations::experimental {

namespace {

// ───────────────────────────────────────────────────────────────────────
// Two-pass Cooley–Tukey composite (commit 3c, corrected commit 5c)
//
// For pow-2 N with 1024 < N ≤ 1M, factor N = N1 · N2 (both pow-2, both
// in [32, 1024]).  We use the standard mixed-radix DIT decomposition
// that — for natural-order input AND natural-order output — requires
// pre- and post-transposes so each pass FFTs along the LAST axis.
//
// Index packing (input n natural, output K natural):
//   n = n1·N2 + n2   (n1 OUTER, n2 INNER of the (B, N) row)
//   K = k2·N1 + k1   (k1 INNER, k2 OUTER  of the (B, N) row)
//
// With this packing every (n_i, k_j) cross-term in n·K/N is either
// integer (vanishes) or matches a clean FFT/twiddle factor:
//
//   X[k2·N1 + k1] = Σ_{n2} W_{N2}^{n2·k2} · ( exp(-2πi·n2·k1/N) ·
//                       Σ_{n1} W_{N1}^{n1·k1} · x[n1, n2] )
//                     ╰── Pass-2 ──╯ ╰─ twiddle ─╯  ╰── Pass-1 ──╯
//
// Implementation chain (3 transposes + 2 fft_radix_pass dispatches):
//   1. reshape (B, N) → (B, N1, N2)              [view, free]
//   2. transpose_rm   → (B, N2, N1)              [data movement]
//   3. view as (B·N2, N1)                        [view]
//   4. fft_radix_pass(P=N1, twiddle_N2=N2)       [Pass-1 FFT_N1 fused with
//                                                 twiddle exp(-2πi·n2·k1/N)]
//   5. view + transpose_rm + view → (B·N1, N2)   [data movement]
//   6. fft_radix_pass(P=N2, twiddle_N2=0)        [Pass-2 pure FFT_N2]
//   7. view + transpose_rm → (B, N2, N1)         [data movement]
//   8. reshape → (B, N)                          [view, free]
//
// NOTE: the EARLIER version of fft_two_pass (commit 4) did the inner
// FFT first WITHOUT the initial transpose and applied the twiddle on
// pass 2 with arguments (P=N2, twiddle_N2=N1).  That doesn't correspond
// to any valid Cooley–Tukey decomposition with natural I/O; on N=4 it
// produced [10, -1+i, -4, -1-i] instead of the correct
// [10, -2+2i, -2, -2-2i].  The diagnostic at N=2048/4096 showed
// rel_err ≈ √2 (output uncorrelated with reference) — symptom of an
// algorithm that sums elements correctly (DC bin matched) but applies
// the wrong twiddles everywhere else.
//
// Gated by TT_FFT_NATIVE=1.  Falls back to legacy CachedProgram path
// (prim::fft) when not enabled.
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

    // ── Step 1: reshape input (B, N) → (B, N1, N2).  View, free.
    //   x_3d[b, n1, n2] = x_orig[b, n1·N2 + n2].
    auto x_3d = ttnn::reshape(input_real, make_shape({B, N1, N2}));

    // ── Step 2: initial transpose (B, N1, N2) → (B, N2, N1).
    //   So that Pass-1 (FFT_N1) sees stride-N1 sub-samples as
    //   contiguous rows.  This is the bit-reversal-equivalent step
    //   that the earlier (commit 4) version was missing.
    auto x_t = ttnn::prim::transpose_rm(x_3d);
    auto x_p1 = ttnn::reshape(x_t, make_shape({B * N2, N1}));

    // ── Step 3: Pass-1 batched length-N1 real FFT + between-pass
    //   twiddle.  Row r = b·N2 + n2, so (r % twiddle_N2=N2) = n2:
    //       post-twiddle = exp(-2πi · n2 · k1 / (N1·N2))
    //                    = exp(-2πi · n2 · k1 / N)        ← Cooley–Tukey twiddle
    auto [r1, i1] = ttnn::prim::fft_radix_pass(
        x_p1, /*input_imag=*/std::nullopt,
        /*P=*/N1, /*twiddle_N2=*/N2);

    // ── Step 4: transpose (B, N2, N1) → (B, N1, N2) to bring n2 to
    //   the last axis ready for Pass-2.
    auto r1_3d = ttnn::reshape(r1, make_shape({B, N2, N1}));
    auto i1_3d = ttnn::reshape(i1, make_shape({B, N2, N1}));
    auto r2t = ttnn::prim::transpose_rm(r1_3d);
    auto i2t = ttnn::prim::transpose_rm(i1_3d);
    auto r2 = ttnn::reshape(r2t, make_shape({B * N1, N2}));
    auto i2 = ttnn::reshape(i2t, make_shape({B * N1, N2}));

    // ── Step 5: Pass-2 batched length-N2 complex FFT, NO twiddle
    //   (all Cooley–Tukey twiddles were absorbed into Pass-1 above).
    auto [r3, i3] = ttnn::prim::fft_radix_pass(
        r2, /*input_imag=*/i2,
        /*P=*/N2, /*twiddle_N2=*/0u);

    // ── Step 6: final transpose (B, N1, N2) → (B, N2, N1) to put
    //   the output in natural-K order under flat reshape.
    //   Recall: algorithm produces X[K = k2·N1 + k1] at position
    //   (b, k1, k2) of the (B, N1, N2) post-Pass-2 tensor.  After this
    //   transpose, element (b, k2, k1) lives at flat (b·N + k2·N1 + k1)
    //   = (b·N + K), i.e. natural K-ordered output.
    auto r3_3d = ttnn::reshape(r3, make_shape({B, N1, N2}));
    auto i3_3d = ttnn::reshape(i3, make_shape({B, N1, N2}));
    auto r4t = ttnn::prim::transpose_rm(r3_3d);
    auto i4t = ttnn::prim::transpose_rm(i3_3d);

    auto out_r = ttnn::reshape(r4t, in_shape);
    auto out_i = ttnn::reshape(i4t, in_shape);
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

// ───────────────────────────────────────────────────────────────────────
// Three-pass Cooley–Tukey composite (commit 5)
//
// ⚠ API NOTE: fft_three_pass takes its input ALREADY PRE-SHAPED as
//   (B·N1·N2, N3) [i.e. with last dim = N3 ≤ 1024], NOT as (B, N).
//   This is because the (B, N) → (B·N1·N2, N3) reshape requires
//   moving an N-element row through a single CB tile per core, which
//   blows L1 for N > ~256K.  The caller is expected to do the equivalent
//   `torch.view(B·N1·N2, N3)` on the host (it's a metadata-only torch
//   view) BEFORE `ttnn.from_torch`, so the device buffer is allocated
//   with small page_size from the start.  Output is returned in the
//   factored shape (B·N1, N2, N3) — caller can reshape on host to
//   recover (B, N) since the data is in natural Cooley–Tukey order.
//
//   TODO (commit 7): write an L1-friendly DRAM→DRAM rebank kernel that
//   handles the page-size change in chunks, so the public `fft()` API
//   can transparently route (B, N) inputs into the three-pass composite.
//
// For pow-2 N with 2^20 < N ≤ 2^30, factor N = N1 · N2 · N3 (each pow-2
// in [32, 1024], picked "max-N3 then balance N1/N2" for best memory
// coalescing on the innermost pass).  Decompose the length-N DFT as:
//
//   X[k1·N2·N3 + k2·N3 + k3]
//     = Σ_n1 W_N1^(n1·k1) ·
//           W_{N1N2}^(n1·k2) ·
//             Σ_n2 W_N2^(n2·k2) ·
//                W_N^((n1·N2+n2)·k3) ·
//                  Σ_n3 x[n1,n2,n3] · W_N3^(n3·k3)
//
// where ω = exp(-2πi / ·).  Implementation chain (12 device dispatches):
//
//   1. reshape (B, N) → (B·N1·N2, N3)          [zero-cost]
//   2. fft_radix_pass(P=N3, twiddle_N2=0)      [pass-1, no twiddle]
//   3. apply_twiddles_xl(P=N3, big_mod=N1·N2)  [twiddle-1, large modulus]
//   4. reshape (B·N1·N2, N3) → (B·N1, N2, N3)  [zero-cost]
//   5. transpose_rm: (B·N1, N2, N3) → (B·N1, N3, N2)         [×2 r,i]
//   6. reshape → (B·N1·N3, N2)                 [zero-cost]
//   7. fft_radix_pass(P=N2, twiddle_N2=N1, stride=N3)  [pass-2 + twiddle-2]
//   8. reshape → (B, N1, N3*N2) view of (B, N1, N3, N2)  [zero-cost]
//   9. transpose_rm: (B, N1, N3·N2) → (B, N3·N2, N1)         [×2 r,i]
//  10. reshape → (B·N3·N2, N1)                 [zero-cost]
//  11. fft(P=N1)                               [pass-3, complex input]
//  12. reshape + two transpose_rm pairs to undo k-axis reversal
//      (B, N3, N2, N1) → (B, N1, N2, N3); final reshape → (B, N).
//
// Above 2^30, we'd need a 4-pass or Bluestein composite (commit 6).
// ───────────────────────────────────────────────────────────────────────

// Max-N3 then balanced N1/N2 split.  Both N1, N2, N3 ∈ [32, 1024], pow-2.
//   N3 = min(1024, N / 32^2)   ← cap by tile-size on innermost
//   then split remaining log2 between N1 and N2 (N1 gets ceil-half).
std::tuple<uint32_t, uint32_t, uint32_t> pick_three_factorization(uint32_t N) {
    uint32_t log2N = 0u;
    while ((1u << log2N) < N) ++log2N;
    TT_FATAL((1u << log2N) == N,
        "fft_three_pass: N must be a power of two (got {}).", N);
    TT_FATAL(log2N >= 15u && log2N <= 30u,
        "fft_three_pass: N must be in [2^15, 2^30] (got 2^{}).", log2N);

    // Cap log2(N3) at 10 (= 1024, the per-row FFT length limit); also
    // leave ≥ 10 bits for N1+N2 split (= 32 · 32 minimum).
    uint32_t log2_N3 = 10u;
    if (log2N - log2_N3 < 10u) {
        // Pathological tiny case (log2N < 20).  Shouldn't happen since
        // routing kicks in at log2N > 20, but be safe.
        log2_N3 = (log2N >= 10u) ? (log2N - 10u) : 5u;
    }
    const uint32_t log2_rest = log2N - log2_N3;
    const uint32_t log2_N1 = (log2_rest + 1u) / 2u;   // ceil half → N1
    const uint32_t log2_N2 = log2_rest - log2_N1;
    TT_FATAL(log2_N1 >= 5u && log2_N1 <= 10u &&
             log2_N2 >= 5u && log2_N2 <= 10u &&
             log2_N3 >= 5u && log2_N3 <= 10u,
        "fft_three_pass: N=2^{} factorization N1=2^{} N2=2^{} N3=2^{} "
        "out of supported [32, 1024] range.",
        log2N, log2_N1, log2_N2, log2_N3);
    return {1u << log2_N1, 1u << log2_N2, 1u << log2_N3};
}

}  // namespace

// ────────────────────────────────────────────────────────────────────
// Public entrypoint — caller-visible.  Input is REQUIRED to be pre-
// shaped as (B·N1·N2, N3) [last dim = N3 ≤ 1024]; the (B, N) → factored
// reshape would otherwise blow L1 (commit 7 will add a rebank kernel
// to lift this restriction).  Output is returned in the factored shape
// (B·N1, N2, N3) — caller can `to_torch().reshape(B, N)` to recover
// natural-order (B, N) on host.
// ────────────────────────────────────────────────────────────────────
std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass(
    const ttnn::Tensor& input_real,
    uint32_t full_N,
    FFTPrecision precision) {
    const auto& in_shape = input_real.padded_shape();
    TT_FATAL(in_shape.size() >= 2,
        "fft_three_pass: pre-shaped input must be ≥2-D, e.g. (M, N3). Got {}-D.",
        in_shape.size());
    const uint32_t P_in = static_cast<uint32_t>(in_shape[-1]);
    const uint32_t M_in = static_cast<uint32_t>(in_shape[-2]);
    // Allow extra leading dims as implicit batch (rare; tests use 2-D inputs).
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(in_shape.size()) - 2; ++d) {
        B *= static_cast<uint32_t>(in_shape[d]);
    }

    const auto [N1, N2, N3] = pick_three_factorization(full_N);
    TT_FATAL(P_in == N3 && M_in == N1 * N2,
        "fft_three_pass: pre-shaped input mismatch — expected (..., M=N1·N2={}, "
        "P=N3={}) for full_N={} (N1={}, N2={}, N3={}), got (..., {}, {}).",
        N1 * N2, N3, full_N, N1, N2, N3, M_in, P_in);

    // ── Step 2: Pass-1 batched length-N3 real FFT, NO twiddle.
    //   fft_radix_pass treats input as (M=B·N1·N2 rows, N3 cols) and
    //   emits FFT_N3 of each row.  twiddle_N2=0 → pure FFT path.
    //   (Input is already in this shape — no reshape needed.)
    auto [r1, i1] = ttnn::prim::fft_radix_pass(
        input_real, /*input_imag=*/std::nullopt,
        /*P=*/N3, /*twiddle_N2=*/0u);

    // ── Step 3: between-pass-1-and-2 twiddle (LARGE modulus N1·N2).
    //   Multiplies y[r=b·N1·N2 + n1·N2 + n2, k3] by
    //       exp(-2πi · (r % (N1·N2)) · k3 / full_N)
    //     = exp(-2πi · (n1·N2 + n2) · k3 / full_N)              (for B=1)
    //   which is Cooley–Tukey twiddle 1 broadcast over B replicas.
    auto [r2, i2] = ttnn::prim::apply_twiddles_xl(
        r1, i1, /*P=*/N3, /*big_modulus=*/N1 * N2, /*full_N=*/full_N);

    // ── Steps 4-6: bring n2 to last axis, ready for Pass-2.
    //   (B·N1·N2, N3) → (B·N1, N2, N3) → transpose → (B·N1, N3, N2)
    //   → (B·N1·N3, N2).
    auto r2_3d = ttnn::reshape(r2, make_shape({B * N1, N2, N3}));
    auto i2_3d = ttnn::reshape(i2, make_shape({B * N1, N2, N3}));
    auto r2t   = ttnn::prim::transpose_rm(r2_3d);
    auto i2t   = ttnn::prim::transpose_rm(i2_3d);
    auto r2f   = ttnn::reshape(r2t, make_shape({B * N1 * N3, N2}));
    auto i2f   = ttnn::reshape(i2t, make_shape({B * N1 * N3, N2}));

    // ── Step 7: Pass-2 batched length-N2 complex FFT  +  small twiddle.
    //   Rows enumerate (b, n1, k3) at stride N3 along the n1 axis, so
    //   (r / stride=N3) % twiddle_N2=N1 picks the right n1 twiddle row
    //   without needing an extra transpose.  Twiddle factor:
    //       exp(-2πi · n1 · k2 / (N1·N2))   = Cooley–Tukey twiddle 2.
    auto [r3, i3] = ttnn::prim::fft_radix_pass(
        r2f, /*input_imag=*/i2f,
        /*P=*/N2, /*twiddle_N2=*/N1, /*stride=*/N3);

    // ── Steps 8-10: bring n1 to last axis, ready for Pass-3.
    //   Layout after Step 7: (B·N1·N3, N2).  View as (B, N1, N3, N2),
    //   collapse (N3, N2) → flat dim of length N3·N2, then transpose
    //   (B, N1, N3·N2) → (B, N3·N2, N1), reshape → (B·N3·N2, N1).
    auto r3_3d = ttnn::reshape(r3, make_shape({B, N1, N3 * N2}));
    auto i3_3d = ttnn::reshape(i3, make_shape({B, N1, N3 * N2}));
    auto r3t   = ttnn::prim::transpose_rm(r3_3d);   // (B, N3·N2, N1)
    auto i3t   = ttnn::prim::transpose_rm(i3_3d);
    auto r3f   = ttnn::reshape(r3t, make_shape({B * N3 * N2, N1}));
    auto i3f   = ttnn::reshape(i3t, make_shape({B * N3 * N2, N1}));

    // ── Step 11: Pass-3 batched length-N1 complex FFT, no twiddle.
    auto [r4, i4] = ttnn::prim::fft(
        r3f, /*inverse=*/false, /*input_imag=*/i3f, precision);

    // ── Step 12: undo the k-axis reversal (k3, k2, k1) → (k1, k2, k3)
    //   so the final flatten naturally gives the DIF-natural index
    //   k = k1·N2·N3 + k2·N3 + k3.  Two transposes via the same
    //   collapse-then-swap trick as Step 9.
    //
    //   T1: view (B, N3, N2, N1) as (B, N3·N2, N1) → transpose
    //       → (B, N1, N3·N2) → reshape (B, N1, N3, N2).
    auto r4_3d  = ttnn::reshape(r4, make_shape({B, N3 * N2, N1}));
    auto i4_3d  = ttnn::reshape(i4, make_shape({B, N3 * N2, N1}));
    auto r4t1   = ttnn::prim::transpose_rm(r4_3d);   // (B, N1, N3·N2)
    auto i4t1   = ttnn::prim::transpose_rm(i4_3d);
    //   T2: reshape (B, N1, N3·N2) → (B·N1, N3, N2) → transpose
    //       → (B·N1, N2, N3) → reshape (B, N1, N2, N3).
    auto r4t1f  = ttnn::reshape(r4t1, make_shape({B * N1, N3, N2}));
    auto i4t1f  = ttnn::reshape(i4t1, make_shape({B * N1, N3, N2}));
    auto r4t2   = ttnn::prim::transpose_rm(r4t1f);
    auto i4t2   = ttnn::prim::transpose_rm(i4t1f);

    // NOTE: NO final reshape back to (B, N).  That would require
    // an L1-busting page_size change (N3·elem → full_N·elem).  Output
    // shape is the factored (B·N1, N2, N3) — caller is expected to do
    // `to_torch().reshape(B, full_N)` on host to recover natural-order
    // (B, full_N).  This is cheap (just a torch view) since the FFT
    // chain already arranges k = k1·N2·N3 + k2·N3 + k3 naturally.
    return {std::move(r4t2), std::move(i4t2)};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real, FFTPrecision precision) {
    // Routing (TT_FFT_NATIVE=1):
    //   N ≤ 1024            → SingleTileStockhamFactory  (via prim::fft)
    //   1024 < N ≤ 2^20     → fft_two_pass               (commit 3/4)
    //   N > 2^20            → caller MUST invoke fft_three_pass
    //                         explicitly with pre-shaped input
    //                         (until commit 7 adds rebank kernel)
    //   N > 2^30            → falls through to prim::fft (will TT_FATAL)
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
