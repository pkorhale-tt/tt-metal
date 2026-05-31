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
    // precision is currently unused — both passes route through
    // prim::fft_radix_pass which keeps its compute precision implicit
    // (fp32 input → fp32 compute, bf16 input → packed bf16).  Kept in
    // the signature for API symmetry with fft_three_pass and for the
    // future tile-quant lowering knob (commit 6+).
    (void)precision;
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
// Three-pass Cooley–Tukey composite (commit 5, corrected commit 5c)
//
// ⚠ API NOTES:
//   (1) INPUT pre-shape: fft_three_pass takes its input ALREADY PRE-SHAPED
//       as (B·N1·N2, N3) [last dim = N3 ≤ 1024], NOT as (B, N).  This is
//       because the (B, N) → (B·N1·N2, N3) reshape requires moving an
//       N-element row through a CB per core, blowing L1 for N > ~256K.
//       Caller does `torch.view(B·N1·N2, N3)` on host (metadata-only)
//       BEFORE `ttnn.from_torch` so the device buffer is allocated with
//       small page_size from the start.
//   (2) OUTPUT shape change (commit 5c): output is now (B·N3, N2, N1)
//       instead of (B·N1, N2, N3).  Caller's `to_torch().reshape(B, N)`
//       on host still gives natural-order X[k] because the (N3, N2, N1)
//       dim layout encodes K = k3·N1·N2 + k2·N1 + k1 = natural flat K.
//       The OLD shape was returning index-permuted (wrong-order) data
//       due to the underlying algorithmic bug (see ALGORITHM section).
//
//   TODO (commit 7): write an L1-friendly DRAM→DRAM rebank kernel that
//   handles the page-size change in chunks, so the public `fft()` API
//   can transparently route (B, N) inputs into the three-pass composite.
//
// ── ALGORITHM ───────────────────────────────────────────────────────────
// For pow-2 N with 2^20 < N ≤ 2^30, factor N = N1 · N2 · N3 (each pow-2
// in [32, 1024]).  We use the standard mixed-radix DIT decomposition.
//
//   Input packing  : n = n1·N2·N3 + n2·N3 + n3   (n1 OUTER, n3 INNER)
//   Output packing : K = k3·N1·N2 + k2·N1 + k1   (k1 INNER, k3 OUTER)
//
// Crucially, K is the REVERSED-digit packing (k_i factor positions
// swapped relative to n_i).  Empirically (and provably) the natural-K
// packing K = k1·N2·N3 + k2·N3 + k3 has a non-integer phase term
// (n2·k2·N3/(N1·N2) is fractional when N3/(N1·N2) is not integer), so
// it does NOT admit a clean Cooley-Tukey decomposition for asymmetric
// (N1, N2, N3).  The reversed packing makes every cross-term integer-
// vanishing or assignable to a clean FFT/twiddle factor:
//
//   n·K/N  ≡  n1·k1/N1
//          + n2·k2/N2  +  n2·k1/(N1·N2)
//          + n3·k3/N3  +  n3·k2/(N2·N3)  +  n3·k1/N        (mod 1)
//
// Three FFT stages with assignable twiddles:
//
//   Stage 1: FFT_N1 over n1 (→ k1).
//   Twiddle-1 (post Stage 1): exp(-2πi · (n2·N3 + n3) · k1 / N)
//     fuses the n2·k1/(N1·N2) and n3·k1/N cross-terms.
//   Stage 2: FFT_N2 over n2 (→ k2).
//   Twiddle-2 (post Stage 2, fused into Stage 2's post-twiddle):
//     exp(-2πi · n3 · k2 / (N2·N3))
//   Stage 3: FFT_N3 over n3 (→ k3).
//
// Output naturally lands at (B, k1, k2, k3) after Stage 3.  A final
// transpose chain reverses the last 3 dims → (B, N3, N2, N1) so that
// `.reshape(B, N)` on host yields X[K] at flat K.
//
// ── DISPATCH CHAIN (8 device ops) ──────────────────────────────────────
//
//   Initial rearrangement (input is (B·N1·N2, N3) with n1 OUTER):
//     1. reshape (B·N1·N2, N3) → (B, N1, N2·N3)        [page: N3·elem
//                                                        → N2·N3·elem]
//     2. transpose_rm        → (B, N2·N3, N1)          [×2 r,i]
//     3. reshape             → (B·N2·N3, N1)           [free]
//
//   Stage 1 + Twiddle-1:
//     4. fft_radix_pass(P=N1, twiddle_N2=0)            [pure FFT_N1]
//     5. apply_twiddles_xl(P=N1, big_mod=N2·N3,
//                          full_N=N)                   [twiddle-1]
//
//   Bring n2 to inner (was at position 1 of (B, N2, N3, k1)):
//     6. reshape → (B, N2, N3·N1)                      [free]
//     7. transpose_rm → (B, N3·N1, N2)                 [×2 r,i]
//     8. reshape → (B·N3·N1, N2)                       [free]
//
//   Stage 2 + Twiddle-2 (FUSED in fft_radix_pass post-twiddle):
//     9. fft_radix_pass(P=N2, twiddle_N2=N3,
//                       stride=N1)                     [FFT_N2 + tw-2]
//
//   Bring n3 to inner (was at position 1 of (B, N3, k1, k2)):
//    10. reshape → (B, N3, N1·N2)                      [free]
//    11. transpose_rm → (B, N1·N2, N3)                 [×2 r,i]
//    12. reshape → (B·N1·N2, N3)                       [free]
//
//   Stage 3 (no twiddle):
//    13. fft_radix_pass(P=N3, twiddle_N2=0)            [pure FFT_N3]
//
//   Final dim-reverse to natural-K order:
//    14. reshape → (B, N1·N2, N3)                      [free]
//    15. transpose_rm → (B, N3, N1·N2)                 [×2 r,i]
//    16. reshape → (B, N3, N1, N2)                     [page change]
//    17. transpose_rm → (B, N3, N2, N1)                [×2 r,i, small]
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
// (B·N3, N2, N1) — caller does `to_torch().reshape(B, N)` on host to
// recover natural-order X[k] (the (N3, N2, N1) dim order encodes
// K = k3·N1·N2 + k2·N1 + k1, which IS the natural flat K).
// ────────────────────────────────────────────────────────────────────
std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass(
    const ttnn::Tensor& input_real,
    uint32_t full_N,
    FFTPrecision precision) {
    (void)precision;  // see fft_two_pass note.
    const auto& in_shape = input_real.padded_shape();
    TT_FATAL(in_shape.size() >= 2,
        "fft_three_pass: pre-shaped input must be ≥2-D, e.g. (M, N3). Got {}-D.",
        in_shape.size());
    const uint32_t P_in = static_cast<uint32_t>(in_shape[-1]);
    // Sum total rows = product of all dims except last.  Caller may pass
    // either flat 2-D (B·N1·N2, N3) or N-D (..., N1·N2, N3) — we treat
    // everything except the last dim as a contiguous row-stream of length
    // B·N1·N2 and derive B from there.
    uint32_t M_in = 1u;
    for (int d = 0; d < static_cast<int>(in_shape.size()) - 1; ++d) {
        M_in *= static_cast<uint32_t>(in_shape[d]);
    }

    const auto [N1, N2, N3] = pick_three_factorization(full_N);
    TT_FATAL(P_in == N3,
        "fft_three_pass: pre-shaped input last dim must be N3={} for full_N={} "
        "(N1={}, N2={}, N3={}); got {}.",
        N3, full_N, N1, N2, N3, P_in);
    TT_FATAL(M_in % (N1 * N2) == 0u,
        "fft_three_pass: total rows {} must be a multiple of N1·N2={} for "
        "full_N={} (N1={}, N2={}, N3={}).",
        M_in, N1 * N2, full_N, N1, N2, N3);
    const uint32_t B = M_in / (N1 * N2);

    // ── Initial rearrangement (input n1 OUTER → n1 to LAST axis).
    //   Input (B·N1·N2, N3) is row-major with n = n1·N2·N3 + n2·N3 + n3.
    //   View as (B, N1, N2·N3) [merges (N2, N3) → page = N2·N3·elem,
    //   page-changing reshape], then transpose_rm → (B, N2·N3, N1).
    auto x_3d = ttnn::reshape(input_real, make_shape({B, N1, N2 * N3}));
    auto x_t  = ttnn::prim::transpose_rm(x_3d);                       // (B, N2·N3, N1)
    auto x_p1 = ttnn::reshape(x_t, make_shape({B * N2 * N3, N1}));

    // ── Stage 1: pure FFT_N1 over the (now-inner) n1 axis.
    auto [r1, i1] = ttnn::prim::fft_radix_pass(
        x_p1, /*input_imag=*/std::nullopt,
        /*P=*/N1, /*twiddle_N2=*/0u);

    // ── Twiddle-1: exp(-2πi · (n2·N3 + n3) · k1 / N).
    //   Row r = b·N2·N3 + n2·N3 + n3, so (r % (N2·N3)) = n2·N3 + n3.
    //   apply_twiddles_xl with big_modulus=N2·N3 picks exactly that.
    //   Combines the n2·k1/(N1·N2) and n3·k1/N cross-terms of the
    //   Cooley-Tukey decomposition into a single dispatch.
    auto [r1t, i1t] = ttnn::prim::apply_twiddles_xl(
        r1, i1, /*P=*/N1, /*big_modulus=*/N2 * N3, /*full_N=*/full_N);

    // ── Bring n2 to inner for Stage 2.
    //   Logical (B, N2, N3, k1) → (B, N3, k1, N2).
    //   View as (B, N2, N3·N1) [merge last two — page change], then
    //   transpose_rm → (B, N3·N1, N2).
    auto r2_3d = ttnn::reshape(r1t, make_shape({B, N2, N3 * N1}));
    auto i2_3d = ttnn::reshape(i1t, make_shape({B, N2, N3 * N1}));
    auto r2t   = ttnn::prim::transpose_rm(r2_3d);                     // (B, N3·N1, N2)
    auto i2t   = ttnn::prim::transpose_rm(i2_3d);
    auto r2p   = ttnn::reshape(r2t, make_shape({B * N3 * N1, N2}));
    auto i2p   = ttnn::reshape(i2t, make_shape({B * N3 * N1, N2}));

    // ── Stage 2 + Twiddle-2 fused: FFT_N2 + post-twiddle
    //       exp(-2πi · n3 · k2 / (N2·N3)).
    //   Row r' = b·N3·N1 + n3·N1 + k1.  (r' / stride=N1) % twiddle_N2=N3
    //   = (b·N3 + n3) % N3 = n3.  P·twiddle_N2 = N2·N3.  ✓
    auto [r2, i2] = ttnn::prim::fft_radix_pass(
        r2p, /*input_imag=*/i2p,
        /*P=*/N2, /*twiddle_N2=*/N3, /*stride=*/N1);

    // ── Bring n3 to inner for Stage 3.
    //   Logical (B, N3, k1, k2) → (B, k1, k2, N3).
    //   View as (B, N3, N1·N2) [merge last two — page change], then
    //   transpose_rm → (B, N1·N2, N3).
    auto r3_3d = ttnn::reshape(r2, make_shape({B, N3, N1 * N2}));
    auto i3_3d = ttnn::reshape(i2, make_shape({B, N3, N1 * N2}));
    auto r3t   = ttnn::prim::transpose_rm(r3_3d);                     // (B, N1·N2, N3)
    auto i3t   = ttnn::prim::transpose_rm(i3_3d);
    auto r3p   = ttnn::reshape(r3t, make_shape({B * N1 * N2, N3}));
    auto i3p   = ttnn::reshape(i3t, make_shape({B * N1 * N2, N3}));

    // ── Stage 3: pure FFT_N3.
    auto [r3, i3] = ttnn::prim::fft_radix_pass(
        r3p, /*input_imag=*/i3p,
        /*P=*/N3, /*twiddle_N2=*/0u);

    // ── FINAL rearrangement (k1, k2, k3) → (k3, k2, k1) so that
    //   `.reshape(B, N)` on host gives natural-order X[K] at flat K.
    //
    //   After Stage 3 we have (B·N1·N2, N3) ≡ (B, k1, k2, k3).
    //   Target: (B, N3, N2, N1) ≡ (B, k3, k2, k1).
    //
    //   (a) view → (B, N1·N2, N3)         [free, last dim unchanged]
    //   (b) transpose_rm → (B, N3, N1·N2) [page_out = N1·N2·elem]
    //   (c) view → (B, N3, N1, N2)        [page change: N1·N2 → N2]
    //   (d) transpose_rm → (B, N3, N2, N1)[page_out = N1·elem, tiny]
    auto r4_3d = ttnn::reshape(r3, make_shape({B, N1 * N2, N3}));
    auto i4_3d = ttnn::reshape(i3, make_shape({B, N1 * N2, N3}));
    auto r4t1  = ttnn::prim::transpose_rm(r4_3d);                     // (B, N3, N1·N2)
    auto i4t1  = ttnn::prim::transpose_rm(i4_3d);
    auto r4s   = ttnn::reshape(r4t1, make_shape({B, N3, N1, N2}));
    auto i4s   = ttnn::reshape(i4t1, make_shape({B, N3, N1, N2}));
    auto r_out = ttnn::prim::transpose_rm(r4s);                       // (B, N3, N2, N1)
    auto i_out = ttnn::prim::transpose_rm(i4s);

    return {std::move(r_out), std::move(i_out)};
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
