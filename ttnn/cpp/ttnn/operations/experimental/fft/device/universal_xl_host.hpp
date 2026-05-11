// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_host.cpp — XL FFT dispatcher (Option B: host outer twiddle).
//
// Handles power-of-two N from 2 up to 1,073,741,824 (2^30) by chaining
// existing fft_stockham kernels with HOST-SIDE outer twiddle multiply
// AND HOST-SIDE outer length-F1 butterfly.
//
// Why the outer butterfly is on host (not on batch_fft):
//   The existing fft_stockham::batch_fft kernel allocates ONE FULL
//   1024-element tile per sub-FFT regardless of sub_N. For K=3 cases
//   the outer pass would be batch=M sub-FFTs of length F1, where M
//   can be up to 1,048,576. That's 4 buffers × 1M tiles × 4 KB =
//   16 GB of DRAM — impossible on a 12 GB Wormhole. Until a packed
//   batch_fft_xl kernel (many short FFTs / tile) lands, we have to
//   keep this step on the host. F1 is by construction the SMALLEST
//   factor in the plan (typically 2 or 4), so the per-element cost
//   is at most a handful of FMAs — trivial vs the host outer-twiddle.
//
// Trade-off vs the eventual on-device path (Option A / pass2_xl):
//   * Pros: works today with NO new device kernels, accepts any pow2 N.
//   * Cons: outer twiddle multiply + outer length-F1 butterfly are on
//           the host. For N <= 1M we never hit this path (we delegate
//           straight to fft_stockham). For N > 1M, host arithmetic is
//           O(N) with a tiny constant — bounded by the host-twiddle
//           step which is ~150 ms / GB-elem. Big-N runtime is dominated
//           by the F1 sequential inner fft_stockham calls anyway.
//
// Algorithm for K=3 (N = F1 * M, F1 = SMALLEST factor, M = N / F1):
//
//   Step 0  : strided pre-pack — reshape signal so row n1 (length M) is
//             T[n1, n2] = signal[n2 * F1 + n1].  Pure host memory shuffle
//             (no arithmetic).  This is REQUIRED for the standard 2-step
//             Cooley-Tukey decomposition; doing contiguous chunks would
//             give the wrong inner-FFT input.
//   Step 1  : F1 sequential calls to fft_stockham::fft on rows of length M.
//   Step 2  : host outer twiddle multiply
//                 Y[n1, k_inner] *= w_N^(n1 * k_inner)
//             with w_N = exp(-2*pi*i / N), table cached per N.
//   Step 3  : host length-F1 butterfly per inner index k_inner (M of them).
//             Writes directly into the natural-order output X — fuses with
//             the final reorder. F1 <= 1024 by planner; in practice F1 is
//             the SMALLEST factor (2 or 4 for almost all N <= 1G), so this
//             is a few FMAs per output element.
//   Step 4  : (N/A — fused into Step 3.)
//   Step 5  : (N/A — fused into Step 3.)
//
// The host twiddle table is cached so the SECOND call for the same N
// is cos/sin-free.

#pragma once

#include "universal_xl_planner.hpp"
#include "stockham_host.hpp"

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fft_universal_xl {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

namespace detail {

// Pick the SMALLEST factor in the plan as F1 (outer dimension).
// Rationale: outer Step 1 is F1 sequential fft_stockham calls; F1 small
// minimises sequential cost.  Inner length M = N / F1 still has to fit
// fft_stockham (i.e., M <= 1M), which is guaranteed because the planner
// produces factors <= 1024 and M = product of remaining factors so the
// largest possible M = 1024 * 1024 = 1M.
inline uint32_t pick_outer_factor(const XLPlan& p) {
    assert(!p.factors.empty());
    return *std::min_element(p.factors.begin(), p.factors.end());
}

// Outer twiddle table: w[n1*M + k_inner] = exp(-2*pi*i * n1 * k_inner / N).
// Cached per (N, F1) since it's reused on every call with the same shape.
struct OuterTwiddle {
    uint32_t N{0};
    uint32_t F1{0};
    uint32_t M{0};
    std::vector<Complex> w;  // size F1 * M
};

inline std::unordered_map<uint64_t, std::shared_ptr<OuterTwiddle>>&
twiddle_cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<OuterTwiddle>> c;
    return c;
}
inline uint64_t twiddle_key(uint32_t N, uint32_t F1) {
    return (static_cast<uint64_t>(N) << 32) | F1;
}

inline std::shared_ptr<OuterTwiddle> get_outer_twiddle(uint32_t N, uint32_t F1) {
    auto key = twiddle_key(N, F1);
    auto& c  = twiddle_cache();
    auto it  = c.find(key);
    if (it != c.end()) return it->second;

    auto tw = std::make_shared<OuterTwiddle>();
    tw->N  = N;
    tw->F1 = F1;
    tw->M  = N / F1;
    tw->w.assign(static_cast<size_t>(F1) * tw->M, Complex{1.0f, 0.0f});

    const double two_pi_over_N = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        Complex* row = tw->w.data() + static_cast<size_t>(n1) * tw->M;
        for (uint32_t k = 0; k < tw->M; ++k) {
            const double ang = two_pi_over_N
                             * static_cast<double>(n1)
                             * static_cast<double>(k);
            row[k] = Complex{
                static_cast<float>(std::cos(ang)),
                static_cast<float>(std::sin(ang))
            };
        }
    }

    c.emplace(key, tw);
    return tw;
}

// Practical N ceiling for the K=3 host-arithmetic path.
//
// Algorithmically the K=3 dispatcher works for any N up to 1G (the K=3 cap
// of the planner), but Step 3 is a host-side length-F1 DFT costing
// O(F1^2) ops per inner index = O(F1 * N) host ops total. For F1 <= 16
// (i.e. N <= 16M) the host time is bounded at a few seconds; above that
// it dominates wall-clock and the path becomes unusable as a public op.
//
// We gate the dispatcher at N <= 16M for now and emit a clear error
// pointing at the kernel work that lifts the ceiling. Once the packed
// batch_fft_xl kernel ships (option_a_pass2_xl_design.md), Step 3 moves
// to device and this constant should be raised to 1G (1u << 30).
inline constexpr uint32_t kXlMaxNFp32_WH = 16u * 1024u * 1024u;   // 16M  (Wormhole)
inline constexpr uint32_t kXlMaxNFp32_BH = 64u * 1024u * 1024u;   // 64M  (Blackhole — ~2x DRAM BW + 2x cores)

// Pick the practical N ceiling for the running device. Blackhole has
// roughly 2x the DRAM bandwidth and 2x the compute cores of Wormhole,
// so the host-side Step-3 outer DFT stays in the seconds range up to
// ~64M. Anything beyond still requires the packed batch_fft_xl kernel.
inline uint32_t xl_max_n_fp32(const std::shared_ptr<MeshDevice>& md) {
    return (md->arch() == tt::ARCH::BLACKHOLE) ? kXlMaxNFp32_BH
                                               : kXlMaxNFp32_WH;
}

// Back-compat alias used by call sites and error messages — set to the
// most conservative (Wormhole) value so any compile-time use is safe.
inline constexpr uint32_t kXlMaxNFp32 = kXlMaxNFp32_WH;

// Recursive entry: handles pow2 N up to kXlMaxNFp32 by falling through to
// fft_stockham for N <= 1M (k <= 2 in the plan), and applying the
// XL Steps 0-3 above for 1M < N <= 16M.
inline std::vector<Complex> fft_impl(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal,
    const XLPlan&                p)
{
    if (p.single_pass() || p.two_pass()) {
        // N <= 1M: existing fft_stockham handles it directly.
        return fft_stockham::fft(md, signal);
    }

    // K >= 3: outer split.
    const uint32_t F1 = pick_outer_factor(p);
    const uint32_t M  = p.N / F1;
    assert(F1 * M == p.N);
    assert(F1 <= kFactorCap);

    // K >= 4 (N > 1G) needs recursion through fft_universal_xl::fft for
    // the inner length-M sub-problem; not implemented yet.
    if (M > 1024u * 1024u) {
        std::fprintf(stderr,
            "[fft_universal_xl] N=%u: inner length M=%u exceeds the 1M cap of "
            "fft_stockham. K=%u plans (N > 1G) require recursive dispatch — "
            "not yet implemented.\n",
            p.N, M, p.k());
        std::abort();
    }

    // Practical host-runtime gate. Above the per-arch ceiling the host-side
    // length-F1 DFT (Step 3) dominates wall-clock; refuse with a clear,
    // actionable error rather than silently running for minutes.
    //
    //   Wormhole : 16M
    //   Blackhole: 64M (2x DRAM BW + 2x cores → host Step-3 amortises further)
    const uint32_t n_ceiling = xl_max_n_fp32(md);
    if (p.N > n_ceiling) {
        std::fprintf(stderr,
            "[fft_universal_xl] N=%u above the practical %uM ceiling for this arch "
            "(F1=%u, host Step-3 cost ~F1^2 * N ops). The algorithm is "
            "correct here, but the host outer DFT would dominate wall-clock. "
            "To lift this ceiling, implement the packed batch_fft_xl kernel "
            "described in fft_universal_xl/option_a_pass2_xl_design.md and "
            "raise kXlMaxNFp32_{WH,BH} to 1u << 30.\n",
            p.N, n_ceiling >> 20, F1);
        std::abort();
    }

    // (dev-time stdout printf removed.)

    // ── Step 0: strided pre-pack ──────────────────────────────────────
    // T[n1, n2] = signal[n2 * F1 + n1].  Each row n1 (length M) holds the
    // STRIDED gather signal[n1], signal[F1+n1], signal[2*F1+n1], ...
    // This is the input layout the inner FFT_M expects in standard 2-step
    // Cooley-Tukey decomposition.  Pure memory shuffle, no arithmetic.
    std::vector<Complex> T(p.N);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        Complex* tr = T.data() + static_cast<size_t>(n1) * M;
        for (uint32_t n2 = 0; n2 < M; ++n2) {
            tr[n2] = signal[static_cast<size_t>(n2) * F1 + n1];
        }
    }

    // ── Step 1: F1 row-FFTs of length M (sequential) ──────────────────
    // Each row of T is a length-M contiguous buffer ready for fft_stockham.
    std::vector<Complex> Y(p.N);
    std::vector<Complex> row(M);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        const Complex* src = T.data() + static_cast<size_t>(n1) * M;
        std::copy(src, src + M, row.begin());

        std::vector<Complex> y_n1 = fft_stockham::fft(md, row);

        Complex* dst = Y.data() + static_cast<size_t>(n1) * M;
        std::copy(y_n1.begin(), y_n1.end(), dst);
    }

    // ── Step 2: host outer twiddle multiply ───────────────────────────
    auto tw = get_outer_twiddle(p.N, F1);
    for (size_t i = 0; i < Y.size(); ++i) Y[i] *= tw->w[i];

    // ── Step 3: host length-F1 DFT, fused with the final reorder ──────
    // For each inner output index c in [0, M), pull the F1 values
    //   v[a] = Y[a, c]         (a in [0, F1))
    // do an F1-point DFT
    //   Vb[d] = sum_a v[a] * w_F1^(d * a)
    // and write
    //   X[c + M * d] = Vb[d]    for d in [0, F1)
    //
    // Why F1-point DFT (O(F1^2)) and not FFT (O(F1 log F1)): F1 is by
    // planner construction the SMALLEST plan factor.  For every N up to
    // 2^30 we currently produce, F1 ∈ {2, 4} — so F1^2 is at most 16
    // multiplies per output element.  Going to FFT would not change the
    // big-O bottleneck (host outer twiddle Step 2 dominates).  Doing it
    // straight as DFT also lets us trivially fuse with the output reorder
    // and avoid a separate transpose buffer.
    std::vector<Complex> X(p.N);
    if (F1 == 2u) {
        // Special case: length-2 DFT is just (a + b, a - b).  Tightest
        // possible host loop — one add and one sub per output pair.
        for (uint32_t c = 0; c < M; ++c) {
            const Complex a = Y[/*n1=0*/ 0u * M + c];
            const Complex b = Y[/*n1=1*/ 1u * M + c];
            X[c]            = a + b;          // d = 0
            X[M + c]        = a - b;          // d = 1
        }
    } else {
        // General length-F1 DFT.  Use a tiny per-N cached F1-point twiddle
        // table — same amortisation pattern as the outer twiddle.
        static thread_local std::vector<Complex> sub_tw;
        static thread_local uint32_t             sub_tw_F1 = 0;
        if (sub_tw_F1 != F1) {
            sub_tw.assign(static_cast<size_t>(F1) * F1, Complex{1.0f, 0.0f});
            const double two_pi_over_F1 = -2.0 * M_PI
                                        / static_cast<double>(F1);
            for (uint32_t d = 0; d < F1; ++d) {
                Complex* trow = sub_tw.data() + static_cast<size_t>(d) * F1;
                for (uint32_t a = 0; a < F1; ++a) {
                    const double ang = two_pi_over_F1
                                     * static_cast<double>(d)
                                     * static_cast<double>(a);
                    trow[a] = Complex{
                        static_cast<float>(std::cos(ang)),
                        static_cast<float>(std::sin(ang))
                    };
                }
            }
            sub_tw_F1 = F1;
        }

        std::vector<Complex> v(F1);
        for (uint32_t c = 0; c < M; ++c) {
            for (uint32_t a = 0; a < F1; ++a) {
                v[a] = Y[static_cast<size_t>(a) * M + c];
            }
            for (uint32_t d = 0; d < F1; ++d) {
                const Complex* trow = sub_tw.data()
                                    + static_cast<size_t>(d) * F1;
                Complex acc{0.0f, 0.0f};
                for (uint32_t a = 0; a < F1; ++a) acc += v[a] * trow[a];
                X[static_cast<size_t>(d) * M + c] = acc;
            }
        }
    }
    return X;
}

}  // namespace detail

// ─── Public API ────────────────────────────────────────────────────────────

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 2u && "FFT requires N >= 2");
    assert(is_pow2(N) && "fft_universal_xl currently supports pow2 N only");

    const XLPlan p = plan(N);
    return detail::fft_impl(md, signal, p);
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    signal)
{
    std::vector<Complex> cx(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) cx[i] = { signal[i], 0.0f };
    return fft(md, cx);
}

// IFFT via the conjugate trick: ifft(X) = conj(fft(conj(X))) / N.
inline std::vector<Complex> ifft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  X)
{
    const uint32_t N = static_cast<uint32_t>(X.size());
    std::vector<Complex> Xc(N);
    for (uint32_t i = 0; i < N; ++i) Xc[i] = std::conj(X[i]);

    std::vector<Complex> y = fft(md, Xc);

    const float inv_N = 1.0f / static_cast<float>(N);
    for (uint32_t i = 0; i < N; ++i) y[i] = std::conj(y[i]) * inv_N;
    return y;
}

}  // namespace fft_universal_xl
