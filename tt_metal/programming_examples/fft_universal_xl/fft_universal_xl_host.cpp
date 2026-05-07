// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_host.cpp — XL FFT dispatcher (Option B: host outer twiddle).
//
// Handles power-of-two N from 2 up to 1,073,741,824 (2^30) by chaining
// existing fft_stockham kernels with a HOST-SIDE outer twiddle multiply.
//
// Trade-off vs the eventual on-device path (Option A / pass2_xl):
//   * Pros: works today with NO new device kernels, accepts any pow2 N.
//   * Cons: the outer twiddle multiply is a host-side complex multiply
//           per element. For N <= 1M we never hit this path (we delegate
//           straight to fft_stockham). For N > 1M each element costs
//           ~10 ns on the host — at N=1G that's ~10 s of host arithmetic
//           on top of device time. Big-N runtime is dominated by the F1
//           sequential outer fft_stockham calls anyway, so the host
//           twiddle is rarely the bottleneck in practice.
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
//   Step 3  : transpose Y (F1 x M) -> Z (M x F1) on host (no arithmetic).
//   Step 4  : ONE batched device dispatch: M sub-FFTs of length F1 via
//             fft_stockham::batch_fft (F1 <= 1024 by planner construction).
//   Step 5  : host reorder W (M x F1) -> X (length N).
//
// The host twiddle table is cached so the SECOND call for the same N
// is cos/sin-free.

#pragma once

#include "fft_universal_xl_planner.hpp"
#include "../fft_stockham/fft_stockham_host.cpp"

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

// Recursive entry: handles any pow2 N <= 1G by falling through to
// fft_stockham for N <= 1M (k <= 2 in the plan), and applying the
// XL Steps 1-5 above for N > 1M.
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
    // Smallest factor as F1 means M is the product of the rest, which
    // is <= 1024^(k-1).  For our supported regime k <= 3 so M <= 1M.
    // For k=4+ we'd recurse here; emit a clear error for now.
    if (M > 1024u * 1024u) {
        std::fprintf(stderr,
            "[fft_universal_xl] N=%u: inner length M=%u exceeds the 1M cap of "
            "fft_stockham. K=%u plans not yet supported.\n",
            p.N, M, p.k());
        std::abort();
    }

    std::printf(
        "[fft_universal_xl] N=%u  factors=[", p.N);
    for (size_t i = 0; i < p.factors.size(); ++i) {
        std::printf("%s%u", (i ? ", " : ""), p.factors[i]);
    }
    std::printf("]  outer F1=%u, inner M=%u  (Option B: host twiddle)\n",
                F1, M);

    auto checksum = [](const std::vector<Complex>& v) -> std::pair<double, double> {
        double s_abs = 0.0, s_sq = 0.0;
        for (const auto& c : v) {
            s_abs += std::fabs(c.real()) + std::fabs(c.imag());
            s_sq  += static_cast<double>(c.real()) * c.real()
                   + static_cast<double>(c.imag()) * c.imag();
        }
        return { s_abs, s_sq };
    };
    auto p_chk = [&](const char* tag, const std::vector<Complex>& v) {
        auto [s_abs, s_sq] = checksum(v);
        std::printf("    [chk] %-12s  size=%-9zu  L1=%.6e  L2^2=%.6e  "
                    "v[0]=(%.4g,%.4g)  v[N-1]=(%.4g,%.4g)\n",
                    tag, v.size(), s_abs, s_sq,
                    v.front().real(), v.front().imag(),
                    v.back().real(),  v.back().imag());
    };
    p_chk("signal_in", signal);

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
    p_chk("T_strided", T);

    // ── Step 1: F1 row-FFTs of length M (sequential) ──────────────────
    // Each row of T is a length-M contiguous buffer ready for fft_stockham.
    std::vector<Complex> Y(p.N);
    std::vector<Complex> row(M);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        const Complex* src = T.data() + static_cast<size_t>(n1) * M;
        std::copy(src, src + M, row.begin());

        // Per-row checksum BEFORE fft to confirm the right input is sent in.
        auto [in_l1, in_l2] = checksum(row);
        std::printf("    [chk] row[%u] in     L1=%.6e  L2^2=%.6e  "
                    "row[0]=(%.4g,%.4g)\n",
                    n1, in_l1, in_l2, row.front().real(), row.front().imag());

        std::vector<Complex> y_n1 = fft_stockham::fft(md, row);

        auto [out_l1, out_l2] = checksum(y_n1);
        std::printf("    [chk] row[%u] fft    size=%zu  L1=%.6e  L2^2=%.6e  "
                    "y[0]=(%.4g,%.4g)  y[1]=(%.4g,%.4g)\n",
                    n1, y_n1.size(), out_l1, out_l2,
                    y_n1.empty() ? 0.0f : y_n1[0].real(),
                    y_n1.empty() ? 0.0f : y_n1[0].imag(),
                    y_n1.size() < 2 ? 0.0f : y_n1[1].real(),
                    y_n1.size() < 2 ? 0.0f : y_n1[1].imag());

        Complex* dst = Y.data() + static_cast<size_t>(n1) * M;
        std::copy(y_n1.begin(), y_n1.end(), dst);
    }
    p_chk("Y_after_fft", Y);

    // ── Step 2: host outer twiddle multiply ───────────────────────────
    auto tw = get_outer_twiddle(p.N, F1);
    for (size_t i = 0; i < Y.size(); ++i) Y[i] *= tw->w[i];
    p_chk("Y_after_tw ", Y);

    // ── Step 3: transpose Y (F1 x M) -> Z (M x F1) on host ─────────────
    std::vector<Complex> Z(p.N);
    for (uint32_t n1 = 0; n1 < F1; ++n1) {
        const Complex* src = Y.data() + static_cast<size_t>(n1) * M;
        for (uint32_t k = 0; k < M; ++k) {
            Z[static_cast<size_t>(k) * F1 + n1] = src[k];
        }
    }
    p_chk("Z_transp   ", Z);

    // ── Step 4: M sub-FFTs of length F1 (one batched device dispatch) ─
    // F1 <= 1024 by planner construction so batch_fft accepts it.
    std::vector<Complex> W;
    fft_stockham::batch_fft(md, /*sub_N=*/F1, /*batch=*/M, Z, W);
    p_chk("W_batchfft ", W);

    // ── Step 5: final reorder W (M x F1) -> X (length N) ──────────────
    // After Step 4 each row j in W is a length-F1 spectrum corresponding
    // to "column j" of the F1 x M matrix. The natural 1D output index k
    // in [0, N) corresponds to (n1=k%F1, k_inner=k/F1) in the original
    // Cooley-Tukey recipe (k = n1 * M + k_inner is the input index;
    // the output index is k_out = k_inner * F1 + k1 — so a simple
    // strided gather).  Using the symmetric mapping consistent with
    // fft_stockham::final_reorder:
    //
    //   X[k] = W[(k % M) * F1 + (k / M)]
    std::vector<Complex> X(p.N);
    for (uint32_t k = 0; k < p.N; ++k) {
        const uint32_t k_inner = k % M;
        const uint32_t k1      = k / M;
        X[k] = W[static_cast<size_t>(k_inner) * F1 + k1];
    }
    p_chk("X_out      ", X);
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
