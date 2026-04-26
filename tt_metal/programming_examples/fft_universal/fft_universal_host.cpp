// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_host.cpp — Host-side FFT that accepts ANY N >= 2.
//
// Reuses existing device kernels via fft_stockham::fft (which itself routes to
// fft_example::fft for N <= 65,536 and runs the 4-pass Stockham for larger
// powers of two). No new device kernels are introduced here.
//
// Dispatch tree (every compute path ends on Wormhole):
//   * N == 1                -> identity.
//   * N is a power of two   -> fft_stockham::fft (direct pass-through, device).
//   * N factors as 2^k * q  -> Cooley-Tukey split on (2^k, q), then recurse.
//   * N is odd composite    -> Cooley-Tukey split on (smallest-prime, rest),
//                              then recurse.
//   * N is prime (>=3)      -> Bluestein (chirp-z): one forward + one inverse
//                              pow2 FFT on the device (M = next pow2 >= 2N-1).
//
// Cost summary:
//   * pow2 N                -> same as fft_stockham (no extra overhead).
//   * composite N           -> pow2 sub-FFTs plus O(N) host twiddle + transpose.
//                              Whenever a side of the split (N1 or N2) is pow2
//                              and fits in a tile (<= 1024), that entire pass
//                              runs in ONE batched device dispatch across all
//                              64 cores via fft_stockham::batch_fft. The other
//                              side (non-pow2) still recurses serially.
//   * prime N               -> ~2x the cost of a length-M Stockham FFT plus
//                              two O(N) host passes (pre/post-multiply).
//
// Caches (globals; single-threaded use):
//   * bluestein_cache keyed on N keeps the chirp table and B_fft so the second
//     call for the same N skips all host pre-work.
//   * fft_stockham::fft carries its own program-build cache; we piggy-back.

#pragma once

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "../fft_stockham/fft_stockham_host.cpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fft_universal {

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

// ─── Tunables ────────────────────────────────────────────────────────────────
// Largest power-of-two that fft_stockham::fft currently accepts. Bluestein
// requires M = next_pow2(2N - 1) <= this ceiling, i.e. prime N <= 524,288.
constexpr uint32_t kStockhamMaxPow2 = 1048576u;

// fft_stockham::batch_fft requires each sub-FFT to fit in a single Tensix
// tile (1024 complex elements). Above this we cannot batch and fall back to
// serial recursion.
constexpr uint32_t kBatchMaxSubN = 1024u;

// ─── Small helpers ───────────────────────────────────────────────────────────
inline bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

inline uint32_t next_pow2(uint32_t n) {
    uint32_t p = 1u;
    while (p < n) p <<= 1;
    return p;
}

inline uint32_t smallest_prime_factor(uint32_t n) {
    if (n < 2u) return n;
    if ((n & 1u) == 0u) return 2u;
    for (uint32_t p = 3u; static_cast<uint64_t>(p) * p <= n; p += 2u) {
        if (n % p == 0u) return p;
    }
    return n;   // n itself is prime
}

inline bool is_prime(uint32_t n) {
    if (n < 2u) return false;
    return smallest_prime_factor(n) == n;
}

// Pick (N1, N2) with N = N1 * N2. Prefer the largest pow2 factor as N1 so
// pass-1 sub-FFTs are pow2 and hit the optimised device path directly.
// Fall back to (smallest-prime-factor, rest) for odd composites.
inline std::pair<uint32_t, uint32_t> pick_factors(uint32_t N) {
    uint32_t pow2 = 1u, odd = N;
    while ((odd & 1u) == 0u) { odd >>= 1; pow2 <<= 1; }
    if (pow2 > 1u && odd > 1u) {
        return {pow2, odd};
    }
    // N is odd; peel off its smallest prime factor.
    const uint32_t p = smallest_prime_factor(N);
    return {p, N / p};
}

// ─── Bluestein (chirp-z) plan ────────────────────────────────────────────────
//
// Identity used:
//   X[k] = sum_{n=0}^{N-1} x[n] exp(-2πi k n / N)
//        = w[k] * sum_n (x[n] * w[n]) * conj(w)[k - n]
//   where w[n] = exp(-i π n² / N).
//
// The inner sum is the linear convolution of (x * w) with conj(w), computed
// as a length-M cyclic convolution with M = next_pow2(2N - 1):
//   A = FFT_M(a), B = FFT_M(b_ext), c = IFFT_M(A * B), X[k] = w[k] * c[k]
// where a is (x * w) zero-padded and b_ext is conj(w) symmetrically extended.
//
// B_fft, chirp_fwd, and M depend only on N — cache them.
struct BluesteinPlan {
    uint32_t             N = 0u;
    uint32_t             M = 0u;
    std::vector<Complex> chirp_fwd;   // w[n] = exp(-i π n² / N), length N
    std::vector<Complex> B_fft;       // FFT_M(b_ext), length M
};

inline std::unordered_map<uint32_t, std::shared_ptr<BluesteinPlan>>&
bluestein_cache() {
    static std::unordered_map<uint32_t, std::shared_ptr<BluesteinPlan>> m;
    return m;
}

inline std::shared_ptr<BluesteinPlan> get_bluestein_plan(
    std::shared_ptr<MeshDevice> md,
    uint32_t                    N)
{
    auto& cache = bluestein_cache();
    if (auto it = cache.find(N); it != cache.end()) return it->second;

    auto plan = std::make_shared<BluesteinPlan>();
    plan->N = N;
    plan->M = next_pow2(2u * N - 1u);
    const uint32_t M = plan->M;

    // chirp_fwd[n] = exp(-i π n² / N). Reduce (n²) mod 2N to keep the trig
    // argument bounded and preserve precision for large n.
    plan->chirp_fwd.resize(N);
    const double   pi_over_N = M_PI / static_cast<double>(N);
    const uint64_t mod2N     = 2ull * static_cast<uint64_t>(N);
    for (uint32_t n = 0; n < N; ++n) {
        const uint64_t nn = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
        const double   a  = pi_over_N * static_cast<double>(nn % mod2N);
        plan->chirp_fwd[n] = Complex(static_cast<float>( std::cos(a)),
                                     static_cast<float>(-std::sin(a)));
    }

    // b_ext: length-M symmetric extension of conj(w).
    //   b_ext[0]   = 1
    //   b_ext[n]   = conj(w[n])    for n = 1..N-1
    //   b_ext[M-n] = conj(w[n])    for n = 1..N-1   (negative-index mirror)
    //   b_ext[n]   = 0             otherwise
    std::vector<Complex> b_ext(M, Complex(0.0f, 0.0f));
    b_ext[0] = Complex(1.0f, 0.0f);
    for (uint32_t n = 1; n < N; ++n) {
        const Complex g = std::conj(plan->chirp_fwd[n]);
        b_ext[n]     = g;
        b_ext[M - n] = g;
    }

    // M is a power of two by construction, so Stockham handles it directly.
    plan->B_fft = fft_stockham::fft(md, b_ext);

    cache[N] = plan;
    return plan;
}

// Forward declaration — helpers below may need to recurse through dispatch.
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal);

// ─── Batched sibling sub-FFTs (Opt #2) ───────────────────────────────────────
//
// Compute `count` independent FFTs of length `len`, all laid out back-to-back
// in a single row-major buffer: in[i * len + k] is element k of sub-FFT i.
//
//   * len is pow2 and <= kBatchMaxSubN        -> ONE device dispatch via
//                                                 fft_stockham::batch_fft
//                                                 (64 cores run
//                                                 padded_count/cores sub-FFTs
//                                                 each in parallel).
//   * otherwise                                -> fall back to `count` serial
//                                                 recursive fft_universal calls
//                                                 (still correct, still on
//                                                 Wormhole, just not batched).
//
// NOTE: fft_stockham::batch_fft requires BOTH `sub_N` AND `batch` to be pow2.
// When `count` isn't pow2 we pad up to `next_pow2(count)` with zero-signal
// sub-FFTs (whose DFTs are all zero, so they're harmless). The wasted work is
// bounded by 2x in the worst case (count = 2^k + 1) and is vastly outweighed
// by collapsing `count` serial dispatches into one.
//
// This is the workhorse for Cooley-Tukey: whichever side (N1 or N2) is pow2
// collapses from `count` dispatches to 1, the dominant win on the
// composite-non-pow2 regime (typical: pow2 × small_odd like 1024 × 7).
inline void batched_siblings_fft(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          len,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out)
{
    assert(in.size() == static_cast<size_t>(count) * len);

    if (len == 1u || count == 0u) {
        out = in;
        return;
    }

    if (is_pow2(len) && len <= kBatchMaxSubN) {
        const uint32_t padded = next_pow2(count);
        if (padded == count) {
            fft_stockham::batch_fft(md, len, count, in, out);
            return;
        }
        // Pad with (padded - count) zero sub-FFTs, batch, drop padding.
        std::vector<Complex> in_padded(static_cast<size_t>(padded) * len,
                                       Complex{0.0f, 0.0f});
        std::copy(in.begin(), in.end(), in_padded.begin());
        std::vector<Complex> out_padded;
        fft_stockham::batch_fft(md, len, padded, in_padded, out_padded);
        out.assign(out_padded.begin(),
                   out_padded.begin() + static_cast<size_t>(count) * len);
        return;
    }

    // Serial fallback: sibling sub-FFTs aren't directly batch-able
    // (non-pow2 length or bigger than a tile). Each recursive call still
    // lands on Wormhole via Bluestein or nested Cooley-Tukey.
    out.resize(static_cast<size_t>(count) * len);
    std::vector<Complex> tmp(len);
    for (uint32_t i = 0; i < count; ++i) {
        const size_t base = static_cast<size_t>(i) * len;
        for (uint32_t k = 0; k < len; ++k) tmp[k] = in[base + k];
        const std::vector<Complex> Yi = fft(md, tmp);
        for (uint32_t k = 0; k < len; ++k) out[base + k] = Yi[k];
    }
}

// ─── Bluestein forward FFT ───────────────────────────────────────────────────
inline std::vector<Complex> bluestein_fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  x)
{
    const uint32_t N    = static_cast<uint32_t>(x.size());
    auto           plan = get_bluestein_plan(md, N);
    const uint32_t M    = plan->M;

    // a[n] = x[n] * w[n], zero-padded to length M.
    std::vector<Complex> a(M, Complex(0.0f, 0.0f));
    for (uint32_t n = 0; n < N; ++n) {
        a[n] = x[n] * plan->chirp_fwd[n];
    }

    // A = FFT_M(a).
    std::vector<Complex> A = fft_stockham::fft(md, a);

    // C = A * B (elementwise; done in place on A).
    for (uint32_t k = 0; k < M; ++k) {
        A[k] *= plan->B_fft[k];
    }

    // c = IFFT_M(C) via the conjugate trick:
    //     IFFT(X) = conj(FFT(conj(X))) / M.
    for (uint32_t k = 0; k < M; ++k) A[k] = std::conj(A[k]);
    std::vector<Complex> c = fft_stockham::fft(md, A);
    const float inv_M = 1.0f / static_cast<float>(M);
    for (uint32_t k = 0; k < M; ++k) c[k] = std::conj(c[k]) * inv_M;

    // X[k] = w[k] * c[k], for k = 0..N-1.
    std::vector<Complex> X(N);
    for (uint32_t k = 0; k < N; ++k) X[k] = plan->chirp_fwd[k] * c[k];
    return X;
}

// ─── Cooley-Tukey split: N = N1 * N2 ─────────────────────────────────────────
//
// Mixed-radix decomposition, matching the scheme used by fft_stockham
// (so the recursion on pow2 sub-FFTs lines up exactly):
//
//   index map:   n  = n1 + N1 * n2     (n1 in [0,N1),   n2 in [0,N2))
//                k  = N2 * k1 + k2     (k1 in [0,N1),   k2 in [0,N2))
//
//   1. transposed reshape:  A[i=n1, j=n2] = x[n1 + N1 * n2]
//   2. pass-1  (length-N2):  for each fixed n1, FFT along n2 axis     -- recurse
//   3. twiddle:              A[n1, k2] *= exp(-2πi · n1 · k2 / N)
//   4. transpose:            C[k2, n1] = A[n1, k2]   (shape N2 x N1)
//   5. pass-2  (length-N1):  for each fixed k2, FFT along n1 axis     -- recurse
//   6. output permute:       X[N2 * k1 + k2] = C[k2, k1]
inline std::vector<Complex> cooley_tukey_split(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  x,
    uint32_t                     N1,
    uint32_t                     N2)
{
    const uint32_t N = N1 * N2;
    assert(static_cast<uint32_t>(x.size()) == N);

    // Step 1: transposed reshape into (N1, N2) row-major. Row i of the
    // packed buffer is the length-N2 strided slice {x[i], x[N1+i], x[2N1+i], ...}.
    std::vector<Complex> pass1_in(N);
    for (uint32_t i = 0; i < N1; ++i) {
        const size_t base = static_cast<size_t>(i) * N2;
        for (uint32_t j = 0; j < N2; ++j) pass1_in[base + j] = x[j * N1 + i];
    }

    // Step 2: N1 sibling sub-FFTs of length N2.
    //   - If N2 is pow2 and <= 1024: ONE batched device dispatch (64-core fan-out).
    //   - Else: N1 serial recursive fft(md, ...) calls (still on Wormhole via
    //     Bluestein or a nested Cooley-Tukey split).
    std::vector<Complex> A;
    batched_siblings_fft(md, N1, N2, pass1_in, A);

    // Step 3: twiddle multiply. A is in (n1, k2) = (i, j) layout now.
    // Double-precision angle keeps fp32 fidelity even at large Ns.
    {
        const double tau_over_N = -2.0 * M_PI / static_cast<double>(N);
        for (uint32_t i = 0; i < N1; ++i) {
            const uint32_t base = i * N2;
            for (uint32_t j = 0; j < N2; ++j) {
                const double  a = tau_over_N * static_cast<double>(i)
                                             * static_cast<double>(j);
                const Complex w(static_cast<float>(std::cos(a)),
                                static_cast<float>(std::sin(a)));
                A[base + j] *= w;
            }
        }
    }

    // Step 4: transpose into shape (N2, N1) row-major — C[k2, n1] = A[n1, k2].
    std::vector<Complex> C(N);
    for (uint32_t i = 0; i < N1; ++i) {
        for (uint32_t j = 0; j < N2; ++j) {
            C[j * N1 + i] = A[i * N2 + j];
        }
    }

    // Step 5: N2 sibling sub-FFTs of length N1. Same batching rule as pass-1
    // — if N1 is pow2 and <= 1024, the whole pass collapses to ONE dispatch.
    std::vector<Complex> D;
    batched_siblings_fft(md, N2, N1, C, D);

    // Step 6: permute to output order. With k = N2*k1 + k2 (scheme (c)),
    // X[N2*k1 + k2] lives at D[k2, k1] = D[k2*N1 + k1].
    std::vector<Complex> X(N);
    for (uint32_t k1 = 0; k1 < N1; ++k1) {
        for (uint32_t k2 = 0; k2 < N2; ++k2) {
            X[k1 * N2 + k2] = D[k2 * N1 + k1];
        }
    }
    return X;
}

// ─── Top-level dispatch ──────────────────────────────────────────────────────
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 1u && "FFT requires N >= 1");

    if (N == 1u)   return signal;
    if (is_pow2(N)) return fft_stockham::fft(md, signal);

    if (is_prime(N)) {
        const uint32_t M = next_pow2(2u * N - 1u);
        assert(M <= kStockhamMaxPow2 &&
               "Bluestein M exceeds fft_stockham's max pow2. Raise the "
               "Stockham ceiling (multi-pass) before using larger prime N.");
        (void)M;
        return bluestein_fft(md, signal);
    }

    const auto [N1, N2] = pick_factors(N);
    return cooley_tukey_split(md, signal, N1, N2);
}

// Convenience overload for purely real input.
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    real_signal)
{
    std::vector<Complex> cx(real_signal.size());
    for (size_t i = 0; i < real_signal.size(); ++i) {
        cx[i] = Complex(real_signal[i], 0.0f);
    }
    return fft(md, cx);
}

}  // namespace fft_universal
