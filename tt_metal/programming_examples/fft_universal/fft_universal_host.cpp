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
// Batched-recursion architecture (Opt #2 + #2b, complete):
//
//   Every recursion level operates on `count` sibling signals laid out
//   back-to-back in row-major order. A single sub-FFT pass at depth d always
//   becomes ONE of:
//     * one fft_stockham::batch_fft dispatch (pow2 length <= 1024, any count),
//     * one batched Bluestein → 2 pow2 batch_fft dispatches (prime length),
//     * one batched Cooley-Tukey split → two recursive calls with count *= Nk.
//
//   Because composite splits MULTIPLY `count` by N1 or N2, batching width
//   grows as we descend; by the time we reach the leaves, we're typically
//   running a single pow2 batch_fft dispatch over thousands of sibling
//   sub-FFTs — regardless of the original N's structure.
//
// Cost summary:
//   * pow2 N                -> same as fft_stockham (no extra overhead).
//   * prime N (single call) -> 2 batch_fft dispatches + O(N) host multiplies.
//   * composite N           -> O(log N) batch_fft dispatches + O(N) host
//                              reshape/twiddle/transpose per level. Dispatch
//                              count is independent of how many sibling
//                              sub-FFTs live at any level.
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

// ─── Forward declarations ────────────────────────────────────────────────────
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal);

inline void batched_siblings_fft(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          len,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out);

inline void batched_bluestein(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out);

inline void cooley_tukey_split_batched(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N1,
    uint32_t                          N2,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out);

// ─── batched_siblings_fft: the universal dispatcher ──────────────────────────
//
// Computes `count` independent FFTs of length `len`, stored back-to-back in
// row-major order (in[r * len + k] is element k of sibling r). Every recursive
// level of the engine passes through here.
//
// Dispatch (in order of priority):
//   * len == 1 or count == 0         -> trivial copy.
//   * len pow2 and <= 1024 (tile)    -> ONE fft_stockham::batch_fft dispatch,
//                                        padding `count` up to next_pow2 with
//                                        zero-signal sibling rows (whose DFTs
//                                        are also zero — harmless, discarded).
//   * len pow2 but > 1024            -> serial fft_stockham::fft per row (the
//                                        kernel's multi-pass Stockham handles
//                                        up to N=1,048,576 for a single row).
//   * len prime (>= 3)               -> batched_bluestein: ONE pre-mul, TWO
//                                        batched length-M FFTs (M=next_pow2(2N-1)),
//                                        ONE pointwise mul, ONE post-mul.
//   * len composite non-pow2         -> cooley_tukey_split_batched, which does
//                                        two recursive calls with count *= Nk.
//                                        Batching width grows with depth.
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

    // Path A: pow2 length fitting in a tile → single batched dispatch.
    if (is_pow2(len) && len <= kBatchMaxSubN) {
        const uint32_t padded = next_pow2(count);
        if (padded == count) {
            fft_stockham::batch_fft(md, len, count, in, out);
            return;
        }
        std::vector<Complex> in_padded(static_cast<size_t>(padded) * len,
                                       Complex{0.0f, 0.0f});
        std::copy(in.begin(), in.end(), in_padded.begin());
        std::vector<Complex> out_padded;
        fft_stockham::batch_fft(md, len, padded, in_padded, out_padded);
        out.assign(out_padded.begin(),
                   out_padded.begin() + static_cast<size_t>(count) * len);
        return;
    }

    // Path B: pow2 length too big for a tile — batch_fft can't help, so
    // serialize across the `count` rows. Each row uses multi-pass Stockham.
    if (is_pow2(len)) {
        out.resize(static_cast<size_t>(count) * len);
        std::vector<Complex> row(len);
        for (uint32_t r = 0; r < count; ++r) {
            const size_t base = static_cast<size_t>(r) * len;
            std::copy(in.begin() + base, in.begin() + base + len, row.begin());
            const std::vector<Complex> Yr = fft_stockham::fft(md, row);
            std::copy(Yr.begin(), Yr.end(), out.begin() + base);
        }
        return;
    }

    // Path C: prime length → batched Bluestein (fuses all `count` siblings
    // into exactly 2 batched pow2 FFT dispatches of length M).
    if (is_prime(len)) {
        batched_bluestein(md, count, len, in, out);
        return;
    }

    // Path D: composite non-pow2 → batched Cooley-Tukey split. The two
    // recursive calls inside propagate `count` multiplied by N1 / N2, so
    // batching width GROWS with recursion depth.
    const auto [N1, N2] = pick_factors(len);
    cooley_tukey_split_batched(md, count, N1, N2, in, out);
}

// ─── batched_bluestein ───────────────────────────────────────────────────────
//
// Compute `count` sibling length-N Bluestein FFTs. Instead of `count`
// independent Bluestein chains (2 × count length-M dispatches), we run:
//   1. batched pre-multiply by chirp w   (host, O(count·N))
//   2. ONE batched length-M forward FFT  (device, via batched_siblings_fft)
//   3. batched pointwise × B_fft          (host, O(count·M))
//   4. ONE batched length-M inverse FFT  (device, via conjugate trick)
//   5. batched post-multiply by chirp w  (host, O(count·N))
// Total device dispatches: 2 — independent of `count`.
inline void batched_bluestein(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out)
{
    assert(in.size() == static_cast<size_t>(count) * N);
    auto           plan = get_bluestein_plan(md, N);
    const uint32_t M    = plan->M;
    const auto&    w    = plan->chirp_fwd;   // chirp, length N
    const auto&    B    = plan->B_fft;       // FFT of mirrored conj(w), length M

    // Step 1: pre-multiply each sibling by w, zero-pad to length M.
    std::vector<Complex> A(static_cast<size_t>(count) * M, Complex{0.0f, 0.0f});
    for (uint32_t r = 0; r < count; ++r) {
        const size_t in_base  = static_cast<size_t>(r) * N;
        const size_t out_base = static_cast<size_t>(r) * M;
        for (uint32_t n = 0; n < N; ++n) {
            A[out_base + n] = in[in_base + n] * w[n];
        }
    }

    // Step 2: batched forward FFT of length M (pow2 — falls into Path A/B).
    std::vector<Complex> A_fft;
    batched_siblings_fft(md, count, M, A, A_fft);

    // Step 3: elementwise A_fft[r, k] *= B[k] (same B for every sibling).
    for (uint32_t r = 0; r < count; ++r) {
        const size_t base = static_cast<size_t>(r) * M;
        for (uint32_t k = 0; k < M; ++k) A_fft[base + k] *= B[k];
    }

    // Step 4: batched inverse FFT via conjugate trick —
    //         IFFT(X) = conj(FFT(conj(X))) / M.
    for (auto& z : A_fft) z = std::conj(z);
    std::vector<Complex> c;
    batched_siblings_fft(md, count, M, A_fft, c);
    const float inv_M = 1.0f / static_cast<float>(M);
    for (auto& z : c) z = std::conj(z) * inv_M;

    // Step 5: post-multiply first N samples of each row by w, drop padding.
    out.resize(static_cast<size_t>(count) * N);
    for (uint32_t r = 0; r < count; ++r) {
        const size_t in_base  = static_cast<size_t>(r) * M;
        const size_t out_base = static_cast<size_t>(r) * N;
        for (uint32_t k = 0; k < N; ++k) {
            out[out_base + k] = c[in_base + k] * w[k];
        }
    }
}

// ─── cooley_tukey_split_batched ──────────────────────────────────────────────
//
// Compute `count` sibling length-(N1·N2) FFTs via one mixed-radix split.
// Every host-side reshape/twiddle/transpose is loop-extended over `count`,
// and every sub-FFT call scales `count` by N1 or N2 — so batching width
// accumulates as we recurse deeper.
//
// Index schemes (matching fft_stockham, scheme (c)):
//   n  = n1 + N1 * n2     (n1 ∈ [0,N1), n2 ∈ [0,N2))
//   k  = N2 * k1 + k2     (k1 ∈ [0,N1), k2 ∈ [0,N2))
//
//   1. transposed reshape:  A[r, n1, n2] = in[r, n1 + N1 * n2]
//   2. pass-1  len N2:      batched_siblings_fft(count * N1, N2)
//   3. twiddle:             A[r, n1, k2] *= exp(-2πi · n1 · k2 / N)
//   4. transpose:           C[r, k2, n1] = A[r, n1, k2]
//   5. pass-2  len N1:      batched_siblings_fft(count * N2, N1)
//   6. output permute:      out[r, N2·k1 + k2] = D[r, k2, k1]
inline void cooley_tukey_split_batched(
    std::shared_ptr<MeshDevice>       md,
    uint32_t                          count,
    uint32_t                          N1,
    uint32_t                          N2,
    const std::vector<Complex>&       in,
    std::vector<Complex>&             out)
{
    const uint32_t N     = N1 * N2;
    const size_t   total = static_cast<size_t>(count) * N;
    assert(in.size() == total);

    // Step 1: per-row transposed reshape.
    //   pass1_in[(r * N1 + n1) * N2 + n2] = in[r * N + n2 * N1 + n1]
    std::vector<Complex> pass1_in(total);
    for (uint32_t r = 0; r < count; ++r) {
        const size_t in_base = static_cast<size_t>(r) * N;
        for (uint32_t n1 = 0; n1 < N1; ++n1) {
            const size_t out_base =
                (static_cast<size_t>(r) * N1 + n1) * N2;
            for (uint32_t n2 = 0; n2 < N2; ++n2) {
                pass1_in[out_base + n2] = in[in_base + n2 * N1 + n1];
            }
        }
    }

    // Step 2: (count * N1) sibling sub-FFTs of length N2.
    std::vector<Complex> A;
    batched_siblings_fft(md, count * N1, N2, pass1_in, A);

    // Step 3: per-row twiddle  A[r, n1, k2] *= exp(-2πi · n1 · k2 / N).
    {
        const double tau_over_N = -2.0 * M_PI / static_cast<double>(N);
        for (uint32_t r = 0; r < count; ++r) {
            for (uint32_t n1 = 0; n1 < N1; ++n1) {
                const size_t base =
                    (static_cast<size_t>(r) * N1 + n1) * N2;
                for (uint32_t k2 = 0; k2 < N2; ++k2) {
                    const double  a  = tau_over_N
                                      * static_cast<double>(n1)
                                      * static_cast<double>(k2);
                    const Complex tw(static_cast<float>(std::cos(a)),
                                     static_cast<float>(std::sin(a)));
                    A[base + k2] *= tw;
                }
            }
        }
    }

    // Step 4: per-row transpose  C[r, k2, n1] = A[r, n1, k2].
    std::vector<Complex> C(total);
    for (uint32_t r = 0; r < count; ++r) {
        for (uint32_t n1 = 0; n1 < N1; ++n1) {
            const size_t A_base = (static_cast<size_t>(r) * N1 + n1) * N2;
            for (uint32_t k2 = 0; k2 < N2; ++k2) {
                const size_t C_idx =
                    (static_cast<size_t>(r) * N2 + k2) * N1 + n1;
                C[C_idx] = A[A_base + k2];
            }
        }
    }

    // Step 5: (count * N2) sibling sub-FFTs of length N1.
    std::vector<Complex> D;
    batched_siblings_fft(md, count * N2, N1, C, D);

    // Step 6: per-row output permute  out[r, N2·k1 + k2] = D[r, k2, k1].
    out.resize(total);
    for (uint32_t r = 0; r < count; ++r) {
        for (uint32_t k1 = 0; k1 < N1; ++k1) {
            for (uint32_t k2 = 0; k2 < N2; ++k2) {
                const size_t out_idx =
                    static_cast<size_t>(r) * N + k1 * N2 + k2;
                const size_t D_idx =
                    (static_cast<size_t>(r) * N2 + k2) * N1 + k1;
                out[out_idx] = D[D_idx];
            }
        }
    }
}

// ─── Top-level single-signal dispatch ────────────────────────────────────────
inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 1u && "FFT requires N >= 1");

    if (N == 1u)    return signal;
    // Large pow2 Ns go directly to fft_stockham so we keep its optimised
    // multi-pass path (up to N = 1M). Everything else — including small pow2
    // — funnels through the batched engine with count=1, which still uses
    // batch_fft internally for tile-sized pow2s.
    if (is_pow2(N)) return fft_stockham::fft(md, signal);

    if (is_prime(N)) {
        const uint32_t M = next_pow2(2u * N - 1u);
        assert(M <= kStockhamMaxPow2 &&
               "Bluestein M exceeds fft_stockham's max pow2. Raise the "
               "Stockham ceiling (multi-pass) before using larger prime N.");
        (void)M;
    }

    std::vector<Complex> out;
    batched_siblings_fft(md, /*count=*/1u, /*len=*/N, signal, out);
    return out;
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
