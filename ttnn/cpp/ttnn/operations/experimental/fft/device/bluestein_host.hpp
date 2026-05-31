// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bluestein_host.hpp — host-side precomputation + per-device cache for the
// Bluestein composite (commit 6d).
//
// Bluestein's identity rewrites the length-N DFT (N arbitrary, not just
// pow-2) as a length-M cyclic convolution where M = next_pow2(2*N - 1).
// We can therefore handle ANY N using only our pow-2 FFT machinery.
//
//   X[k] = chirp_k[k] · (a * b)[k]      for k = 0..N-1
//
//   chirp_n[n] = exp(-π·i · n² / N)                  ; length N
//   chirp_k[k] = exp(-π·i · k² / N) = chirp_n[k]     ; length N
//   b[m]       = exp(+π·i · m² / N)                  ; length N
//   a[n]       = x[n] · chirp_n[n]                   ; length N
//
//   (a * b)[k] = sum_n a[n] · b[k - n]               ; linear convolution
//
// To compute that linear convolution via cyclic-FFT-conv:
//   1. Zero-pad a to length M : a_pad = [a, 0, 0, ..., 0]               (M floats)
//   2. Build a cyclic b kernel: b_cyc[m] = b[m]   for m =  0..N-1
//                               b_cyc[M-m] = b[m] for m =  1..N-1
//                               b_cyc[m] = 0      for m = N..M-N
//   3. (a * b)_cyc = IFFT( FFT(a_pad) ⊙ FFT(b_cyc) )
//   4. Take the FIRST N elements; these are the LINEAR convolution result
//      because the zero-padding guarantees the cyclic wrap-around is zero
//      for k = 0..N-1.
//
// PER-N PRECOMPUTATION (cached):
//   chirp_n  : (1, N)  device Tensor pair (real, imag)
//   chirp_k  : (1, N)  device Tensor pair (real, imag)
//   B = FFT(b_cyc) : (1, M) device Tensor pair (real, imag)
//
// All three live on the device they were uploaded to and are returned by
// reference (shared_ptr) from get_or_create.  Cache key is
// (device-ptr, N, dtype) — separate entries for fp32 and bf16.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/experimental/fft/fft.hpp"   // ttnn::operations::experimental::fft

namespace ttnn::experimental::prim::bluestein_host {

struct BluesteinPlan {
    uint32_t N = 0;   // logical FFT length (the user's `N`).
    uint32_t M = 0;   // padded length, next_pow2(2*N - 1).
    uint32_t B = 1;   // batch dim of the chirp / B tensors (matches caller).

    // chirp_n, chirp_k : (B, N) ROW_MAJOR, dtype = caller's dtype.
    //   Each of the B rows holds the SAME chirp sequence (replicated on
    //   the host before upload).  Replication is required because our
    //   complex_mul is shape-strict (no broadcast).
    ttnn::Tensor chirp_n_re;
    ttnn::Tensor chirp_n_im;
    ttnn::Tensor chirp_k_re;
    ttnn::Tensor chirp_k_im;

    // B_fft = FFT_M(b_cyc) replicated to (B, M) ROW_MAJOR.
    //   (Same value across batch rows; replicated for the same reason.)
    ttnn::Tensor B_re;
    ttnn::Tensor B_im;
};

// ── Helpers ──────────────────────────────────────────────────────────────

inline constexpr uint32_t next_pow2(uint32_t v) {
    if (v <= 1u) return 1u;
    --v;
    v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
    v |= v >> 8;  v |= v >> 16;
    return v + 1u;
}

// Bluestein padding length: smallest pow-2 ≥ 2*N - 1.
inline constexpr uint32_t bluestein_M(uint32_t N) {
    return next_pow2(2u * N - 1u);
}

// chirp[n] = exp(±π·i · n² / N), returned as two length-N float vectors.
// Use double precision for the angle accumulation; angle = π * (n²/N) can
// grow large for big N but the modular reduction by 2π is implicit in
// std::cos / std::sin so we don't need an explicit mod-2 step.
inline std::pair<std::vector<float>, std::vector<float>>
build_chirp(uint32_t N, int sign) {
    std::vector<float> r(N), im(N);
    const double pi_over_N = M_PI / static_cast<double>(N);
    const double s         = static_cast<double>(sign);
    for (uint32_t n = 0; n < N; ++n) {
        // Reduce n² mod 2N once on the integer side to keep the angle
        // bounded — π · (n² mod 2N) / N ∈ [0, 2π).  Improves trig
        // precision for large N (n² can overflow uint32_t at N ≈ 2^16
        // otherwise; we use uint64_t below).
        const uint64_t n_sq      = static_cast<uint64_t>(n) * n;
        const uint64_t two_N     = static_cast<uint64_t>(2) * N;
        const uint64_t n_sq_mod  = n_sq % two_N;
        const double   angle     = s * pi_over_N * static_cast<double>(n_sq_mod);
        r[n]  = static_cast<float>(std::cos(angle));
        im[n] = static_cast<float>(std::sin(angle));
    }
    return {std::move(r), std::move(im)};
}

// Build the cyclic b-kernel of length M:
//   b_cyc[m]      =  exp(+π·i · m² / N)   for m = 0..N-1
//   b_cyc[M - m]  =  exp(+π·i · m² / N)   for m = 1..N-1   (b is even)
//   b_cyc[m]      =  0                    for m = N..M-N
inline std::pair<std::vector<float>, std::vector<float>>
build_b_cyc(uint32_t N, uint32_t M) {
    auto [b_r, b_i] = build_chirp(N, /*sign=*/+1);
    std::vector<float> r(M, 0.0f), im(M, 0.0f);
    for (uint32_t m = 0; m < N; ++m) {
        r[m]  = b_r[m];
        im[m] = b_i[m];
    }
    for (uint32_t m = 1; m < N; ++m) {
        r[M - m]  = b_r[m];
        im[M - m] = b_i[m];
    }
    return {std::move(r), std::move(im)};
}

// ── Device-tensor upload helper ──────────────────────────────────────────
//
// Builds a (B, length) ROW_MAJOR Tensor of `dtype` by REPLICATING the
// length-`length` host buffer across B rows.  Narrows fp32 → bf16 on the
// host when dtype = BFLOAT16.
//
// Replication is done on the host before upload — simpler than a per-row
// device-side broadcast, and chirp tensors are tiny relative to per-call
// activations (chirp = O(B*N) floats, activations = O(B*M) ≥ 4× larger).
inline ttnn::Tensor upload_replicated_rows(
    const std::vector<float>& row,
    uint32_t B,
    uint32_t length,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::distributed::MeshDevice* device)
{
    using namespace tt::tt_metal;
    TT_FATAL(row.size() == length,
        "bluestein_host::upload_replicated_rows: row size {} != expected "
        "length {}.", row.size(), length);

    ttnn::Shape shape{ttnn::SmallVector<uint32_t>{B, length}};
    TensorSpec spec(
        shape,
        TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    const size_t total = static_cast<size_t>(B) * length;

    if (dtype == DataType::BFLOAT16) {
        std::vector<bfloat16> bf(total);
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t i = 0; i < length; ++i) {
                bf[static_cast<size_t>(b) * length + i] = bfloat16(row[i]);
            }
        }
        return Tensor::from_vector(std::move(bf), spec, device);
    }

    std::vector<float> rep(total);
    for (uint32_t b = 0; b < B; ++b) {
        std::copy(row.begin(), row.end(),
                  rep.begin() + static_cast<ptrdiff_t>(b) * length);
    }
    return Tensor::from_vector(std::move(rep), spec, device);
}

// ── Cache ────────────────────────────────────────────────────────────────

inline std::unordered_map<uint64_t, std::shared_ptr<BluesteinPlan>>& cache() {
    static std::unordered_map<uint64_t, std::shared_ptr<BluesteinPlan>> c;
    return c;
}

// Hash key: (device-ptr, N, dtype, B).  We include B because chirp / B
// tensors are stored at the caller's batch shape — complex_mul is
// shape-strict, so a (1, N) chirp can't be multiplied against a (4, N)
// activation.  Different B values get their own cache entries; in
// practice most workloads use a single fixed B so this is fine.
inline uint64_t make_key(
    tt::tt_metal::distributed::MeshDevice* md,
    uint32_t N,
    tt::tt_metal::DataType dtype,
    uint32_t B)
{
    return reinterpret_cast<uint64_t>(md)
         ^ (static_cast<uint64_t>(N)              * 0x9E3779B97F4A7C15ull)
         ^ (static_cast<uint64_t>(dtype)          * 0xBF58476D1CE4E5B9ull)
         ^ (static_cast<uint64_t>(B)              * 0x94D049BB133111EBull);
}

// Get or build the per-(N, dtype, B) Bluestein plan for `device`.
//
// On miss we:
//   1. Build chirp_n, chirp_k host arrays (cos/sin of -π·n²/N), length N.
//   2. Build b_cyc host array (cyclic kernel of length M).
//   3. Replicate each to B rows on the host and upload as (B, N) and
//      (B, M) ROW_MAJOR tensors.
//   4. Run a SINGLE forward FFT_M on the (B, M) b_cyc to produce B.
//      The FFT operates on each batch row independently, so all B rows
//      of the output are identical — that's intentional, they'll be
//      used to multiply the per-call (B, M) activations row-wise.
//      Precision matches by construction (same on-device FFT chain).
inline std::shared_ptr<BluesteinPlan> get_or_create(
    tt::tt_metal::distributed::MeshDevice* md,
    uint32_t N,
    tt::tt_metal::DataType dtype,
    uint32_t B = 1u,
    ttnn::operations::experimental::FFTPrecision precision =
        ttnn::operations::experimental::FFTPrecision::Precise)
{
    TT_FATAL(N >= 2u, "bluestein_host: N must be ≥ 2 (got {}).", N);
    TT_FATAL(B >= 1u, "bluestein_host: B must be ≥ 1 (got {}).", B);

    const uint32_t M = bluestein_M(N);
    const uint64_t key = make_key(md, N, dtype, B);
    auto& c = cache();
    auto it = c.find(key);
    if (it != c.end()) return it->second;

    auto plan = std::make_shared<BluesteinPlan>();
    plan->N = N;
    plan->M = M;
    plan->B = B;

    // ── (1) chirp_n  (sign = -1).  Build host arrays ONCE; chirp_k uses
    //   the same math so we reuse those vectors.
    auto [chirp_r, chirp_i] = build_chirp(N, /*sign=*/-1);

    plan->chirp_n_re = upload_replicated_rows(chirp_r, B, N, dtype, md);
    plan->chirp_n_im = upload_replicated_rows(chirp_i, B, N, dtype, md);

    // ── (2) chirp_k = chirp_n (same expression).
    plan->chirp_k_re = upload_replicated_rows(chirp_r, B, N, dtype, md);
    plan->chirp_k_im = upload_replicated_rows(chirp_i, B, N, dtype, md);

    // ── (3) b_cyc → upload (replicated to B rows) → FFT_M  ⇒  B_fft.
    ttnn::Tensor b_cyc_re;
    ttnn::Tensor b_cyc_im;
    {
        auto [r, i] = build_b_cyc(N, M);
        b_cyc_re = upload_replicated_rows(r, B, M, dtype, md);
        b_cyc_im = upload_replicated_rows(i, B, M, dtype, md);
    }

    // Run the same device FFT chain we'll use per-call so the B we
    // multiply against has exactly matching numerics.  For M ≤ 1024
    // this is the SingleTileStockham path; for 1024 < M ≤ 1M it's
    // fft_two_pass.  M > 1M not yet supported here (Bluestein for
    // N > ~500K → M > 1M is the 6e-2 extension via fft_three_pass).
    auto [B_re, B_im] =
        ttnn::operations::experimental::fft(b_cyc_re, b_cyc_im, precision);
    plan->B_re = std::move(B_re);
    plan->B_im = std::move(B_im);

    c.emplace(key, plan);
    return plan;
}

}  // namespace ttnn::experimental::prim::bluestein_host
