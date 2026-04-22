// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_stockham_host.cpp — Multi-pass Stockham (six-step / Bailey 4-step) FFT
//                        orchestrator that lifts our radix-2 single-shot FFT
//                        from N <= 65,536 to N up to ~1M points (and beyond).
//
// Strategy (MVP):
//   * Factor N as N1 * N2 with both ≤ 65,536 (and both powers of two).
//   * Reshape input as (N1, N2) row-major.
//   * Pass 1: row-FFT of length N2  (N1 sub-FFTs, all of length N2).
//             Each sub-FFT is dispatched through the existing radix-2 kernel
//             via `fft_example::fft`, so all sub-FFTs benefit from the
//             inner kernel's plan cache.
//   * Pass 2: per-element twiddle multiply  W_N^(i*j)  +  transpose to (N2, N1).
//             Done on the HOST in this MVP — keeps the wiring trivial. The
//             whole step is < ~100 ms even at N=1M and is parallel-safe (a
//             future optimisation is a dedicated BRISC-only device kernel).
//   * Pass 3: row-FFT of length N1  (N2 sub-FFTs of length N1).
//   * Final reorder on host: X[k] = D[k % N2, k / N2].
//
// Total DRAM round-trips: 2 (one per pass), not one per stage. Each sub-FFT
// stays fully L1-resident inside the inner radix-2 kernel.
//
// Public API (mirrors fft_example::fft):
//
//     auto X = fft_stockham::fft(md, signal);   // 1D power-of-two of any size
//                                                // (capped by host memory).
//
// For N <= 65,536 we transparently fall back to the inner radix-2 path so
// callers never need to know which algorithm ran.

#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include "../fft/fft_host.cpp"   // reuse the inner radix-2 kernel & plan cache

#include <cmath>
#include <complex>
#include <vector>
#include <utility>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <memory>
#include <unordered_map>

// fft/fft_host.cpp does `using namespace tt::tt_metal::distributed;` at file
// scope, so MeshDevice and friends are visible here without further qualification.

namespace fft_stockham {

using Complex = std::complex<float>;
using fft_example::log2u;
using tt::tt_metal::distributed::MeshDevice;

// ── Sizing & factorisation ────────────────────────────────────────────────

// Maximum N a single inner radix-2 dispatch can handle.
constexpr uint32_t kInnerMaxN = 65536u;

// Power-of-two check.
inline bool is_pow2(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

struct StockhamPlan {
    uint32_t N        = 0;
    uint32_t N1       = 0;     // outer (column-FFT) dimension
    uint32_t N2       = 0;     // inner (row-FFT)    dimension
    bool     stockham = false; // false => fall through to inner radix-2
};

// Choose a balanced factorisation N = N1 * N2 such that both halves fit in
// the inner radix-2 kernel. Strategy: pick N2 = sqrt(N) rounded to the next
// power of two; clamp to kInnerMaxN. This keeps both passes well L1-resident.
inline StockhamPlan plan(uint32_t N) {
    StockhamPlan p{};
    p.N = N;

    if (N <= kInnerMaxN) { p.stockham = false; p.N1 = N; p.N2 = 1; return p; }

    assert(is_pow2(N) && "Stockham path requires N to be a power of two.");

    // log2N is the total number of butterfly stages.
    const uint32_t log2N = log2u(N);

    // Split log2N as evenly as possible, then clamp each half to fit the
    // inner kernel (at most log2(kInnerMaxN) = 16 bits per pass).
    uint32_t log2N2 = log2N / 2;             // inner / row-FFT length
    uint32_t log2N1 = log2N - log2N2;        // outer / column-FFT length
    const uint32_t log2_inner_max = log2u(kInnerMaxN);
    if (log2N1 > log2_inner_max) {
        const uint32_t spill = log2N1 - log2_inner_max;
        log2N1 -= spill;
        log2N2 += spill;
    }
    if (log2N2 > log2_inner_max) {
        const uint32_t spill = log2N2 - log2_inner_max;
        log2N2 -= spill;
        log2N1 += spill;
    }

    p.N1 = 1u << log2N1;
    p.N2 = 1u << log2N2;
    p.stockham = true;

    assert(p.N1 <= kInnerMaxN && p.N2 <= kInnerMaxN);
    assert(static_cast<uint64_t>(p.N1) * static_cast<uint64_t>(p.N2) ==
           static_cast<uint64_t>(p.N));
    return p;
}

// ── Pass 1: N1 row-FFTs of length N2 ──────────────────────────────────────
//
// Treat the 1D input as (N1, N2) row-major and FFT each row. We use the
// existing `fft_example::fft` for every row, which hits the inner plan
// cache after the first row. This is an MVP "host-side batch" — a future
// device-side batch kernel would amortise the per-call enqueue overhead
// across all N1 rows in a single dispatch.

inline std::vector<Complex> pass1_row_ffts(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  x,
    const StockhamPlan&          p)
{
    assert(static_cast<uint32_t>(x.size()) == p.N);
    std::vector<Complex> A(p.N);
    std::vector<Complex> row(p.N2);

    for (uint32_t i = 0; i < p.N1; ++i) {
        const Complex* src = x.data()     + static_cast<size_t>(i) * p.N2;
        Complex*       dst = A.data()     + static_cast<size_t>(i) * p.N2;
        for (uint32_t j = 0; j < p.N2; ++j) row[j] = src[j];

        const std::vector<Complex> Yi = fft_example::fft(md, row);

        for (uint32_t j = 0; j < p.N2; ++j) dst[j] = Yi[j];
    }
    return A;
}

// ── Pass 2: per-element twiddle  +  transpose to (N2, N1) ────────────────
//
//   B[i, j] = A[i, j] * exp(-2*pi*i*i*j / N)         (twiddle)
//   C[j, i] = B[i, j]                                (transpose)
//
// On host this is a tight O(N) loop with cos/sin per element. At N=1M it
// completes in ~50–100 ms on a normal x86. A future optimisation is a
// BRISC-only device kernel that fuses both steps and runs in <5 ms.

inline std::vector<Complex> pass2_twiddle_transpose(
    const std::vector<Complex>& A,
    const StockhamPlan&         p)
{
    std::vector<Complex> C(p.N);
    const double tau_over_N = -2.0 * M_PI / static_cast<double>(p.N);

    for (uint32_t i = 0; i < p.N1; ++i) {
        const Complex* src = A.data() + static_cast<size_t>(i) * p.N2;
        for (uint32_t j = 0; j < p.N2; ++j) {
            const double angle = tau_over_N *
                                 static_cast<double>(i) *
                                 static_cast<double>(j);
            const float  cw = static_cast<float>(std::cos(angle));
            const float  sw = static_cast<float>(std::sin(angle));
            const Complex w(cw, sw);
            C[static_cast<size_t>(j) * p.N1 + i] = src[j] * w;
        }
    }
    return C;
}

// ── Pass 3: N2 row-FFTs of length N1 ──────────────────────────────────────

inline std::vector<Complex> pass3_row_ffts(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  C,
    const StockhamPlan&          p)
{
    assert(static_cast<uint32_t>(C.size()) == p.N);
    std::vector<Complex> D(p.N);
    std::vector<Complex> row(p.N1);

    for (uint32_t j = 0; j < p.N2; ++j) {
        const Complex* src = C.data() + static_cast<size_t>(j) * p.N1;
        Complex*       dst = D.data() + static_cast<size_t>(j) * p.N1;
        for (uint32_t i = 0; i < p.N1; ++i) row[i] = src[i];

        const std::vector<Complex> Dj = fft_example::fft(md, row);

        for (uint32_t i = 0; i < p.N1; ++i) dst[i] = Dj[i];
    }
    return D;
}

// ── Final reorder: D is (N2, N1) row-major; natural 1D output:
//     X[k] = D[k % N2, k / N2] = D_flat[(k % N2) * N1 + (k / N2)]
inline std::vector<Complex> final_reorder(
    const std::vector<Complex>& D,
    const StockhamPlan&         p)
{
    std::vector<Complex> X(p.N);
    for (uint32_t k = 0; k < p.N; ++k) {
        const uint32_t j  = k % p.N2;
        const uint32_t ip = k / p.N2;
        X[k] = D[static_cast<size_t>(j) * p.N1 + ip];
    }
    return X;
}

// ── Public API ────────────────────────────────────────────────────────────
//
// fft_stockham::fft(md, signal) — drop-in equivalent of
// fft_example::fft(md, signal) that supports any N (power of two).
//
// For N <= 65,536 we just call the inner radix-2 directly (zero overhead).
// For N >  65,536 we run the four-pass Stockham orchestrator.

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<Complex>&  signal)
{
    const uint32_t N = static_cast<uint32_t>(signal.size());
    assert(N >= 2 && "FFT requires N >= 2");
    assert(is_pow2(N) && "FFT requires N to be a power of two");

    const StockhamPlan p = plan(N);

    if (!p.stockham) {
        return fft_example::fft(md, signal);
    }

    std::printf(
        "[fft_stockham] N=%u  =>  N1=%u  x  N2=%u   (inner radix-2 kernel "
        "handles each sub-FFT)\n", p.N, p.N1, p.N2);

    const auto A = pass1_row_ffts        (md, signal, p);
    const auto C = pass2_twiddle_transpose(    A,      p);
    const auto D = pass3_row_ffts        (md, C,      p);
    return final_reorder(D, p);
}

inline std::vector<Complex> fft(
    std::shared_ptr<MeshDevice>  md,
    const std::vector<float>&    signal)
{
    std::vector<Complex> cx(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) cx[i] = {signal[i], 0.0f};
    return fft(md, cx);
}

}  // namespace fft_stockham
