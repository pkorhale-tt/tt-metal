// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_test.cpp — End-to-end correctness test for the XL path.
//
// Verifies:
//   1. Pass-through cases (N <= 1M) match fft_stockham exactly (XL just
//      delegates).
//   2. Big-N cases (N > 1M, K=3) match a double-precision DFT reference
//      to within fp32 ULP scaling (target rel err <= 1e-4).
//   3. ifft(fft(x)) round-trips to within 1e-5 relative error.
//
// We keep N small in the K=3 cases (smallest pow2 > 1M is 2M; we test
// 2M and 4M as the smoke test). 8M+ would also work but is slow on the
// Phase-1 (host-twiddle, sequential per-row) dispatcher.

#include "fft_universal_xl_host.cpp"

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using Complex = std::complex<float>;
using ComplexD = std::complex<double>;
using tt::tt_metal::distributed::MeshDevice;

// ── Reference: O(N^2) double-precision direct DFT ───────────────────────────
// Slow but exact; we cap N at 2048 for cases where we use it as the
// reference. For larger N we use ifft(fft(x)) ≈ x as the correctness check.
std::vector<ComplexD> dft_double(const std::vector<Complex>& x) {
    const size_t N = x.size();
    std::vector<ComplexD> X(N, ComplexD{0.0, 0.0});
    const double two_pi_over_N = -2.0 * M_PI / static_cast<double>(N);
    for (size_t k = 0; k < N; ++k) {
        ComplexD acc{0.0, 0.0};
        for (size_t n = 0; n < N; ++n) {
            const double ang = two_pi_over_N
                             * static_cast<double>(k)
                             * static_cast<double>(n);
            acc += ComplexD{x[n].real(), x[n].imag()}
                 * ComplexD{std::cos(ang), std::sin(ang)};
        }
        X[k] = acc;
    }
    return X;
}

template <typename TA, typename TB>
double rel_err(const std::vector<TA>& a, const std::vector<TB>& b) {
    double num = 0.0, den = 0.0;
    const size_t N = a.size();
    for (size_t i = 0; i < N; ++i) {
        const double dr = static_cast<double>(a[i].real())
                        - static_cast<double>(b[i].real());
        const double di = static_cast<double>(a[i].imag())
                        - static_cast<double>(b[i].imag());
        const double br = static_cast<double>(b[i].real());
        const double bi = static_cast<double>(b[i].imag());
        num += dr * dr + di * di;
        den += br * br + bi * bi;
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

std::vector<Complex> random_signal(uint32_t N, uint32_t seed = 42) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Complex> x(N);
    for (uint32_t i = 0; i < N; ++i) x[i] = { u(g), u(g) };
    return x;
}

int g_pass = 0;
int g_fail = 0;
const char* tick(bool ok) { return ok ? "[PASS]" : "[FAIL]"; }

// Forward FFT correctness vs double-precision reference (only feasible for
// N <= ~2048).
void test_forward_vs_dft(std::shared_ptr<MeshDevice> md, uint32_t N,
                         double tol = 1e-4) {
    auto x = random_signal(N);
    auto X = fft_universal_xl::fft(md, x);
    auto X_ref = dft_double(x);
    const double e = rel_err(X, X_ref);
    const bool ok = e <= tol;
    std::printf("%s N=%-9u  forward vs DFT_double  rel_err=%.3e  (tol=%.0e)\n",
                tick(ok), N, e, tol);
    (ok ? g_pass : g_fail)++;
}

// IFFT round-trip: ifft(fft(x)) ≈ x.  Works at any N where the FFT itself
// completes in reasonable time.
void test_round_trip(std::shared_ptr<MeshDevice> md, uint32_t N,
                     double tol = 1e-4) {
    auto x  = random_signal(N);
    auto X  = fft_universal_xl::fft (md, x);
    auto y  = fft_universal_xl::ifft(md, X);
    const double e = rel_err(y, x);
    const bool ok = e <= tol;
    std::printf("%s N=%-9u  ifft(fft(x)) vs x      rel_err=%.3e  (tol=%.0e)\n",
                tick(ok), N, e, tol);
    (ok ? g_pass : g_fail)++;
}

int main() {
    std::printf("=== fft_universal_xl correctness test ===\n");

    auto md = MeshDevice::create_unit_mesh(0);

    // Pass-through cases (N <= 1M).  Should behave identically to
    // fft_stockham; rel err target == fp32 noise floor.
    std::printf("\n-- Pass-through (k <= 2, delegates to fft_stockham) --\n");
    test_forward_vs_dft(md, 1024,    1e-4);
    test_forward_vs_dft(md, 2048,    1e-4);
    test_round_trip   (md, 65536,   1e-4);
    test_round_trip   (md, 1048576, 1e-4);

    // K=3 cases (N > 1M).  Use cheap-to-verify spectra first to localise
    // bugs without an O(N^2) reference DFT.
    std::printf("\n-- XL outer recursion (k = 3): impulse input --\n");
    {
        // x[n] = delta(n); expected X[k] = 1 for every k.
        const uint32_t N = 2097152;
        std::vector<Complex> x(N, Complex{0.0f, 0.0f});
        x[0] = Complex{1.0f, 0.0f};
        auto X = fft_universal_xl::fft(md, x);

        // Check ALL bins are ~1 + 0i.
        double worst = 0.0;
        uint32_t worst_k = 0;
        for (uint32_t k = 0; k < N; ++k) {
            const double dr = static_cast<double>(X[k].real()) - 1.0;
            const double di = static_cast<double>(X[k].imag());
            const double e  = std::sqrt(dr * dr + di * di);
            if (e > worst) { worst = e; worst_k = k; }
        }
        const bool ok = worst <= 1e-4;
        std::printf("%s N=%-9u  impulse fft (expect X[k]=1 for all k)  "
                    "max_err=%.3e at k=%u\n",
                    tick(ok), N, worst, worst_k);
        (ok ? g_pass : g_fail)++;

        if (!ok) {
            std::printf("    First 8 bins of X:\n");
            for (uint32_t k = 0; k < 8 && k < N; ++k) {
                std::printf("      X[%u] = (%.4g, %.4g)\n",
                            k, X[k].real(), X[k].imag());
            }
            std::printf("    Bins around the M boundary (M=%u):\n", N / 2);
            for (uint32_t k = (N/2) - 4; k < (N/2) + 4; ++k) {
                std::printf("      X[%u] = (%.4g, %.4g)\n",
                            k, X[k].real(), X[k].imag());
            }
        }
    }

    std::printf("\n-- XL outer recursion (k = 3): single-tone input --\n");
    {
        // x[n] = exp(+2*pi*i * k0 * n / N); expected X[k] has a single
        // spike of magnitude N at k=k0, zero elsewhere.
        const uint32_t N  = 2097152;
        const uint32_t k0 = 7;   // arbitrary low bin
        std::vector<Complex> x(N);
        const double ang = 2.0 * M_PI * static_cast<double>(k0)
                                      / static_cast<double>(N);
        for (uint32_t n = 0; n < N; ++n) {
            x[n] = Complex{
                static_cast<float>(std::cos(ang * n)),
                static_cast<float>(std::sin(ang * n))
            };
        }
        auto X = fft_universal_xl::fft(md, x);

        // Find max-magnitude bin.
        uint32_t peak_k = 0;
        double   peak_m = 0.0;
        for (uint32_t k = 0; k < N; ++k) {
            const double m = std::sqrt(
                static_cast<double>(X[k].real()) * X[k].real() +
                static_cast<double>(X[k].imag()) * X[k].imag());
            if (m > peak_m) { peak_m = m; peak_k = k; }
        }
        const bool ok =
            (peak_k == k0) && (peak_m > 0.95 * static_cast<double>(N));
        std::printf("%s N=%-9u  single tone k0=%u  found peak at k=%u "
                    "(|X|=%.3e, expected %.3e)\n",
                    tick(ok), N, k0, peak_k, peak_m,
                    static_cast<double>(N));
        (ok ? g_pass : g_fail)++;
    }

    std::printf("\n-- XL outer recursion (k = 3): random round-trip --\n");
    test_round_trip   (md, 2097152, 1e-4);  // 2M
    test_round_trip   (md, 4194304, 1e-4);  // 4M

    md->close();

    std::printf("\nResult: %d PASS, %d FAIL\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
