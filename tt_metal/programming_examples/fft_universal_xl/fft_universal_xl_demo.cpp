// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_demo.cpp — Minimal "how to call it" example.
//
// Usage:
//   metal_example_fft_universal_xl_demo <N>
//
//   N defaults to 2097152 (2M) — the smallest size that exercises the
//   K=3 outer-recursion path. Smaller N just falls through to
//   fft_stockham.

#include "fft_universal_xl_host.cpp"

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Complex = std::complex<float>;
using tt::tt_metal::distributed::MeshDevice;

int main(int argc, char** argv) {
    uint32_t N = (argc > 1) ? static_cast<uint32_t>(std::atoll(argv[1]))
                            : 2097152u;  // default: 2M (smallest K=3 size)

    if ((N & (N - 1u)) != 0u || N < 2u) {
        std::fprintf(stderr,
            "Error: N must be a power of two and >= 2.  Got %u.\n", N);
        return 1;
    }

    std::printf("=== fft_universal_xl demo: N=%u ===\n", N);
    std::printf("    plan: ");
    auto p = fft_universal_xl::plan(N);
    std::printf("k=%u  factors=[", p.k());
    for (size_t i = 0; i < p.factors.size(); ++i) {
        std::printf("%s%u", (i ? ", " : ""), p.factors[i]);
    }
    std::printf("]\n");

    // Build a single-tone test signal so we can eyeball correctness:
    // x[n] = cos(2*pi * k0 * n / N), expected spectrum has spikes at
    // k=k0 and k=N-k0 of magnitude N/2.
    const uint32_t k0 = 7;  // arbitrary low bin
    std::vector<Complex> x(N);
    const double two_pi_k0_over_N =
        2.0 * M_PI * static_cast<double>(k0) / static_cast<double>(N);
    for (uint32_t n = 0; n < N; ++n) {
        x[n] = Complex{
            static_cast<float>(std::cos(two_pi_k0_over_N
                                        * static_cast<double>(n))),
            0.0f
        };
    }

    auto md = MeshDevice::create_unit_mesh(0);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto X = fft_universal_xl::fft(md, x);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto y = fft_universal_xl::ifft(md, X);
    auto t3 = std::chrono::high_resolution_clock::now();

    const double fft_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ifft_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Find peak bin to confirm correctness.
    uint32_t  peak_bin = 0;
    float     peak_mag = 0.0f;
    for (uint32_t k = 0; k < N; ++k) {
        const float m = std::abs(X[k]);
        if (m > peak_mag) { peak_mag = m; peak_bin = k; }
    }

    // Round-trip error.
    double num = 0.0, den = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        const double dr = static_cast<double>(y[i].real() - x[i].real());
        const double di = static_cast<double>(y[i].imag() - x[i].imag());
        num += dr * dr + di * di;
        den += static_cast<double>(x[i].real()) * x[i].real()
             + static_cast<double>(x[i].imag()) * x[i].imag();
    }
    const double rel_err = std::sqrt(num / std::max(den, 1e-30));

    std::printf("\n--- Result ---\n");
    std::printf("  FFT  time    : %.2f ms\n", fft_ms);
    std::printf("  IFFT time    : %.2f ms\n", ifft_ms);
    std::printf("  Expected peak: bin %u  (and %u via Hermitian symmetry)\n",
                k0, N - k0);
    std::printf("  Found peak   : bin %u   |X|=%g   (expected ~%.0f)\n",
                peak_bin, peak_mag, static_cast<double>(N) / 2.0);
    std::printf("  Round-trip   : rel_err=%.3e\n", rel_err);

    md->close();
    return 0;
}
