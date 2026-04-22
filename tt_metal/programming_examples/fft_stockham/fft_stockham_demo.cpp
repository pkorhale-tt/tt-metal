// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_stockham_demo.cpp — minimal usage example.
//
// Shows the public API for the Stockham FFT and prints a few output bins
// to sanity-check the result. This is the simplest possible "how do I call
// it" template — copy and edit for your own signal.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_stockham_host.cpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

int main() {
    auto md = MeshDevice::create_unit_mesh(0);

    // Build a clean test signal: pure tone at bin 5 of an N=131072 FFT.
    constexpr uint32_t N = 131072u;
    constexpr uint32_t k_in = 5u;
    std::vector<Complex> signal(N);
    for (uint32_t n = 0; n < N; ++n) {
        const double angle = 2.0 * M_PI * static_cast<double>(k_in) *
                             static_cast<double>(n) / static_cast<double>(N);
        signal[n] = {static_cast<float>(std::cos(angle)),
                     static_cast<float>(std::sin(angle))};
    }

    std::printf("Running Stockham FFT on N=%u pure tone at bin %u...\n", N, k_in);

    const std::vector<Complex> X = fft_stockham::fft(md, signal);

    std::printf("\nTop bins (expected: a single spike of magnitude N=%u at bin %u):\n", N, k_in);
    for (int delta = -2; delta <= 2; ++delta) {
        const int idx = static_cast<int>(k_in) + delta;
        if (idx < 0 || idx >= static_cast<int>(N)) continue;
        std::printf("  X[%d] = (%+.3f, %+.3f)   |X|=%.3f\n",
                    idx, X[idx].real(), X[idx].imag(), std::abs(X[idx]));
    }

    md.reset();
    return 0;
}
