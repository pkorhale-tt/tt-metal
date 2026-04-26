// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_demo.cpp — minimal usage example for ANY-N FFT.
//
// Shows the public API and prints a few output bins to sanity-check the
// result on a pure tone. Default N here is 1000 (= 8 * 125, exercises the
// Cooley-Tukey recursion), but any N >= 2 works.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_host.cpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

int main(int argc, char** argv) {
    const uint32_t N    = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 1000u;
    const uint32_t k_in = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 5u;

    auto md = MeshDevice::create_unit_mesh(0);

    // Build a clean test signal: pure tone at bin k_in of an N-point FFT.
    std::vector<Complex> signal(N);
    for (uint32_t n = 0; n < N; ++n) {
        const double angle = 2.0 * M_PI * static_cast<double>(k_in) *
                             static_cast<double>(n) / static_cast<double>(N);
        signal[n] = {static_cast<float>(std::cos(angle)),
                     static_cast<float>(std::sin(angle))};
    }

    std::printf("Running universal FFT on N=%u pure tone at bin %u...\n",
                N, k_in);

    const std::vector<Complex> X = fft_universal::fft(md, signal);

    std::printf(
        "\nTop bins (expected: a single spike of magnitude N=%u at bin %u):\n",
        N, k_in);
    for (int delta = -2; delta <= 2; ++delta) {
        const int idx = static_cast<int>(k_in) + delta;
        if (idx < 0 || idx >= static_cast<int>(N)) continue;
        std::printf("  X[%d] = (%+.3f, %+.3f)   |X|=%.3f\n",
                    idx, X[idx].real(), X[idx].imag(), std::abs(X[idx]));
    }

    md.reset();
    return 0;
}
