// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_demo.cpp — minimal usage example for the TRUE-bf16
// dispatch tree. Defaults to N=32 (Phase 1). Supports N in [2, 32]
// (Phase 1, single-pass) and pow2 N in [64, 1024] (Phase 2a, two-pass CT).
// Feeds a pure tone at bin k_in and prints the top output bins — expect
// a clean spike of magnitude N at bin k_in.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_bf16_host.cpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

int main(int argc, char** argv) {
    const uint32_t N    = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 32u;
    const uint32_t k_in = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 5u;

    auto is_pow2 = [](uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; };
    const bool in_phase1 = (N >= 2u && N <= fft_universal_bf16::kPackedMaxN);
    const bool in_phase2a = (is_pow2(N) && N >= 64u && N <= 1024u);
    if (!in_phase1 && !in_phase2a) {
        std::fprintf(stderr,
            "fft_universal_bf16 currently supports N in [2, 32] (Phase 1) "
            "and pow2 N in [64, 1024] (Phase 2a). Got N=%u.\n", N);
        return 2;
    }

    auto md = MeshDevice::create_unit_mesh(0);

    std::vector<Complex> signal(N);
    for (uint32_t n = 0; n < N; ++n) {
        const double angle = 2.0 * M_PI * static_cast<double>(k_in) *
                             static_cast<double>(n) / static_cast<double>(N);
        signal[n] = {static_cast<float>(std::cos(angle)),
                     static_cast<float>(std::sin(angle))};
    }

    std::printf("Running TRUE-bf16 FFT on N=%u pure tone at bin %u...\n",
                N, k_in);

    const std::vector<Complex> X = fft_universal_bf16::fft(md, signal);

    std::printf("\nTop bins (expected: a spike of magnitude N=%u at bin %u;\n"
                "bf16 precision: ~40-45 dB SNR, so neighbouring bins are\n"
                "non-zero at the ~10^-2 relative level):\n", N, k_in);
    for (int delta = -2; delta <= 2; ++delta) {
        const int idx = static_cast<int>(k_in) + delta;
        if (idx < 0 || idx >= static_cast<int>(N)) continue;
        std::printf("  X[%d] = (%+.3f, %+.3f)   |X|=%.3f\n",
                    idx, X[idx].real(), X[idx].imag(), std::abs(X[idx]));
    }

    md.reset();
    return 0;
}
