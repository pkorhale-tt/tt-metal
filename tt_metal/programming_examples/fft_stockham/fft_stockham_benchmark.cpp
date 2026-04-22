// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_stockham_benchmark.cpp
//
// End-to-end host-to-device-to-host latency measurement for the Stockham
// orchestrator. Mirrors the inner radix-2 benchmark exactly so the two
// numbers can be compared apples-to-apples.
//
// Usage:
//     metal_example_fft_stockham_benchmark [N] [iterations]
// Defaults: N = 131072, iterations = 100.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_stockham_host.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

static std::vector<Complex> make_random(uint32_t N, uint32_t seed = 42) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x) {
        c = {(std::rand() / float(RAND_MAX)) * 2.0f - 1.0f,
             (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f};
    }
    return x;
}

int main(int argc, char** argv) {
    uint32_t N    = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 131072u;
    uint32_t iter = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 100u;

    if (N < 2 || (N & (N - 1)) != 0) {
        std::printf("N must be a power of two and >= 2 (got %u)\n", N);
        return 1;
    }
    if (iter < 2) iter = 2;

    std::printf("\n=== FFT (Stockham) benchmark: N=%u, iterations=%u ===\n\n",
                N, iter);

    auto md     = MeshDevice::create_unit_mesh(0);
    auto signal = make_random(N);

    std::vector<double> dt(iter, 0.0);

    for (uint32_t i = 0; i < iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto X = fft_stockham::fft(md, signal);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        dt[i] = ms;

        if (i == 0) {
            std::printf("  iter %3u  %8.3f ms   <- includes plan build + JIT\n",
                        i, ms);
        } else if (i == 1) {
            std::printf("  iter %3u  %8.3f ms   <- cached\n", i, ms);
        } else if (i < 5 || i == iter - 1) {
            std::printf("  iter %3u  %8.3f ms\n", i, ms);
        } else if (i == 5) {
            std::printf("  ...\n");
        }
    }

    const double cold = dt[0];
    const double cached_sum =
        std::accumulate(dt.begin() + 1, dt.end(), 0.0);
    const double cached_avg = cached_sum / static_cast<double>(iter - 1);
    const double cached_min =
        *std::min_element(dt.begin() + 1, dt.end());
    const double cached_max =
        *std::max_element(dt.begin() + 1, dt.end());
    const double total      =
        std::accumulate(dt.begin(), dt.end(), 0.0);

    std::printf("\n--- Summary ---\n");
    std::printf("  First call (build):       %8.3f ms\n", cold);
    std::printf("  Cached avg (iters 1..%u): %8.3f ms\n", iter - 1, cached_avg);
    std::printf("  Cached min / max:         %8.3f / %8.3f ms\n",
                cached_min, cached_max);
    std::printf("  Speedup of cached vs. first call: %.1fx\n",
                cold / cached_avg);
    std::printf("  Total wall: %.3f ms  (without cache would be ~%.0f ms)\n\n",
                total, cold * static_cast<double>(iter));

    md.reset();
    return 0;
}
