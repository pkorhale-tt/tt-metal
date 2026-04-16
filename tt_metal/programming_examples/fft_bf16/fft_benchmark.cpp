// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_benchmark.cpp (bf16) — demonstrates the impact of the plan cache
// on the bfloat16 FFT path.
//
// Usage:
//   ./build/programming_examples/fft_bf16/metal_example_fft_bf16_benchmark [N] [iters]

#include "fft_host.cpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const uint32_t N     = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 4096u;
    const uint32_t iters = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 100u;

    if ((N & (N - 1)) != 0 || N < 2 || N > 65536) {
        std::fprintf(stderr, "N must be a power of two in [2, 65536], got %u\n", N);
        return 1;
    }

    std::printf("\n=== FFT (bf16) benchmark: N=%u, iterations=%u ===\n\n", N, iters);

    auto md = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> signal(N);
    for (auto& v : signal) v = {dist(rng), dist(rng)};

    std::vector<double> times_ms;
    times_ms.reserve(iters);

    for (uint32_t i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = fft_example_bf16::fft(md, signal);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);

        if (i < 5 || i >= iters - 3) {
            std::printf("  iter %3u  %8.3f ms%s\n",
                        i, ms, (i == 0) ? "   <- includes plan build + JIT"
                             : (i == 1) ? "   <- cached"
                             : "");
        } else if (i == 5) {
            std::printf("  ...\n");
        }
        (void)out;
    }

    double sum_cached = 0, min_c = 1e18, max_c = 0;
    for (uint32_t i = 1; i < iters; ++i) {
        sum_cached += times_ms[i];
        if (times_ms[i] < min_c) min_c = times_ms[i];
        if (times_ms[i] > max_c) max_c = times_ms[i];
    }
    const double avg_cached = (iters > 1) ? sum_cached / (iters - 1) : 0.0;

    std::printf("\n--- Summary ---\n");
    std::printf("  First call (build):   %.3f ms\n", times_ms.front());
    if (iters > 1) {
        std::printf("  Cached avg (iters 1..%u): %.3f ms\n", iters - 1, avg_cached);
        std::printf("  Cached min / max:     %.3f / %.3f ms\n", min_c, max_c);
        std::printf("  Speedup of cached vs. first call: %.1fx\n",
                    times_ms.front() / avg_cached);
        std::printf("  Total wall: %.3f ms  (without cache would be ~%.0f ms)\n",
                    times_ms.front() + sum_cached,
                    times_ms.front() * iters);
    }
    std::printf("\n");

    return 0;
}
