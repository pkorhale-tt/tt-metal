// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_benchmark.cpp — end-to-end timing for the TRUE-bf16
// dispatch tree. Accepts any N supported by the current fft() dispatcher
// (Phase 1: N in [2, 32]; Phase 2a: pow2 N in [64, 1024]).
//
// Reports:
//   * Cold (plan build + JIT) and cached/steady-state per-call latency
//   * Cached speedup vs cold
//   * Total wall time actually observed
//
// Not reported yet (Phase 2b/2c): CPU baseline comparison, fp32
// fft_universal cached timing side-by-side, SNR-vs-iteration sweep.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_bf16_host.cpp"

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

static std::vector<Complex> make_random(uint32_t N, uint32_t seed = 1234) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x) {
        c = {(std::rand() / float(RAND_MAX)) * 2.0f - 1.0f,
             (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f};
    }
    return x;
}

int main(int argc, char** argv) {
    const uint32_t N     = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 32u;
    const uint32_t iters = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 100u;

    auto is_pow2 = [](uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; };
    const bool in_phase1 = (N >= 2u && N <= fft_universal_bf16::kPackedMaxN);
    const bool in_phase2a = (is_pow2(N) && N >= 64u && N <= 1024u);
    if (!in_phase1 && !in_phase2a) {
        std::fprintf(stderr,
            "fft_universal_bf16 currently supports N in [2, 32] (Phase 1) "
            "and pow2 N in [64, 1024] (Phase 2a). Got N=%u.\n"
            "Phase 2b will cover pow2 N > 1024; Phase 2c will cover "
            "primes (Bluestein) and composite non-pow2 (mixed-radix).\n", N);
        return 2;
    }

    auto md = MeshDevice::create_unit_mesh(0);
    const std::vector<Complex> signal = make_random(N);

    std::printf("Benchmarking fft_universal_bf16::fft (TRUE bf16, FPU matmul):\n"
                "  N=%u, iters=%u\n\n", N, iters);

    std::vector<double> ms(iters);
    const auto t_all0 = std::chrono::high_resolution_clock::now();
    for (uint32_t it = 0; it < iters; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto X  = fft_universal_bf16::fft(md, signal);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ms[it] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        (void)X;
        if (it < 5 || it == iters - 1) {
            std::printf("  iter %3u  %8.3f ms  %s\n",
                it, ms[it],
                it == 0u ? "<- includes plan build + JIT"
                         : "<- cached");
        } else if (it == 5) {
            std::printf("  ...\n");
        }
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_all0).count();

    const double cold = ms.front();
    const double warm_sum = std::accumulate(ms.begin() + 1, ms.end(), 0.0);
    const double warm_avg = (iters > 1) ? warm_sum / (iters - 1) : cold;
    const double warm_min = (iters > 1) ? *std::min_element(ms.begin() + 1, ms.end()) : cold;
    const double warm_max = (iters > 1) ? *std::max_element(ms.begin() + 1, ms.end()) : cold;

    std::printf("\n  --- Summary ---\n");
    std::printf("  First call (build):        %8.3f ms\n", cold);
    std::printf("  Cached avg (iters 1..%u):  %8.3f ms\n", iters - 1, warm_avg);
    std::printf("  Cached min / max:          %8.3f / %8.3f ms\n", warm_min, warm_max);
    std::printf("  Speedup of cached vs cold: %.1fx\n", cold / std::max(warm_avg, 1e-9));
    std::printf("  Total wall time:           %8.3f ms\n", total_ms);

    md.reset();
    return 0;
}
