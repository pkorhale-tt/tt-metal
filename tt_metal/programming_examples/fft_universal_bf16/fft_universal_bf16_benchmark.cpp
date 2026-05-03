// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_benchmark.cpp — end-to-end timing for the TRUE-bf16
// dispatch tree. Accepts ANY N ≥ 2 supported by the fft() dispatcher:
//   * Phase 1   : N in [2, 32]                   (packed direct-DFT bf16)
//   * Phase 2b  : pow2 N > 32                    (recursive CT 32x…)
//                 composite N > 32 w/ ÷ ≤ 32     (mixed-radix CT)
//   * Phase 2c  : prime / hard-composite N > 32  (Bluestein)
//
// Reports:
//   * Cold (plan build + JIT) and cached/steady-state per-call latency
//   * Cached speedup vs cold
//   * Total wall time actually observed

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

    if (N < 2u) {
        std::fprintf(stderr, "fft_universal_bf16 requires N >= 2. Got N=%u.\n", N);
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

    // ─────────────────────────────────────────────────────────────────────
    // IFFT timing pass. Conjugate-trick IFFT internally calls fft(); all
    // PackedDFTBf16Plan entries are already cached from the forward loop
    // above, so the first iter only sees the small host conj/scale cost
    // on top of a steady-state forward call.
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n=== IFFT (universal_bf16, conjugate trick) ===\n");
    const auto spectrum = fft_universal_bf16::fft(md, signal);
    std::vector<double> ims(iters);
    for (uint32_t it = 0; it < iters; ++it) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto x  = fft_universal_bf16::ifft(md, spectrum);
        ims[it] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        (void)x;
        if (it < 3 || it == iters - 1) {
            std::printf("  iter %3u  %8.3f ms\n", it, ims[it]);
        } else if (it == 3) {
            std::printf("  ...\n");
        }
    }
    const double ifft_avg =
        std::accumulate(ims.begin() + 1, ims.end(), 0.0)
            / static_cast<double>(std::max(iters - 1, 1u));
    const double ifft_min = (iters > 1) ? *std::min_element(ims.begin() + 1, ims.end()) : ims[0];
    const double ifft_max = (iters > 1) ? *std::max_element(ims.begin() + 1, ims.end()) : ims[0];
    std::printf("\n  --- IFFT summary ---\n");
    std::printf("  Cached avg (iters 1..%u):  %8.3f ms\n", iters - 1, ifft_avg);
    std::printf("  Cached min / max:          %8.3f / %8.3f ms\n", ifft_min, ifft_max);
    std::printf("  IFFT/FFT ratio:            %.2fx\n", ifft_avg / std::max(warm_avg, 1e-9));

    // Round-trip sanity check.
    {
        const auto rt = fft_universal_bf16::ifft(md, spectrum);
        double max_in = 0.0, max_e = 0.0;
        for (size_t i = 0; i < signal.size(); ++i) {
            max_in = std::max<double>(max_in, std::abs(signal[i]));
            max_e  = std::max<double>(max_e,  std::abs(rt[i] - signal[i]));
        }
        std::printf("  Round-trip rel err:        %8.2e   (ifft(fft(x)) vs x)\n",
                    max_e / std::max(1e-30, max_in));
    }

    md.reset();
    return 0;
}
