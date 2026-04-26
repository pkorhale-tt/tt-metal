// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_benchmark.cpp
//
// End-to-end host-to-device-to-host latency for fft_universal::fft on ANY N,
// plus an in-process single-threaded CPU fp32 radix-2 baseline for apples-to-
// apples comparison on pow2 N. Report format mirrors Table 1 of Brown, Davies
// & Le Clair, "Exploring Fast Fourier Transforms on the Tenstorrent Wormhole"
// (arXiv:2506.15437).
//
// Usage:
//     metal_example_fft_universal_benchmark [N] [iterations]
// Defaults: N = 1000, iterations = 100. For a Table 1 replica run:
//     metal_example_fft_universal_benchmark 16384 100

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_host.cpp"

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

static const char* describe_path(uint32_t N) {
    if (N == 1u)                    return "identity";
    if (fft_universal::is_pow2(N))  return "pow2 pass-through (fft_stockham, Wormhole)";
    if (fft_universal::is_prime(N)) return "Bluestein on Wormhole (prime)";
    return "Cooley-Tukey split on Wormhole (composite non-pow2)";
}

// ─────────────────────────────────────────────────────────────────────────
// CPU baseline: single-threaded fp32 iterative radix-2 Cooley-Tukey.
//
// Honest comparison point. Twiddles and bit-reverse indices are built once
// per N and reused across iterations, exactly like the device plan cache so
// neither side is paying setup cost inside the timed region.
// ─────────────────────────────────────────────────────────────────────────
namespace cpu_fft {

struct Plan {
    uint32_t              N = 0;
    std::vector<Complex>  w;   // size N/2, w[k] = exp(-2*pi*i * k / N)
    std::vector<uint32_t> br;  // bit-reverse permutation, size N
};

static Plan make_plan(uint32_t N) {
    Plan p;
    p.N = N;
    p.w.resize(N / 2u);
    const double tau = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t k = 0; k < N / 2u; ++k) {
        p.w[k] = Complex(static_cast<float>(std::cos(tau * static_cast<double>(k))),
                         static_cast<float>(std::sin(tau * static_cast<double>(k))));
    }
    p.br.resize(N);
    uint32_t logN = 0;
    for (uint32_t t = N; t > 1u; t >>= 1) ++logN;
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t r = 0;
        uint32_t v = i;
        for (uint32_t b = 0; b < logN; ++b) { r = (r << 1) | (v & 1u); v >>= 1; }
        p.br[i] = r;
    }
    return p;
}

// In-place radix-2 DIT. Caller passes a fresh scratch buffer holding the input.
static void fft_inplace(const Plan& p, Complex* x) {
    const uint32_t N = p.N;
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t j = p.br[i];
        if (j > i) std::swap(x[i], x[j]);
    }
    for (uint32_t m = 2u; m <= N; m <<= 1) {
        const uint32_t m2   = m >> 1;
        const uint32_t step = N / m;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m2; ++j) {
                const Complex w = p.w[j * step];
                const Complex t = w * x[k + j + m2];
                const Complex u = x[k + j];
                x[k + j]      = u + t;
                x[k + j + m2] = u - t;
            }
        }
    }
}

}  // namespace cpu_fft

int main(int argc, char** argv) {
    uint32_t N    = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 1000u;
    uint32_t iter = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 100u;

    if (N < 2u) {
        std::printf("N must be >= 2 (got %u)\n", N);
        return 1;
    }
    if (iter < 2u) iter = 2u;

    std::printf("\n=== FFT (universal) benchmark: N=%u, iterations=%u ===\n",
                N, iter);
    std::printf("    dispatch path: %s\n\n", describe_path(N));

    auto md     = MeshDevice::create_unit_mesh(0);
    auto signal = make_random(N);

    std::vector<double> dt(iter, 0.0);

    for (uint32_t i = 0; i < iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto X = fft_universal::fft(md, signal);
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
    const double total =
        std::accumulate(dt.begin(), dt.end(), 0.0);

    std::printf("\n--- Wormhole summary ---\n");
    std::printf("  First call (build):       %8.3f ms\n", cold);
    std::printf("  Cached avg (iters 1..%u): %8.3f ms\n", iter - 1, cached_avg);
    std::printf("  Cached min / max:         %8.3f / %8.3f ms\n",
                cached_min, cached_max);
    std::printf("  Speedup of cached vs. first call: %.1fx\n",
                cold / cached_avg);
    std::printf("  Total wall: %.3f ms  (without cache would be ~%.0f ms)\n",
                total, cold * static_cast<double>(iter));

    // ─────────────────────────────────────────────────────────────────────
    // CPU baseline run (single-thread fp32 radix-2). Only meaningful for
    // pow2 N; non-pow2 prints N/A so the table stays honest.
    // ─────────────────────────────────────────────────────────────────────
    double cpu_cached_avg = 0.0;
    double cpu_cached_min = 0.0;
    double cpu_cached_max = 0.0;
    bool   cpu_ran        = false;

    if (fft_universal::is_pow2(N)) {
        std::printf("\n=== CPU baseline: single-thread fp32 radix-2 Cooley-Tukey ===\n");
        const auto plan_t0 = std::chrono::high_resolution_clock::now();
        auto cpu_plan      = cpu_fft::make_plan(N);
        const double plan_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - plan_t0).count();
        std::printf("    plan build (twiddles + bit-reverse): %.3f ms  (outside timed region)\n\n",
                    plan_ms);

        std::vector<Complex> buf(N);
        std::vector<double>  cpu_dt(iter, 0.0);

        for (uint32_t i = 0; i < iter; ++i) {
            std::copy(signal.begin(), signal.end(), buf.begin());
            const auto t0 = std::chrono::high_resolution_clock::now();
            cpu_fft::fft_inplace(cpu_plan, buf.data());
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            cpu_dt[i] = ms;

            if (i == 0) {
                std::printf("  iter %3u  %8.3f ms   <- first call\n", i, ms);
            } else if (i == 1) {
                std::printf("  iter %3u  %8.3f ms   <- steady state\n", i, ms);
            } else if (i < 5 || i == iter - 1) {
                std::printf("  iter %3u  %8.3f ms\n", i, ms);
            } else if (i == 5) {
                std::printf("  ...\n");
            }
        }

        const double cpu_sum = std::accumulate(cpu_dt.begin() + 1, cpu_dt.end(), 0.0);
        cpu_cached_avg = cpu_sum / static_cast<double>(iter - 1);
        cpu_cached_min = *std::min_element(cpu_dt.begin() + 1, cpu_dt.end());
        cpu_cached_max = *std::max_element(cpu_dt.begin() + 1, cpu_dt.end());
        cpu_ran        = true;

        std::printf("\n--- CPU summary ---\n");
        std::printf("  Cached avg (iters 1..%u): %8.3f ms\n", iter - 1, cpu_cached_avg);
        std::printf("  Cached min / max:         %8.3f / %8.3f ms\n",
                    cpu_cached_min, cpu_cached_max);
    } else {
        std::printf("\n=== CPU baseline skipped: N=%u is not a power of two ===\n", N);
        std::printf("    (The in-process CPU FFT is radix-2 only. Use a pow2 N like\n");
        std::printf("     16384 / 65536 / 1048576 to reproduce Table 1 of the paper.)\n");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Paper-style head-to-head table.
    //   Mirrors Table 1 of arXiv:2506.15437 (1D FFT, single-precision,
    //   runtime only). Note: our Wormhole path uses many Tensix cores
    //   in parallel, whereas the paper's Table 1 is single-Tensix; see
    //   Table 3 of the paper for a whole-card vs whole-CPU comparison.
    // ─────────────────────────────────────────────────────────────────────
    std::printf("\n=== Paper-style comparison (Table 1 replica, arXiv:2506.15437) ===\n");
    std::printf("    Problem size : N = %u  (1D FFT, fp32 random complex)\n", N);
    std::printf("    Iterations   : %u  (reporting cached/steady-state avg)\n\n", iter);
    std::printf("    | %-44s | %-5s | %-13s |\n",
                "Version", "Cores", "Runtime (ms)");
    std::printf("    |%s|%s|%s|\n",
                "----------------------------------------------",
                "-------",
                "---------------");
    if (cpu_ran) {
        std::printf("    | %-44s | %5d | %13.3f |\n",
                    "CPU fp32 radix-2 (single-thread, in-process)",
                    1, cpu_cached_avg);
    } else {
        std::printf("    | %-44s | %5s | %13s |\n",
                    "CPU fp32 radix-2 (single-thread, in-process)",
                    "-", "N/A");
    }
    std::printf("    | %-44s | %5s | %13.3f |\n",
                "Wormhole fft_universal (cached, many Tensix)",
                "many", cached_avg);
    if (cpu_ran) {
        const double ratio = cached_avg / cpu_cached_avg;
        std::printf("\n    => CPU is %.2fx %s than Wormhole end-to-end.\n",
                    ratio >= 1.0 ? ratio : 1.0 / ratio,
                    ratio >= 1.0 ? "faster" : "slower");
        std::printf("       (Paper Table 1, single-Tensix, reported 2.8x CPU advantage;\n");
        std::printf("        our number includes PCIe + dispatch, not just kernel time.)\n");
    }
    std::printf("\n");

    md.reset();
    return 0;
}
