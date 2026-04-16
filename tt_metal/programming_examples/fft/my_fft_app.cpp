// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// my_fft_app.cpp — your playground for the tt-metal FFT.
//
// Just edit the signal inside main(), rebuild, run. Equivalent to:
//
//     import torch
//     signal = torch.tensor([10., 20., 30., 40.])
//     spectrum = torch.fft.fft(signal)
//
// Requirements:
//   * signal length N is a power of two, 2..65536.
//   * real or complex input both work (overloads below).
//
// Build & run (from tt-metal repo root):
//   cmake --build build --target my_fft_app -j
//   ARCH_NAME=wormhole_b0 TT_METAL_CLEAR_JIT_CACHE=1 \
//       ./build/programming_examples/fft/my_fft_app

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_host.cpp"   // brings in fft_example::fft(md, signal)

#include <cstdio>
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>

using namespace tt::tt_metal::distributed;
using namespace fft_example;
using Complex = std::complex<float>;

// Pretty-print a complex vector like torch does.
static void print_spectrum(const char* label, const std::vector<Complex>& X) {
    std::printf("\n%s  (N=%zu)\n", label, X.size());
    for (size_t k = 0; k < X.size(); ++k) {
        std::printf("  X[%3zu] = %+11.4f %+11.4fj\n",
                    k, X[k].real(), X[k].imag());
    }
}

int main() {
    // One Tenstorrent device, picked automatically.
    auto md = MeshDevice::create_unit_mesh(0);

    // ─────────────────────────────────────────────────────────────────────
    //  EDIT BELOW:  put your input signal here.
    // ─────────────────────────────────────────────────────────────────────

    // Example 1: real input — exactly the PyTorch snippet
    //   torch.fft.fft(torch.tensor([10., 20., 30., 40.]))
    // std::vector<float> signal = {10.f, 20.f, 30.f, 40.f};
    std::vector<float> signal = {
        10.f, 20.f, 30.f, 40.f,
        73.4f, 12.8f, 55.1f, 88.6f, 34.2f, 67.9f, 21.5f, 49.3f, 95.7f, 8.4f,
        62.1f, 37.8f, 84.5f, 16.3f, 71.2f, 43.9f, 58.7f, 29.4f, 91.6f, 5.2f,
        76.8f, 48.1f, 23.7f, 69.4f, 14.6f, 82.3f, 57.9f, 31.5f, 96.2f, 44.8f,
        18.7f, 63.5f, 39.1f, 85.4f, 27.6f, 72.3f, 51.8f, 9.7f,  66.4f, 33.2f,
        78.9f, 45.6f, 22.3f, 87.1f, 54.7f, 11.9f, 68.5f, 36.2f, 93.8f, 25.4f,
        59.7f, 42.1f, 77.6f, 15.3f, 83.9f, 50.5f, 28.8f, 74.2f, 41.6f, 97.3f,
        6.8f,  64.5f, 32.9f, 79.6f, 47.3f, 24.1f, 89.7f, 56.4f, 13.2f, 70.8f,
        38.5f, 85.1f, 52.7f, 19.4f, 75.3f, 43.8f, 61.5f, 30.2f, 94.6f, 7.9f,
        67.3f, 35.1f, 81.8f, 48.4f, 26.7f, 72.9f, 40.3f, 58.6f, 17.2f, 84.7f,
        53.4f, 21.8f, 76.5f, 44.1f, 92.8f, 29.5f, 65.2f, 33.7f, 80.4f, 11.6f,
        57.3f, 25.9f, 71.6f, 39.2f, 86.9f, 54.5f, 18.1f, 63.8f, 31.4f, 78.1f,
        46.7f, 23.4f, 88.2f, 55.8f, 12.5f, 69.1f, 37.7f, 84.4f, 52.1f, 20.6f,
        75.8f, 43.4f, 61.1f, 28.7f, 93.4f, 8.1f,  66.7f, 34.4f, 81.1f, 48.7f,
        26.3f, 72.1f, 40.7f, 58.4f, 16.9f, 83.6f, 51.2f, 19.8f, 74.5f, 42.2f,
        97.8f, 65.4f, 33.1f, 79.8f, 47.4f, 24.8f, 90.5f, 57.1f, 14.7f, 71.4f,
        39.8f, 86.5f, 53.2f, 21.9f, 76.6f, 44.2f, 62.9f, 30.5f, 95.2f, 9.8f,
        68.4f, 36.1f, 82.8f, 50.4f, 27.1f, 73.8f, 41.4f, 59.1f, 17.7f, 85.4f,
        53.8f, 22.4f, 77.1f, 45.7f, 93.1f, 30.8f, 66.4f, 34.8f, 81.5f, 12.1f,
        58.8f, 26.4f, 72.5f, 40.1f, 87.8f, 55.4f, 19.1f, 64.8f, 32.4f, 79.1f,
        47.8f, 24.5f, 89.2f, 56.8f, 13.5f, 70.1f, 38.8f, 85.5f, 53.1f, 21.7f
    };

    auto spectrum = fft(md, signal);
    print_spectrum("FFT of [10, 20, 30, 40]", spectrum);

    // ─────────────────────────────────────────────────────────────────────
    //  More templates you can uncomment:
    // ─────────────────────────────────────────────────────────────────────

    /*
    // Complex input
    std::vector<Complex> cx = {
        { 1.0f,  0.0f}, { 0.0f,  1.0f}, {-1.0f,  0.0f}, { 0.0f, -1.0f},
    };
    auto spec_cx = fft(md, cx);
    print_spectrum("FFT of [1, j, -1, -j]", spec_cx);
    */

    /*
    // N=8 cosine  (torch.cos(2*pi*n/8).fft()  =>  peaks at bins 1 and 7)
    constexpr uint32_t N = 8;
    std::vector<float> sig(N);
    for (uint32_t n = 0; n < N; ++n) {
        sig[n] = std::cos(2.0f * M_PIf * static_cast<float>(n) /
                          static_cast<float>(N));
    }
    auto spec = fft(md, sig);
    print_spectrum("FFT of cos(2*pi*n/8)", spec);
    */

    /*
    // Large N on the full 8x8 grid. Grid layout & stage count are auto-picked
    // and printed by run_fft().
    constexpr uint32_t N = 65536;
    std::vector<float> big(N);
    for (uint32_t n = 0; n < N; ++n) big[n] = static_cast<float>(n) / N;
    auto big_spec = fft(md, big);
    std::printf("N=%u  X[0]=%+f %+fj\n", N, big_spec[0].real(), big_spec[0].imag());
    */

    md.reset();
    return 0;
}
