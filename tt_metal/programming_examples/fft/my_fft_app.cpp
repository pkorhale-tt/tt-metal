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
    std::vector<float> signal = {10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f};

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
/*
command:
cmake --build build --target my_fft_app -j
ARCH_NAME=wormhole_b0 TT_METAL_CLEAR_JIT_CACHE=1 \
    ./build/programming_examples/fft/my_fft_app


*/