// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// my_fft_app.cpp (bf16) — your playground for the bfloat16 tt-metal FFT.
//
// Edit the signal inside main(), rebuild, run.
//
// Build & run (from tt-metal repo root):
//   cmake --build build --target my_fft_bf16_app -j
//   ARCH_NAME=wormhole_b0 TT_METAL_CLEAR_JIT_CACHE=1 \
//       ./build/programming_examples/fft_bf16/my_fft_bf16_app

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_host.cpp"   // brings in fft_example_bf16::fft(md, signal)

#include <cstdio>
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>

using namespace tt::tt_metal::distributed;
using namespace fft_example_bf16;
using Complex = std::complex<float>;

static void print_spectrum(const char* label, const std::vector<Complex>& X) {
    std::printf("\n%s  (N=%zu, bf16)\n", label, X.size());
    for (size_t k = 0; k < X.size(); ++k) {
        std::printf("  X[%3zu] = %+11.4f %+11.4fj\n",
                    k, X[k].real(), X[k].imag());
    }
}

int main() {
    auto md = MeshDevice::create_unit_mesh(0);

    // ─────────────────────────────────────────────────────────────────────
    //  EDIT BELOW:  put your input signal here.
    // ─────────────────────────────────────────────────────────────────────

    std::vector<float> signal = {10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f};

    auto spectrum = fft(md, signal);
    print_spectrum("FFT (bf16) of [10, 20, 30, 40, 50, 60, 70, 80]", spectrum);

    // ─────────────────────────────────────────────────────────────────────
    //  More templates you can uncomment:
    // ─────────────────────────────────────────────────────────────────────

    /*
    std::vector<Complex> cx = {
        { 1.0f,  0.0f}, { 0.0f,  1.0f}, {-1.0f,  0.0f}, { 0.0f, -1.0f},
    };
    auto spec_cx = fft(md, cx);
    print_spectrum("FFT (bf16) of [1, j, -1, -j]", spec_cx);
    */

    /*
    constexpr uint32_t N = 8;
    std::vector<float> sig(N);
    for (uint32_t n = 0; n < N; ++n) {
        sig[n] = std::cos(2.0f * M_PIf * static_cast<float>(n) /
                          static_cast<float>(N));
    }
    auto spec = fft(md, sig);
    print_spectrum("FFT (bf16) of cos(2*pi*n/8)", spec);
    */

    /*
    constexpr uint32_t N = 65536;
    std::vector<float> big(N);
    for (uint32_t n = 0; n < N; ++n) big[n] = static_cast<float>(n) / N;
    auto big_spec = fft(md, big);
    std::printf("N=%u  X[0]=%+f %+fj\n", N, big_spec[0].real(), big_spec[0].imag());
    */

    md.reset();
    return 0;
}
