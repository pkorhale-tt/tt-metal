// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_demo.cpp (bf16) — minimal PyTorch-style usage of the bf16 FFT.
//
// Equivalent PyTorch:
//     import torch
//     signal      = torch.tensor([10., 20., 30., 40.], dtype=torch.bfloat16)
//     fft_result  = torch.fft.fft(signal.float()).to(torch.bfloat16)
//
// API:
//     std::vector<float> signal = {10.f, 20.f, 30.f, 40.f};
//     auto spectrum = fft_example_bf16::fft(md, signal);

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_host.cpp"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <complex>

using namespace tt::tt_metal::distributed;
using namespace fft_example_bf16;
using Complex = std::complex<float>;

static void print_spectrum(const char* label, const std::vector<Complex>& x) {
    std::printf("\n%s  (N=%zu, bf16)\n", label, x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        std::printf("  X[%3zu] = %+11.4f %+11.4fj\n",
                    k, x[k].real(), x[k].imag());
    }
}

int main() {
    auto md = MeshDevice::create_unit_mesh(0);

    {
        std::vector<float> signal = {10.f, 20.f, 30.f, 40.f};
        auto spectrum = fft(md, signal);
        print_spectrum("FFT of [10, 20, 30, 40]", spectrum);
    }

    {
        std::vector<Complex> signal = {
            { 1.0f,  0.0f},
            { 0.0f,  1.0f},
            {-1.0f,  0.0f},
            { 0.0f, -1.0f},
        };
        auto spectrum = fft(md, signal);
        print_spectrum("FFT of [1, j, -1, -j]  (rotating phasor)", spectrum);
    }

    {
        constexpr uint32_t N = 8;
        std::vector<float> signal(N);
        for (uint32_t n = 0; n < N; ++n) {
            signal[n] = std::cos(2.0f * M_PIf * static_cast<float>(n) /
                                 static_cast<float>(N));
        }
        auto spectrum = fft(md, signal);
        print_spectrum("FFT of cos(2*pi*n/8)  (N=8)", spectrum);
    }

    {
        constexpr uint32_t N = 1024;
        std::vector<float> signal(N);
        for (uint32_t n = 0; n < N; ++n) signal[n] = static_cast<float>(n);
        auto spectrum = fft(md, signal);
        std::printf("\nFFT of ramp [0..1023]  (N=%u, bf16)\n", N);
        std::printf("  X[0]   = %+11.4f %+11.4fj   (DC, expected %.4f)\n",
                    spectrum[0].real(), spectrum[0].imag(),
                    (N - 1.0f) * N / 2.0f);
        std::printf("  X[1]   = %+11.4f %+11.4fj\n",
                    spectrum[1].real(), spectrum[1].imag());
        std::printf("  X[N/2] = %+11.4f %+11.4fj\n",
                    spectrum[N / 2].real(), spectrum[N / 2].imag());
    }

    md.reset();
    std::printf("\nDone.\n");
    return 0;
}
