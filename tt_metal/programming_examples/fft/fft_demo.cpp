// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_demo.cpp — minimal PyTorch-style usage of the tt-metal FFT.
//
// Equivalent PyTorch:
//     import torch
//     signal      = torch.tensor([10., 20., 30., 40.])
//     fft_result  = torch.fft.fft(signal)
//     print(fft_result)
//
// Here:
//     std::vector<float> signal = {10.f, 20.f, 30.f, 40.f};
//     auto spectrum = fft_example::fft(md, signal);
//     for (size_t k = 0; k < spectrum.size(); ++k) {
//         std::printf("X[%zu] = %+f %+fj\n", k, spectrum[k].real(), spectrum[k].imag());
//     }

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_host.cpp"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <complex>

using namespace tt::tt_metal::distributed;
using namespace fft_example;
using Complex = std::complex<float>;

static void print_spectrum(const char* label, const std::vector<Complex>& x) {
    std::printf("\n%s  (N=%zu)\n", label, x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        std::printf("  X[%3zu] = %+11.4f %+11.4fj\n",
                    k, x[k].real(), x[k].imag());
    }
}

int main() {
    auto md = MeshDevice::create_unit_mesh(0);

    // ── Example 1: exactly the PyTorch snippet, FFT of [10, 20, 30, 40] ──
    //
    //   torch.fft.fft(torch.tensor([10., 20., 30., 40.]))
    //     =>  tensor([100.+0.j, -20.+20.j, -20.+0.j, -20.-20.j])
    {
        std::vector<float> signal = {10.f, 20.f, 30.f, 40.f};
        auto spectrum = fft(md, signal);
        print_spectrum("FFT of [10, 20, 30, 40]", spectrum);
    }

    // ── Example 2: complex input ─────────────────────────────────────────
    //
    //   torch.fft.fft(torch.tensor([1+0j, 0+1j, -1+0j, 0-1j]))
    //     =>  tensor([0.+0.j, 0.+0.j, 0.+0.j, 4.+0.j])    (a pure +1 bin)
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

    // ── Example 3: 8-point cosine ────────────────────────────────────────
    //
    // cos(2*pi*k/N) over N=8 samples  =>  energy concentrated at bins 1 & 7
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

    // ── Example 4: larger FFT (shows P>1 kicks in automatically) ─────────
    //
    // FFT of [0, 1, 2, ..., 1023]. The DC bin (X[0]) is sum = 1023*1024/2.
    {
        constexpr uint32_t N = 1024;
        std::vector<float> signal(N);
        for (uint32_t n = 0; n < N; ++n) signal[n] = static_cast<float>(n);
        auto spectrum = fft(md, signal);
        std::printf("\nFFT of ramp [0..1023]  (N=%u)\n", N);
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
