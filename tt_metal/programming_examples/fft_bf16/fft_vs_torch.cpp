// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_vs_torch.cpp (bf16) — runs the bf16 tt-metal FFT and dumps the input
// signal and FFT output to plain-text files for the Python comparison
// script (compare_with_torch.py).
//
// Usage:
//     metal_example_fft_bf16_vs_torch <N> [seed] [in_path] [out_path]

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_host.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <cstdint>
#include <chrono>

using namespace tt::tt_metal::distributed;
using namespace fft_example_bf16;
using Complex = std::complex<float>;

static std::vector<Complex> make_random(uint32_t N, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<Complex> x(N);
    for (auto& c : x) c = Complex(d(rng), d(rng));
    return x;
}

static void write_complex_file(const std::string& path,
                               const std::vector<Complex>& v) {
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path.c_str());
        std::exit(1);
    }
    f.setf(std::ios::scientific);
    f.precision(9);
    for (const auto& c : v) f << c.real() << ' ' << c.imag() << '\n';
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <N> [seed] [in_path] [out_path]\n", argv[0]);
        return 1;
    }
    const uint32_t N    = static_cast<uint32_t>(std::atoi(argv[1]));
    const uint32_t seed = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 42u;
    const std::string in_path  = (argc > 3) ? argv[3] : "fft_input.txt";
    const std::string out_path = (argc > 4) ? argv[4] : "fft_output.txt";

    if (N < 2 || (N & (N - 1)) != 0 || N > 65536) {
        std::fprintf(stderr, "N must be a power of two in [2, 65536], got %u\n", N);
        return 1;
    }

    auto md = MeshDevice::create_unit_mesh(0);
    const auto input = make_random(N, seed);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto output = fft(md, input);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();

    write_complex_file(in_path,  input);
    write_complex_file(out_path, output);

    std::printf("[fft_bf16_vs_torch] wrote %s  and  %s\n",
                in_path.c_str(), out_path.c_str());
    std::printf("[fft_bf16_vs_torch] N=%u  seed=%u  wall=%.1f ms (incl. JIT on cold run)\n",
                N, seed, ms);

    md.reset();
    return 0;
}
