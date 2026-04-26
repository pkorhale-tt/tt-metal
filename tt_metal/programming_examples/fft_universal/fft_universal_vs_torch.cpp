// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_vs_torch.cpp — runs fft_universal::fft on a user-specified N
// (ANY N >= 2, not restricted to powers of two) and writes the input signal
// and FFT output to plain-text files that a Python companion script
// (compare_with_torch.py) reads and compares against torch.fft.fft.
//
// Usage:
//     metal_example_fft_universal_vs_torch <N> [seed] [in_path] [out_path]
//
//   N         : FFT length (any integer >= 2; prime/composite/non-pow2 all OK).
//   seed      : RNG seed for the random complex input (default 42).
//   in_path   : output path for the input signal (default "fft_input.txt").
//   out_path  : output path for the tt-metal FFT result (default
//               "fft_output.txt").
//
// Each file has N lines, two space-separated floats per line: "real imag".

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_host.cpp"

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace tt::tt_metal::distributed;
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

static const char* describe_path(uint32_t N) {
    if (N == 1u)                    return "identity";
    if (fft_universal::is_pow2(N))  return "pow2 pass-through (fft_stockham)";
    if (fft_universal::is_prime(N)) return "Bluestein (prime)";
    return "Cooley-Tukey split (composite non-pow2)";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <N> [seed] [in_path] [out_path]\n", argv[0]);
        return 1;
    }
    const uint32_t    N        = static_cast<uint32_t>(std::atoi(argv[1]));
    const uint32_t    seed     = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 42u;
    const std::string in_path  = (argc > 3) ? argv[3] : "fft_input.txt";
    const std::string out_path = (argc > 4) ? argv[4] : "fft_output.txt";

    if (N < 2u) {
        std::fprintf(stderr, "N must be >= 2, got %u\n", N);
        return 1;
    }

    auto md     = MeshDevice::create_unit_mesh(0);
    const auto  input = make_random(N, seed);

    std::printf("[fft_universal_vs_torch] N=%u  path=%s\n", N, describe_path(N));

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto output = fft_universal::fft(md, input);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();

    write_complex_file(in_path,  input);
    write_complex_file(out_path, output);

    std::printf("[fft_universal_vs_torch] wrote %s  and  %s\n",
                in_path.c_str(), out_path.c_str());
    std::printf("[fft_universal_vs_torch] seed=%u  wall=%.1f ms (incl. JIT on cold run)\n",
                seed, ms);

    md.reset();
    return 0;
}
