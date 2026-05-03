// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_run.cpp — file-IO runner for fft_universal_bf16
// (TRUE bf16 pipeline: bf16 inputs, bf16 multiplier array on the FPU,
// fp32 accumulator).
//
// Reads a complex signal from a text file, runs FFT (or IFFT via --inverse)
// on Wormhole through the bf16 dispatch tree, writes the result to another
// text file. C++ side of the PyTorch-style Python wrapper (tt_fft.py).
//
// Each line of the input/output file is "real imag" (two space-separated
// floats). Line count = N (FFT length). Any N >= 2 is supported.
//
// Usage:
//     metal_example_fft_universal_bf16_run <in_path> <out_path> [--inverse]

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_bf16_host.cpp"

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

static std::vector<Complex> read_complex_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Cannot open %s for reading\n", path.c_str());
        std::exit(1);
    }
    std::vector<Complex> v;
    float r, i;
    while (f >> r >> i) v.emplace_back(r, i);
    if (v.empty()) {
        std::fprintf(stderr, "%s contained no valid 'real imag' lines\n",
                     path.c_str());
        std::exit(1);
    }
    return v;
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
    if (N == 1u) return "identity";
    if (N <= 32u) return "packed direct-DFT (bf16 FPU matmul)";
    if (fft_universal_bf16::is_pow2(N)) return "two-level Cooley-Tukey (bf16)";
    if (fft_universal_bf16::is_prime(N)) return "Bluestein (bf16)";
    return "Cooley-Tukey split (bf16)";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <in_path> <out_path> [--inverse]\n", argv[0]);
        return 1;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    bool inverse = false;
    for (int a = 3; a < argc; ++a) {
        if (std::strcmp(argv[a], "--inverse") == 0) inverse = true;
    }

    const auto input = read_complex_file(in_path);
    const uint32_t N = static_cast<uint32_t>(input.size());
    if (N < 2u) {
        std::fprintf(stderr, "N must be >= 2 (got %u)\n", N);
        return 1;
    }

    auto md = MeshDevice::create_unit_mesh(0);
    std::printf("[fft_universal_bf16_run] N=%u  direction=%s  path=%s\n",
                N, inverse ? "IFFT" : "FFT", describe_path(N));

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto output = inverse ? fft_universal_bf16::ifft(md, input)
                                : fft_universal_bf16::fft(md, input);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();

    write_complex_file(out_path, output);
    std::printf("[fft_universal_bf16_run] wrote %s  wall=%.2f ms (incl. JIT on cold call)\n",
                out_path.c_str(), ms);

    md.reset();
    return 0;
}
