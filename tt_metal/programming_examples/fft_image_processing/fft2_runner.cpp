// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft2_runner.cpp — single-process 2-D FFT runner for the image-processing
// demo. Opens the Wormhole MeshDevice ONCE, performs row-then-column 1-D FFTs
// using the existing fft_universal pipeline, and writes the result back to
// disk. This avoids the 2N subprocess launches the per-row Python wrapper
// would otherwise need.
//
// File format (input and output):
//     line 1 : "H W"        (image dimensions, two unsigned ints)
//     line 2..H*W+1 : "real imag"   (one complex sample per line, row-major)
//
// Usage:
//     metal_example_fft_image_processing_fft2_runner <in_path> <out_path> [--inverse]
//
// Exit codes:
//     0 = success
//     1 = bad CLI args / missing file / bad input format
//     2 = device error

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
#include <string>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

struct Image2D {
    uint32_t H = 0;
    uint32_t W = 0;
    // row-major, length H*W
    std::vector<Complex> data;

    Complex* row(uint32_t r) { return data.data() + static_cast<size_t>(r) * W; }
    const Complex* row(uint32_t r) const { return data.data() + static_cast<size_t>(r) * W; }
};

static Image2D read_image(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Cannot open %s for reading\n", path.c_str());
        std::exit(1);
    }
    Image2D img;
    if (!(f >> img.H >> img.W) || img.H == 0u || img.W == 0u) {
        std::fprintf(stderr, "%s: expected first line 'H W' with H,W>=1\n", path.c_str());
        std::exit(1);
    }
    img.data.resize(static_cast<size_t>(img.H) * img.W);
    float r, i;
    size_t got = 0;
    while (f >> r >> i) {
        if (got >= img.data.size()) break;
        img.data[got++] = Complex(r, i);
    }
    if (got != img.data.size()) {
        std::fprintf(stderr,
                     "%s: expected %zu samples, got %zu\n",
                     path.c_str(), img.data.size(), got);
        std::exit(1);
    }
    return img;
}

static void write_image(const std::string& path, const Image2D& img) {
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "Cannot open %s for writing\n", path.c_str());
        std::exit(1);
    }
    f << img.H << ' ' << img.W << '\n';
    f.setf(std::ios::scientific);
    f.precision(9);
    for (const auto& c : img.data) {
        f << c.real() << ' ' << c.imag() << '\n';
    }
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

    Image2D img = read_image(in_path);
    const uint32_t H = img.H;
    const uint32_t W = img.W;
    std::printf("[fft2_runner] image %ux%u  direction=%s  total samples=%zu\n",
                H, W, inverse ? "IFFT2" : "FFT2", img.data.size());

    // ----- Open the Wormhole MeshDevice ONCE for the whole 2-D pass. -----
    auto md = MeshDevice::create_unit_mesh(0);

    auto run_1d = [&](const std::vector<Complex>& x) {
        return inverse ? fft_universal::ifft(md, x)
                       : fft_universal::fft(md, x);
    };

    const auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Pass 1: H row FFTs of length W ----
    {
        std::vector<Complex> row_buf(W);
        for (uint32_t r = 0; r < H; ++r) {
            for (uint32_t c = 0; c < W; ++c) row_buf[c] = img.row(r)[c];
            const auto y = run_1d(row_buf);
            for (uint32_t c = 0; c < W; ++c) img.row(r)[c] = y[c];
        }
    }

    // ---- Pass 2: W column FFTs of length H ----
    {
        std::vector<Complex> col_buf(H);
        for (uint32_t c = 0; c < W; ++c) {
            for (uint32_t r = 0; r < H; ++r) col_buf[r] = img.row(r)[c];
            const auto y = run_1d(col_buf);
            for (uint32_t r = 0; r < H; ++r) img.row(r)[c] = y[r];
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();

    write_image(out_path, img);
    std::printf("[fft2_runner] wrote %s  wall=%.2f ms  (%u 1-D FFTs, single device session)\n",
                out_path.c_str(), ms, H + W);

    md.reset();
    return 0;
}
