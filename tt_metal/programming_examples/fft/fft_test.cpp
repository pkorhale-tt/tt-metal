// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_test.cpp — correctness + basic timing for the single-core fp32 FFT.

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/constants.hpp"

#include "fft_host.cpp"

#include <cmath>
#include <complex>
#include <vector>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cstdlib>

using namespace tt::tt_metal::distributed;
using namespace fft_example;
using Complex = std::complex<float>;

// ── Reference DFT ─────────────────────────────────────────────────────────
static std::vector<Complex> ref_dft(const std::vector<Complex>& x) {
    const uint32_t N = static_cast<uint32_t>(x.size());
    std::vector<Complex> X(N, {0.0f, 0.0f});
    for (uint32_t k = 0; k < N; ++k) {
        for (uint32_t n = 0; n < N; ++n) {
            const float a = -2.0f * static_cast<float>(M_PI) *
                            static_cast<float>(k) * static_cast<float>(n) /
                            static_cast<float>(N);
            X[k] += x[n] * Complex(std::cos(a), std::sin(a));
        }
    }
    return X;
}

static float max_err(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// ── Input generators ──────────────────────────────────────────────────────
static std::vector<Complex> make_impulse(uint32_t N) {
    std::vector<Complex> x(N, {0.0f, 0.0f});
    x[0] = {1.0f, 0.0f};
    return x;
}

static std::vector<Complex> make_constant(uint32_t N) {
    return std::vector<Complex>(N, {1.0f, 0.0f});
}

static std::vector<Complex> make_random(uint32_t N, uint32_t seed = 42) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x) {
        c = {(std::rand() / float(RAND_MAX)) * 2.0f - 1.0f,
             (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f};
    }
    return x;
}

// ── Single test ───────────────────────────────────────────────────────────
static bool run_test(
    std::shared_ptr<MeshDevice> md,
    const std::vector<Complex>& input,
    const char* name)
{
    const uint32_t N = static_cast<uint32_t>(input.size());
    const auto ref   = ref_dft(input);

    MeshCommandQueue& cq = md->mesh_command_queue();

    auto [in_r, in_i] = pack_input(input);
    auto in_r_buf = make_mesh_buf(md, kTileSizeFp32, kTileSizeFp32);
    auto in_i_buf = make_mesh_buf(md, kTileSizeFp32, kTileSizeFp32);
    auto out_r_buf = make_mesh_buf(md, kTileSizeFp32, kTileSizeFp32);
    auto out_i_buf = make_mesh_buf(md, kTileSizeFp32, kTileSizeFp32);

    WriteShard(cq, in_r_buf, in_r, MeshCoordinate(0, 0), false);
    WriteShard(cq, in_i_buf, in_i, MeshCoordinate(0, 0), false);

    const auto t0 = std::chrono::high_resolution_clock::now();
    run_fft(md, {N}, in_r_buf, in_i_buf, out_r_buf, out_i_buf);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();

    std::vector<float> out_r, out_i;
    ReadShard(cq, out_r, out_r_buf, MeshCoordinate(0, 0), true);
    ReadShard(cq, out_i, out_i_buf, MeshCoordinate(0, 0), true);

    const auto got = unpack_output(out_r, out_i, N);
    const float err = max_err(ref, got);
    const bool  pass = err < 1e-3f;
    std::printf("[%s] N=%-5u FFT | err=%.2e | %.1f ms  %s\n",
                pass ? "PASS" : "FAIL", N, err, ms, name);
    return pass;
}

int main() {
    auto md  = MeshDevice::create_unit_mesh(0);
    bool all = true;

    all &= run_test(md, make_impulse(16),   "impulse N=16");
    all &= run_test(md, make_impulse(64),   "impulse N=64");
    all &= run_test(md, make_constant(64),  "DC N=64");
    all &= run_test(md, make_random(64),    "random N=64");
    all &= run_test(md, make_random(256),   "random N=256");
    all &= run_test(md, make_random(1024),  "random N=1024");

    md.reset();
    std::printf("\n%s\n", all ? "All tests PASSED." : "SOME TESTS FAILED.");
    return all ? 0 : 1;
}
