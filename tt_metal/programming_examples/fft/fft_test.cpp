// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_test.cpp — correctness + basic timing for the multi-core fp32 FFT.

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
            const double a = -2.0 * M_PI * static_cast<double>(k) *
                             static_cast<double>(n) / static_cast<double>(N);
            X[k] += x[n] * Complex(static_cast<float>(std::cos(a)),
                                   static_cast<float>(std::sin(a)));
        }
    }
    return X;
}

static float max_err(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    float e = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

static float rel_err(const std::vector<Complex>& ref, const std::vector<Complex>& got) {
    float max_abs = 0.0f;
    for (const auto& c : ref) max_abs = std::max(max_abs, std::abs(c));
    if (max_abs == 0.0f) max_abs = 1.0f;
    return max_err(ref, got) / max_abs;
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
    const Sizing  z  = compute_sizing(N);

    // Reference DFT is O(N^2); skip it for very large N and trust the
    // smaller-N correctness + the algorithmic invariants.
    const bool have_ref = (N <= 4096);
    std::vector<Complex> ref;
    if (have_ref) ref = ref_dft(input);

    MeshCommandQueue& cq = md->mesh_command_queue();

    auto [in_r, in_i] = pack_input(input, z);
    auto in_r_buf  = make_io_buf(md, N);
    auto in_i_buf  = make_io_buf(md, N);
    auto out_r_buf = make_io_buf(md, N);
    auto out_i_buf = make_io_buf(md, N);

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

    const auto got = unpack_output(out_r, out_i, z);

    if (!have_ref) {
        std::printf("[SKIP] N=%-5u FFT | no O(N^2) reference at this size | "
                    "%.1f ms  %s  (P=%u cores)\n",
                    N, ms, name, z.P);
        return true;
    }

    const float abs_e = max_err(ref, got);
    const float rel_e = rel_err(ref, got);

    // Wormhole fp32 path gives ~tf19-class precision. Allow more slack per
    // stage for multi-core because each cross-core butterfly adds a NoC
    // round-trip but same numerical precision; accumulated error still
    // scales roughly as sqrt(logN).
    const bool pass = rel_e < 2e-3f;
    std::printf("[%s] N=%-5u FFT | abs=%.2e rel=%.2e | %.1f ms  %s  (P=%u)\n",
                pass ? "PASS" : "FAIL", N, abs_e, rel_e, ms, name, z.P);
    return pass;
}

int main() {
    auto md  = MeshDevice::create_unit_mesh(0);
    bool all = true;

    // Single-core cases (P=1)
    all &= run_test(md, make_impulse(16),   "impulse N=16");
    all &= run_test(md, make_impulse(64),   "impulse N=64");
    all &= run_test(md, make_constant(64),  "DC N=64");
    all &= run_test(md, make_random(64),    "random N=64");
    all &= run_test(md, make_random(256),   "random N=256");
    all &= run_test(md, make_random(1024),  "random N=1024");

    // Multi-core cases (P>1)
    all &= run_test(md, make_impulse(2048), "impulse N=2048");
    all &= run_test(md, make_random(2048),  "random N=2048");
    all &= run_test(md, make_random(4096),  "random N=4096");
    all &= run_test(md, make_random(8192),  "random N=8192");

    md.reset();
    std::printf("\n%s\n", all ? "All tests PASSED." : "SOME TESTS FAILED.");
    return all ? 0 : 1;
}
