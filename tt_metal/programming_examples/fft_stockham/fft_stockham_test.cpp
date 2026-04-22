// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_stockham_test.cpp — correctness & timing for the Stockham orchestrator.
//
// Coverage:
//   * The fall-through path (N <= 65,536) — should match fft_example::fft.
//   * The Stockham path (N > 65,536) — verified against a double-precision
//     iterative radix-2 reference. Tested at N = 131,072 / 262,144 / 524,288
//     / 1,048,576 to demonstrate scaling.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_stockham_host.cpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

// ── Reference: double-precision iterative radix-2 ─────────────────────────
// Same routine the inner test uses. Reference error sits well below
// anything the device path could produce, even at N=1M.
static std::vector<Complex> ref_fft_fast(const std::vector<Complex>& x) {
    using CD = std::complex<double>;
    const uint32_t N = static_cast<uint32_t>(x.size());

    uint32_t log2N = 0;
    while ((1u << log2N) < N) ++log2N;

    std::vector<CD> a(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t r = 0;
        for (uint32_t b = 0; b < log2N; ++b) r = (r << 1) | ((i >> b) & 1u);
        a[r] = CD(x[i].real(), x[i].imag());
    }

    for (uint32_t s = 1; s <= log2N; ++s) {
        const uint32_t m  = 1u << s;
        const uint32_t mh = m >> 1;
        const double   theta = -2.0 * M_PI / static_cast<double>(m);
        const CD       wm(std::cos(theta), std::sin(theta));
        for (uint32_t k = 0; k < N; k += m) {
            CD w(1.0, 0.0);
            for (uint32_t j = 0; j < mh; ++j) {
                const CD t = w * a[k + j + mh];
                const CD u = a[k + j];
                a[k + j]      = u + t;
                a[k + j + mh] = u - t;
                w *= wm;
            }
        }
    }

    std::vector<Complex> out(N);
    for (uint32_t i = 0; i < N; ++i)
        out[i] = Complex(static_cast<float>(a[i].real()),
                         static_cast<float>(a[i].imag()));
    return out;
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
//
// Stockham is two passes of fp32 sub-FFTs plus per-element fp32 twiddles, so
// the relative-error budget is roughly ~2x what the inner kernel produces
// alone (each pass contributes its own rounding). 5e-3 is comfortable for
// every size we cover here.
static bool run_test(
    std::shared_ptr<MeshDevice> md,
    const std::vector<Complex>& input,
    const char*                 name)
{
    const uint32_t N = static_cast<uint32_t>(input.size());

    const auto t_ref0 = std::chrono::high_resolution_clock::now();
    const std::vector<Complex> ref = ref_fft_fast(input);
    const double ref_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_ref0).count();

    const auto t0  = std::chrono::high_resolution_clock::now();
    const auto got = fft_stockham::fft(md, input);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    const float abs_e = max_err(ref, got);
    const float rel_e = rel_err(ref, got);

    const float threshold = (N <= 65536u) ? 2e-3f : 5e-3f;
    const bool  pass      = rel_e < threshold;

    std::printf(
        "[%s] N=%-8u | abs=%.2e rel=%.2e | device=%.1f ms  ref=%.1f ms  %s\n",
        pass ? "PASS" : "FAIL", N, abs_e, rel_e, ms, ref_ms, name);
    return pass;
}

int main() {
    auto md  = MeshDevice::create_unit_mesh(0);
    bool all = true;

    // Sanity: fall-through path (must match the inner kernel exactly).
    all &= run_test(md, make_random(1024),    "random N=1024     (fall-through)");
    all &= run_test(md, make_random(16384),   "random N=16384    (fall-through)");
    all &= run_test(md, make_impulse(65536),  "impulse N=65536   (fall-through)");

    // Stockham path — N > 65,536. Each sub-FFT still fits the inner kernel.
    all &= run_test(md, make_impulse(131072), "impulse N=131072  (N1=512  x N2=256)");
    all &= run_test(md, make_constant(131072),"DC      N=131072");
    all &= run_test(md, make_random(131072),  "random  N=131072");

    all &= run_test(md, make_random(262144),  "random  N=262144  (N1=512  x N2=512)");
    all &= run_test(md, make_random(524288),  "random  N=524288  (N1=1024 x N2=512)");
    all &= run_test(md, make_random(1048576), "random  N=1048576 (N1=1024 x N2=1024)");

    md.reset();
    std::printf("\n%s\n", all ? "All Stockham tests PASSED." : "SOME TESTS FAILED.");
    return all ? 0 : 1;
}
