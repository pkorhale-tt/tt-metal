// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_test.cpp — correctness harness for fft_universal::fft.
//
// Every dispatch path runs on Wormhole:
//   * Small pow2 pass-through   (N = 2, 4, 8, 16, 32) -> fft_stockham/fft_example.
//   * Small primes via Bluestein (N = 3, 5, 7, 11, 31) -> two pow2 FFTs on device.
//   * Larger pow2 pass-through  (N = 1024, 16384, 65536, 131072).
//   * Cooley-Tukey composite    (N = 6, 12, 48, 100, 360, 1000, 3072, 5120, 7168).
//   * Cooley-Tukey w/ prime tail (N = 2 * 101, 3 * 97, 2 * 257).
//   * Bluestein primes           (N = 37, 97, 101, 257, 509, 1021).
//
// Reference is a straight double-precision O(N^2) DFT — slow but error-free
// for the sizes we test.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_host.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

// ── Reference: double-precision O(N^2) DFT ────────────────────────────────
static std::vector<Complex> ref_dft(const std::vector<Complex>& x) {
    using CD = std::complex<double>;
    const uint32_t N = static_cast<uint32_t>(x.size());
    std::vector<Complex> X(N);
    const double tau = -2.0 * M_PI / static_cast<double>(N);
    for (uint32_t k = 0; k < N; ++k) {
        CD sum(0.0, 0.0);
        for (uint32_t n = 0; n < N; ++n) {
            const double a = tau * static_cast<double>(k) * static_cast<double>(n);
            sum += CD(static_cast<double>(x[n].real()),
                      static_cast<double>(x[n].imag())) * CD(std::cos(a), std::sin(a));
        }
        X[k] = Complex(static_cast<float>(sum.real()),
                       static_cast<float>(sum.imag()));
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

static std::vector<Complex> make_random(uint32_t N, uint32_t seed = 42) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x) {
        c = {(std::rand() / float(RAND_MAX)) * 2.0f - 1.0f,
             (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f};
    }
    return x;
}

// ── Round-trip test for IFFT (ifft(fft(x)) == x) ──────────────────────────
//
// The conjugate-trick IFFT runs through the EXACT same dispatch tree as
// the forward FFT, so any path-specific bug shows up symmetrically here.
// Round-trip error is bounded by ~2x the single-direction error (one fp32
// rounding cycle each way), so we reuse the same rel-error budget.
static bool run_round_trip(
    std::shared_ptr<MeshDevice> md,
    const std::vector<Complex>& input,
    const char*                 name)
{
    const uint32_t N = static_cast<uint32_t>(input.size());

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto X  = fft_universal::fft(md, input);
    const auto y  = fft_universal::ifft(md, X);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    const float abs_e = max_err(input, y);
    const float rel_e = rel_err(input, y);

    float threshold;
    if (fft_universal::is_pow2(N)) threshold = (N <= 65536u) ? 4e-3f : 1e-2f;
    else                            threshold = 2e-2f;

    const bool pass = rel_e < threshold;

    std::printf(
        "[%s] N=%-8u | abs=%.2e rel=%.2e | round-trip=%.1f ms  %s\n",
        pass ? "PASS" : "FAIL", N, abs_e, rel_e, ms, name);
    return pass;
}

// ── Single test ───────────────────────────────────────────────────────────
//
// Error budget notes:
//   * Bluestein touches each sample through 2 extra fp32 multiplies (chirp
//     pre/post) and one length-M pow2 FFT — use 1e-2 rel for safety at small N.
//   * Cooley-Tukey has depth = log_p(N) extra twiddle passes; 5e-3 handles
//     the deepest factorisations we test (e.g. 1000 = 8 * 125 = 8 * 5 * 5 * 5).
static bool run_test(
    std::shared_ptr<MeshDevice> md,
    const std::vector<Complex>& input,
    const char*                 name)
{
    const uint32_t N = static_cast<uint32_t>(input.size());

    const auto t_ref0 = std::chrono::high_resolution_clock::now();
    const std::vector<Complex> ref = ref_dft(input);
    const double ref_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_ref0).count();

    const auto t0  = std::chrono::high_resolution_clock::now();
    const auto got = fft_universal::fft(md, input);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    const float abs_e = max_err(ref, got);
    const float rel_e = rel_err(ref, got);

    // Looser budget when the path ran Bluestein or deep Cooley-Tukey; the
    // Stockham-only case uses the tighter fft_stockham budget.
    float threshold;
    if (fft_universal::is_pow2(N))       threshold = (N <= 65536u) ? 2e-3f : 5e-3f;
    else                                  threshold = 1e-2f;

    const bool pass = rel_e < threshold;

    std::printf(
        "[%s] N=%-8u | abs=%.2e rel=%.2e | device=%.1f ms  ref=%.1f ms  %s\n",
        pass ? "PASS" : "FAIL", N, abs_e, rel_e, ms, ref_ms, name);
    return pass;
}

int main() {
    auto md  = MeshDevice::create_unit_mesh(0);
    bool all = true;

    // 1) Tiny sizes — still on Wormhole.
    all &= run_test(md, make_random(2),         "random N=2      (pow2, device)");
    all &= run_test(md, make_random(4),         "random N=4      (pow2, device)");
    all &= run_test(md, make_random(8),         "random N=8      (pow2, device)");
    all &= run_test(md, make_random(16),        "random N=16     (pow2, device)");
    all &= run_test(md, make_random(32),        "random N=32     (pow2, device)");
    all &= run_test(md, make_random(3),         "random N=3      (prime,  Bluestein M=8)");
    all &= run_test(md, make_random(5),         "random N=5      (prime,  Bluestein M=16)");
    all &= run_test(md, make_random(7),         "random N=7      (prime,  Bluestein M=16)");
    all &= run_test(md, make_random(11),        "random N=11     (prime,  Bluestein M=32)");
    all &= run_test(md, make_random(31),        "random N=31     (prime,  Bluestein M=64)");

    // 2) Pow2 pass-through (same guarantees as fft_stockham tests).
    all &= run_test(md, make_random(1024),      "random  N=1024    (pow2)");
    all &= run_test(md, make_random(16384),     "random  N=16384   (pow2)");
    all &= run_test(md, make_impulse(65536),    "impulse N=65536   (pow2)");
    all &= run_test(md, make_random(131072),    "random  N=131072  (pow2, Stockham)");

    // 3) Composite non-pow2 via Cooley-Tukey.
    all &= run_test(md, make_random(6),         "random  N=6       (2 x 3)");
    all &= run_test(md, make_random(12),        "random  N=12      (4 x 3)");
    all &= run_test(md, make_random(48),        "random  N=48      (16 x 3)");
    all &= run_test(md, make_random(100),       "random  N=100     (4 x 25)");
    all &= run_test(md, make_random(360),       "random  N=360     (8 x 45)");
    all &= run_test(md, make_random(1000),      "random  N=1000    (8 x 125)");
    all &= run_test(md, make_random(3 * 1024),  "random  N=3072    (1024 x 3)");
    all &= run_test(md, make_random(5 * 1024),  "random  N=5120    (1024 x 5)");
    all &= run_test(md, make_random(7 * 1024),  "random  N=7168    (1024 x 7)");

    // 4) Composite with large prime factor -> recursion hits Bluestein.
    all &= run_test(md, make_random(2 * 101),   "random  N=202     (2 x 101; prime-101 via Bluestein)");
    all &= run_test(md, make_random(3 * 97),    "random  N=291     (3 x 97;  prime-97  via Bluestein)");
    all &= run_test(md, make_random(2 * 257),   "random  N=514     (2 x 257; prime-257 via Bluestein)");

    // 5) Prime N -> Bluestein directly.
    all &= run_test(md, make_random(37),        "random  N=37      (prime,  Bluestein)");
    all &= run_test(md, make_random(97),        "random  N=97      (prime,  Bluestein)");
    all &= run_test(md, make_random(101),       "random  N=101     (prime,  Bluestein)");
    all &= run_test(md, make_random(257),       "random  N=257     (prime,  Bluestein)");
    all &= run_test(md, make_random(509),       "random  N=509     (prime,  Bluestein)");
    all &= run_test(md, make_random(1021),      "random  N=1021    (prime,  Bluestein)");

    // 6) IFFT round-trip across every dispatch path.
    //    Conjugate-trick IFFT reuses the entire forward pipeline; this
    //    exercises pow2 / Bluestein / Cooley-Tukey / packed_dft symmetrically.
    std::printf("\n--- IFFT round-trip (ifft(fft(x)) == x) ---\n");
    all &= run_round_trip(md, make_random(2),     "rt N=2      (pow2)");
    all &= run_round_trip(md, make_random(8),     "rt N=8      (pow2)");
    all &= run_round_trip(md, make_random(32),    "rt N=32     (pow2)");
    all &= run_round_trip(md, make_random(7),     "rt N=7      (prime)");
    all &= run_round_trip(md, make_random(31),    "rt N=31     (prime)");
    all &= run_round_trip(md, make_random(60),    "rt N=60     (composite, 4 x 15)");
    all &= run_round_trip(md, make_random(360),   "rt N=360    (composite)");
    all &= run_round_trip(md, make_random(1024),  "rt N=1024   (pow2)");
    all &= run_round_trip(md, make_random(16384), "rt N=16384  (pow2 six-step)");
    all &= run_round_trip(md, make_random(37),    "rt N=37     (prime, Bluestein)");
    all &= run_round_trip(md, make_random(257),   "rt N=257    (prime, Bluestein)");
    all &= run_round_trip(md, make_random(1021),  "rt N=1021   (prime, Bluestein)");

    md.reset();
    std::printf("\n%s\n",
                all ? "All universal FFT tests PASSED." : "SOME TESTS FAILED.");
    return all ? 0 : 1;
}
