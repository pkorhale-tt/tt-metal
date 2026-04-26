// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_test.cpp — correctness harness for the TRUE-bf16
// packed direct-DFT path (Phase 1). Exercises every Phase 1 N in [2, 32]
// — all pow2, primes, and small composites land in the same kernel.
//
// Threshold choice
// ----------------
// bf16 has ~8 bits of mantissa, so ULP is ~4e-3 relative. A length-N
// DFT accumulates N bf16 × bf16 products in fp32 (log2(N) rounding
// depth), then packs once to bf16 on output. Empirically that gives
// ~3-5e-3 relative error on random |x| <= 1 input at N <= 32.
// We use 1e-2 as a safety margin — if a kernel regresses below bf16's
// inherent floor, the failure is unambiguous.

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_bf16_host.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

// Double-precision O(N²) DFT — slow but exact (negligible rounding) for
// the sizes we test here.
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

// SNR in dB (signal power vs error power) — the standard "how close is bf16
// to the fp32 ideal" metric. 40 dB = ~1% error. 42-45 dB is the practical
// bf16 floor for length-32 DFTs.
static float snr_db(const std::vector<Complex>& ref, const std::vector<Complex>& got) {
    double sig = 0.0, err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const Complex e = got[i] - ref[i];
        sig += static_cast<double>(std::norm(ref[i]));
        err += static_cast<double>(std::norm(e));
    }
    if (err == 0.0) return INFINITY;
    if (sig == 0.0) return 0.0;
    return static_cast<float>(10.0 * std::log10(sig / err));
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

static std::vector<Complex> make_impulse(uint32_t N) {
    std::vector<Complex> x(N, {0.0f, 0.0f});
    x[0] = {1.0f, 0.0f};
    return x;
}

static bool run_test(
    std::shared_ptr<MeshDevice> md,
    const std::vector<Complex>& input,
    const char*                 name)
{
    const uint32_t N = static_cast<uint32_t>(input.size());

    const std::vector<Complex> ref = ref_dft(input);

    const auto t0  = std::chrono::high_resolution_clock::now();
    const auto got = fft_universal_bf16::fft(md, input);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    const float abs_e = max_err(ref, got);
    const float rel_e = rel_err(ref, got);
    const float snr   = snr_db(ref, got);

    // 1e-2 is the Phase 1 bf16 threshold (see file header).
    const float threshold = 1e-2f;
    const bool pass = rel_e < threshold;

    std::printf(
        "[%s] N=%-3u | abs=%.2e rel=%.2e snr=%.1f dB | device=%.1f ms  %s\n",
        pass ? "PASS" : "FAIL", N, abs_e, rel_e, snr, ms, name);
    return pass;
}

int main() {
    auto md  = MeshDevice::create_unit_mesh(0);
    bool all = true;

    // All Phase 1 sizes go through the packed direct-DFT bf16 kernel.
    // Pow2:
    all &= run_test(md, make_random(2),   "random N=2  (pow2)");
    all &= run_test(md, make_random(4),   "random N=4  (pow2)");
    all &= run_test(md, make_random(8),   "random N=8  (pow2)");
    all &= run_test(md, make_random(16),  "random N=16 (pow2)");
    all &= run_test(md, make_random(32),  "random N=32 (pow2)");
    // Primes:
    all &= run_test(md, make_random(3),   "random N=3  (prime)");
    all &= run_test(md, make_random(5),   "random N=5  (prime)");
    all &= run_test(md, make_random(7),   "random N=7  (prime)");
    all &= run_test(md, make_random(11),  "random N=11 (prime)");
    all &= run_test(md, make_random(13),  "random N=13 (prime)");
    all &= run_test(md, make_random(17),  "random N=17 (prime)");
    all &= run_test(md, make_random(19),  "random N=19 (prime)");
    all &= run_test(md, make_random(23),  "random N=23 (prime)");
    all &= run_test(md, make_random(29),  "random N=29 (prime)");
    all &= run_test(md, make_random(31),  "random N=31 (prime)");
    // Composites:
    all &= run_test(md, make_random(6),   "random N=6  (2 x 3)");
    all &= run_test(md, make_random(9),   "random N=9  (3 x 3)");
    all &= run_test(md, make_random(10),  "random N=10 (2 x 5)");
    all &= run_test(md, make_random(12),  "random N=12 (4 x 3)");
    all &= run_test(md, make_random(15),  "random N=15 (3 x 5)");
    all &= run_test(md, make_random(21),  "random N=21 (3 x 7)");
    all &= run_test(md, make_random(25),  "random N=25 (5 x 5)");
    all &= run_test(md, make_random(27),  "random N=27 (3 x 9)");
    // Impulse: flat spectrum. Every bin = 1 regardless of N.
    all &= run_test(md, make_impulse(32), "impulse N=32");
    all &= run_test(md, make_impulse(17), "impulse N=17");

    // Phase 2 smoke check: N > 32 should throw — we exercise the guard
    // path so it doesn't silently regress later.
    bool threw_as_expected = false;
    try {
        (void)fft_universal_bf16::fft(md, make_random(64));
    } catch (const std::runtime_error&) {
        threw_as_expected = true;
    }
    std::printf("[%s] N=64 (Phase 2 guard): %s threw runtime_error\n",
                threw_as_expected ? "PASS" : "FAIL",
                threw_as_expected ? "correctly" : "unexpectedly did NOT");
    all &= threw_as_expected;

    md.reset();
    std::printf("\n%s\n",
                all ? "All fft_universal_bf16 Phase 1 tests PASSED."
                    : "SOME TESTS FAILED.");
    return all ? 0 : 1;
}
