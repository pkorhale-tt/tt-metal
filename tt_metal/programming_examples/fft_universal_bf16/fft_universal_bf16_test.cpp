// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_test.cpp — correctness harness for the TRUE-bf16
// dispatch tree. Exercises:
//   * Phase 1   : N in [2, 32]               (packed direct-DFT on the FPU)
//   * Phase 2b  : pow2 N > 32                (recursive CT, N1 = 32)
//                 composite N > 32 with ÷≤32 (recursive CT, N1 = largest ÷≤32)
//   * Phase 2c  : prime N > 32               (Bluestein → Phase 2b)
//                 composite with no ÷ ≤ 32   (Bluestein → Phase 2b)
//
// Threshold choice
// ----------------
// bf16 has ~8 bits of mantissa, so ULP is ~4e-3 relative.
//   * Phase 1              : ~3-5e-3 empirical.
//   * Phase 2b, N ≤ 1024   : ~5-8e-3   (depth-1 recursion).
//   * Phase 2b, N ≤ 32768  : ~8-15e-3  (depth-2/3 recursion).
//   * Phase 2c Bluestein   : ~5-15e-3  (3 length-M FFTs + fp32 chirps).
// Thresholds below are generous safety margins. If any one regresses below
// bf16's inherent floor, the failure is unambiguous.

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
    const char*                 name,
    float                       threshold = 1e-2f)
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

    const bool pass = rel_e < threshold;

    std::printf(
        "[%s] N=%-4u | abs=%.2e rel=%.2e snr=%.1f dB | device=%.1f ms  %s\n",
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

    // ── Phase 2b: pow2 N via two-level Cooley-Tukey (N1 = 32) ──────────────
    // N ≤ 1024  : depth-1 recursion (two Phase-1 passes).
    // N > 1024  : depth-2+ recursion. Each extra level doubles the bf16
    //             rounding depth, so we loosen the threshold step-wise.
    constexpr float kPow2ShallowThreshold = 2e-2f;   // N ∈ [64, 1024]
    constexpr float kPow2DeepThreshold    = 3e-2f;   // N ∈ [2048, 16384]
    constexpr float kPow2VeryDeepThreshold = 5e-2f;  // N ∈ [32768, 65536]

    // Shallow pow2 (Phase 2a sizes).
    all &= run_test(md, make_random(64),    "random N=64    (pow2 CT 32x2)",     kPow2ShallowThreshold);
    all &= run_test(md, make_random(128),   "random N=128   (pow2 CT 32x4)",     kPow2ShallowThreshold);
    all &= run_test(md, make_random(256),   "random N=256   (pow2 CT 32x8)",     kPow2ShallowThreshold);
    all &= run_test(md, make_random(512),   "random N=512   (pow2 CT 32x16)",    kPow2ShallowThreshold);
    all &= run_test(md, make_random(1024),  "random N=1024  (pow2 CT 32x32)",    kPow2ShallowThreshold);
    all &= run_test(md, make_impulse(1024), "impulse N=1024 (pow2)",             kPow2ShallowThreshold);

    // Deep pow2 (new in this phase).
    all &= run_test(md, make_random(2048),  "random N=2048  (pow2 CT 32x64)",    kPow2DeepThreshold);
    all &= run_test(md, make_random(4096),  "random N=4096  (pow2 CT 32x128)",   kPow2DeepThreshold);
    all &= run_test(md, make_random(8192),  "random N=8192  (pow2 CT 32x256)",   kPow2DeepThreshold);
    all &= run_test(md, make_random(16384), "random N=16384 (pow2 CT 32x512)",   kPow2DeepThreshold);
    all &= run_test(md, make_impulse(16384),"impulse N=16384 (pow2)",            kPow2DeepThreshold);
    // Very-deep pow2.
    all &= run_test(md, make_random(32768), "random N=32768 (pow2 CT 32x1024)",  kPow2VeryDeepThreshold);
    all &= run_test(md, make_random(65536), "random N=65536 (pow2 depth-3)",     kPow2VeryDeepThreshold);

    // ── Phase 2b: composite non-pow2 via mixed-radix CT ────────────────────
    // Each of these has a divisor ≤ 32 so the dispatcher picks mixed-radix.
    // 36  = 4 × 9      (N1 = 4,  N2 = 9, both ≤ 32)
    // 48  = 16 × 3     (N1 = 16, N2 = 3, both ≤ 32)
    // 60  = 30 × 2     (N1 = 30, N2 = 2, both ≤ 32)
    // 100 = 25 × 4     (N1 = 25, N2 = 4, both ≤ 32)
    // 3600 = 30 × 120  → 120 = 30 × 4 (depth-2 recursion)
    constexpr float kMixedThreshold = 2e-2f;
    all &= run_test(md, make_random(36),    "random N=36    (mixed 4x9)",        kMixedThreshold);
    all &= run_test(md, make_random(48),    "random N=48    (mixed 16x3)",       kMixedThreshold);
    all &= run_test(md, make_random(60),    "random N=60    (mixed 30x2)",       kMixedThreshold);
    all &= run_test(md, make_random(100),   "random N=100   (mixed 25x4)",       kMixedThreshold);
    all &= run_test(md, make_random(3600),  "random N=3600  (mixed, depth-2)",   3e-2f);

    // ── Phase 2c: Bluestein for primes > 32 ─────────────────────────────────
    // Each prime N uses length-M = next_pow2(2N-1) internally:
    //   N=37  → M=128,   depth-1 pow2
    //   N=41  → M=128,   depth-1 pow2
    //   N=43  → M=128,   depth-1 pow2
    //   N=47  → M=128,   depth-1 pow2
    //   N=101 → M=256,   depth-1 pow2
    //   N=251 → M=512,   depth-1 pow2
    //   N=1009→ M=2048,  depth-2 pow2   (bigger M → looser threshold)
    constexpr float kBluesteinThreshold = 3e-2f;
    all &= run_test(md, make_random(37),    "random N=37    (prime, Bluestein M=128)",   kBluesteinThreshold);
    all &= run_test(md, make_random(41),    "random N=41    (prime, Bluestein M=128)",   kBluesteinThreshold);
    all &= run_test(md, make_random(43),    "random N=43    (prime, Bluestein M=128)",   kBluesteinThreshold);
    all &= run_test(md, make_random(47),    "random N=47    (prime, Bluestein M=128)",   kBluesteinThreshold);
    all &= run_test(md, make_random(101),   "random N=101   (prime, Bluestein M=256)",   kBluesteinThreshold);
    all &= run_test(md, make_random(251),   "random N=251   (prime, Bluestein M=512)",   kBluesteinThreshold);
    all &= run_test(md, make_random(1009),  "random N=1009  (prime, Bluestein M=2048)",  5e-2f);

    // Hard composite: 37² = 1369 has no divisor ≤ 32, so it MUST route
    // through Bluestein, not mixed-radix. If the dispatcher mis-routes
    // it, the test will either hang or throw — either way, PASS will not
    // print. M = next_pow2(2·1369-1) = 4096.
    all &= run_test(md, make_random(1369),  "random N=1369  (37x37, Bluestein M=4096)",  5e-2f);

    md.reset();
    std::printf("\n%s\n",
                all ? "All fft_universal_bf16 tests PASSED."
                    : "SOME TESTS FAILED.");
    return all ? 0 : 1;
}
