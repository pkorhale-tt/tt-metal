// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_bf16_precision.cpp — head-to-head precision comparison of
// the TRUE-bf16 FFT vs the fp32 FFT (fft_universal), both measured against
// a double-precision O(N²) DFT ground truth.
//
// Why this binary exists
// ----------------------
// The "true bf16 compute" claim is only interesting if we can quantify
// *what it costs in precision*. This tool runs the EXACT same random
// input through both paths and reports, for each N:
//
//   * SNR_fp32      : dB gap between fp32 FFT and double-precision truth
//   * SNR_bf16      : dB gap between bf16 FFT and double-precision truth
//   * ΔSNR          : how much precision bf16 sacrificed vs fp32
//   * mantissa bits : back-of-envelope bit budget implied by each SNR
//   * rel_err (L∞)  : maximum per-bin relative error
//
// Interpretation
// --------------
// Every 6 dB of SNR ≈ 1 bit of effective mantissa precision.
//   fp32 mantissa: 23 bits → expected SNR ≈ 138 dB theoretical, ~120 dB practical.
//   bf16 mantissa:  7 bits → expected SNR ≈  42 dB theoretical, per multiply.
// A length-N FFT accumulates log₂N rounding stages on the critical path,
// so SNR degrades roughly as theoretical - 3·log₂(depth). Anything within
// 2-3 dB of that floor is the bf16 format doing its job — not a bug.
//
// What a healthy output looks like
// --------------------------------
// For random unit-modulus input:
//   * fp32 SNR should be 90-115 dB across all N.
//   * bf16 SNR should be 36-56 dB, declining by ~3 dB per recursion level.
//   * ΔSNR = SNR_fp32 - SNR_bf16 should be 50-80 dB.
//     That 50-80 dB gap ≈ 8-13 mantissa bits, which is the actual width
//     difference between fp32 (23 bits) and bf16 (7 bits). So the gap is
//     approximately the *correct* one — not "bf16 is broken", not "bf16
//     is secretly fp32".

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_bf16_host.cpp"
#include "../fft_universal/fft_universal_host.cpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

// Double-precision O(N²) DFT. This is the "truth" — slow but exact enough
// that its own floating-point error is negligible against bf16 / fp32.
static std::vector<Complex> ref_dft_dbl(const std::vector<Complex>& x) {
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

// Signal-to-Noise Ratio in dB. sig = ‖ref‖², err = ‖got - ref‖².
static double snr_db(const std::vector<Complex>& ref, const std::vector<Complex>& got) {
    double sig = 0.0, err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const Complex e = got[i] - ref[i];
        sig += static_cast<double>(std::norm(ref[i]));
        err += static_cast<double>(std::norm(e));
    }
    if (err == 0.0) return INFINITY;
    if (sig == 0.0) return 0.0;
    return 10.0 * std::log10(sig / err);
}

static float rel_err_linf(const std::vector<Complex>& ref, const std::vector<Complex>& got) {
    float max_abs = 0.0f;
    float max_e   = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(ref[i]));
        max_e   = std::max(max_e,   std::abs(got[i] - ref[i]));
    }
    if (max_abs == 0.0f) max_abs = 1.0f;
    return max_e / max_abs;
}

// Approximate effective mantissa bits implied by the measured SNR:
//   SNR_dB ≈ 6.02 · b + 1.76   (standard ADC quantisation formula)
// We invert to back out b. Real floating-point FFTs are dominated by
// accumulated rounding, not single-rounding quantisation, so this is a
// rough upper bound — "the number of bits that would have produced
// this SNR if all error came from a single uniform rounding stage".
static double effective_bits(double snr_dB) {
    if (std::isinf(snr_dB)) return 24.0;   // floor at "fp32 perfect"
    if (snr_dB < 0.0)      return 0.0;
    return std::max(0.0, (snr_dB - 1.76) / 6.02);
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

static void run_compare(std::shared_ptr<MeshDevice> md, uint32_t N) {
    const std::vector<Complex> x = make_random(N);

    const std::vector<Complex> ref  = ref_dft_dbl(x);
    const std::vector<Complex> fp32 = fft_universal::fft(md, x);
    const std::vector<Complex> bf16 = fft_universal_bf16::fft(md, x);

    const double snr_fp32 = snr_db(ref, fp32);
    const double snr_bf16 = snr_db(ref, bf16);
    const double d_snr    = snr_fp32 - snr_bf16;

    const float  re_fp32  = rel_err_linf(ref, fp32);
    const float  re_bf16  = rel_err_linf(ref, bf16);

    const double bits_fp32 = effective_bits(snr_fp32);
    const double bits_bf16 = effective_bits(snr_bf16);

    std::printf(
        "N=%-6u | fp32 %6.1f dB (≈%4.1f bits, rel=%.1e) | "
        "bf16 %6.1f dB (≈%4.1f bits, rel=%.1e) | ΔSNR = %5.1f dB  (%.1f bits lost)\n",
        N,
        snr_fp32, bits_fp32, re_fp32,
        snr_bf16, bits_bf16, re_bf16,
        d_snr, bits_fp32 - bits_bf16);
}

int main() {
    auto md = MeshDevice::create_unit_mesh(0);

    std::printf(
        "=====================================================================\n"
        "  fft_universal_bf16 precision audit\n"
        "  Random complex input, |x| <= 1, fixed seed. Truth = double-precision\n"
        "  O(N^2) DFT. Δbits computed from Δ(6.02·b + 1.76 dB) quantisation law.\n"
        "=====================================================================\n\n");

    std::printf("--- Phase 1: N in [2, 32] (packed direct-DFT, single dispatch) ---\n");
    for (uint32_t N : {2u, 3u, 5u, 8u, 11u, 16u, 17u, 23u, 31u, 32u}) run_compare(md, N);

    std::printf("\n--- Phase 2b: pow2 N > 32 (recursive Cooley-Tukey, N1=32) ---\n");
    for (uint32_t N : {64u, 128u, 256u, 512u, 1024u, 2048u, 4096u,
                       8192u, 16384u, 32768u, 65536u}) run_compare(md, N);

    std::printf("\n--- Phase 2b: composite non-pow2 (mixed-radix) ---\n");
    for (uint32_t N : {36u, 48u, 60u, 100u, 3600u}) run_compare(md, N);

    std::printf("\n--- Phase 2c: Bluestein (prime / hard-composite N) ---\n");
    for (uint32_t N : {37u, 41u, 47u, 101u, 251u, 1009u, 1369u}) run_compare(md, N);

    std::printf(
        "\n=====================================================================\n"
        "  Takeaways\n"
        "  --------\n"
        "  * ΔSNR (fp32 − bf16) should be 50-80 dB. That ≈ 8-13 mantissa bits\n"
        "    — the *actual* width gap between fp32 (23 bits) and bf16 (7 bits).\n"
        "  * bf16 SNR should degrade by ~3 dB per recursion level (each extra\n"
        "    device round-trip adds 2 bf16 roundings on the critical path).\n"
        "  * If ΔSNR < 10 dB on any row, bf16 is secretly upcasting to fp32\n"
        "    somewhere — that would be a bug to track down.\n"
        "  * If fp32 SNR < 90 dB on any row, the fp32 path has a bug (not bf16).\n"
        "=====================================================================\n");

    md.reset();
    return 0;
}
