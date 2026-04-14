// ============================================================
//  fft_test.cpp  –  correctness + performance test harness
//
//  Tests:
//    1. Single-core FFT, N=64   (all local stages)
//    2. Multi-core FFT, N=1024, 8 cores (local + NOC stages)
//    3. Multi-core FFT, N=1024, 8 cores, bf16
//    4. Round-trip: FFT then IFFT, check reconstruction
//    5. Impulse response: FFT of delta function = constant
// ============================================================

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "fft_host.cpp"   // include host impl directly for single-TU build

#include <cmath>
#include <complex>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdio>

using Complex = std::complex<float>;

// ── Reference DFT (O(N^2), for validation only) ─────────────
std::vector<Complex> reference_dft(const std::vector<Complex>& x) {
    uint32_t N = x.size();
    std::vector<Complex> X(N, {0,0});
    for (uint32_t k = 0; k < N; k++)
        for (uint32_t n = 0; n < N; n++) {
            float angle = -2.0f * M_PI * k * n / N;
            X[k] += x[n] * Complex(std::cos(angle), std::sin(angle));
        }
    return X;
}

std::vector<Complex> reference_idft(const std::vector<Complex>& X) {
    uint32_t N = X.size();
    std::vector<Complex> x(N, {0,0});
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k = 0; k < N; k++) {
            float angle = +2.0f * M_PI * k * n / N;
            x[n] += X[k] * Complex(std::cos(angle), std::sin(angle));
        }
        x[n] /= static_cast<float>(N);
    }
    return x;
}

// ── Read/write buffer helpers ────────────────────────────────
// Input layout: interleaved [r0,i0,r1,i1,...] as float32
std::vector<float> complex_to_interleaved(const std::vector<Complex>& v) {
    std::vector<float> out;
    out.reserve(v.size() * 2);
    for (auto& c : v) { out.push_back(c.real()); out.push_back(c.imag()); }
    return out;
}

std::vector<Complex> interleaved_to_complex(const std::vector<float>& v) {
    std::vector<Complex> out;
    out.reserve(v.size() / 2);
    for (size_t i = 0; i < v.size(); i += 2)
        out.push_back({v[i], v[i+1]});
    return out;
}

// ── Error metric ─────────────────────────────────────────────
float max_abs_error(
    const std::vector<Complex>& ref,
    const std::vector<Complex>& got)
{
    float err = 0;
    for (size_t i = 0; i < ref.size(); i++)
        err = std::max(err, std::abs(ref[i] - got[i]));
    return err;
}

// ── Single test case ─────────────────────────────────────────
bool run_test(
    Device* device, CommandQueue& cq,
    const std::vector<Complex>& input,
    uint32_t num_cores,
    bool use_bf16,
    bool is_ifft,
    const char* test_name)
{
    uint32_t N = input.size();
    uint32_t total_bytes = N * 2 * (use_bf16 ? 2 : 4);

    // Reference output
    auto ref_out = is_ifft ? reference_idft(input) : reference_dft(input);

    // Create DRAM buffers
    auto in_buf = CreateBuffer(device, {
        .size        = total_bytes,
        .page_size   = use_bf16 ? 2u : 4u,
        .buffer_type = BufferType::DRAM
    });
    auto out_buf = CreateBuffer(device, {
        .size        = total_bytes,
        .page_size   = use_bf16 ? 2u : 4u,
        .buffer_type = BufferType::DRAM
    });

    // Upload input
    auto in_flat = complex_to_interleaved(input);
    if (use_bf16) {
        auto in_bf16 = to_bf16(in_flat);
        EnqueueWriteBuffer(cq, in_buf,
            reinterpret_cast<const void*>(in_bf16.data()), false);
    } else {
        EnqueueWriteBuffer(cq, in_buf,
            reinterpret_cast<const void*>(in_flat.data()), false);
    }

    // Run FFT
    auto t0 = std::chrono::high_resolution_clock::now();
    run_fft(device, cq,
        {.N = N, .num_cores = num_cores,
         .is_ifft = is_ifft, .use_bf16 = use_bf16},
        in_buf, out_buf);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Read back output
    std::vector<float> out_flat(N * 2);
    if (use_bf16) {
        std::vector<bfloat16> out_bf16(N * 2);
        EnqueueReadBuffer(cq, out_buf,
            reinterpret_cast<void*>(out_bf16.data()), true);
        for (size_t i = 0; i < out_bf16.size(); i++)
            out_flat[i] = out_bf16[i].to_float();
    } else {
        EnqueueReadBuffer(cq, out_buf,
            reinterpret_cast<void*>(out_flat.data()), true);
    }
    auto got = interleaved_to_complex(out_flat);

    // Validate
    float tol    = use_bf16 ? 1e-2f : 1e-4f;
    float err    = max_abs_error(ref_out, got);
    bool  passed = err < tol;

    std::printf("[%s] N=%-5u cores=%-2u %s %s | err=%.2e tol=%.2e | %.2f ms  %s\n",
        passed ? "PASS" : "FAIL",
        N, num_cores,
        use_bf16 ? "bf16" : "fp32",
        is_ifft  ? "IFFT" : "FFT ",
        err, tol, ms,
        test_name);

    return passed;
}

// ── Generate test signals ────────────────────────────────────
std::vector<Complex> make_impulse(uint32_t N) {
    std::vector<Complex> x(N, {0,0});
    x[0] = {1, 0};
    return x;
}

std::vector<Complex> make_sine(uint32_t N, uint32_t freq) {
    std::vector<Complex> x(N);
    for (uint32_t n = 0; n < N; n++) {
        float t = 2.0f * M_PI * freq * n / N;
        x[n] = {std::cos(t), std::sin(t)};
    }
    return x;
}

std::vector<Complex> make_random(uint32_t N, uint32_t seed = 42) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x)
        c = {(std::rand() / float(RAND_MAX)) * 2 - 1,
             (std::rand() / float(RAND_MAX)) * 2 - 1};
    return x;
}

// ── Main ─────────────────────────────────────────────────────
int main() {
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();

    bool all_pass = true;

    // ── Test 1: Single-core, small N ────────────────────────
    all_pass &= run_test(device, cq, make_impulse(64),
        1, false, false, "impulse → constant spectrum");

    // ── Test 2: Impulse IFFT ─────────────────────────────────
    // FFT of impulse = all-ones; IFFT of all-ones = impulse/N
    {
        std::vector<Complex> ones(64, {1.0f, 0.0f});
        all_pass &= run_test(device, cq, ones,
            1, false, true, "all-ones IFFT → impulse");
    }

    // ── Test 3: Multi-core FFT, fp32 ────────────────────────
    all_pass &= run_test(device, cq, make_random(1024),
        8, false, false, "random fp32 multicore");

    // ── Test 4: Multi-core FFT, bf16 ────────────────────────
    all_pass &= run_test(device, cq, make_random(1024),
        8, true, false, "random bf16 multicore");

    // ── Test 5: Round-trip FFT → IFFT ───────────────────────
    {
        auto x = make_random(1024);
        uint32_t N = x.size();
        uint32_t total_bytes = N * 2 * 4;

        auto in_buf = CreateBuffer(device, {
            .size = total_bytes, .page_size = 4,
            .buffer_type = BufferType::DRAM});
        auto mid_buf = CreateBuffer(device, {
            .size = total_bytes, .page_size = 4,
            .buffer_type = BufferType::DRAM});
        auto out_buf = CreateBuffer(device, {
            .size = total_bytes, .page_size = 4,
            .buffer_type = BufferType::DRAM});

        auto flat = complex_to_interleaved(x);
        EnqueueWriteBuffer(cq, in_buf,
            reinterpret_cast<const void*>(flat.data()), false);

        fft( device, cq, N, 8, in_buf,  mid_buf);
        ifft(device, cq, N, 8, mid_buf, out_buf);

        std::vector<float> out_flat(N * 2);
        EnqueueReadBuffer(cq, out_buf,
            reinterpret_cast<void*>(out_flat.data()), true);
        auto got = interleaved_to_complex(out_flat);

        float err = max_abs_error(x, got);
        bool pass = err < 1e-4f;
        all_pass &= pass;
        std::printf("[%s] round-trip FFT→IFFT N=1024 cores=8 | err=%.2e\n",
            pass ? "PASS" : "FAIL", err);
    }

    // ── Test 6: Sine wave → single frequency bin ─────────────
    {
        auto x = make_sine(256, 10);   // freq bin 10
        all_pass &= run_test(device, cq, x,
            4, false, false, "sine freq=10, expect peak at bin 10");
    }

    // ── Test 7: Large N, 32 cores ────────────────────────────
    all_pass &= run_test(device, cq, make_random(4096),
        32, false, false, "large N=4096 fp32 32-core");

    all_pass &= run_test(device, cq, make_random(4096),
        32, true,  false, "large N=4096 bf16 32-core");

    CloseDevice(device);

    std::printf("\n%s\n", all_pass ? "All tests PASSED." : "SOME TESTS FAILED.");
    return all_pass ? 0 : 1;
}
