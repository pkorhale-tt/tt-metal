// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fftw_baseline.cpp — HPEC 2026 paper, host-CPU FFTW3 baseline.
//
// Mirrors the N list and CSV schema of metal_example_fft_universal_sweep
// so combine_results.py can join the two CSVs and overlay them on the
// same Fig. 1/2 axes. We use single-precision (fftwf_*) to match the
// fp32 path of fft_universal.
//
// We measure FFTW under the "library FFT" convention that mirrors how
// real applications use it: plan-once-execute-many.
//   * Plan with FFTW_MEASURE  (per-N, one-time cost, NOT timed)
//   * Reuse the plan across `iters` executes; reported time is execute-only
//   * Single-thread by default; pass --threads N to enable libfftw3f_threads
//
// This is the most-favourable-to-FFTW configuration. If our Wormhole
// numbers beat THIS, we beat FFTW honestly.
//
// Build (on the box): ninja -C build_Release fftw_baseline
// Run: build_Release/programming_examples/fftw_baseline/fftw_baseline \
//          --csv paper_results/fftw_baseline.csv --iters 200

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using Complex = std::complex<float>;

// Must match fft_universal_sweep.cpp default sweep exactly.
static const std::vector<unsigned> kDefaultNs = {
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    524288, 1048576,
    3, 5, 7, 11, 13, 17, 23, 31, 127, 257, 1009, 7919, 65537,
    33, 65, 129, 257, 1025,
    6, 10, 12, 15, 24, 100, 384, 1000, 6144, 100003,
};

static std::vector<Complex> make_random(unsigned N, unsigned seed = 42) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x) {
        c = {(std::rand() / float(RAND_MAX)) * 2.0f - 1.0f,
             (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f};
    }
    return x;
}

static const char* describe_path(unsigned N) {
    // Mirror the categories used in the Wormhole sweep so the figures
    // line up. FFTW does not internally label its dispatch like ours,
    // but for plotting purposes the N classification is the same.
    auto is_pow2  = [](unsigned n){ return n && !(n & (n - 1u)); };
    auto is_prime = [](unsigned n) {
        if (n < 2u) return false;
        if (n < 4u) return true;
        if ((n & 1u) == 0u) return false;
        for (unsigned d = 3; (unsigned long long)d * d <= (unsigned long long)n; d += 2)
            if (n % d == 0u) return false;
        return true;
    };
    if (N == 1u)        return "identity";
    if (N <= 32u)       return "packed_dft";
    if (is_pow2(N))     return "pow2_stockham";
    if (is_prime(N))    return "bluestein";
    return "cooley_tukey";
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(std::round(p * (v.size() - 1)))];
}
static double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double mean(const std::vector<double>& v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
static double gflops_count(unsigned N) {
    if (N <= 1u) return 0.0;
    unsigned log2N = 0; for (unsigned t = N; t > 1u; t >>= 1) ++log2N;
    if ((1u << log2N) != N) {
        // Use natural log scaling for non-pow-2 — same convention as
        // the cuFFT / Brown reports (5 N log2 N is an upper bound).
        log2N = static_cast<unsigned>(std::ceil(std::log2(double(N))));
    }
    return 5.0 * double(N) * double(log2N);
}

struct Row {
    unsigned    N;
    std::string path;
    double      plan_ms;
    double      median_ms;
    double      p05_ms;
    double      p95_ms;
    double      min_ms;
    double      max_ms;
    double      mean_ms;
    unsigned    n_iters;
    double      gflops_median;
    double      msamples_per_sec_median;
    int         threads;
};

static Row measure_one(unsigned N, unsigned iters, int threads) {
    auto signal = make_random(N);

    // Allocate FFTW-aligned buffers.
    fftwf_complex* in  = reinterpret_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * N));
    fftwf_complex* out = reinterpret_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * N));

    // Plan with FFTW_MEASURE (real wall-time tuned, cached by FFTW
    // wisdom for the process lifetime).
    const auto plan_t0 = std::chrono::high_resolution_clock::now();
    fftwf_plan plan = fftwf_plan_dft_1d(
        static_cast<int>(N), in, out, FFTW_FORWARD, FFTW_MEASURE);
    const double plan_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - plan_t0).count();

    // FFTW_MEASURE clobbers the input array — refill after planning.
    for (unsigned i = 0; i < N; ++i) {
        in[i][0] = signal[i].real();
        in[i][1] = signal[i].imag();
    }

    std::vector<double> dt(iters, 0.0);
    for (unsigned i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fftwf_execute(plan);
        dt[i] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    Row r;
    r.N         = N;
    r.path      = describe_path(N);
    r.plan_ms   = plan_ms;
    r.median_ms = median(dt);
    r.p05_ms    = percentile(dt, 0.05);
    r.p95_ms    = percentile(dt, 0.95);
    r.min_ms    = *std::min_element(dt.begin(), dt.end());
    r.max_ms    = *std::max_element(dt.begin(), dt.end());
    r.mean_ms   = mean(dt);
    r.n_iters   = iters;
    const double t_sec = r.median_ms * 1e-3;
    r.gflops_median           = (t_sec > 0.0) ? gflops_count(N) / t_sec / 1e9 : 0.0;
    r.msamples_per_sec_median = (t_sec > 0.0) ? double(N) / t_sec / 1e6     : 0.0;
    r.threads   = threads;
    return r;
}

static bool parse_uint_csv(const std::string& s, std::vector<unsigned>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        try {
            long v = std::stol(item);
            if (v <= 0) return false;
            out.push_back(static_cast<unsigned>(v));
        } catch (...) { return false; }
    }
    return !out.empty();
}

int main(int argc, char** argv) {
    std::string csv_path = "";
    unsigned    iters    = 200u;
    int         threads  = 1;
    std::vector<unsigned> Ns = kDefaultNs;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--N-list" && i + 1 < argc) {
            if (!parse_uint_csv(argv[++i], Ns)) {
                std::fprintf(stderr, "Bad --N-list\n");
                return 1;
            }
        }
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--csv path] [--iters N] [--threads T]\n"
                        "         [--N-list \"n1,n2,...\"]\n", argv[0]);
            return 0;
        }
        else { std::fprintf(stderr, "Unknown: %s\n", argv[1]); return 1; }
    }
    if (iters < 2u) iters = 2u;
    if (threads < 1) threads = 1;

#ifdef FFTW_HAS_THREADS
    fftwf_init_threads();
    fftwf_plan_with_nthreads(threads);
#else
    if (threads != 1) {
        std::fprintf(stderr,
            "warn: built without fftw3f_threads, ignoring --threads %d\n",
            threads);
        threads = 1;
    }
#endif

    std::printf("=== fftw_baseline (host CPU) ===\n");
    std::printf("  N values   : %zu\n", Ns.size());
    std::printf("  iters/N    : %u\n", iters);
    std::printf("  threads    : %d\n", threads);
    std::printf("  CSV output : %s\n",
                csv_path.empty() ? "(stdout only)" : csv_path.c_str());
    std::printf("\n");
    std::printf("  %8s  %-15s  %10s  %10s  %10s  %10s  %10s\n",
                "N", "path", "plan_ms", "med_ms", "p05_ms", "p95_ms", "GFLOPs");

    std::vector<Row> rows;
    rows.reserve(Ns.size());
    for (unsigned N : Ns) {
        try {
            Row r = measure_one(N, iters, threads);
            rows.push_back(r);
            std::printf(
                "  %8u  %-15s  %10.3f  %10.3f  %10.3f  %10.3f  %10.2f\n",
                r.N, r.path.c_str(),
                r.plan_ms, r.median_ms, r.p05_ms, r.p95_ms,
                r.gflops_median);
            std::fflush(stdout);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  %8u  SKIP (exception: %s)\n", N, e.what());
        }
    }

    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        if (!f) {
            std::fprintf(stderr, "Could not write %s\n", csv_path.c_str());
        } else {
            f << "N,path,plan_ms,median_ms,p05_ms,p95_ms,min_ms,max_ms,"
                 "mean_ms,n_iters,gflops_median,msamples_per_sec_median,threads\n";
            for (const auto& r : rows) {
                f << r.N << ',' << r.path << ',' << r.plan_ms << ','
                  << r.median_ms << ',' << r.p05_ms << ',' << r.p95_ms << ','
                  << r.min_ms << ',' << r.max_ms << ',' << r.mean_ms << ','
                  << r.n_iters << ',' << r.gflops_median << ','
                  << r.msamples_per_sec_median << ',' << r.threads << '\n';
            }
            std::printf("\n  wrote %zu rows to %s\n",
                        rows.size(), csv_path.c_str());
        }
    }

#ifdef FFTW_HAS_THREADS
    fftwf_cleanup_threads();
#endif
    return 0;
}
