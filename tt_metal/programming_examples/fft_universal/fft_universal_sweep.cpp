// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_sweep.cpp — HPEC 2026 paper benchmark driver.
//
// Sweeps `fft_universal::fft` across many N values in one device-open
// session (so plan-cache + JIT cost is amortized exactly the way the
// paper claims), and emits one CSV row per N with the numbers needed
// for the paper figures:
//   * cold wall time (first call, includes plan build + JIT)
//   * cached wall time (median / p05 / p95 over N-1 subsequent iters)
//   * GFLOPs  (5·N·log2(N) / time, the Brown / cuFFT convention)
//   * samples/sec  (N / time)
//   * dispatch path label  (pow2 / packed-DFT / Bluestein / Cooley-Tukey)
//
// Companion file `plot_universal.py` reads the CSV and emits the
// log-log latency-vs-N plot (paper Fig. 1), GFLOPs-vs-N (Fig. 2), and
// the Brown Table-1 replica bar chart (Fig. 3).
//
// Usage:
//   metal_example_fft_universal_sweep [--csv path] [--iters N]
//                                     [--N-list "n1,n2,n3,..."]
//                                     [--round-trip] [--include-cold]
//
//   Default sweep covers the full paper range:
//     pow-2: 2,4,8,...,1048576  (21 points, log2-spaced)
//     packed-DFT corners: 3,5,7,11,13,17,23,31
//     just-above pow2: 33,65,129,257,1025
//     primes for Bluestein: 127,257,1009,7919,65537
//     composite non-pow2 (mixed-radix): 6,10,12,15,24,100,384,6144,100003
//
// Build & run (on the Wormhole box):
//   ninja -C build metal_example_fft_universal_sweep
//   build/programming_examples/fft_universal/metal_example_fft_universal_sweep \
//       --csv results/universal_sweep.csv --iters 50

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "fft_universal_host.cpp"

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

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

// ────────────────────────── default sweep ────────────────────────────
static const std::vector<uint32_t> kDefaultNs = {
    // pow-2 (single-tile, six-step, four-step paths)
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    524288, 1048576,
    // primes (Bluestein chirp-Z path)
    3, 5, 7, 11, 13, 17, 23, 31, 127, 257, 1009, 7919, 65537,
    // just-above-pow2 to stress padding logic
    33, 65, 129, 257, 1025,
    // composite non-pow2 (mixed-radix Cooley-Tukey)
    6, 10, 12, 15, 24, 100, 384, 1000, 6144, 100003,
};

// ────────────────────────── helpers ──────────────────────────────────
static std::vector<Complex> make_random(uint32_t N, uint32_t seed = 42) {
    std::vector<Complex> x(N);
    std::srand(seed);
    for (auto& c : x) {
        c = {(std::rand() / float(RAND_MAX)) * 2.0f - 1.0f,
             (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f};
    }
    return x;
}

static const char* describe_path(uint32_t N) {
    if (N == 1u)                    return "identity";
    if (N <= 32u)                   return "packed_dft";
    if (fft_universal::is_pow2(N))  return "pow2_stockham";
    if (fft_universal::is_prime(N)) return "bluestein";
    return "cooley_tukey";
}

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(std::round(p * (v.size() - 1)));
    return v[idx];
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

static double gflops_count(uint32_t N) {
    // 5·N·log2(N) — standard FFT FLOP count (matches Brown 2025, cuFFT,
    // FFTW).  log2(1) = 0 → identity reports 0 GFLOPs.
    if (N <= 1u) return 0.0;
    uint32_t log2N = 0;
    for (uint32_t t = N; t > 1u; t >>= 1) ++log2N;
    return 5.0 * static_cast<double>(N) * static_cast<double>(log2N);
}

// ────────────────────────── one-N measurement ────────────────────────
struct Row {
    uint32_t    N;
    std::string path;
    double      cold_ms;
    double      cached_median_ms;
    double      cached_p05_ms;
    double      cached_p95_ms;
    double      cached_min_ms;
    double      cached_max_ms;
    double      cached_mean_ms;
    uint32_t    n_iters;
    double      gflops_median;
    double      msamples_per_sec_median;
    double      host_pct_median;       // 100 * host_ns / wall_ns, median across iters
    double      device_pct_median;     // 100 * device_ns / wall_ns, median across iters
    double      dispatches_per_call;   // mean dispatches per call (cached iters)
    double      roundtrip_rel_err;     // NaN if --round-trip not set
};

static Row measure_one(
    std::shared_ptr<MeshDevice> md,
    uint32_t                    N,
    uint32_t                    iters,
    bool                        do_round_trip)
{
    auto signal = make_random(N);

    std::vector<double> dt(iters, 0.0);
    std::vector<double> host_pct(iters, 0.0);
    std::vector<double> dev_pct(iters, 0.0);
    std::vector<double> ndisp(iters, 0.0);
    for (uint32_t i = 0; i < iters; ++i) {
        fft_universal::profile::current().reset();
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto X = fft_universal::fft(md, signal);
        const auto wall_ns = std::chrono::duration<double, std::nano>(
            std::chrono::high_resolution_clock::now() - t0).count();
        dt[i] = wall_ns * 1e-6;

        const auto& p = fft_universal::profile::current();
        const double dev_ns = std::chrono::duration<double, std::nano>(p.device_ns).count();
        dev_pct[i]  = (wall_ns > 0.0) ? 100.0 * dev_ns / wall_ns : 0.0;
        host_pct[i] = std::max(0.0, 100.0 - dev_pct[i]);
        ndisp[i]    = static_cast<double>(p.n_dispatches);
    }

    const double cold = dt[0];
    std::vector<double> cached(dt.begin() + 1, dt.end());
    std::vector<double> host_cached(host_pct.begin() + 1, host_pct.end());
    std::vector<double> dev_cached(dev_pct.begin() + 1, dev_pct.end());
    std::vector<double> ndisp_cached(ndisp.begin() + 1, ndisp.end());

    Row r;
    r.N                = N;
    r.path             = describe_path(N);
    r.cold_ms          = cold;
    r.cached_median_ms = median(cached);
    r.cached_p05_ms    = percentile(cached, 0.05);
    r.cached_p95_ms    = percentile(cached, 0.95);
    r.cached_min_ms    = *std::min_element(cached.begin(), cached.end());
    r.cached_max_ms    = *std::max_element(cached.begin(), cached.end());
    r.cached_mean_ms   = mean(cached);
    r.n_iters          = iters;

    const double flops    = gflops_count(N);
    const double t_sec    = r.cached_median_ms * 1e-3;
    r.gflops_median           = (t_sec > 0.0) ? flops / t_sec / 1e9 : 0.0;
    r.msamples_per_sec_median = (t_sec > 0.0) ? double(N) / t_sec / 1e6 : 0.0;
    r.host_pct_median         = median(host_cached);
    r.device_pct_median       = median(dev_cached);
    r.dispatches_per_call     = mean(ndisp_cached);

    r.roundtrip_rel_err = std::numeric_limits<double>::quiet_NaN();
    if (do_round_trip) {
        auto X  = fft_universal::fft(md, signal);
        auto rt = fft_universal::ifft(md, X);
        double max_abs_in = 0.0, max_abs_err = 0.0;
        for (size_t i = 0; i < signal.size(); ++i) {
            max_abs_in  = std::max<double>(max_abs_in, std::abs(signal[i]));
            max_abs_err = std::max<double>(max_abs_err,
                                           std::abs(rt[i] - signal[i]));
        }
        r.roundtrip_rel_err = max_abs_err / std::max(1e-30, max_abs_in);
    }

    return r;
}

// ────────────────────────── CLI parsing ──────────────────────────────
static bool parse_uint_csv(const std::string& s, std::vector<uint32_t>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        try {
            const long v = std::stol(item);
            if (v <= 0) return false;
            out.push_back(static_cast<uint32_t>(v));
        } catch (...) { return false; }
    }
    return !out.empty();
}

// ────────────────────────── main ─────────────────────────────────────
int main(int argc, char** argv) {
    std::string csv_path     = "";
    uint32_t    iters        = 50u;
    bool        do_round_trip = false;
    bool        include_cold = false;
    std::vector<uint32_t> Ns = kDefaultNs;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (a == "--iters" && i + 1 < argc) {
            iters = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (a == "--N-list" && i + 1 < argc) {
            if (!parse_uint_csv(argv[++i], Ns)) {
                std::fprintf(stderr,
                    "Bad --N-list (expected comma-separated positive ints)\n");
                return 1;
            }
        } else if (a == "--round-trip") {
            do_round_trip = true;
        } else if (a == "--include-cold") {
            include_cold = true;
            (void)include_cold;  // reserved for future use
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "Usage: %s [--csv path] [--iters N] [--N-list \"n1,n2,...\"]\n"
                "         [--round-trip] [--include-cold]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown arg: %s (use --help)\n", argv[0]);
            return 1;
        }
    }
    if (iters < 2u) iters = 2u;

    std::printf("=== fft_universal_sweep ===\n");
    std::printf("  N values     : %zu\n", Ns.size());
    std::printf("  iters per N  : %u\n", iters);
    std::printf("  round trip   : %s\n", do_round_trip ? "yes" : "no");
    std::printf("  CSV output   : %s\n",
                csv_path.empty() ? "(stdout only)" : csv_path.c_str());
    std::printf("\n");

    auto md = MeshDevice::create_unit_mesh(0);

    std::vector<Row> rows;
    rows.reserve(Ns.size());

    // Table header for stdout
    std::printf("  %8s  %-15s  %10s  %10s  %10s  %10s  %7s  %7s  %6s\n",
                "N", "path", "cold_ms", "med_ms", "p05_ms", "p95_ms",
                "GFLOPs", "host%", "ndisp");

    for (uint32_t N : Ns) {
        try {
            Row r = measure_one(md, N, iters, do_round_trip);
            rows.push_back(r);

            std::printf("  %8u  %-15s  %10.3f  %10.3f  %10.3f  %10.3f  %7.2f  %7.1f  %6.1f\n",
                r.N, r.path.c_str(),
                r.cold_ms, r.cached_median_ms,
                r.cached_p05_ms, r.cached_p95_ms,
                r.gflops_median,
                r.host_pct_median,
                r.dispatches_per_call);
            std::fflush(stdout);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "  %8u  SKIP (exception: %s)\n", N, e.what());
        } catch (...) {
            std::fprintf(stderr, "  %8u  SKIP (unknown exception)\n", N);
        }
    }

    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        if (!f) {
            std::fprintf(stderr, "Could not open %s for writing\n",
                         csv_path.c_str());
        } else {
            f << "N,path,cold_ms,cached_median_ms,cached_p05_ms,cached_p95_ms,"
                 "cached_min_ms,cached_max_ms,cached_mean_ms,n_iters,"
                 "gflops_median,msamples_per_sec_median,host_pct_median,"
                 "device_pct_median,dispatches_per_call,roundtrip_rel_err\n";
            for (const auto& r : rows) {
                f << r.N << ',' << r.path << ','
                  << r.cold_ms << ',' << r.cached_median_ms << ','
                  << r.cached_p05_ms << ',' << r.cached_p95_ms << ','
                  << r.cached_min_ms << ',' << r.cached_max_ms << ','
                  << r.cached_mean_ms << ',' << r.n_iters << ','
                  << r.gflops_median << ',' << r.msamples_per_sec_median << ','
                  << r.host_pct_median << ',' << r.device_pct_median << ','
                  << r.dispatches_per_call << ',';
                if (std::isnan(r.roundtrip_rel_err)) f << "";
                else f << r.roundtrip_rel_err;
                f << '\n';
            }
            std::printf("\n  wrote %zu rows to %s\n",
                        rows.size(), csv_path.c_str());
        }
    }

    md.reset();
    return 0;
}
