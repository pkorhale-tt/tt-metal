// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_universal_xl_planner_test.cpp
//
// Pure-host unit test for the K-level factorisation planner.  Does NOT
// touch a device, so it builds and runs anywhere — useful for catching
// planner regressions in CI without burning Wormhole time.
//
// Test cases cover:
//   * single-pass regime (N <= 1024)
//   * two-pass regime (N up to 1M, mirrors fft_stockham today)
//   * three-pass regime (N up to 1G, the new XL territory)
//   * boundary cases at the kFactorCap edges
//
// Verifies for every N:
//   1. factor product == N
//   2. every factor is pow2 and <= kFactorCap
//   3. number of factors equals ceil(log2N / log2(kFactorCap))

#include "fft_universal_xl_planner.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using fft_universal_xl::is_pow2;
using fft_universal_xl::kFactorCap;
using fft_universal_xl::log2u;
using fft_universal_xl::plan;
using fft_universal_xl::XLPlan;

int g_pass = 0;
int g_fail = 0;

const char* tick(bool ok) { return ok ? "[PASS]" : "[FAIL]"; }

void check(bool cond, const char* what, uint32_t N, const XLPlan& p) {
    if (cond) { ++g_pass; return; }
    ++g_fail;
    std::printf("%s N=%-12u  %s   factors=[", tick(false), N, what);
    for (size_t i = 0; i < p.factors.size(); ++i) {
        std::printf("%s%u", (i ? ", " : ""), p.factors[i]);
    }
    std::printf("]\n");
}

void verify(uint32_t N) {
    XLPlan p = plan(N);

    // (1) factor product == N
    uint64_t prod = 1ull;
    for (uint32_t f : p.factors) prod *= f;
    check(prod == static_cast<uint64_t>(N), "prod(factors) != N", N, p);

    // (2) all factors pow2 and <= cap
    bool ok_caps = true;
    for (uint32_t f : p.factors) {
        if (!is_pow2(f) || f < 2u || f > kFactorCap) ok_caps = false;
    }
    check(ok_caps, "factor not pow2 or out of [2, kFactorCap]", N, p);

    // (3) k matches ceil(log2N / log2(cap))
    const uint32_t log2N    = log2u(N);
    const uint32_t log2_cap = log2u(kFactorCap);
    const uint32_t k_expect = std::max(1u, (log2N + log2_cap - 1u) / log2_cap);
    check(p.k() == k_expect, "k != ceil(log2N / log2(cap))", N, p);

    // Print one summary line per N.
    std::printf("%s N=%-12u  k=%u  factors=[", tick(true), N, p.k());
    for (size_t i = 0; i < p.factors.size(); ++i) {
        std::printf("%s%u", (i ? ", " : ""), p.factors[i]);
    }
    std::printf("]\n");
}

}  // namespace

int main() {
    std::printf("=== fft_universal_xl planner unit test ===\n");

    // Single-pass (k=1): existing kernels handle directly.
    for (uint32_t N : {2u, 4u, 16u, 64u, 256u, 1024u}) verify(N);
    std::printf("---\n");

    // Two-pass (k=2): the regime fft_stockham covers today.
    for (uint32_t N : {2048u, 4096u, 16384u, 65536u, 262144u, 1048576u}) verify(N);
    std::printf("---\n");

    // Three-pass (k=3): the new XL territory.
    for (uint32_t N : {2097152u, 4194304u, 8388608u, 16777216u,
                       33554432u, 67108864u, 134217728u, 268435456u,
                       536870912u, 1073741824u}) verify(N);
    std::printf("---\n");

    // Boundary cases.
    verify(kFactorCap);                     // exactly 1 factor
    verify(kFactorCap * 2u);                // forces k=2 with [1024, 2]
    verify(kFactorCap * kFactorCap);        // [1024, 1024]
    verify(kFactorCap * kFactorCap * 2u);   // [1024, 1024, 2]

    std::printf("\nResult: %d PASS, %d FAIL\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
