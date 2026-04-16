// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_host.cpp — single-core radix-2 DIT FFT on Wormhole.
//
// Constraints:
//   * N is a power of two, 2 <= N <= 1024 (fits in one 32x32 fp32 tile).
//   * FFT only (no IFFT).
//
// Flow:
//   1. pack_input bit-reverses and splits into (real_tile, imag_tile).
//   2. precompute_twiddles builds one tile per stage, each holding N/2 DIT
//      twiddle factors for that stage.
//   3. run_fft uploads input+twiddles, launches reader/compute/writer, and
//      returns the output tiles in out_r_buf/out_i_buf.
//
// The heavy lifting (butterfly math) happens on TRISC as tile ops; the reader
// on BRISC0 does the stage-dependent gather/scatter and DRAM loads; the
// writer on BRISC1 flushes the final state tile to DRAM.

#pragma once

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include <cmath>
#include <vector>
#include <cassert>
#include <complex>
#include <utility>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace fft_example {

constexpr uint32_t kTileHW       = 32;
constexpr uint32_t kTileElems    = kTileHW * kTileHW;            // 1024
constexpr uint32_t kTileSizeFp32 = kTileElems * sizeof(float);   // 4096 bytes

// CB layout — must match kernel/fft_common.h
constexpr uint32_t CB_EVEN_R   = 0;
constexpr uint32_t CB_EVEN_I   = 1;
constexpr uint32_t CB_ODD_R    = 2;
constexpr uint32_t CB_ODD_I    = 3;
constexpr uint32_t CB_TW_R     = 4;
constexpr uint32_t CB_TW_I     = 5;
constexpr uint32_t CB_OUT0_R   = 6;
constexpr uint32_t CB_OUT0_I   = 7;
constexpr uint32_t CB_OUT1_R   = 8;
constexpr uint32_t CB_OUT1_I   = 9;
constexpr uint32_t CB_TMP_R    = 10;
constexpr uint32_t CB_TMP_I    = 11;
constexpr uint32_t CB_TW_ODD_R = 12;
constexpr uint32_t CB_TW_ODD_I = 13;
constexpr uint32_t CB_STATE_R  = 14;
constexpr uint32_t CB_STATE_I  = 15;
constexpr uint32_t CB_SYNC     = 16;
constexpr uint32_t NUM_CBS     = 17;

struct FFTConfig { uint32_t N; };

// ── Helpers ───────────────────────────────────────────────────────────────

inline std::shared_ptr<MeshBuffer> make_mesh_buf(
    std::shared_ptr<MeshDevice> md, uint32_t size, uint32_t page_size)
{
    ReplicatedBufferConfig rep{.size = size};
    DeviceLocalBufferConfig dev{.page_size = page_size, .buffer_type = BufferType::DRAM};
    return MeshBuffer::create(rep, dev, md.get());
}

inline uint32_t buf_addr(const std::shared_ptr<MeshBuffer>& mb) {
    return mb->get_device_buffer(MeshCoordinate(0, 0))->address();
}

inline uint32_t log2u(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) ++r;
    return r;
}

inline uint32_t bit_rev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) { r = (r << 1) | (x & 1u); x >>= 1u; }
    return r;
}

// Pack an N-point complex signal into two fp32 tiles (real, imag), already
// bit-reversed so the DIT butterflies start in stage-0 layout. Slots [N..1023]
// are zero-padded.
inline std::pair<std::vector<float>, std::vector<float>> pack_input(
    const std::vector<std::complex<float>>& x)
{
    const uint32_t N     = static_cast<uint32_t>(x.size());
    const uint32_t log2N = log2u(N);
    std::vector<float> r(kTileElems, 0.0f), i(kTileElems, 0.0f);
    for (uint32_t k = 0; k < N; ++k) {
        const uint32_t src = bit_rev(k, log2N);
        r[k] = x[src].real();
        i[k] = x[src].imag();
    }
    return {std::move(r), std::move(i)};
}

inline std::vector<std::complex<float>> unpack_output(
    const std::vector<float>& r, const std::vector<float>& i, uint32_t N)
{
    std::vector<std::complex<float>> out(N);
    for (uint32_t k = 0; k < N; ++k) out[k] = {r[k], i[k]};
    return out;
}

// Stage-s twiddle for pair index p (0 <= p < N/2):
//   k = p mod 2^s,   M = 2^(s+1)
//   W^k = cos(-2pi*k/M) + i*sin(-2pi*k/M)
// One full tile per stage; pair index p lives at slot p; slots [N/2..1023]
// are zero (harmless: they pair with zero-padded even/odd lanes).
inline std::pair<std::vector<float>, std::vector<float>> precompute_twiddles(
    uint32_t N, uint32_t num_stages)
{
    const uint32_t num_pairs = N / 2;
    std::vector<float> r(num_stages * kTileElems, 0.0f);
    std::vector<float> i(num_stages * kTileElems, 0.0f);
    for (uint32_t s = 0; s < num_stages; ++s) {
        const uint32_t stride = 1u << s;
        const double   Mf     = static_cast<double>(stride) * 2.0;
        float* const tile_r = r.data() + s * kTileElems;
        float* const tile_i = i.data() + s * kTileElems;
        for (uint32_t p = 0; p < num_pairs; ++p) {
            const uint32_t k     = p & (stride - 1u);
            const double   angle = -2.0 * M_PI * static_cast<double>(k) / Mf;
            tile_r[p] = static_cast<float>(std::cos(angle));
            tile_i[p] = static_cast<float>(std::sin(angle));
        }
    }
    return {std::move(r), std::move(i)};
}

// ── Launch ────────────────────────────────────────────────────────────────

inline void run_fft(
    std::shared_ptr<MeshDevice> md,
    const FFTConfig& cfg,
    std::shared_ptr<MeshBuffer> in_r_buf,
    std::shared_ptr<MeshBuffer> in_i_buf,
    std::shared_ptr<MeshBuffer> out_r_buf,
    std::shared_ptr<MeshBuffer> out_i_buf)
{
    assert(cfg.N >= 2 && cfg.N <= kTileElems);
    assert((cfg.N & (cfg.N - 1)) == 0);

    MeshCommandQueue& cq = md->mesh_command_queue();
    const uint32_t log2N = log2u(cfg.N);

    // --- twiddle tables in DRAM --------------------------------------------
    auto [tw_r_data, tw_i_data] = precompute_twiddles(cfg.N, log2N);
    auto tw_r_buf = make_mesh_buf(md, log2N * kTileSizeFp32, kTileSizeFp32);
    auto tw_i_buf = make_mesh_buf(md, log2N * kTileSizeFp32, kTileSizeFp32);
    WriteShard(cq, tw_r_buf, tw_r_data, MeshCoordinate(0, 0), false);
    WriteShard(cq, tw_i_buf, tw_i_data, MeshCoordinate(0, 0), false);

    // --- program & CBs -----------------------------------------------------
    Program prog = CreateProgram();
    CoreCoord core{0, 0};
    CoreRange cr(core, core);

    // Pipelined CBs: EVEN/ODD/TW/OUT get 2 tiles, scratch/state/sync get 1.
    // Indices:                        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    constexpr uint32_t kCbTiles[17] = {2,2,2,2,2,2,2,2,2,2, 1, 1, 1, 1, 1, 1, 1};
    static_assert(sizeof(kCbTiles) / sizeof(kCbTiles[0]) == NUM_CBS);

    for (uint32_t id = 0; id < NUM_CBS; ++id) {
        CircularBufferConfig c(
            kCbTiles[id] * kTileSizeFp32,
            {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        CreateCircularBuffer(prog, cr, c);
    }

    // --- kernels -----------------------------------------------------------
    auto rk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft/kernel/fft_reader.cpp",
        cr,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = {cfg.N, log2N}});

    auto wk = CreateKernel(
        prog,
        "tt_metal/programming_examples/fft/kernel/fft_writer.cpp",
        cr,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    CreateKernel(
        prog,
        "tt_metal/programming_examples/fft/kernel/fft_compute.cpp",
        cr,
        ComputeConfig{
            .math_fidelity     = MathFidelity::HiFi4,
            .fp32_dest_acc_en  = true,
            .compile_args      = {log2N}});

    // --- runtime args ------------------------------------------------------
    SetRuntimeArgs(prog, rk, core, {
        buf_addr(in_r_buf), buf_addr(in_i_buf),
        buf_addr(tw_r_buf), buf_addr(tw_i_buf),
    });
    SetRuntimeArgs(prog, wk, core, {
        buf_addr(out_r_buf), buf_addr(out_i_buf),
    });

    MeshWorkload workload;
    workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)),
        std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    cq.finish();
}

}  // namespace fft_example
