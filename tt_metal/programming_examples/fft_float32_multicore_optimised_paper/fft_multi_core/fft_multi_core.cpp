// SPDX-FileCopyrightText: © 2026 OpenAI
// SPDX-License-Identifier: Apache-2.0
//
// Paper-style 1D FFT host code for Wormhole / TT-Metalium.
// Safe version that follows the paper's basic execution model:
//   stage source row-major buffer -> reader reorder -> butterfly compute -> writer scatter back.
//
// Notes:
// - This keeps the paper's explicit per-stage reordering model because it is the least brittle option
//   under time pressure.
// - Twiddles are precomputed once on host and bulk-read by the reader.
// - The code assumes FP32, power-of-two FFT length, and row-parallel decomposition.
// - Depending on your TT-Metal branch, you may need small namespace / queue API adjustments.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tt_metal/api/tt-metalium/host_api.hpp"

using namespace tt::tt_metal;

namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr uint32_t TILE_ELEMS = 32u * 32u;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

inline uint32_t ceil_div_u32(const uint32_t a, const uint32_t b) {
    return (a + b - 1u) / b;
}

inline uint32_t round_up_u32(const uint32_t value, const uint32_t multiple) {
    return ceil_div_u32(value, multiple) * multiple;
}

inline bool is_power_of_two_u32(const uint32_t x) {
    return x != 0u && ((x & (x - 1u)) == 0u);
}

inline uint32_t log2_u32(const uint32_t x) {
    uint32_t out = 0;
    uint32_t v = x;
    while (v > 1u) {
        v >>= 1u;
        ++out;
    }
    return out;
}

std::vector<float> pack_rows_to_tiled(const std::vector<float>& src, const uint32_t rows, const uint32_t cols) {
    const uint32_t padded_cols = round_up_u32(cols, TILE_ELEMS);
    std::vector<float> out(rows * padded_cols, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        std::memcpy(out.data() + r * padded_cols, src.data() + r * cols, cols * sizeof(float));
    }
    return out;
}

std::vector<float> unpack_rows_from_tiled(const std::vector<float>& src, const uint32_t rows, const uint32_t cols) {
    const uint32_t padded_cols = round_up_u32(cols, TILE_ELEMS);
    std::vector<float> out(rows * cols, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        std::memcpy(out.data() + r * cols, src.data() + r * padded_cols, cols * sizeof(float));
    }
    return out;
}

std::vector<float> build_twiddle_table(
    const uint32_t n_row,
    const uint32_t batch_size,
    const uint32_t num_stages,
    const uint32_t pair_stride_elems,
    const bool inverse) {

    const uint32_t pair_count = n_row >> 1u;
    const float sign = inverse ? 1.0f : -1.0f;
    std::vector<float> twiddles(static_cast<size_t>(num_stages) * batch_size * pair_stride_elems, 0.0f);

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        const uint32_t m = 1u << (stage + 1u);
        const uint32_t half_m = m >> 1u;

        for (uint32_t row = 0; row < batch_size; ++row) {
            float* row_ptr = twiddles.data() + (static_cast<size_t>(stage) * batch_size + row) * pair_stride_elems;
            for (uint32_t p = 0; p < pair_count; ++p) {
                const uint32_t j = p % half_m;
                const uint32_t k = j * (n_row / m);
                const float angle = sign * 2.0f * PI_F * static_cast<float>(k) / static_cast<float>(n_row);
                row_ptr[p] = std::cos(angle);
            }
        }
    }

    return twiddles;
}

std::vector<float> build_twiddle_table_imag(
    const uint32_t n_row,
    const uint32_t batch_size,
    const uint32_t num_stages,
    const uint32_t pair_stride_elems,
    const bool inverse) {

    const uint32_t pair_count = n_row >> 1u;
    const float sign = inverse ? 1.0f : -1.0f;
    std::vector<float> twiddles(static_cast<size_t>(num_stages) * batch_size * pair_stride_elems, 0.0f);

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        const uint32_t m = 1u << (stage + 1u);
        const uint32_t half_m = m >> 1u;

        for (uint32_t row = 0; row < batch_size; ++row) {
            float* row_ptr = twiddles.data() + (static_cast<size_t>(stage) * batch_size + row) * pair_stride_elems;
            for (uint32_t p = 0; p < pair_count; ++p) {
                const uint32_t j = p % half_m;
                const uint32_t k = j * (n_row / m);
                const float angle = sign * 2.0f * PI_F * static_cast<float>(k) / static_cast<float>(n_row);
                row_ptr[p] = std::sin(angle);
            }
        }
    }

    return twiddles;
}

}  // namespace

struct FFTPaperPlan {
    uint32_t n_row = 0;
    uint32_t batch_size = 0;
    uint32_t num_stages = 0;
    uint32_t row_tiles = 0;
    uint32_t pair_tiles = 0;
    uint32_t row_stride_elems = 0;
    uint32_t pair_stride_elems = 0;
    uint32_t active_cores = 0;
    uint32_t rows_per_core = 0;

    IDevice* device = nullptr;
    Program program;
    KernelHandle reader_kernel;
    KernelHandle compute_kernel;
    KernelHandle writer_kernel;

    std::shared_ptr<Buffer> stage0_r;
    std::shared_ptr<Buffer> stage0_i;
    std::shared_ptr<Buffer> stage1_r;
    std::shared_ptr<Buffer> stage1_i;
    std::shared_ptr<Buffer> twiddle_r;
    std::shared_ptr<Buffer> twiddle_i;
    std::shared_ptr<Buffer> output_r;
    std::shared_ptr<Buffer> output_i;
};

FFTPaperPlan create_fft_paper_plan(IDevice* device, const uint32_t n_row, const uint32_t batch_size, const bool inverse) {
    if (device == nullptr) {
        throw std::invalid_argument("device must not be null");
    }
    if (!is_power_of_two_u32(n_row)) {
        throw std::invalid_argument("n_row must be a power of two");
    }
    if (n_row < 2u) {
        throw std::invalid_argument("n_row must be at least 2");
    }
    if (batch_size == 0u) {
        throw std::invalid_argument("batch_size must be > 0");
    }

    FFTPaperPlan plan{};
    plan.device = device;
    plan.n_row = n_row;
    plan.batch_size = batch_size;
    plan.num_stages = log2_u32(n_row);
    plan.row_stride_elems = round_up_u32(n_row, TILE_ELEMS);
    plan.pair_stride_elems = round_up_u32(n_row / 2u, TILE_ELEMS);
    plan.row_tiles = plan.row_stride_elems / TILE_ELEMS;
    plan.pair_tiles = plan.pair_stride_elems / TILE_ELEMS;

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t max_cores = std::max(1u, static_cast<uint32_t>(grid.x * grid.y));
    plan.active_cores = std::min(batch_size, max_cores);
    plan.rows_per_core = ceil_div_u32(batch_size, plan.active_cores);

    const uint32_t stage_bytes = batch_size * plan.row_tiles * TILE_BYTES;
    const uint32_t output_bytes = stage_bytes;
    const uint32_t twiddle_bytes = plan.num_stages * batch_size * plan.pair_tiles * TILE_BYTES;

    plan.stage0_r = CreateBuffer(InterleavedBufferConfig{device, stage_bytes, TILE_BYTES, BufferType::DRAM});
    plan.stage0_i = CreateBuffer(InterleavedBufferConfig{device, stage_bytes, TILE_BYTES, BufferType::DRAM});
    plan.stage1_r = CreateBuffer(InterleavedBufferConfig{device, stage_bytes, TILE_BYTES, BufferType::DRAM});
    plan.stage1_i = CreateBuffer(InterleavedBufferConfig{device, stage_bytes, TILE_BYTES, BufferType::DRAM});
    plan.output_r = CreateBuffer(InterleavedBufferConfig{device, output_bytes, TILE_BYTES, BufferType::DRAM});
    plan.output_i = CreateBuffer(InterleavedBufferConfig{device, output_bytes, TILE_BYTES, BufferType::DRAM});
    plan.twiddle_r = CreateBuffer(InterleavedBufferConfig{device, twiddle_bytes, TILE_BYTES, BufferType::DRAM});
    plan.twiddle_i = CreateBuffer(InterleavedBufferConfig{device, twiddle_bytes, TILE_BYTES, BufferType::DRAM});

    const auto tw_r = build_twiddle_table(n_row, batch_size, plan.num_stages, plan.pair_stride_elems, inverse);
    const auto tw_i = build_twiddle_table_imag(n_row, batch_size, plan.num_stages, plan.pair_stride_elems, inverse);
    WriteToBuffer(plan.twiddle_r, tw_r);
    WriteToBuffer(plan.twiddle_i, tw_i);

    plan.program = CreateProgram();

    auto make_cb_config = [&](const uint32_t pages, std::initializer_list<uint32_t> ids) {
        std::map<uint8_t, tt::DataFormat> formats;
        for (const auto id : ids) {
            formats.emplace(static_cast<uint8_t>(id), tt::DataFormat::Float32);
        }
        CircularBufferConfig cfg(pages * TILE_BYTES, formats);
        for (const auto id : ids) {
            cfg.set_page_size(id, TILE_BYTES);
        }
        return cfg;
    };

    // Launch kernels on the full rectangle; extra cores simply receive rows_this_core = 0.
    const uint32_t used_rows = ceil_div_u32(plan.active_cores, static_cast<uint32_t>(grid.x));
    const CoreRange all_cores({0, 0}, {grid.x - 1u, used_rows - 1u});

    CreateCircularBuffer(plan.program, all_cores, make_cb_config(2u * plan.pair_tiles, {
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_3,
        tt::CBIndex::c_4, tt::CBIndex::c_5,
        tt::CBIndex::c_16, tt::CBIndex::c_17, tt::CBIndex::c_18, tt::CBIndex::c_19
    }));
    CreateCircularBuffer(plan.program, all_cores, make_cb_config(1u, {
        tt::CBIndex::c_20, tt::CBIndex::c_21, tt::CBIndex::c_22, tt::CBIndex::c_23
    }));
    CreateCircularBuffer(plan.program, all_cores, make_cb_config(plan.row_tiles, {
        tt::CBIndex::c_24, tt::CBIndex::c_25
    }));

    plan.reader_kernel = CreateKernel(
        plan.program,
        "/mnt/data/reader_fft_f32_paper.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    plan.writer_kernel = CreateKernel(
        plan.program,
        "/mnt/data/writer_fft_f32_paper.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    plan.compute_kernel = CreateKernel(
        plan.program,
        "/mnt/data/fft_compute_f32_paper.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});

    for (uint32_t core_index = 0; core_index < used_rows * static_cast<uint32_t>(grid.x); ++core_index) {
        const CoreCoord core{core_index % static_cast<uint32_t>(grid.x), core_index / static_cast<uint32_t>(grid.x)};
        const uint32_t row_start = core_index * plan.rows_per_core;
        const uint32_t rows_this_core = row_start < batch_size
            ? std::min(plan.rows_per_core, batch_size - row_start)
            : 0u;

        SetRuntimeArgs(plan.program, plan.reader_kernel, core, {
            plan.stage0_r->address(),
            plan.stage0_i->address(),
            plan.stage1_r->address(),
            plan.stage1_i->address(),
            plan.twiddle_r->address(),
            plan.twiddle_i->address(),
            plan.row_tiles,
            plan.pair_tiles,
            plan.n_row,
            plan.num_stages,
            plan.batch_size,
            row_start,
            rows_this_core,
        });

        SetRuntimeArgs(plan.program, plan.compute_kernel, core, {
            plan.num_stages,
            rows_this_core,
            plan.pair_tiles,
        });

        SetRuntimeArgs(plan.program, plan.writer_kernel, core, {
            plan.stage0_r->address(),
            plan.stage0_i->address(),
            plan.stage1_r->address(),
            plan.stage1_i->address(),
            plan.output_r->address(),
            plan.output_i->address(),
            plan.row_tiles,
            plan.pair_tiles,
            plan.n_row,
            plan.num_stages,
            row_start,
            rows_this_core,
        });
    }

    return plan;
}

void run_fft_paper_plan(
    FFTPaperPlan& plan,
    const std::vector<float>& input_real,
    const std::vector<float>& input_imag,
    std::vector<float>& output_real,
    std::vector<float>& output_imag) {

    const size_t expected = static_cast<size_t>(plan.batch_size) * plan.n_row;
    if (input_real.size() != expected || input_imag.size() != expected) {
        throw std::invalid_argument("input sizes must be batch_size * n_row");
    }

    const auto packed_real = pack_rows_to_tiled(input_real, plan.batch_size, plan.n_row);
    const auto packed_imag = pack_rows_to_tiled(input_imag, plan.batch_size, plan.n_row);

    WriteToBuffer(plan.stage0_r, packed_real);
    WriteToBuffer(plan.stage0_i, packed_imag);

    EnqueueProgram(plan.device->command_queue(), plan.program, false);
    Finish(plan.device->command_queue());

    std::vector<float> packed_out_real(plan.batch_size * plan.row_stride_elems, 0.0f);
    std::vector<float> packed_out_imag(plan.batch_size * plan.row_stride_elems, 0.0f);
    ReadFromBuffer(plan.output_r, packed_out_real);
    ReadFromBuffer(plan.output_i, packed_out_imag);

    output_real = unpack_rows_from_tiled(packed_out_real, plan.batch_size, plan.n_row);
    output_imag = unpack_rows_from_tiled(packed_out_imag, plan.batch_size, plan.n_row);
}
