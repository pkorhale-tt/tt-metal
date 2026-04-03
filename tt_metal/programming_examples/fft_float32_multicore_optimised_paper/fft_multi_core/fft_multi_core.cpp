// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_multi_core.cpp
//
// Host code matched to these kernels:
//   - reader_fft_f32_prod.cpp
//   - writer_fft_f32_prod.cpp
//   - fft_compute_f32_prod.cpp
//
// This file fixes the host API mismatch by using MeshDevice / MeshBuffer APIs.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "tt_metal/api/tt-metalium/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;            // 1024
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float); // 4096

constexpr uint32_t CB_EVEN_R = 0;
constexpr uint32_t CB_EVEN_I = 1;
constexpr uint32_t CB_ODD_R  = 2;
constexpr uint32_t CB_ODD_I  = 3;
constexpr uint32_t CB_TW_R   = 4;
constexpr uint32_t CB_TW_I   = 5;

constexpr uint32_t CB_OUT0_R = 16;
constexpr uint32_t CB_OUT0_I = 17;
constexpr uint32_t CB_OUT1_R = 18;
constexpr uint32_t CB_OUT1_I = 19;

constexpr uint32_t CB_TMP_R  = 20;
constexpr uint32_t CB_TMP_I  = 21;

inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

inline bool is_power_of_two(uint32_t x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

inline uint32_t log2_u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) {
        ++r;
    }
    return r;
}

inline uint32_t float_to_u32(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
}

inline float u32_to_float(uint32_t v) {
    float out;
    std::memcpy(&out, &v, sizeof(uint32_t));
    return out;
}

std::shared_ptr<distributed::MeshBuffer> create_dram_mesh_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t size_bytes) {

    distributed::DeviceLocalBufferConfig dram_cfg{
        .page_size = TILE_BYTES,
        .buffer_type = BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig rep_cfg{
        .size = size_bytes
    };

    return distributed::MeshBuffer::create(rep_cfg, dram_cfg, mesh_device.get());
}

// Pack stage-0 input into even/odd layout expected by your reader kernel.
// For stage 0 of radix-2 FFT:
//   even[k] = x[2k]
//   odd[k]  = x[2k+1]
void pack_stage0_even_odd(
    const std::vector<float>& input_r,
    const std::vector<float>& input_i,
    uint32_t batch_size,
    uint32_t N_row,
    uint32_t num_cores,
    uint32_t rows_per_core,
    std::vector<uint32_t>& even_r_packed,
    std::vector<uint32_t>& even_i_packed,
    std::vector<uint32_t>& odd_r_packed,
    std::vector<uint32_t>& odd_i_packed) {

    const uint32_t batch_padded = num_cores * rows_per_core;
    const uint32_t half_N = N_row / 2;
    const uint32_t tiles_per_row = ceil_div(half_N, TILE_ELEMS);
    const uint32_t elems_per_row_padded = tiles_per_row * TILE_ELEMS;

    even_r_packed.assign(batch_padded * elems_per_row_padded, 0);
    even_i_packed.assign(batch_padded * elems_per_row_padded, 0);
    odd_r_packed.assign(batch_padded * elems_per_row_padded, 0);
    odd_i_packed.assign(batch_padded * elems_per_row_padded, 0);

    for (uint32_t row = 0; row < batch_size; ++row) {
        const uint32_t dst_base = row * elems_per_row_padded;
        const uint32_t src_base = row * N_row;

        for (uint32_t k = 0; k < half_N; ++k) {
            even_r_packed[dst_base + k] = float_to_u32(input_r[src_base + 2 * k]);
            even_i_packed[dst_base + k] = float_to_u32(input_i[src_base + 2 * k]);
            odd_r_packed [dst_base + k] = float_to_u32(input_r[src_base + 2 * k + 1]);
            odd_i_packed [dst_base + k] = float_to_u32(input_i[src_base + 2 * k + 1]);
        }
    }
}

// Build per-stage twiddle tiles in the exact flattened order your reader uses:
//   twiddle_tile_base = (stage * rows_per_core + local_row) * tiles_per_row
// per core, with tile_offset added independently by runtime args.
// Since twiddles are the same for every row, we simply replicate them for all padded rows.
void build_twiddle_tiles(
    uint32_t N_row,
    uint32_t num_stages,
    uint32_t num_cores,
    uint32_t rows_per_core,
    uint32_t direction,
    std::vector<uint32_t>& tw_r_packed,
    std::vector<uint32_t>& tw_i_packed) {

    const uint32_t batch_padded = num_cores * rows_per_core;
    const uint32_t half_N = N_row / 2;
    const uint32_t tiles_per_row = ceil_div(half_N, TILE_ELEMS);
    const uint32_t elems_per_row_padded = tiles_per_row * TILE_ELEMS;
    const float sign = (direction == 1) ? 1.0f : -1.0f;

    tw_r_packed.assign(num_stages * batch_padded * elems_per_row_padded, 0);
    tw_i_packed.assign(num_stages * batch_padded * elems_per_row_padded, 0);

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        const uint32_t m = 1u << (stage + 1);
        const uint32_t half_m = m >> 1;

        for (uint32_t row = 0; row < batch_padded; ++row) {
            const uint32_t base = (stage * batch_padded + row) * elems_per_row_padded;

            for (uint32_t b = 0; b < half_N; ++b) {
                const uint32_t j = b % half_m;
                const uint32_t k = j * (N_row / m);

                const float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(N_row);
                tw_r_packed[base + b] = float_to_u32(std::cos(angle));
                tw_i_packed[base + b] = float_to_u32(std::sin(angle));
            }
        }
    }
}

void make_test_input(
    uint32_t batch_size,
    uint32_t N_row,
    std::vector<float>& input_r,
    std::vector<float>& input_i) {

    input_r.resize(batch_size * N_row);
    input_i.resize(batch_size * N_row);

    for (uint32_t row = 0; row < batch_size; ++row) {
        for (uint32_t i = 0; i < N_row; ++i) {
            float x = std::sin(2.0f * PI * static_cast<float>(i) / static_cast<float>(N_row))
                    + 0.25f * std::cos(6.0f * PI * static_cast<float>(i) / static_cast<float>(N_row));
            input_r[row * N_row + i] = x + 0.01f * static_cast<float>(row);
            input_i[row * N_row + i] = 0.0f;
        }
    }
}

void print_first_outputs(
    const std::vector<float>& out_r,
    const std::vector<float>& out_i,
    uint32_t batch_size,
    uint32_t N_row,
    uint32_t count = 16) {

    const uint32_t n = std::min(count, N_row);
    for (uint32_t row = 0; row < std::min(batch_size, 2u); ++row) {
        std::cout << "row " << row << ":\n";
        for (uint32_t i = 0; i < n; ++i) {
            std::cout << "  [" << i << "] = (" << out_r[row * N_row + i]
                      << ", " << out_i[row * N_row + i] << ")\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        // Args:
        //   argv[1] = device_id     (default 0)
        //   argv[2] = N_row         (default 1024)
        //   argv[3] = batch_size    (default 256)
        //   argv[4] = num_cores     (default 8)
        //   argv[5] = direction     (default 0; 0=forward, 1=inverse)
        const int device_id = (argc > 1) ? std::stoi(argv[1]) : 0;
        const uint32_t N_row = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batch_size = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 256;
        const uint32_t num_cores = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!is_power_of_two(N_row)) {
            throw std::runtime_error("N_row must be a power of two");
        }
        if (N_row < 2) {
            throw std::runtime_error("N_row must be >= 2");
        }
        if (num_cores == 0 || num_cores > 64) {
            throw std::runtime_error("num_cores must be in [1, 64]");
        }

        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        const uint32_t num_stages = log2_u32(N_row);
        const uint32_t half_N = N_row / 2;
        const uint32_t tiles_per_row = ceil_div(half_N, TILE_ELEMS);
        const uint32_t elems_per_row_padded = tiles_per_row * TILE_ELEMS;

        const uint32_t rows_per_core = ceil_div(batch_size, num_cores);
        const uint32_t batch_padded = num_cores * rows_per_core;
        const uint32_t total_tile_count = batch_padded * tiles_per_row;
        const uint32_t total_bytes = total_tile_count * TILE_BYTES;

        std::cout << "[fft_paper_host]\n";
        std::cout << "  N_row        = " << N_row << "\n";
        std::cout << "  batch_size   = " << batch_size << "\n";
        std::cout << "  num_cores    = " << num_cores << "\n";
        std::cout << "  rows_per_core= " << rows_per_core << "\n";
        std::cout << "  num_stages   = " << num_stages << "\n";
        std::cout << "  tiles_per_row= " << tiles_per_row << "\n";

        // Input
        std::vector<float> input_r;
        std::vector<float> input_i;
        make_test_input(batch_size, N_row, input_r, input_i);

        std::vector<uint32_t> even_r_packed, even_i_packed, odd_r_packed, odd_i_packed;
        pack_stage0_even_odd(
            input_r, input_i, batch_size, N_row, num_cores, rows_per_core,
            even_r_packed, even_i_packed, odd_r_packed, odd_i_packed);

        std::vector<uint32_t> tw_r_packed, tw_i_packed;
        build_twiddle_tiles(
            N_row, num_stages, num_cores, rows_per_core, direction,
            tw_r_packed, tw_i_packed);

        // Allocate DRAM mesh buffers
        auto b_even_r = create_dram_mesh_buffer(mesh_device, total_bytes);
        auto b_even_i = create_dram_mesh_buffer(mesh_device, total_bytes);
        auto b_odd_r  = create_dram_mesh_buffer(mesh_device, total_bytes);
        auto b_odd_i  = create_dram_mesh_buffer(mesh_device, total_bytes);

        auto b_out0_r = create_dram_mesh_buffer(mesh_device, total_bytes);
        auto b_out0_i = create_dram_mesh_buffer(mesh_device, total_bytes);
        auto b_out1_r = create_dram_mesh_buffer(mesh_device, total_bytes);
        auto b_out1_i = create_dram_mesh_buffer(mesh_device, total_bytes);

        const uint32_t twiddle_total_bytes = static_cast<uint32_t>(tw_r_packed.size() * sizeof(uint32_t));
        auto b_twiddle_r = create_dram_mesh_buffer(mesh_device, twiddle_total_bytes);
        auto b_twiddle_i = create_dram_mesh_buffer(mesh_device, twiddle_total_bytes);

        // Program + workload
        Program program = CreateProgram();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());

        // Core range: first row of the grid
        CoreRange core_range({0, 0}, {num_cores - 1, 0});

        // Circular buffers
        auto make_cb = [&](uint32_t cb_id, uint32_t num_pages) {
            CircularBufferConfig cb_cfg =
                CircularBufferConfig(num_pages * TILE_BYTES, {{cb_id, tt::DataFormat::Float32}})
                    .set_page_size(cb_id, TILE_BYTES);
            CreateCircularBuffer(program, core_range, cb_cfg);
        };

        // double-buffered I/O CBs
        make_cb(CB_EVEN_R, 2 * tiles_per_row);
        make_cb(CB_EVEN_I, 2 * tiles_per_row);
        make_cb(CB_ODD_R,  2 * tiles_per_row);
        make_cb(CB_ODD_I,  2 * tiles_per_row);
        make_cb(CB_TW_R,   2 * tiles_per_row);
        make_cb(CB_TW_I,   2 * tiles_per_row);

        make_cb(CB_OUT0_R, 2 * tiles_per_row);
        make_cb(CB_OUT0_I, 2 * tiles_per_row);
        make_cb(CB_OUT1_R, 2 * tiles_per_row);
        make_cb(CB_OUT1_I, 2 * tiles_per_row);

        // scratch
        make_cb(CB_TMP_R, 1);
        make_cb(CB_TMP_I, 1);

        // Kernels
        KernelHandle reader_kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader_fft_f32_prod.cpp",
            core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default
            });

        KernelHandle writer_kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer_fft_f32_prod.cpp",
            core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default
            });

        KernelHandle compute_kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/fft_compute_f32_prod.cpp",
            core_range,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true
            });

        const uint32_t tiles_per_core = rows_per_core * tiles_per_row;

        // Runtime args
        for (uint32_t c = 0; c < num_cores; ++c) {
            CoreCoord cc{c, 0};
            const uint32_t tile_offset = c * tiles_per_core;

            SetRuntimeArgs(
                program,
                reader_kernel,
                cc,
                {
                    b_even_r->address(),
                    b_even_i->address(),
                    b_odd_r->address(),
                    b_odd_i->address(),
                    b_twiddle_r->address(),
                    b_twiddle_i->address(),
                    tiles_per_row,
                    tile_offset,
                    num_stages,
                    rows_per_core
                });

            SetRuntimeArgs(
                program,
                compute_kernel,
                cc,
                {
                    num_stages,
                    tiles_per_row
                });

            SetRuntimeArgs(
                program,
                writer_kernel,
                cc,
                {
                    b_out0_r->address(),
                    b_out0_i->address(),
                    b_out1_r->address(),
                    b_out1_i->address(),
                    tiles_per_row,
                    num_stages,
                    tile_offset,
                    rows_per_core
                });
        }

        // Upload inputs + twiddles
        distributed::EnqueueWriteMeshBuffer(cq, b_even_r, even_r_packed, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_even_i, even_i_packed, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_odd_r,  odd_r_packed,  false);
        distributed::EnqueueWriteMeshBuffer(cq, b_odd_i,  odd_i_packed,  false);

        distributed::EnqueueWriteMeshBuffer(cq, b_twiddle_r, tw_r_packed, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_twiddle_i, tw_i_packed, false);

        // Execute
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        // Read back final outputs
        std::vector<uint32_t> out0_r_packed;
        std::vector<uint32_t> out0_i_packed;
        std::vector<uint32_t> out1_r_packed;
        std::vector<uint32_t> out1_i_packed;

        distributed::EnqueueReadMeshBuffer(cq, out0_r_packed, b_out0_r, true);
        distributed::EnqueueReadMeshBuffer(cq, out0_i_packed, b_out0_i, true);
        distributed::EnqueueReadMeshBuffer(cq, out1_r_packed, b_out1_r, true);
        distributed::EnqueueReadMeshBuffer(cq, out1_i_packed, b_out1_i, true);

        // Reconstruct final row-major output.
        // With the current writer, final-stage out0 and out1 correspond to the two butterfly halves.
        std::vector<float> output_r(batch_size * N_row, 0.0f);
        std::vector<float> output_i(batch_size * N_row, 0.0f);

        for (uint32_t row = 0; row < batch_size; ++row) {
            const uint32_t packed_base = row * elems_per_row_padded;
            const uint32_t dst_base = row * N_row;

            for (uint32_t k = 0; k < half_N; ++k) {
                output_r[dst_base + k]          = u32_to_float(out0_r_packed[packed_base + k]);
                output_i[dst_base + k]          = u32_to_float(out0_i_packed[packed_base + k]);
                output_r[dst_base + half_N + k] = u32_to_float(out1_r_packed[packed_base + k]);
                output_i[dst_base + half_N + k] = u32_to_float(out1_i_packed[packed_base + k]);
            }
        }

        print_first_outputs(output_r, output_i, batch_size, N_row);

        if (!mesh_device->close()) {
            throw std::runtime_error("mesh_device->close() failed");
        }

        std::cout << "FFT host run finished.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FFT host failed: " << e.what() << "\n";
        return 1;
    }
}