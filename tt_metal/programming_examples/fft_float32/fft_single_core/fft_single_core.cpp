// fft_single_core.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fft_single_core.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>

// TT-Metal Host API - Try this format
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/detail/util.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;

//----------------------------------
// Bit-reversal permutation
//----------------------------------
uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void bit_reverse_permutation(std::vector<float>& real, std::vector<float>& imag) {
    uint32_t n = real.size();
    uint32_t log2n = 0;
    while ((1u << log2n) < n) log2n++;
    
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = bit_reverse(i, log2n);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
}

//----------------------------------
// Reference CPU FFT (for validation)
//----------------------------------
void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inverse = false) {
    uint32_t N = real.size();
    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;
    
    bit_reverse_permutation(real, imag);
    
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u << (s + 1);
        float angle_base = (inverse ? 2.0f : -2.0f) * PI / m;
        
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float angle = angle_base * j;
                float tw_r = std::cos(angle);
                float tw_i = std::sin(angle);
                
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + m / 2;
                
                float t_r = tw_r * real[idx_odd] - tw_i * imag[idx_odd];
                float t_i = tw_r * imag[idx_odd] + tw_i * real[idx_odd];
                
                float e_r = real[idx_even];
                float e_i = imag[idx_even];
                
                real[idx_even] = e_r + t_r;
                imag[idx_even] = e_i + t_i;
                real[idx_odd] = e_r - t_r;
                imag[idx_odd] = e_i - t_i;
            }
        }
    }
    
    if (inverse) {
        for (uint32_t i = 0; i < N; i++) {
            real[i] /= N;
            imag[i] /= N;
        }
    }
}

//----------------------------------
// Pack float to uint32
//----------------------------------
inline uint32_t float_to_uint32(float f) {
    uint32_t result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

//----------------------------------
// Unpack uint32 to float
//----------------------------------
inline float uint32_to_float(uint32_t u) {
    float result;
    std::memcpy(&result, &u, sizeof(float));
    return result;
}

//----------------------------------
// Convert float vector to tilized uint32 vector
//----------------------------------
std::vector<uint32_t> tilize_float_vector(const std::vector<float>& data, uint32_t num_tiles, uint32_t tile_size) {
    std::vector<uint32_t> result(num_tiles * tile_size, 0);
    
    for (uint32_t i = 0; i < data.size() && i < result.size(); i++) {
        result[i] = float_to_uint32(data[i]);
    }
    
    return result;
}

//----------------------------------
// Convert tilized uint32 vector to float vector
//----------------------------------
std::vector<float> untilize_to_float(const std::vector<uint32_t>& data, uint32_t num_elements) {
    std::vector<float> result(num_elements);
    
    for (uint32_t i = 0; i < num_elements && i < data.size(); i++) {
        result[i] = uint32_to_float(data[i]);
    }
    
    return result;
}

//----------------------------------
// Main function
//----------------------------------
int main(int argc, char** argv) {
    // FFT parameters
    constexpr uint32_t N = 1024;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_SIZE = TILE_HEIGHT * TILE_WIDTH;
    
    uint32_t direction = 0;  // 0 = forward, 1 = inverse
    
    if (argc > 1) {
        direction = static_cast<uint32_t>(std::atoi(argv[1]));
    }
    
    std::cout << "=== TT-Metal FFT (Float32) ===" << std::endl;
    std::cout << "FFT Size: " << N << std::endl;
    std::cout << "Direction: " << (direction == 0 ? "Forward" : "Inverse") << std::endl;
    
    // Calculate dimensions
    uint32_t num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t padded_size = num_tiles * TILE_SIZE;
    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;
    
    std::cout << "Number of tiles: " << num_tiles << std::endl;
    std::cout << "Number of stages: " << log2N << std::endl;
    
    // Initialize input data (test signal)
    std::vector<float> input_real(padded_size, 0.0f);
    std::vector<float> input_imag(padded_size, 0.0f);
    
    for (uint32_t i = 0; i < N; i++) {
        input_real[i] = std::sin(2.0f * PI * 4.0f * i / N) + 0.5f * std::sin(2.0f * PI * 8.0f * i / N);
        input_imag[i] = 0.0f;
    }
    
    // Create reference copy for CPU validation
    std::vector<float> ref_real(input_real.begin(), input_real.begin() + N);
    std::vector<float> ref_imag(input_imag.begin(), input_imag.begin() + N);
    
    // Compute reference FFT on CPU
    cpu_fft(ref_real, ref_imag, direction == 1);
    
    // Initialize device
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    
    CoreCoord core = {0, 0};
    
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(tt::DataFormat::Float32);
    
    // Create circular buffers
    // Input CBs
    uint32_t cb_even_r_id = tt::CBIndex::c_0;
    uint32_t cb_even_i_id = tt::CBIndex::c_1;
    uint32_t cb_odd_r_id = tt::CBIndex::c_2;
    uint32_t cb_odd_i_id = tt::CBIndex::c_3;
    uint32_t cb_tw_r_id = tt::CBIndex::c_4;
    uint32_t cb_tw_i_id = tt::CBIndex::c_5;
    
    // Output CBs
    uint32_t cb_out0_r_id = tt::CBIndex::c_16;
    uint32_t cb_out0_i_id = tt::CBIndex::c_17;
    uint32_t cb_out1_r_id = tt::CBIndex::c_18;
    uint32_t cb_out1_i_id = tt::CBIndex::c_19;
    
    // Intermediate CBs
    uint32_t cb_tmp0_id = tt::CBIndex::c_20;
    uint32_t cb_tmp1_id = tt::CBIndex::c_21;
    uint32_t cb_tw_odd_r_id = tt::CBIndex::c_22;
    uint32_t cb_tw_odd_i_id = tt::CBIndex::c_23;
    uint32_t cb_neg_tmp_id = tt::CBIndex::c_24;
    
    uint32_t num_cb_tiles = 2;
    
    // Input circular buffers
    CircularBufferConfig cb_even_r_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_even_r_id, tt::DataFormat::Float32}})
        .set_page_size(cb_even_r_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_even_r_config);
    
    CircularBufferConfig cb_even_i_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_even_i_id, tt::DataFormat::Float32}})
        .set_page_size(cb_even_i_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_even_i_config);
    
    CircularBufferConfig cb_odd_r_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_odd_r_id, tt::DataFormat::Float32}})
        .set_page_size(cb_odd_r_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_odd_r_config);
    
    CircularBufferConfig cb_odd_i_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_odd_i_id, tt::DataFormat::Float32}})
        .set_page_size(cb_odd_i_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_odd_i_config);
    
    CircularBufferConfig cb_tw_r_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_tw_r_id, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_r_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_r_config);
    
    CircularBufferConfig cb_tw_i_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_tw_i_id, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_i_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_i_config);
    
    // Output circular buffers
    CircularBufferConfig cb_out0_r_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_out0_r_id, tt::DataFormat::Float32}})
        .set_page_size(cb_out0_r_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_out0_r_config);
    
    CircularBufferConfig cb_out0_i_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_out0_i_id, tt::DataFormat::Float32}})
        .set_page_size(cb_out0_i_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_out0_i_config);
    
    CircularBufferConfig cb_out1_r_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_out1_r_id, tt::DataFormat::Float32}})
        .set_page_size(cb_out1_r_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_out1_r_config);
    
    CircularBufferConfig cb_out1_i_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_out1_i_id, tt::DataFormat::Float32}})
        .set_page_size(cb_out1_i_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_out1_i_config);
    
    // Intermediate circular buffers
    CircularBufferConfig cb_tmp0_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_tmp0_id, tt::DataFormat::Float32}})
        .set_page_size(cb_tmp0_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_tmp0_config);
    
    CircularBufferConfig cb_tmp1_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_tmp1_id, tt::DataFormat::Float32}})
        .set_page_size(cb_tmp1_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_tmp1_config);
    
    CircularBufferConfig cb_tw_odd_r_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_tw_odd_r_id, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_odd_r_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_odd_r_config);
    
    CircularBufferConfig cb_tw_odd_i_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_tw_odd_i_id, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_odd_i_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_odd_i_config);
    
    CircularBufferConfig cb_neg_tmp_config = CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_neg_tmp_id, tt::DataFormat::Float32}})
        .set_page_size(cb_neg_tmp_id, single_tile_size);
    CreateCircularBuffer(program, core, cb_neg_tmp_config);
    
        // Create DRAM buffers
    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = padded_size * sizeof(float),
        .page_size = single_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    
    auto even_r_buffer = CreateBuffer(dram_config);
    auto even_i_buffer = CreateBuffer(dram_config);
    auto odd_r_buffer = CreateBuffer(dram_config);
    auto odd_i_buffer = CreateBuffer(dram_config);
    auto tw_r_buffer = CreateBuffer(dram_config);
    auto tw_i_buffer = CreateBuffer(dram_config);
    auto out0_r_buffer = CreateBuffer(dram_config);
    auto out0_i_buffer = CreateBuffer(dram_config);
    auto out1_r_buffer = CreateBuffer(dram_config);
    auto out1_i_buffer = CreateBuffer(dram_config);
    
    // Create kernels
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );
    
    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );
    
    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false
        }
    );
    
    // Apply bit-reversal to input
    bit_reverse_permutation(input_real, input_imag);
    
    // Working buffers
    std::vector<float> work_real = input_real;
    std::vector<float> work_imag = input_imag;
    std::vector<float> result_real(padded_size, 0.0f);
    std::vector<float> result_imag(padded_size, 0.0f);
    
    // Process FFT stage by stage
    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m / 2;
        
        std::cout << "Processing stage " << stage << " / " << log2N - 1 << std::endl;
        
        // Prepare butterfly data for this stage
        std::vector<float> even_real(padded_size, 0.0f);
        std::vector<float> even_imag(padded_size, 0.0f);
        std::vector<float> odd_real(padded_size, 0.0f);
        std::vector<float> odd_imag(padded_size, 0.0f);
        std::vector<float> tw_real(padded_size, 0.0f);
        std::vector<float> tw_imag(padded_size, 0.0f);
        
        uint32_t butterfly_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + half_m;
                
                even_real[butterfly_idx] = work_real[idx_even];
                even_imag[butterfly_idx] = work_imag[idx_even];
                odd_real[butterfly_idx] = work_real[idx_odd];
                odd_imag[butterfly_idx] = work_imag[idx_odd];
                
                float angle = (direction == 0 ? -2.0f : 2.0f) * PI * j / m;
                tw_real[butterfly_idx] = std::cos(angle);
                tw_imag[butterfly_idx] = std::sin(angle);
                
                butterfly_idx++;
            }
        }
        
        uint32_t num_butterflies = N / 2;
        uint32_t butterfly_tiles = (num_butterflies + TILE_SIZE - 1) / TILE_SIZE;
        
        // Tilize data
        auto even_r_data = tilize_float_vector(even_real, butterfly_tiles, TILE_SIZE);
        auto even_i_data = tilize_float_vector(even_imag, butterfly_tiles, TILE_SIZE);
        auto odd_r_data = tilize_float_vector(odd_real, butterfly_tiles, TILE_SIZE);
        auto odd_i_data = tilize_float_vector(odd_imag, butterfly_tiles, TILE_SIZE);
        auto tw_r_data = tilize_float_vector(tw_real, butterfly_tiles, TILE_SIZE);
        auto tw_i_data = tilize_float_vector(tw_imag, butterfly_tiles, TILE_SIZE);
        
        // Write to DRAM
        EnqueueWriteBuffer(cq, even_r_buffer, even_r_data, false);
        EnqueueWriteBuffer(cq, even_i_buffer, even_i_data, false);
        EnqueueWriteBuffer(cq, odd_r_buffer, odd_r_data, false);
        EnqueueWriteBuffer(cq, odd_i_buffer, odd_i_data, false);
        EnqueueWriteBuffer(cq, tw_r_buffer, tw_r_data, false);
        EnqueueWriteBuffer(cq, tw_i_buffer, tw_i_data, false);
        Finish(cq);
        
        // Set runtime args for reader
        std::vector<uint32_t> reader_args = {
            even_r_buffer->address(),
            even_i_buffer->address(),
            odd_r_buffer->address(),
            odd_i_buffer->address(),
            tw_r_buffer->address(),
            tw_i_buffer->address(),
            butterfly_tiles,
            0  // start_tile
        };
        SetRuntimeArgs(program, reader_kernel, core, reader_args);
        
        // Set runtime args for writer
        std::vector<uint32_t> writer_args = {
            out0_r_buffer->address(),
            out0_i_buffer->address(),
            out1_r_buffer->address(),
            out1_i_buffer->address(),
            butterfly_tiles,
            0  // start_tile
        };
        SetRuntimeArgs(program, writer_kernel, core, writer_args);
        
        // Set runtime args for compute
        std::vector<uint32_t> compute_args = {
            direction,
            butterfly_tiles
        };
        SetRuntimeArgs(program, compute_kernel, core, compute_args);
        
        // Execute program
        EnqueueProgram(cq, program, false);
        Finish(cq);
        
        // Read results
        std::vector<uint32_t> out0_r_data(butterfly_tiles * TILE_SIZE);
        std::vector<uint32_t> out0_i_data(butterfly_tiles * TILE_SIZE);
        std::vector<uint32_t> out1_r_data(butterfly_tiles * TILE_SIZE);
        std::vector<uint32_t> out1_i_data(butterfly_tiles * TILE_SIZE);
        
        EnqueueReadBuffer(cq, out0_r_buffer, out0_r_data, true);
        EnqueueReadBuffer(cq, out0_i_buffer, out0_i_data, true);
        EnqueueReadBuffer(cq, out1_r_buffer, out1_r_data, true);
        EnqueueReadBuffer(cq, out1_i_buffer, out1_i_data, true);
        
        // Untilize results
        auto out0_real = untilize_to_float(out0_r_data, num_butterflies);
        auto out0_imag = untilize_to_float(out0_i_data, num_butterflies);
        auto out1_real = untilize_to_float(out1_r_data, num_butterflies);
        auto out1_imag = untilize_to_float(out1_i_data, num_butterflies);
        
        // Reassemble into working buffer
        butterfly_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + half_m;
                
                result_real[idx_even] = out0_real[butterfly_idx];
                result_imag[idx_even] = out0_imag[butterfly_idx];
                result_real[idx_odd] = out1_real[butterfly_idx];
                result_imag[idx_odd] = out1_imag[butterfly_idx];
                
                butterfly_idx++;
            }
        }
        
        // Swap for next stage
        work_real = result_real;
        work_imag = result_imag;
    }
    
    // Apply scaling for inverse FFT
    if (direction == 1) {
        for (uint32_t i = 0; i < N; i++) {
            work_real[i] /= N;
            work_imag[i] /= N;
        }
    }
    
    // Validate results
    std::cout << "\n=== Validation ===" << std::endl;
    
    float max_error_real = 0.0f;
    float max_error_imag = 0.0f;
    
    for (uint32_t i = 0; i < N; i++) {
        float err_r = std::abs(work_real[i] - ref_real[i]);
        float err_i = std::abs(work_imag[i] - ref_imag[i]);
        max_error_real = std::max(max_error_real, err_r);
        max_error_imag = std::max(max_error_imag, err_i);
    }
    
    std::cout << "Max error (real): " << max_error_real << std::endl;
    std::cout << "Max error (imag): " << max_error_imag << std::endl;
    
    bool passed = (max_error_real < 1e-3f) && (max_error_imag < 1e-3f);
    std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    // Print first 16 results
    std::cout << "\n=== First 16 FFT Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    for (uint32_t i = 0; i < 16 && i < N; i++) {
        std::cout << "X[" << std::setw(2) << i << "] = "
                  << std::setw(12) << work_real[i]
                  << (work_imag[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(work_imag[i]) << "j"
                  << "  |  ref: "
                  << std::setw(12) << ref_real[i]
                  << (ref_imag[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(ref_imag[i]) << "j"
                  << std::endl;
    }
    
    // Cleanup
    CloseDevice(device);
    
    std::cout << "\n=== Done ===" << std::endl;
    
    return passed ? 0 : 1;
}