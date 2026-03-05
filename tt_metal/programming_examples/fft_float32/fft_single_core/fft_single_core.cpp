// fft_single_core.cpp
#include <cmath>
#include <vector>
#include <algorithm>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"

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
// Precompute twiddle factors
// W_N^k = exp(-2*pi*i*k/N) for forward FFT
//----------------------------------
void compute_twiddle_factors(
    uint32_t N,
    uint32_t stage,
    std::vector<float>& tw_real,
    std::vector<float>& tw_imag,
    bool inverse = false
) {
    uint32_t m = 1u << (stage + 1);  // Butterfly group size
    float sign = inverse ? 1.0f : -1.0f;
    
    tw_real.resize(N / 2);
    tw_imag.resize(N / 2);
    
    uint32_t idx = 0;
    for (uint32_t k = 0; k < N; k += m) {
        for (uint32_t j = 0; j < m / 2; j++) {
            float angle = sign * 2.0f * PI * j / m;
            tw_real[idx] = std::cos(angle);
            tw_imag[idx] = std::sin(angle);
            idx++;
        }
    }
}

//----------------------------------
// Create FFT program
//----------------------------------
tt_metal::Program create_fft_program(
    Device* device,
    const CoreCoord& core,
    uint32_t num_tiles,
    uint32_t direction,
    Buffer* even_r_buffer,
    Buffer* even_i_buffer,
    Buffer* odd_r_buffer,
    Buffer* odd_i_buffer,
    Buffer* tw_r_buffer,
    Buffer* tw_i_buffer,
    Buffer* out0_r_buffer,
    Buffer* out0_i_buffer,
    Buffer* out1_r_buffer,
    Buffer* out1_i_buffer
) {
    tt_metal::Program program = CreateProgram();
    
    uint32_t single_tile_size = detail::TileSize(tt::DataFormat::Float32);
    
    // Create Circular Buffers
    // Input CBs
    uint32_t cb_even_r = tt::CBIndex::c_0;
    uint32_t cb_even_i = tt::CBIndex::c_1;
    uint32_t cb_odd_r = tt::CBIndex::c_2;
    uint32_t cb_odd_i = tt::CBIndex::c_3;
    uint32_t cb_tw_r = tt::CBIndex::c_4;
    uint32_t cb_tw_i = tt::CBIndex::c_5;
    
    // Output CBs
    uint32_t cb_out0_r = tt::CBIndex::c_16;
    uint32_t cb_out0_i = tt::CBIndex::c_17;
    uint32_t cb_out1_r = tt::CBIndex::c_18;
    uint32_t cb_out1_i = tt::CBIndex::c_19;
    
    // Intermediate CBs
    uint32_t cb_tmp0 = tt::CBIndex::c_20;
    uint32_t cb_tmp1 = tt::CBIndex::c_21;
    uint32_t cb_tw_odd_r = tt::CBIndex::c_22;
    uint32_t cb_tw_odd_i = tt::CBIndex::c_23;
    uint32_t cb_neg_tmp = tt::CBIndex::c_24;
    
    CircularBufferConfig cb_even_r_config = CircularBufferConfig(single_tile_size * 2, {{cb_even_r, tt::DataFormat::Float32}})
        .set_page_size(cb_even_r, single_tile_size);
    CreateCircularBuffer(program, core, cb_even_r_config);
    
        CircularBufferConfig cb_even_i_config = CircularBufferConfig(single_tile_size * 2, {{cb_even_i, tt::DataFormat::Float32}})
        .set_page_size(cb_even_i, single_tile_size);
    CreateCircularBuffer(program, core, cb_even_i_config);
    
    CircularBufferConfig cb_odd_r_config = CircularBufferConfig(single_tile_size * 2, {{cb_odd_r, tt::DataFormat::Float32}})
        .set_page_size(cb_odd_r, single_tile_size);
    CreateCircularBuffer(program, core, cb_odd_r_config);
    
    CircularBufferConfig cb_odd_i_config = CircularBufferConfig(single_tile_size * 2, {{cb_odd_i, tt::DataFormat::Float32}})
        .set_page_size(cb_odd_i, single_tile_size);
    CreateCircularBuffer(program, core, cb_odd_i_config);
    
    CircularBufferConfig cb_tw_r_config = CircularBufferConfig(single_tile_size * 2, {{cb_tw_r, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_r, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_r_config);
    
    CircularBufferConfig cb_tw_i_config = CircularBufferConfig(single_tile_size * 2, {{cb_tw_i, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_i, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_i_config);
    
    // Output CBs
    CircularBufferConfig cb_out0_r_config = CircularBufferConfig(single_tile_size * 2, {{cb_out0_r, tt::DataFormat::Float32}})
        .set_page_size(cb_out0_r, single_tile_size);
    CreateCircularBuffer(program, core, cb_out0_r_config);
    
    CircularBufferConfig cb_out0_i_config = CircularBufferConfig(single_tile_size * 2, {{cb_out0_i, tt::DataFormat::Float32}})
        .set_page_size(cb_out0_i, single_tile_size);
    CreateCircularBuffer(program, core, cb_out0_i_config);
    
    CircularBufferConfig cb_out1_r_config = CircularBufferConfig(single_tile_size * 2, {{cb_out1_r, tt::DataFormat::Float32}})
        .set_page_size(cb_out1_r, single_tile_size);
    CreateCircularBuffer(program, core, cb_out1_r_config);
    
    CircularBufferConfig cb_out1_i_config = CircularBufferConfig(single_tile_size * 2, {{cb_out1_i, tt::DataFormat::Float32}})
        .set_page_size(cb_out1_i, single_tile_size);
    CreateCircularBuffer(program, core, cb_out1_i_config);
    
    // Intermediate CBs
    CircularBufferConfig cb_tmp0_config = CircularBufferConfig(single_tile_size * 2, {{cb_tmp0, tt::DataFormat::Float32}})
        .set_page_size(cb_tmp0, single_tile_size);
    CreateCircularBuffer(program, core, cb_tmp0_config);
    
    CircularBufferConfig cb_tmp1_config = CircularBufferConfig(single_tile_size * 2, {{cb_tmp1, tt::DataFormat::Float32}})
        .set_page_size(cb_tmp1, single_tile_size);
    CreateCircularBuffer(program, core, cb_tmp1_config);
    
    CircularBufferConfig cb_tw_odd_r_config = CircularBufferConfig(single_tile_size * 2, {{cb_tw_odd_r, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_odd_r, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_odd_r_config);
    
    CircularBufferConfig cb_tw_odd_i_config = CircularBufferConfig(single_tile_size * 2, {{cb_tw_odd_i, tt::DataFormat::Float32}})
        .set_page_size(cb_tw_odd_i, single_tile_size);
    CreateCircularBuffer(program, core, cb_tw_odd_i_config);
    
    CircularBufferConfig cb_neg_tmp_config = CircularBufferConfig(single_tile_size * 2, {{cb_neg_tmp, tt::DataFormat::Float32}})
        .set_page_size(cb_neg_tmp, single_tile_size);
    CreateCircularBuffer(program, core, cb_neg_tmp_config);
    
    // Reader kernel
    std::vector<uint32_t> reader_compile_args = {};
    KernelHandle reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    
    // Writer kernel
    std::vector<uint32_t> writer_compile_args = {};
    KernelHandle writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    
    // Compute kernel
    std::vector<uint32_t> compute_compile_args = {};
    KernelHandle compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_compile_args
        }
    );
    
    // Set reader runtime args
    std::vector<uint32_t> reader_runtime_args = {
        even_r_buffer->address(),
        even_i_buffer->address(),
        odd_r_buffer->address(),
        odd_i_buffer->address(),
        tw_r_buffer->address(),
        tw_i_buffer->address(),
        num_tiles,
        0  // start_tile
    };
    SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
    
    // Set writer runtime args
    std::vector<uint32_t> writer_runtime_args = {
        out0_r_buffer->address(),
        out0_i_buffer->address(),
        out1_r_buffer->address(),
        out1_i_buffer->address(),
        num_tiles,
        0  // start_tile
    };
    SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);
    
    // Set compute runtime args
    std::vector<uint32_t> compute_runtime_args = {
        direction,
        num_tiles
    };
    SetRuntimeArgs(program, compute_kernel, core, compute_runtime_args);
    
    return program;
}

//----------------------------------
// Convert float vector to tilized format
//----------------------------------
std::vector<uint32_t> tilize_float_vector(const std::vector<float>& data, uint32_t tile_height, uint32_t tile_width) {
    uint32_t num_elements = data.size();
    uint32_t num_tiles = (num_elements + tile_height * tile_width - 1) / (tile_height * tile_width);
    
    std::vector<uint32_t> tilized_data(num_tiles * tile_height * tile_width);
    
    for (uint32_t t = 0; t < num_tiles; t++) {
        for (uint32_t h = 0; h < tile_height; h++) {
            for (uint32_t w = 0; w < tile_width; w++) {
                uint32_t src_idx = t * tile_height * tile_width + h * tile_width + w;
                uint32_t dst_idx = t * tile_height * tile_width + h * tile_width + w;
                
                float val = (src_idx < num_elements) ? data[src_idx] : 0.0f;
                uint32_t* val_ptr = reinterpret_cast<uint32_t*>(&val);
                tilized_data[dst_idx] = *val_ptr;
            }
        }
    }
    
    return tilized_data;
}

//----------------------------------
// Convert tilized format back to float vector
//----------------------------------
std::vector<float> untilize_to_float_vector(const std::vector<uint32_t>& tilized_data, uint32_t num_elements, uint32_t tile_height, uint32_t tile_width) {
    std::vector<float> data(num_elements);
    
    uint32_t num_tiles = (num_elements + tile_height * tile_width - 1) / (tile_height * tile_width);
    
    for (uint32_t t = 0; t < num_tiles; t++) {
        for (uint32_t h = 0; h < tile_height; h++) {
            for (uint32_t w = 0; w < tile_width; w++) {
                uint32_t src_idx = t * tile_height * tile_width + h * tile_width + w;
                uint32_t dst_idx = t * tile_height * tile_width + h * tile_width + w;
                
                if (dst_idx < num_elements) {
                    uint32_t val_bits = tilized_data[src_idx];
                    float* val_ptr = reinterpret_cast<float*>(&val_bits);
                    data[dst_idx] = *val_ptr;
                }
            }
        }
    }
    
    return data;
}

//----------------------------------
// Reference CPU FFT (for validation)
//----------------------------------
void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inverse = false) {
    uint32_t N = real.size();
    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;
    
    // Bit-reversal permutation
    bit_reverse_permutation(real, imag);
    
    // Cooley-Tukey iterative FFT
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
                
                // t = twiddle * odd
                float t_r = tw_r * real[idx_odd] - tw_i * imag[idx_odd];
                float t_i = tw_r * imag[idx_odd] + tw_i * real[idx_odd];
                
                // Butterfly
                float e_r = real[idx_even];
                float e_i = imag[idx_even];
                
                real[idx_even] = e_r + t_r;
                imag[idx_even] = e_i + t_i;
                real[idx_odd] = e_r - t_r;
                imag[idx_odd] = e_i - t_i;
            }
        }
    }
    
    // Scale for inverse FFT
    if (inverse) {
        for (uint32_t i = 0; i < N; i++) {
            real[i] /= N;
            imag[i] /= N;
        }
    }
}

//----------------------------------
// Main Function
//----------------------------------
int main(int argc, char** argv) {
    // FFT parameters
    constexpr uint32_t N = 1024;  // FFT size (must be power of 2)
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_SIZE = TILE_HEIGHT * TILE_WIDTH;
    
    uint32_t direction = 0;  // 0 = forward, 1 = inverse
    
    if (argc > 1) {
        direction = std::atoi(argv[1]);
    }
    
    std::cout << "Running " << (direction == 0 ? "Forward" : "Inverse") << " FFT with N = " << N << std::endl;
    
    // Initialize device
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    
    // Compute number of tiles
    uint32_t num_elements = N;
    uint32_t num_tiles = (num_elements + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t padded_size = num_tiles * TILE_SIZE;
    
    // Initialize input data (test signal: combination of sine waves)
    std::vector<float> input_real(padded_size, 0.0f);
    std::vector<float> input_imag(padded_size, 0.0f);
    
    for (uint32_t i = 0; i < N; i++) {
        // Test signal: sum of two frequencies
        input_real[i] = std::sin(2.0f * PI * 4.0f * i / N) + 0.5f * std::sin(2.0f * PI * 8.0f * i / N);
        input_imag[i] = 0.0f;
    }
    
    // Create reference copy for CPU validation
    std::vector<float> ref_real = input_real;
    std::vector<float> ref_imag = input_imag;
    ref_real.resize(N);
    ref_imag.resize(N);
    
    // Compute reference FFT on CPU
    cpu_fft(ref_real, ref_imag, direction == 1);
    
    // Apply bit-reversal to input for hardware FFT
    bit_reverse_permutation(input_real, input_imag);
    
    // For Cooley-Tukey FFT, we need to process stage by stage
    // Each stage processes N/2 butterflies
    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;
    
    uint32_t single_tile_size = detail::TileSize(tt::DataFormat::Float32);
    
    // Create device buffers
    auto even_r_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
        auto even_i_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto odd_r_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto odd_i_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto tw_r_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto tw_i_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    
    // Output buffers
    auto out0_r_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto out0_i_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto out1_r_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    auto out1_i_buffer = CreateBuffer(InterleavedBufferConfig{
        device, padded_size * sizeof(float), single_tile_size, BufferType::DRAM});
    
    // Working buffers for ping-pong between stages
    std::vector<float> work_real = input_real;
    std::vector<float> work_imag = input_imag;
    std::vector<float> result_real(padded_size, 0.0f);
    std::vector<float> result_imag(padded_size, 0.0f);
    
    CoreCoord core = {0, 0};
    
    // Process FFT stage by stage
    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m = 1u << (stage + 1);      // Butterfly group size
        uint32_t half_m = m / 2;
        
        std::cout << "Processing stage " << stage << " (group size = " << m << ")" << std::endl;
        
        // Prepare data for this stage
        std::vector<float> even_real(padded_size, 0.0f);
        std::vector<float> even_imag(padded_size, 0.0f);
        std::vector<float> odd_real(padded_size, 0.0f);
        std::vector<float> odd_imag(padded_size, 0.0f);
        std::vector<float> tw_real(padded_size, 0.0f);
        std::vector<float> tw_imag(padded_size, 0.0f);
        
        // Reorganize data for butterfly operations
        uint32_t butterfly_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + half_m;
                
                // Even (top) input
                even_real[butterfly_idx] = work_real[idx_even];
                even_imag[butterfly_idx] = work_imag[idx_even];
                
                // Odd (bottom) input
                odd_real[butterfly_idx] = work_real[idx_odd];
                odd_imag[butterfly_idx] = work_imag[idx_odd];
                
                // Twiddle factor: W_m^j = exp(-2*pi*i*j/m)
                float angle = (direction == 0 ? -2.0f : 2.0f) * PI * j / m;
                tw_real[butterfly_idx] = std::cos(angle);
                tw_imag[butterfly_idx] = std::sin(angle);
                
                butterfly_idx++;
            }
        }
        
        uint32_t num_butterflies = N / 2;
        uint32_t butterfly_tiles = (num_butterflies + TILE_SIZE - 1) / TILE_SIZE;
        
        // Tilize and write to device
        auto even_r_tilized = tilize_float_vector(even_real, TILE_HEIGHT, TILE_WIDTH);
        auto even_i_tilized = tilize_float_vector(even_imag, TILE_HEIGHT, TILE_WIDTH);
        auto odd_r_tilized = tilize_float_vector(odd_real, TILE_HEIGHT, TILE_WIDTH);
        auto odd_i_tilized = tilize_float_vector(odd_imag, TILE_HEIGHT, TILE_WIDTH);
        auto tw_r_tilized = tilize_float_vector(tw_real, TILE_HEIGHT, TILE_WIDTH);
        auto tw_i_tilized = tilize_float_vector(tw_imag, TILE_HEIGHT, TILE_WIDTH);
        
        EnqueueWriteBuffer(cq, even_r_buffer, even_r_tilized, false);
        EnqueueWriteBuffer(cq, even_i_buffer, even_i_tilized, false);
        EnqueueWriteBuffer(cq, odd_r_buffer, odd_r_tilized, false);
        EnqueueWriteBuffer(cq, odd_i_buffer, odd_i_tilized, false);
        EnqueueWriteBuffer(cq, tw_r_buffer, tw_r_tilized, false);
        EnqueueWriteBuffer(cq, tw_i_buffer, tw_i_tilized, false);
        Finish(cq);
        
        // Create and run program for this stage
        Program program = create_fft_program(
            device, core, butterfly_tiles, direction,
            even_r_buffer.get(), even_i_buffer.get(),
            odd_r_buffer.get(), odd_i_buffer.get(),
            tw_r_buffer.get(), tw_i_buffer.get(),
            out0_r_buffer.get(), out0_i_buffer.get(),
            out1_r_buffer.get(), out1_i_buffer.get()
        );
        
        EnqueueProgram(cq, program, false);
        Finish(cq);
        
        // Read back results
        std::vector<uint32_t> out0_r_tilized(padded_size);
        std::vector<uint32_t> out0_i_tilized(padded_size);
        std::vector<uint32_t> out1_r_tilized(padded_size);
        std::vector<uint32_t> out1_i_tilized(padded_size);
        
        EnqueueReadBuffer(cq, out0_r_buffer, out0_r_tilized, true);
        EnqueueReadBuffer(cq, out0_i_buffer, out0_i_tilized, true);
        EnqueueReadBuffer(cq, out1_r_buffer, out1_r_tilized, true);
        EnqueueReadBuffer(cq, out1_i_buffer, out1_i_tilized, true);
        
        // Untilize results
        auto out0_real = untilize_to_float_vector(out0_r_tilized, padded_size, TILE_HEIGHT, TILE_WIDTH);
        auto out0_imag = untilize_to_float_vector(out0_i_tilized, padded_size, TILE_HEIGHT, TILE_WIDTH);
        auto out1_real = untilize_to_float_vector(out1_r_tilized, padded_size, TILE_HEIGHT, TILE_WIDTH);
        auto out1_imag = untilize_to_float_vector(out1_i_tilized, padded_size, TILE_HEIGHT, TILE_WIDTH);
        
        // Reassemble results back to working buffer
        butterfly_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                uint32_t idx_even = k + j;
                uint32_t idx_odd = k + j + half_m;
                
                // out0 = even + twiddle * odd (goes to idx_even)
                // out1 = even - twiddle * odd (goes to idx_odd)
                result_real[idx_even] = out0_real[butterfly_idx];
                result_imag[idx_even] = out0_imag[butterfly_idx];
                result_real[idx_odd] = out1_real[butterfly_idx];
                result_imag[idx_odd] = out1_imag[butterfly_idx];
                
                butterfly_idx++;
            }
        }
        
        // Swap buffers for next stage
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
    
    // Validate results against CPU reference
    std::cout << "\n--- FFT Results Validation ---" << std::endl;
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
    
    bool passed = (max_error_real < 1e-4f) && (max_error_imag < 1e-4f);
    std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    // Print first few results for visual inspection
    std::cout << "\n--- First 16 FFT Output Values ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (uint32_t i = 0; i < std::min(16u, N); i++) {
        std::cout << "X[" << i << "] = " << work_real[i];
        if (work_imag[i] >= 0) std::cout << " + ";
        else std::cout << " - ";
        std::cout << std::abs(work_imag[i]) << "j";
        std::cout << "  (ref: " << ref_real[i];
        if (ref_imag[i] >= 0) std::cout << " + ";
        else std::cout << " - ";
        std::cout << std::abs(ref_imag[i]) << "j)" << std::endl;
    }
    
    // Cleanup
    CloseDevice(device);
    
    return passed ? 0 : 1;
}