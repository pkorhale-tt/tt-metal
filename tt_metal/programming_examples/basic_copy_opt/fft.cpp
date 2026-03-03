// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <cmath>
#include <memory>
#include <cstdlib>
#include <cstdio>

#define PI 3.14159265358979323846264338327950288

using namespace tt;
using namespace tt::tt_metal;

enum FFTDirection {
    FFT_FORWARD = 0,
    FFT_BACKWARD = 1
};

// Function declarations
void compare(float*, float*, float*, float*, int, float);
void descale(float*, float*, int);
int checkIfPowerOfTwo(int);
CBHandle createCB(Program&, CoreCoord&, uint32_t, uint32_t, uint32_t);
float* computeTwiddleFactors(int);
static double getElapsedTime(struct timeval);

tt::tt_metal::Program create_fft_program(
    CoreCoord core,
    uint32_t domain_size,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_i_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> twiddle_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_i_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> step_results_r_buffer,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> step_results_i_buffer,
    FFTDirection direction) 
{
    Program program = CreateProgram();

    uint32_t cb_tile_size = 512 * 4;

    // Create all circular buffers
    createCB(program, core, CBIndex::c_0, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_1, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_2, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_3, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_4, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_5, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_6, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_7, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_8, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_9, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_16, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_17, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_18, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_19, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_20, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_21, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_22, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_23, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_24, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_25, 1, cb_tile_size);

    // Create reader kernel
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/basic_copy_opt/kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, 
            .noc = NOC::RISCV_1_default
        });

    // Create writer kernel
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/basic_copy_opt/kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, 
            .noc = NOC::RISCV_0_default
        });

    // Create compute kernel
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/basic_copy_opt/kernels/compute/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = {},
        });

    // Set runtime arguments for reader
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {
            in_data_r_dram->address(),
            in_data_i_dram->address(),
            twiddle_dram->address(),
            0, 0, 0,
            domain_size
        });

    // Set runtime arguments for compute
    SetRuntimeArgs(
        program,
        compute_kernel_id,
        core,
        {
            (uint32_t)direction, 
            domain_size, 
            step_results_r_buffer->address(), 
            step_results_i_buffer->address()
        });

    // Set runtime arguments for writer
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {
            result_r_dram->address(),
            result_i_dram->address(),
            0, 0,
            domain_size
        });

    return program;
}

void fft_mesh(
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device,
    tt::tt_metal::Program program,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_i_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> twiddle_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_data_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_data_i_dram,
    std::vector<float>& input_r,
    std::vector<float>& input_i,
    std::vector<float>& twiddles,
    std::vector<float>& result_r,
    std::vector<float>& result_i,
    uint32_t domain_size,
    FFTDirection direction) 
{
    struct timeval start_time;

    // Transfer input data to device
    gettimeofday(&start_time, NULL);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, in_data_r_dram, input_r, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, in_data_i_dram, input_i, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, twiddle_dram, twiddles, false);
    tt::tt_metal::distributed::Finish(cq);
    double xfer_on_time = getElapsedTime(start_time);

    // Execute program
    gettimeofday(&start_time, NULL);
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(device->shape()), std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);
    double exec_time = getElapsedTime(start_time);

    // Transfer results back
    gettimeofday(&start_time, NULL);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, result_r, result_data_r_dram, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, result_i, result_data_i_dram, true);
    tt::tt_metal::distributed::Finish(cq);
    double xfer_off_time = getElapsedTime(start_time);

    double total_time = xfer_on_time + exec_time + xfer_off_time;
    printf("%s FFT of size %d: total=%.4fs (xfer_on=%.4fs, exec=%.4fs, xfer_off=%.4fs)\n",
           direction == FFT_FORWARD ? "Forward" : "Backward", 
           domain_size, total_time, xfer_on_time, exec_time, xfer_off_time);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <domain_size>\n", argv[0]);
        fprintf(stderr, "  domain_size must be a power of 2 (e.g., 64, 128, 256, 512, 1024)\n");
        return -1;
    }

    int domain_size = atoi(argv[1]);
    if (!checkIfPowerOfTwo(domain_size)) {
        fprintf(stderr, "Error: %d is not a power of two\n", domain_size);
        return -1;
    }

    printf("========================================\n");
    printf("FFT Test - Domain Size: %d\n", domain_size);
    printf("========================================\n\n");

    // Create device
    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    tt::tt_metal::distributed::MeshCommandQueue& cq = device->mesh_command_queue();

    CoreCoord core = {0, 0};
    uint32_t dram_tile_size = 4 * domain_size;

    // Create DRAM buffer config
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
        .page_size = dram_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_tile_size
    };

    // Create DRAM buffers
    auto in_data_r_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto in_data_i_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto result_data_r_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto result_data_i_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto twiddle_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());

    // Create L1 buffer config
    uint32_t cb_tile_size = 512 * 4;
    tt::tt_metal::distributed::DeviceLocalBufferConfig l1_config{
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig l1_buffer_config{
        .size = cb_tile_size
    };

    // Create L1 buffers for intermediate results
    auto step_results_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(l1_buffer_config, l1_config, device.get());
    auto step_results_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(l1_buffer_config, l1_config, device.get());

    // Compute twiddle factors
    float* twiddle_factors = computeTwiddleFactors(domain_size);
    std::vector<float> twiddle_vec(twiddle_factors, twiddle_factors + domain_size);

    //==========================================================
    // TEST 1: Impulse Response Test
    //==========================================================
    printf("---------- TEST 1: Impulse Response ----------\n");
    printf("Input: delta function at index 0\n");
    printf("Expected: FFT should produce all 1s\n\n");

    std::vector<float> impulse_r(domain_size, 0.0f);
    std::vector<float> impulse_i(domain_size, 0.0f);
    impulse_r[0] = 1.0f;

    std::vector<float> result_r(domain_size, 0.0f);
    std::vector<float> result_i(domain_size, 0.0f);

    tt::tt_metal::Program program_impulse = create_fft_program(
        core, domain_size,
        in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
        result_data_r_dram_buffer, result_data_i_dram_buffer,
        step_results_r_buffer, step_results_i_buffer,
        FFT_FORWARD
    );

    fft_mesh(cq, device, std::move(program_impulse),
             in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
             result_data_r_dram_buffer, result_data_i_dram_buffer,
             impulse_r, impulse_i, twiddle_vec,
             result_r, result_i, domain_size, FFT_FORWARD);

    printf("Impulse FFT Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] (%.6f, %.6f)\n", i, result_r[i], result_i[i]);
    }

    int impulse_correct = 0;
    float tolerance = 0.5f;
    for (int i = 0; i < domain_size; i++) {
        if (fabs(result_r[i] - 1.0f) < tolerance && fabs(result_i[i]) < tolerance) {
            impulse_correct++;
        }
    }
    printf("Impulse Test: %d/%d correct (tolerance=%.2f)\n\n", 
           impulse_correct, domain_size, tolerance);

    //==========================================================
    // TEST 2: Round Trip Test (

        //==========================================================
    // TEST 2: Round Trip Test (Forward + Backward)
    //==========================================================
    printf("---------- TEST 2: Round Trip Test ----------\n");
    printf("Input -> Forward FFT -> Backward FFT -> Output\n");
    printf("Expected: Output should match Input\n\n");

    // Create test signal
    std::vector<float> test_r(domain_size);
    std::vector<float> test_i(domain_size, 0.0f);
    
    for (int i = 0; i < domain_size; i++) {
        test_r[i] = (float)(i % 10);  // Pattern: 0,1,2,3,4,5,6,7,8,9,0,1,2,...
    }

    printf("Input (first 15 elements):\n");
    for (int i = 0; i < 15; i++) {
        printf("  [%d] (%.2f, %.2f)\n", i, test_r[i], test_i[i]);
    }
    printf("\n");

    // Forward FFT
    std::vector<float> fwd_result_r(domain_size, 0.0f);
    std::vector<float> fwd_result_i(domain_size, 0.0f);

    tt::tt_metal::Program program_fwd = create_fft_program(
        core, domain_size,
        in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
        result_data_r_dram_buffer, result_data_i_dram_buffer,
        step_results_r_buffer, step_results_i_buffer,
        FFT_FORWARD
    );

    fft_mesh(cq, device, std::move(program_fwd),
             in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
             result_data_r_dram_buffer, result_data_i_dram_buffer,
             test_r, test_i, twiddle_vec,
             fwd_result_r, fwd_result_i, domain_size, FFT_FORWARD);

    printf("After Forward FFT (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] (%.6f, %.6f)\n", i, fwd_result_r[i], fwd_result_i[i]);
    }
    printf("\n");

    // Backward FFT
    std::vector<float> bwd_result_r(domain_size, 0.0f);
    std::vector<float> bwd_result_i(domain_size, 0.0f);

    tt::tt_metal::Program program_bwd = create_fft_program(
        core, domain_size,
        in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
        result_data_r_dram_buffer, result_data_i_dram_buffer,
        step_results_r_buffer, step_results_i_buffer,
        FFT_BACKWARD
    );

    fft_mesh(cq, device, std::move(program_bwd),
             in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
             result_data_r_dram_buffer, result_data_i_dram_buffer,
             fwd_result_r, fwd_result_i, twiddle_vec,
             bwd_result_r, bwd_result_i, domain_size, FFT_BACKWARD);

    printf("After Backward FFT (raw, first 15 elements):\n");
    for (int i = 0; i < 15; i++) {
        printf("  [%d] (%.6f, %.6f)\n", i, bwd_result_r[i], bwd_result_i[i]);
    }
    printf("\n");

    // Scale by 1/N for inverse FFT
    descale(bwd_result_r.data(), bwd_result_i.data(), domain_size);

    printf("After scaling by 1/N (first 15 elements):\n");
    for (int i = 0; i < 15; i++) {
        printf("  [%d] got (%.4f, %.4f), expected (%.2f, %.2f)\n", 
               i, bwd_result_r[i], bwd_result_i[i], test_r[i], test_i[i]);
    }
    printf("\n");

    // Compare round trip result with original
    printf("Round Trip Comparison:\n");
    compare(bwd_result_r.data(), bwd_result_i.data(), 
            test_r.data(), test_i.data(), domain_size, 1.0f);

    //==========================================================
    // Summary
    //==========================================================
    printf("\n========================================\n");
    printf("FFT Test Summary\n");
    printf("========================================\n");
    printf("Domain size: %d\n", domain_size);
    printf("Test 1 (Impulse): %d/%d correct\n", impulse_correct, domain_size);
    printf("Test 2 (Round trip): See comparison above\n");
    printf("========================================\n");

    // Cleanup
    device->close();
    free(twiddle_factors);

    return 0;
}

//==========================================================
// Helper Functions
//==========================================================

void compare(float* a_data_r, float* a_data_i, 
             float* b_data_r, float* b_data_i, 
             int domain_size, float tolerance) 
{
    int matching = 0, mismatching = 0;
    
    for (int i = 0; i < domain_size; i++) {
        float diff_r = fabs(a_data_r[i] - b_data_r[i]);
        float diff_i = fabs(a_data_i[i] - b_data_i[i]);
        
        if (diff_r > tolerance || diff_i > tolerance) {
            if (mismatching < 10) {
                printf("  Mismatch [%d]: got (%.4f, %.4f), expected (%.4f, %.4f)\n",
                       i, a_data_r[i], a_data_i[i], b_data_r[i], b_data_i[i]);
            }
            mismatching++;
        } else {
            matching++;
        }
    }
    
    if (mismatching > 10) {
        printf("  ... and %d more mismatches\n", mismatching - 10);
    }
    
    printf("Result: %d/%d match, %d mismatch (tolerance=%.4f)\n", 
           matching, domain_size, mismatching, tolerance);
    
    if (matching == domain_size) {
        printf("*** TEST PASSED ***\n");
    } else {
        printf("*** TEST FAILED ***\n");
    }
}

void descale(float* data_r, float* data_i, int domain_size) {
    float scale = 1.0f / domain_size;
    for (int i = 0; i < domain_size; i++) {
        data_r[i] = data_r[i] * scale;
        data_i[i] = data_i[i] * scale;
    }
}

int checkIfPowerOfTwo(int v) {
    return (v != 0) && ((v & (v - 1)) == 0);
}

CBHandle createCB(Program& program, CoreCoord& core, 
                  uint32_t cb_index, uint32_t num_tiles, uint32_t tile_size) 
{
    CircularBufferConfig cb_config = 
        CircularBufferConfig(num_tiles * tile_size, {{cb_index, tt::DataFormat::Float32}})
            .set_page_size(cb_index, tile_size);
    CBHandle cb = tt_metal::CreateCircularBuffer(program, core, cb_config);
    return cb;
}

float* computeTwiddleFactors(int n) {
    int num_twiddle_factors = n / 2;
    float* twiddle_factors = (float*)malloc(sizeof(float) * num_twiddle_factors * 2);
    
    for (int i = 0; i < num_twiddle_factors; i++) {
        float angle = (2.0 * PI * i) / (float)n;
        twiddle_factors[i * 2] = (float)cos((double)angle);
        twiddle_factors[(i * 2) + 1] = (float)(-sin((double)angle));
    }
    
    return twiddle_factors;
}

static double getElapsedTime(struct timeval start_time) {
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    long int elapsedtime = (curr_time.tv_sec * 1000000 + curr_time.tv_usec) - 
                          (start_time.tv_sec * 1000000 + start_time.tv_usec);
    return elapsedtime / 1000000.0;
}