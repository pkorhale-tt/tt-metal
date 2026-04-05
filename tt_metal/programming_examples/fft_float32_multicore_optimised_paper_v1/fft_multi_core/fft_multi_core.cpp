// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#define PI 3.14159265358979323846264338327950288

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

enum FFTDirection {
    FFT_FORWARD  = 0,
    FFT_BACKWARD = 1
};

struct TTExecution {
    Program*      program;       // kept alive for SetRuntimeArgs
    MeshWorkload* workload;      // owns the moved program for dispatch
    CoreCoord*    core;
    KernelHandle* read_kernel;
    KernelHandle* write_kernel;
    KernelHandle* compute_kernel;
    std::shared_ptr<MeshBuffer> in_data_r_dram_buffer;
    std::shared_ptr<MeshBuffer> in_data_i_dram_buffer;
    std::shared_ptr<MeshBuffer> twiddle_dram_buffer;
    std::shared_ptr<MeshBuffer> result_data_r_dram_buffer;
    std::shared_ptr<MeshBuffer> result_data_i_dram_buffer;
    std::shared_ptr<tt::tt_metal::Buffer> step_results_r_buffer;
    std::shared_ptr<tt::tt_metal::Buffer> step_results_i_buffer;
};

// Forward declarations
void fft(MeshCommandQueue&, TTExecution*, float*, float*, float*, float*, float*, uint32_t, enum FFTDirection);
void compare(float*, float*, float*, float*, int);
void moveorigin(float*, float*, int);
void descale(float*, float*, int);
int  checkIfPowerOfTwo(int);
CBHandle createCB(Program&, CoreCoord&, uint32_t, uint32_t, uint32_t);
float* computeTwiddleFactors(int);
static double getElapsedTime(struct timeval);

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "You must provide the size of the domain as an argument\n");
        return -1;
    }

    int domain_size = atoi(argv[1]);
    if (!checkIfPowerOfTwo(domain_size)) {
        fprintf(stderr, "%d provided as domain size, but this must be a power of two\n", domain_size);
        return -1;
    }

    // --- Device setup ---
    auto mesh = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices()[0];   // for L1 buffer allocation only

    Program   program = CreateProgram();
    CoreCoord core    = {0, 0};

    uint32_t problem_mem_size = 4 * domain_size;

    // --- DRAM MeshBuffers ---
    DeviceLocalBufferConfig dram_local{
        .page_size   = problem_mem_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    ReplicatedBufferConfig dram_replicated{.size = problem_mem_size};

    auto in_data_r_dram_buffer     = MeshBuffer::create(dram_replicated, dram_local, mesh.get());
    auto in_data_i_dram_buffer     = MeshBuffer::create(dram_replicated, dram_local, mesh.get());
    auto result_data_r_dram_buffer = MeshBuffer::create(dram_replicated, dram_local, mesh.get());
    auto result_data_i_dram_buffer = MeshBuffer::create(dram_replicated, dram_local, mesh.get());
    auto twiddle_dram_buffer       = MeshBuffer::create(dram_replicated, dram_local, mesh.get());

    // --- CBs ---
    uint32_t cb_tile_size  = 2048 * 4;
    uint32_t cb_total_size = problem_mem_size > cb_tile_size ? problem_mem_size : cb_tile_size;

    createCB(program, core, tt::CBIndex::c_0,  1, cb_tile_size);   // data0 real in
    createCB(program, core, tt::CBIndex::c_1,  1, cb_tile_size);   // data0 imag in
    createCB(program, core, tt::CBIndex::c_2,  1, cb_tile_size);   // data1 real in
    createCB(program, core, tt::CBIndex::c_3,  1, cb_tile_size);   // data1 imag in
    createCB(program, core, tt::CBIndex::c_4,  1, cb_tile_size);   // twiddle real
    createCB(program, core, tt::CBIndex::c_5,  1, cb_tile_size);   // twiddle imag
    createCB(program, core, tt::CBIndex::c_6,  1, cb_tile_size);   // f0
    createCB(program, core, tt::CBIndex::c_7,  1, cb_tile_size);   // f1
    createCB(program, core, tt::CBIndex::c_8,  1, cb_total_size);  // out real to writer
    createCB(program, core, tt::CBIndex::c_9,  1, cb_total_size);  // out imag to writer
    createCB(program, core, tt::CBIndex::c_16, 1, cb_tile_size);   // data0 real out
    createCB(program, core, tt::CBIndex::c_17, 1, cb_tile_size);   // data0 imag out
    createCB(program, core, tt::CBIndex::c_18, 1, cb_tile_size);   // data1 real out
    createCB(program, core, tt::CBIndex::c_19, 1, cb_tile_size);   // data1 imag out
    createCB(program, core, tt::CBIndex::c_20, 1, cb_total_size);  // DDR real in
    createCB(program, core, tt::CBIndex::c_21, 1, cb_total_size);  // DDR imag in
    createCB(program, core, tt::CBIndex::c_22, 1, cb_total_size);  // DDR twiddle in
    createCB(program, core, tt::CBIndex::c_23, 1, cb_tile_size);   // intermediate0
    createCB(program, core, tt::CBIndex::c_24, 1, cb_tile_size);   // intermediate1
    createCB(program, core, tt::CBIndex::c_25, 1, cb_tile_size);   // intermediate2

    // --- L1 scratch buffers (plain Buffer on the single device) ---
    tt::tt_metal::InterleavedBufferConfig l1_config{
        .device      = device,
        .size        = problem_mem_size,
        .page_size   = problem_mem_size,
        .buffer_type = tt::tt_metal::BufferType::L1
    };
    auto step_results_r_buffer = tt::tt_metal::CreateBuffer(l1_config);
    auto step_results_i_buffer = tt::tt_metal::CreateBuffer(l1_config);

    // --- Kernels ---
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        std::string(OVERRIDE_KERNEL_PREFIX) +
            "fft_float32_multicore_optimised_paper_v1/fft_multi_core/kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        std::string(OVERRIDE_KERNEL_PREFIX) +
            "fft_float32_multicore_optimised_paper_v1/fft_multi_core/kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        std::string(OVERRIDE_KERNEL_PREFIX) +
            "fft_float32_multicore_optimised_paper_v1/fft_multi_core/kernels/compute/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args     = {}});

    // --- Build MeshWorkload once (program is moved into it) ---
    // We keep a raw reference to program before moving for SetRuntimeArgs
    Program& program_ref = program;
    MeshWorkload workload;
    MeshCoordinateRange rng = MeshCoordinateRange(mesh->shape());
    workload.add_program(rng, std::move(program));

    // --- Input data ---
    float* golden_r = (float*)malloc(sizeof(float) * domain_size);
    float* golden_i = (float*)malloc(sizeof(float) * domain_size);
    for (int i = 0; i < domain_size; i++) {
        golden_r[i] = 0.0f;
        golden_i[i] = 0.0f;
    }
    golden_r[domain_size / 2] = (float)domain_size;
    golden_i[domain_size / 2] = (float)domain_size * 2;

    float* twiddle_factors = computeTwiddleFactors(domain_size);
    float* data_r = (float*)malloc(sizeof(float) * domain_size);
    float* data_i = (float*)malloc(sizeof(float) * domain_size);
    memcpy(data_r, golden_r, sizeof(float) * domain_size);
    memcpy(data_i, golden_i, sizeof(float) * domain_size);

    TTExecution exec = {
        .program                   = &program_ref,
        .workload                  = &workload,
        .core                      = &core,
        .read_kernel               = &reader_kernel_id,
        .write_kernel              = &writer_kernel_id,
        .compute_kernel            = &compute_kernel_id,
        .in_data_r_dram_buffer     = in_data_r_dram_buffer,
        .in_data_i_dram_buffer     = in_data_i_dram_buffer,
        .twiddle_dram_buffer       = twiddle_dram_buffer,
        .result_data_r_dram_buffer = result_data_r_dram_buffer,
        .result_data_i_dram_buffer = result_data_i_dram_buffer,
        .step_results_r_buffer     = step_results_r_buffer,
        .step_results_i_buffer     = step_results_i_buffer,
    };

    fft(cq, &exec, data_r, data_i, twiddle_factors, data_r, data_i, domain_size, FFT_FORWARD);
    fft(cq, &exec, data_r, data_i, twiddle_factors, data_r, data_i, domain_size, FFT_BACKWARD);

    moveorigin(data_r, data_i, domain_size);
    descale(data_r, data_i, domain_size);

    // mesh closes automatically when shared_ptr goes out of scope

    free(data_r);
    free(data_i);
    free(twiddle_factors);
    free(golden_r);
    free(golden_i);
    return 0;
}

// ---------------------------------------------------------------------------
void fft(MeshCommandQueue& cq, TTExecution* d,
         float* input_r, float* input_i, float* twiddle_factors,
         float* result_r, float* result_i,
         uint32_t domain_size, enum FFTDirection direction)
{
    uint32_t bank_id = 0;

    const std::vector<uint32_t> read_args = {
        d->in_data_r_dram_buffer->address(),
        d->in_data_i_dram_buffer->address(),
        d->twiddle_dram_buffer->address(),
        bank_id, bank_id, bank_id,
        domain_size
    };
    const std::vector<uint32_t> write_args = {
        d->result_data_r_dram_buffer->address(),
        d->result_data_i_dram_buffer->address(),
        bank_id, bank_id,
        domain_size
    };

    SetRuntimeArgs(*d->program, *d->read_kernel,    *d->core, read_args);
    SetRuntimeArgs(*d->program, *d->compute_kernel, *d->core,
                   {(uint32_t)direction, domain_size,
                    d->step_results_r_buffer->address(),
                    d->step_results_i_buffer->address()});
    SetRuntimeArgs(*d->program, *d->write_kernel, *d->core, write_args);

    struct timeval t0;

    gettimeofday(&t0, NULL);
    EnqueueWriteMeshBuffer(cq, d->in_data_r_dram_buffer, input_r,         false);
    EnqueueWriteMeshBuffer(cq, d->in_data_i_dram_buffer, input_i,         false);
    EnqueueWriteMeshBuffer(cq, d->twiddle_dram_buffer,   twiddle_factors,  false);
    Finish(cq);
    double xfer_on = getElapsedTime(t0);

    gettimeofday(&t0, NULL);
    EnqueueMeshWorkload(cq, *d->workload, false);
    Finish(cq);
    double exec_t = getElapsedTime(t0);

    gettimeofday(&t0, NULL);
    EnqueueReadMeshBuffer(cq, result_r, d->result_data_r_dram_buffer, false);
    EnqueueReadMeshBuffer(cq, result_i, d->result_data_i_dram_buffer, false);
    Finish(cq);
    double xfer_off = getElapsedTime(t0);

    printf("%s FFT size %d: total %.4f s  (xfer_on %.4f  exec %.4f  xfer_off %.4f)\n",
           direction == FFT_FORWARD ? "Forwards" : "Backwards",
           domain_size,
           xfer_on + exec_t + xfer_off,
           xfer_on, exec_t, xfer_off);
}

// ---------------------------------------------------------------------------
void compare(float* a_r, float* a_i, float* b_r, float* b_i, int n) {
    int ok = 0, bad = 0;
    for (int i = 0; i < n; i++) {
        if (a_r[i] != b_r[i] || a_i[i] != b_i[i]) {
            printf("Mismatch [%d]: (%.2f,%.2f) vs (%.2f,%.2f)\n",
                   i, a_r[i], a_i[i], b_r[i], b_i[i]);
            bad++;
        } else {
            ok++;
        }
    }
    printf("Checked %d: %d match, %d mismatch\n", n, ok, bad);
}

void moveorigin(float* r, float* im, int n) {
    for (int i = 0; i < n; i++) {
        float sign = (i % 2 == 0) ? 1.0f : -1.0f;
        r[i]  *= sign;
        im[i] *= sign;
    }
}

void descale(float* r, float* im, int n) {
    for (int i = 0; i < n; i++) {
        r[i]  =  r[i]  / (float)n;
        im[i] = -(im[i] / (float)n);
    }
}

int checkIfPowerOfTwo(int v) {
    return (v != 0) && ((v & (v - 1)) == 0);
}

CBHandle createCB(Program& program, CoreCoord& core,
                  uint32_t cb_index, uint32_t num_tiles, uint32_t tile_size) {
    CircularBufferConfig cfg(
        num_tiles * tile_size,
        {{cb_index, tt::DataFormat::Float32}});
    cfg.set_page_size(cb_index, tile_size);
    return tt_metal::CreateCircularBuffer(program, core, cfg);
}

float* computeTwiddleFactors(int n) {
    int    m  = n / 2;
    float* tf = (float*)malloc(sizeof(float) * m * 2);
    for (int i = 0; i < m; i++) {
        float a       = (2.0f * (float)PI * (float)i) / (float)n;
        tf[i * 2]     =  cosf(a);
        tf[i * 2 + 1] = -sinf(a);
    }
    return tf;
}

static double getElapsedTime(struct timeval s) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return ((now.tv_sec  * 1000000 + now.tv_usec) -
            (s.tv_sec    * 1000000 + s.tv_usec  )) / 1e6;
}