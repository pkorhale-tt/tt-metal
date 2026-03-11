// fft_single_core.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Changes vs original:
//  1. IO CBs (input + output) use depth=2 for true double-buffering.
//     Intermediate CBs (tmp0/1, tw_odd_r/i, neg_tmp) use depth=1 —
//     they are transient and never hold more than one tile at a time.
//
//  2. DRAM buffer size is now sized to butterfly_tiles * TILE_SIZE * sizeof(float)
//     (i.e. num_butterflies rounded up to tile boundary) rather than
//     padded_size * sizeof(float). This avoids over-allocating DRAM when
//     N is small relative to TILE_SIZE.
//
//  3. Host EnqueueWriteMeshBuffer calls are issued without Finish() between
//     them — all 6 writes are enqueued before a single Finish(), mirroring
//     the batched-NOC-read pattern in the optimized reader kernel.
//
//  4. butterfly_tiles is passed as num_butterflies (actual butterfly count)
//     to reader/writer/compute so the kernels process exactly the right
//     number of tiles with no wasted iterations.
//
//  5. unpack_to_dest_mode is set for ALL CB indices that the compute kernel
//     reads from, including both input and intermediate CBs.

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;

// ─────────────────────────────────────────────────────────────────────────────
// Bit-reversal permutation
// ─────────────────────────────────────────────────────────────────────────────
uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void bit_reverse_permutation(std::vector<float>& real, std::vector<float>& imag, uint32_t n) {
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

// ─────────────────────────────────────────────────────────────────────────────
// Reference CPU FFT (for validation)
// ─────────────────────────────────────────────────────────────────────────────
void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inverse = false) {
    uint32_t N = real.size();
    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;

    bit_reverse_permutation(real, imag, N);

    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u << (s + 1);
        float angle_base = (inverse ? 2.0f : -2.0f) * PI / m;

        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float angle = angle_base * j;
                float tw_r  = std::cos(angle);
                float tw_i  = std::sin(angle);

                uint32_t idx_even = k + j;
                uint32_t idx_odd  = k + j + m / 2;

                float t_r = tw_r * real[idx_odd] - tw_i * imag[idx_odd];
                float t_i = tw_r * imag[idx_odd] + tw_i * real[idx_odd];
                float e_r = real[idx_even];
                float e_i = imag[idx_even];

                real[idx_even] = e_r + t_r;
                imag[idx_even] = e_i + t_i;
                real[idx_odd]  = e_r - t_r;
                imag[idx_odd]  = e_i - t_i;
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

// ─────────────────────────────────────────────────────────────────────────────
// Float <-> uint32 bit-cast helpers
// ─────────────────────────────────────────────────────────────────────────────
inline uint32_t float_to_uint32(float f) {
    uint32_t u; std::memcpy(&u, &f, sizeof(float)); return u;
}
inline float uint32_to_float(uint32_t u) {
    float f; std::memcpy(&f, &u, sizeof(float)); return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tilize / untilize
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint32_t> tilize_float_vector(
    const std::vector<float>& data, uint32_t num_tiles, uint32_t tile_size)
{
    std::vector<uint32_t> result(num_tiles * tile_size, 0);
    for (uint32_t i = 0; i < data.size() && i < result.size(); i++)
        result[i] = float_to_uint32(data[i]);
    return result;
}

std::vector<float> untilize_to_float(
    const std::vector<uint32_t>& data, uint32_t num_elements)
{
    std::vector<float> result(num_elements);
    for (uint32_t i = 0; i < num_elements && i < data.size(); i++)
        result[i] = uint32_to_float(data[i]);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Load input from file
// Format: one line per sample "real imag" (imag defaults to 0 if absent)
// ─────────────────────────────────────────────────────────────────────────────
bool load_input_from_file(
    const std::string& filename,
    std::vector<float>& real, std::vector<float>& imag,
    uint32_t N)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open input file: " << filename << std::endl;
        return false;
    }
    std::string line;
    uint32_t count = 0;
    while (std::getline(infile, line) && count < N) {
        std::istringstream iss(line);
        float r = 0.0f, im = 0.0f;
        iss >> r >> im;
        real[count] = r;
        imag[count] = im;
        count++;
    }
    std::cout << "Loaded " << count << " samples from " << filename << std::endl;
    if (count < N)
        std::cout << "  (remaining " << (N - count) << " samples zero-padded)" << std::endl;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create a DRAM MeshBuffer sized for butterfly_tiles tiles
// ─────────────────────────────────────────────────────────────────────────────
static std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> make_dram_buffer(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_bytes,
    uint32_t page_size_bytes)
{
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_cfg{
        .page_size   = page_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig buf_cfg{
        .size = num_bytes
    };
    return tt::tt_metal::distributed::MeshBuffer::create(buf_cfg, dram_cfg, mesh_device);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH  = 32;
    constexpr uint32_t TILE_SIZE   = TILE_HEIGHT * TILE_WIDTH;  // 1024 elements

    // ── CLI args ──────────────────────────────────────────────────────────
    uint32_t    direction  = 0;
    std::string input_file = "";
    uint32_t    N          = 1024;

    if (argc > 1) direction  = static_cast<uint32_t>(std::atoi(argv[1]));
    if (argc > 2) input_file = argv[2];
    if (argc > 3) {
        N = static_cast<uint32_t>(std::atoi(argv[3]));
        if (N == 0 || (N & (N - 1)) != 0) {
            std::cerr << "Error: N must be a power of 2, got " << N << std::endl;
            return 1;
        }
    }

    uint32_t log2N = 0;
    while ((1u << log2N) < N) log2N++;

    // num_butterflies per stage = N/2 (constant across all stages)
    const uint32_t num_butterflies  = N / 2;
    // How many tiles do num_butterflies elements fill?
    const uint32_t butterfly_tiles  = (num_butterflies + TILE_SIZE - 1) / TILE_SIZE;
    // Padded element count so data fits evenly into tiles
    const uint32_t padded_butterfly = butterfly_tiles * TILE_SIZE;

    std::cout << "=== TT-Metal FFT (Float32) ===" << std::endl;
    std::cout << "FFT Size       : " << N          << std::endl;
    std::cout << "Direction      : " << (direction == 0 ? "Forward" : "Inverse") << std::endl;
    std::cout << "Stages         : " << log2N       << std::endl;
    std::cout << "Butterflies/stage: " << num_butterflies << std::endl;
    std::cout << "Butterfly tiles: " << butterfly_tiles   << std::endl;
    std::cout << "Usage: " << argv[0] << " [direction: 0|1] [input_file.txt] [N]" << std::endl;

    // ── Input signal ──────────────────────────────────────────────────────
    // Size padded_size for bit-reversal (operates on N elements)
    uint32_t num_tiles_n  = (N + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t padded_size  = num_tiles_n * TILE_SIZE;

    std::vector<float> input_real(padded_size, 0.0f);
    std::vector<float> input_imag(padded_size, 0.0f);

    if (!input_file.empty()) {
        if (!load_input_from_file(input_file, input_real, input_imag, N))
            return 1;
    } else {
        for (uint32_t i = 0; i < N; i++) {
            input_real[i] = std::sin(2.0f * PI * 4.0f * i / N)
                          + 0.5f * std::sin(2.0f * PI * 8.0f * i / N);
            input_imag[i] = 0.0f;
        }
    }

    // ── CPU reference FFT ─────────────────────────────────────────────────
    std::vector<float> ref_real(input_real.begin(), input_real.begin() + N);
    std::vector<float> ref_imag(input_imag.begin(), input_imag.begin() + N);
    cpu_fft(ref_real, ref_imag, direction == 1);

    // ── Device init ───────────────────────────────────────────────────────
    int device_id = 0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto& cq        = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core  = {0, 0};

    const uint32_t single_tile_size =
        tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH * sizeof(float);

    // ── CB index declarations ─────────────────────────────────────────────
    // Input CBs
    constexpr uint32_t cb_even_r_id   = tt::CBIndex::c_0;
    constexpr uint32_t cb_even_i_id   = tt::CBIndex::c_1;
    constexpr uint32_t cb_odd_r_id    = tt::CBIndex::c_2;
    constexpr uint32_t cb_odd_i_id    = tt::CBIndex::c_3;
    constexpr uint32_t cb_tw_r_id     = tt::CBIndex::c_4;
    constexpr uint32_t cb_tw_i_id     = tt::CBIndex::c_5;
    // Output CBs
    constexpr uint32_t cb_out0_r_id   = tt::CBIndex::c_16;
    constexpr uint32_t cb_out0_i_id   = tt::CBIndex::c_17;
    constexpr uint32_t cb_out1_r_id   = tt::CBIndex::c_18;
    constexpr uint32_t cb_out1_i_id   = tt::CBIndex::c_19;
    // Intermediate CBs
    constexpr uint32_t cb_tmp0_id     = tt::CBIndex::c_20;
    constexpr uint32_t cb_tmp1_id     = tt::CBIndex::c_21;
    constexpr uint32_t cb_tw_odd_r_id = tt::CBIndex::c_22;
    constexpr uint32_t cb_tw_odd_i_id = tt::CBIndex::c_23;
    constexpr uint32_t cb_neg_tmp_id  = tt::CBIndex::c_24;

    // ── CB depth constants ────────────────────────────────────────────────
    //
    //  IO CBs (depth=2):
    //    Reader pre-fetches tile i+1 while compute processes tile i.
    //    Writer drains tile i while compute produces tile i+1.
    //    Without depth=2 the pipeline collapses to single-buffered and
    //    the NOC idles 50% of the time waiting for compute to pop.
    //
    //  Intermediate CBs (depth=1):
    //    cb_tmp0/1     : hold one mul result, immediately consumed by SUB/ADD.
    //    cb_tw_odd_r/i : hold W·O[k], immediately consumed by butterfly ADD/SUB.
    //    cb_neg_tmp    : holds negated twiddle_i, immediately consumed by butterfly.
    //    Setting depth=2 here wastes L1 and creates false slack that can
    //    hide bugs (double-produce goes undetected until CB wraps).
    //
    //  L1 total:
    //    Before: 13 CBs × 2 × 4096 = 106,496 bytes
    //    After : 10 CBs × 2 × 4096
    //          +  5 CBs × 1 × 4096 = 102,400 bytes  (saves 4 KB)

    constexpr uint32_t CB_DEPTH_IO  = 2;  // input + output CBs
    constexpr uint32_t CB_DEPTH_TMP = 1;  // intermediate CBs

    // ── Create circular buffers ───────────────────────────────────────────
    auto make_cb = [&](uint32_t cb_id, uint32_t depth) {
        CircularBufferConfig cfg =
            CircularBufferConfig(depth * single_tile_size,
                                 {{cb_id, tt::DataFormat::Float32}})
            .set_page_size(cb_id, single_tile_size);
        CreateCircularBuffer(program, core, cfg);
    };

    // Input CBs — depth 2 (double-buffered, reader prefetch)
    make_cb(cb_even_r_id,   CB_DEPTH_IO);
    make_cb(cb_even_i_id,   CB_DEPTH_IO);
    make_cb(cb_odd_r_id,    CB_DEPTH_IO);
    make_cb(cb_odd_i_id,    CB_DEPTH_IO);
    make_cb(cb_tw_r_id,     CB_DEPTH_IO);
    make_cb(cb_tw_i_id,     CB_DEPTH_IO);

    // Output CBs — depth 2 (double-buffered, writer drain)
    make_cb(cb_out0_r_id,   CB_DEPTH_IO);
    make_cb(cb_out0_i_id,   CB_DEPTH_IO);
    make_cb(cb_out1_r_id,   CB_DEPTH_IO);
    make_cb(cb_out1_i_id,   CB_DEPTH_IO);

    // Intermediate CBs — depth 1 (transient, 1 tile in flight max)
    make_cb(cb_tmp0_id,     CB_DEPTH_TMP);
    make_cb(cb_tmp1_id,     CB_DEPTH_TMP);
    make_cb(cb_tw_odd_r_id, CB_DEPTH_TMP);
    make_cb(cb_tw_odd_i_id, CB_DEPTH_TMP);
    make_cb(cb_neg_tmp_id,  CB_DEPTH_TMP);

    // ── DRAM buffers ──────────────────────────────────────────────────────
    //
    // Sized to butterfly_tiles * TILE_SIZE * sizeof(float).
    // All stages have exactly num_butterflies = N/2 butterfly pairs, so this
    // size is constant and the same buffers are reused each stage.
    const uint32_t dram_buf_bytes = butterfly_tiles * TILE_SIZE * sizeof(float);

    auto even_r_buf  = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto even_i_buf  = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto odd_r_buf   = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto odd_i_buf   = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto tw_r_buf    = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto tw_i_buf    = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto out0_r_buf  = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto out0_i_buf  = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto out1_r_buf  = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);
    auto out1_i_buf  = make_dram_buffer(mesh_device.get(), dram_buf_bytes, single_tile_size);

    // ── Kernels ───────────────────────────────────────────────────────────
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default
        }
    );

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default
        }
    );

    // ── UnpackToDestMode: full Float32 for all CBs read by compute ────────
    //
    // Without UnpackToDestFp32 the unpacker truncates Float32 tile data to
    // Tf32 (10-bit mantissa) before writing to the dest register, losing
    // ~13 bits of precision — catastrophic for FFT accuracy accumulation.
    //
    // Set for: all 6 input CBs + all 5 intermediate CBs.
    // Output CBs (c_16..c_19) are written by the packer, not unpacked,
    // so they do not need this mode.
    std::vector<UnpackToDestMode> unpack_modes(32, UnpackToDestMode::Default);

    // Input CBs
    unpack_modes[cb_even_r_id]   = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_even_i_id]   = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_odd_r_id]    = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_odd_i_id]    = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_tw_r_id]     = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_tw_i_id]     = UnpackToDestMode::UnpackToDestFp32;
    // Intermediate CBs
    unpack_modes[cb_tmp0_id]     = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_tmp1_id]     = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_tw_odd_r_id] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_tw_odd_i_id] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[cb_neg_tmp_id]  = UnpackToDestMode::UnpackToDestFp32;

    auto compute_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{
            .math_fidelity      = MathFidelity::HiFi4,
            .fp32_dest_acc_en   = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode   = false
        }
    );

    // ── Bit-reverse input, set up working buffers ─────────────────────────
    bit_reverse_permutation(input_real, input_imag, N);

    std::vector<float> work_real   = input_real;
    std::vector<float> work_imag   = input_imag;
    std::vector<float> result_real(padded_size, 0.0f);
    std::vector<float> result_imag(padded_size, 0.0f);

    // ── MeshWorkload wrapper ──────────────────────────────────────────────
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    Program& prog = workload.get_programs().begin()->second;

    // ── Stage loop ────────────────────────────────────────────────────────
    //
    // Each stage processes all N/2 butterfly pairs in one device dispatch.
    // The host gathers even/odd/twiddle data, writes 6 DRAM buffers, runs
    // the kernel, reads 4 output buffers, and scatters results back into
    // work_real/work_imag for the next stage.
    //
    // Enqueue pattern (batched, mirrors the optimized reader kernel):
    //   EnqueueWrite × 6  (no Finish between them — pipelined on the CQ)
    //   Finish             (one sync point for all 6 writes)
    //   EnqueueMeshWorkload
    //   Finish
    //   EnqueueRead × 4   (blocking=true on last read — implicit sync)

    for (uint32_t stage = 0; stage < log2N; stage++) {
        const uint32_t m      = 1u << (stage + 1);
        const uint32_t half_m = m / 2;

        std::cout << "Stage " << stage << " / " << log2N - 1
                  << "  (m=" << m << ")" << std::endl;

        // ── Gather butterfly pairs for this stage ─────────────────────────
        // All vectors are pre-sized to padded_butterfly (tile-aligned).
        std::vector<float> even_real(padded_butterfly, 0.0f);
        std::vector<float> even_imag(padded_butterfly, 0.0f);
        std::vector<float> odd_real (padded_butterfly, 0.0f);
        std::vector<float> odd_imag (padded_butterfly, 0.0f);
        std::vector<float> tw_real  (padded_butterfly, 0.0f);
        std::vector<float> tw_imag  (padded_butterfly, 0.0f);

        uint32_t bf = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++, bf++) {
                const uint32_t idx_even = k + j;
                const uint32_t idx_odd  = k + j + half_m;

                even_real[bf] = work_real[idx_even];
                even_imag[bf] = work_imag[idx_even];
                odd_real [bf] = work_real[idx_odd];
                odd_imag [bf] = work_imag[idx_odd];

                // Always use forward-FFT twiddle angles here.
                // The device compute kernel conjugates them for IFFT
                // (negates the imaginary part via SFPU NEG).
                const float angle = -2.0f * PI * j / m;
                tw_real[bf] = std::cos(angle);
                tw_imag[bf] = std::sin(angle);
            }
        }

        // ── Tilize ────────────────────────────────────────────────────────
        auto even_r_data = tilize_float_vector(even_real, butterfly_tiles, TILE_SIZE);
        auto even_i_data = tilize_float_vector(even_imag, butterfly_tiles, TILE_SIZE);
        auto odd_r_data  = tilize_float_vector(odd_real,  butterfly_tiles, TILE_SIZE);
        auto odd_i_data  = tilize_float_vector(odd_imag,  butterfly_tiles, TILE_SIZE);
        auto tw_r_data   = tilize_float_vector(tw_real,   butterfly_tiles, TILE_SIZE);
        auto tw_i_data   = tilize_float_vector(tw_imag,   butterfly_tiles, TILE_SIZE);

        // ── Write all 6 input buffers, then one Finish ────────────────────
        //
        // Enqueueing all 6 writes before Finish() lets the command queue
        // pipeline the DMA transfers — the same philosophy as batching NOC
        // reads in the reader kernel (6 reads, 1 barrier).
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, even_r_buf, even_r_data, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, even_i_buf, even_i_data, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, odd_r_buf,  odd_r_data,  false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, odd_i_buf,  odd_i_data,  false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, tw_r_buf,   tw_r_data,   false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, tw_i_buf,   tw_i_data,   false);
        tt::tt_metal::distributed::Finish(cq);  // one sync for all 6 DMA writes

        // ── Set runtime args ──────────────────────────────────────────────
        //
        // Reader: 6 DRAM addresses + num_tiles + start_tile
        // Writer: 4 DRAM addresses + num_tiles + start_tile
        // Compute: direction + num_butterflies (tile count)
        //
        // NOTE: butterfly_tiles is passed as num_butterflies to all kernels.
        // The compute kernel iterates bf in [0, butterfly_tiles) and processes
        // exactly one tile per butterfly iteration — each tile holds up to
        // TILE_SIZE butterfly pairs.

        SetRuntimeArgs(prog, reader_kernel, core, {
            even_r_buf->address(),
            even_i_buf->address(),
            odd_r_buf->address(),
            odd_i_buf->address(),
            tw_r_buf->address(),
            tw_i_buf->address(),
            butterfly_tiles,   // num_tiles
            0u                 // start_tile
        });

        SetRuntimeArgs(prog, writer_kernel, core, {
            out0_r_buf->address(),
            out0_i_buf->address(),
            out1_r_buf->address(),
            out1_i_buf->address(),
            butterfly_tiles,   // num_tiles
            0u                 // start_tile
        });

        SetRuntimeArgs(prog, compute_kernel, core, {
            direction,
            butterfly_tiles    // num_butterflies (one per tile)
        });

        // ── Execute ───────────────────────────────────────────────────────
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
        tt::tt_metal::distributed::Finish(cq);

        // ── Read 4 output buffers ─────────────────────────────────────────
        //
        // Use blocking=true on the last read so the final Finish is implicit.
        // The first three use blocking=false to pipeline the reads.
        const uint32_t out_elems = butterfly_tiles * TILE_SIZE;
        std::vector<uint32_t> out0_r_raw(out_elems);
        std::vector<uint32_t> out0_i_raw(out_elems);
        std::vector<uint32_t> out1_r_raw(out_elems);
        std::vector<uint32_t> out1_i_raw(out_elems);

        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_r_raw, out0_r_buf, false);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_i_raw, out0_i_buf, false);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_r_raw, out1_r_buf, false);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_i_raw, out1_i_buf, true); // blocking

        // ── Untilize ──────────────────────────────────────────────────────
        auto out0_real = untilize_to_float(out0_r_raw, num_butterflies);
        auto out0_imag = untilize_to_float(out0_i_raw, num_butterflies);
        auto out1_real = untilize_to_float(out1_r_raw, num_butterflies);
        auto out1_imag = untilize_to_float(out1_i_raw, num_butterflies);

        // ── Scatter results back into working buffer ───────────────────────
        bf = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++, bf++) {
                const uint32_t idx_even = k + j;
                const uint32_t idx_odd  = k + j + half_m;

                result_real[idx_even] = out0_real[bf];
                result_imag[idx_even] = out0_imag[bf];
                result_real[idx_odd]  = out1_real[bf];
                result_imag[idx_odd]  = out1_imag[bf];
            }
        }

        work_real = result_real;
        work_imag = result_imag;
    }

    // ── IFFT 1/N scaling ──────────────────────────────────────────────────
    if (direction == 1) {
        const float scale = 1.0f / N;
        for (uint32_t i = 0; i < N; i++) {
            work_real[i] *= scale;
            work_imag[i] *= scale;
        }
    }

    // ── Validation ────────────────────────────────────────────────────────
    std::cout << "\n=== Validation ===" << std::endl;

    float max_err_r = 0.0f, max_err_i = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        max_err_r = std::max(max_err_r, std::abs(work_real[i] - ref_real[i]));
        max_err_i = std::max(max_err_i, std::abs(work_imag[i] - ref_imag[i]));
    }

    std::cout << "Max error (real): " << max_err_r << std::endl;
    std::cout << "Max error (imag): " << max_err_i << std::endl;

    const bool passed = (max_err_r < 1e-3f) && (max_err_i < 1e-3f);
    std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << std::endl;

    // ── Print first 16 results ────────────────────────────────────────────
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

    // ── Cleanup ───────────────────────────────────────────────────────────
    mesh_device->close();
    std::cout << "\n=== Done ===" << std::endl;
    return passed ? 0 : 1;
}