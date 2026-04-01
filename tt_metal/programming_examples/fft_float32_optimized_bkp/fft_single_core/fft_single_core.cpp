// fft_single_core.cpp  — OPTIMAL v2: compact twiddle table  [FIXED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FIX 1 (host): CB 4/5 (twiddle real/imag) depth changed from 1 to num_tiles.
//   The reader now pushes one tile at a time inside its per-tile loop, so
//   compute can pipeline: while compute processes tile k, the reader can
//   already be filling tile k+1 into the next slot. Without this, the
//   CB with depth=1 was safe only for N<=1024 (num_tiles==1). For larger N
//   depth must equal num_tiles so the reader never stalls waiting for compute
//   to drain a slot before it can write the next tile in the same stage.
//
// FIX 2 (host): CB 0-3 (even/odd inputs) depth changed from 1 to num_tiles+1.
//   The extra slot avoids a circular wait between compute and writer across
//   stage boundaries.
//
// FIX 3 (host): compact twiddle table size fixed.
//   precompute_compact_twiddles() must return exactly N/2 entries, because the
//   compact DRAM buffers are allocated with size half_N * sizeof(float).
//   Returning TILE_SIZE entries corrupts the compact twiddle upload and can
//   hang the workload even for N=4.

#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;

constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

std::vector<uint32_t> pack_tiles(const std::vector<float>& data, uint32_t numTiles) {
    std::vector<uint32_t> out(numTiles * TILE_SIZE, 0);
    for (uint32_t i = 0; i < data.size() && i < out.size(); i++) {
        out[i] = f2u(data[i]);
    }
    return out;
}

std::vector<float> unpack_tiles(const std::vector<uint32_t>& data, uint32_t count) {
    std::vector<float> out(count);
    for (uint32_t i = 0; i < count && i < data.size(); i++) {
        out[i] = u2f(data[i]);
    }
    return out;
}

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t reversed = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }
    return reversed;
}

void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inverse) {
    uint32_t N = real.size();
    uint32_t log2N = 0;
    while ((1u << log2N) < N) {
        log2N++;
    }

    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m = 1u << (stage + 1);
        float angleBase = (inverse ? 2.0f : -2.0f) * PI / static_cast<float>(m);

        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < m / 2; j++) {
                float wr = std::cos(angleBase * j);
                float wi = std::sin(angleBase * j);

                uint32_t evenIndex = k + j;
                uint32_t oddIndex = k + j + m / 2;

                float tempReal = wr * real[oddIndex] - wi * imag[oddIndex];
                float tempImag = wr * imag[oddIndex] + wi * real[oddIndex];

                float evenReal = real[evenIndex];
                float evenImag = imag[evenIndex];

                real[evenIndex] = evenReal + tempReal;
                imag[evenIndex] = evenImag + tempImag;
                real[oddIndex] = evenReal - tempReal;
                imag[oddIndex] = evenImag - tempImag;
            }
        }
    }

    if (inverse) {
        for (uint32_t i = 0; i < N; i++) {
            real[i] /= static_cast<float>(N);
            imag[i] /= static_cast<float>(N);
        }
    }
}

// Stage-0 split: bit-reversed input, stride-2 partition into even/odd
void prepare_stage0(
    const std::vector<float>& sourceReal,
    const std::vector<float>& sourceImag,
    uint32_t N,
    uint32_t log2N,
    uint32_t numTiles,
    std::vector<uint32_t>& evenRealTiles,
    std::vector<uint32_t>& evenImagTiles,
    std::vector<uint32_t>& oddRealTiles,
    std::vector<uint32_t>& oddImagTiles) {

    uint32_t half_N = N / 2;
    std::vector<float> evenReal(half_N), evenImag(half_N), oddReal(half_N), oddImag(half_N);

    for (uint32_t i = 0; i < half_N; i++) {
        uint32_t evenSource = bit_reverse(2 * i, log2N);
        uint32_t oddSource = bit_reverse(2 * i + 1, log2N);

        evenReal[i] = sourceReal[evenSource];
        evenImag[i] = sourceImag[evenSource];
        oddReal[i] = sourceReal[oddSource];
        oddImag[i] = sourceImag[oddSource];
    }

    evenRealTiles = pack_tiles(evenReal, numTiles);
    evenImagTiles = pack_tiles(evenImag, numTiles);
    oddRealTiles = pack_tiles(oddReal, numTiles);
    oddImagTiles = pack_tiles(oddImag, numTiles);
}

// Compact twiddle table: exactly N/2 entries, direction-aware sign.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, uint32_t direction) {
    uint32_t half_N = N / 2;
    float sign = (direction == 1) ? 1.0f : -1.0f;

    std::vector<uint32_t> twiddleReal(half_N, 0);
    std::vector<uint32_t> twiddleImag(half_N, 0);

    for (uint32_t k = 0; k < half_N; k++) {
        float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(N);
        twiddleReal[k] = f2u(std::cos(angle));
        twiddleImag[k] = f2u(std::sin(angle));
    }

    return {twiddleReal, twiddleImag};
}

void create_cb(Program& program, CoreCoord core, uint32_t id, uint32_t numTiles, uint32_t pageBytes) {
    CircularBufferConfig config =
        CircularBufferConfig(numTiles * pageBytes, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, pageBytes);
    CreateCircularBuffer(program, core, config);
}

bool read_file(
    const std::string& path,
    uint32_t& N,
    bool fromCommandLine,
    std::vector<float>& inputReal,
    std::vector<float>& inputImag) {

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }

    std::vector<float> values;
    std::string token;
    while (file >> token) {
        if (!token.empty() && token.back() == ',') {
            token.pop_back();
        }
        if (token.empty()) {
            continue;
        }
        try {
            values.push_back(std::stof(token));
        } catch (...) {
            std::cerr << "Bad token\n";
            return false;
        }
    }

    if (values.empty()) {
        std::cerr << "Empty file\n";
        return false;
    }

    uint32_t valueCount = static_cast<uint32_t>(values.size());
    bool interleaved = false;

    if (fromCommandLine) {
        if (valueCount == 2 * N) {
            interleaved = true;
        } else if (valueCount < N) {
            std::cerr << "File has " << valueCount << " values, padding to N=" << N << "\n";
        } else if (valueCount > N) {
            valueCount = N;
            values.resize(N);
        }
    } else {
        N = 1;
        while (N < valueCount) {
            N <<= 1;
        }
    }

    inputReal.assign(N, 0.0f);
    inputImag.assign(N, 0.0f);

    if (interleaved) {
        for (uint32_t i = 0; i < N && (2 * i + 1) < static_cast<uint32_t>(values.size()); i++) {
            inputReal[i] = values[2 * i];
            inputImag[i] = values[2 * i + 1];
        }
    } else {
        for (uint32_t i = 0; i < N && i < static_cast<uint32_t>(values.size()); i++) {
            inputReal[i] = values[i];
        }
    }

    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <0|1> [file] [N]\n";
        return 1;
    }

    uint32_t direction = static_cast<uint32_t>(std::atoi(argv[1]));
    uint32_t N = 1024;
    std::string inputFile;
    bool fromCommandLine = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        bool looksLikeFile = (arg.find('.') != std::string::npos || arg.find('/') != std::string::npos);

        if (looksLikeFile && inputFile.empty()) {
            inputFile = arg;
        } else {
            try {
                N = static_cast<uint32_t>(std::stol(arg));
                fromCommandLine = true;
            } catch (...) {
                if (inputFile.empty()) {
                    inputFile = arg;
                }
            }
        }
    }

    if (fromCommandLine && (N == 0 || (N & (N - 1)))) {
        std::cerr << "N must be power of 2\n";
        return 1;
    }

    uint32_t log2N = 0;
    while ((1u << log2N) < N) {
        log2N++;
    }

    uint32_t half_N = N / 2;
    uint32_t numTiles = (half_N + TILE_SIZE - 1) / TILE_SIZE;

    std::vector<float> inputReal(N, 0.0f), inputImag(N, 0.0f);
    if (!inputFile.empty()) {
        if (!read_file(inputFile, N, fromCommandLine, inputReal, inputImag)) {
            return 1;
        }

        log2N = 0;
        while ((1u << log2N) < N) {
            log2N++;
        }
        half_N = N / 2;
        numTiles = (half_N + TILE_SIZE - 1) / TILE_SIZE;

        inputReal.resize(N, 0.0f);
        inputImag.resize(N, 0.0f);

        if (N < 4 || (N & (N - 1))) {
            std::cerr << "Invalid N=" << N << " (must be power of 2, >= 4)\n";
            return 1;
        }
    } else {
        for (uint32_t i = 0; i < N; i++) {
            inputReal[i] = std::sin(2.0f * PI * 4.0f * i / N) + 0.5f * std::sin(2.0f * PI * 8.0f * i / N);
        }
    }

    uint32_t inputBytes = numTiles * TILE_BYTES;
    uint32_t compactBytes = half_N * sizeof(float);

    std::cout << "════════════════════════════════════════\n";
    std::cout << " TT-Metal FFT  (Optimal v2 — compact twiddles) [FIXED]\n";
    std::cout << "════════════════════════════════════════\n";
    std::cout << " N           : " << N << "\n";
    std::cout << " log2N       : " << log2N << "\n";
    std::cout << " Direction   : " << (direction ? "Inverse" : "Forward") << "\n";
    std::cout << " tiles/stage : " << numTiles << "\n";
    std::cout << " DRAM upload : " << (4 * inputBytes + 2 * compactBytes) / 1024 << " KB"
              << " (input " << 4 * inputBytes / 1024 << "KB + twiddles " << 2 * compactBytes / 1024 << "KB)\n";
    std::cout << " DRAM dl     : " << 4 * inputBytes / 1024 << " KB\n";
    std::cout << "════════════════════════════════════════\n";

    std::vector<float> referenceReal(inputReal), referenceImag(inputImag);
    cpu_fft(referenceReal, referenceImag, direction == 1);

    std::vector<uint32_t> evenRealTiles, evenImagTiles, oddRealTiles, oddImagTiles;
    prepare_stage0(inputReal, inputImag, N, log2N, numTiles, evenRealTiles, evenImagTiles, oddRealTiles, oddImagTiles);

    auto [compactTwiddleReal, compactTwiddleImag] = precompute_compact_twiddles(N, direction);

    int deviceId = 0;
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(deviceId);
    auto& commandQueue = mesh->mesh_command_queue();

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    tt::tt_metal::distributed::DeviceLocalBufferConfig dramConfig{
        .page_size = TILE_BYTES,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    auto makeTileBuffer = [&](uint32_t bytes) {
        tt::tt_metal::distributed::ReplicatedBufferConfig replicatedConfig{.size = bytes};
        return tt::tt_metal::distributed::MeshBuffer::create(replicatedConfig, dramConfig, mesh.get());
    };

    auto bufferEvenReal = makeTileBuffer(inputBytes);
    auto bufferEvenImag = makeTileBuffer(inputBytes);
    auto bufferOddReal  = makeTileBuffer(inputBytes);
    auto bufferOddImag  = makeTileBuffer(inputBytes);

    tt::tt_metal::distributed::DeviceLocalBufferConfig compactDramConfig{
        .page_size = compactBytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig compactReplicatedConfig{.size = compactBytes};
    auto bufferCompactReal = tt::tt_metal::distributed::MeshBuffer::create(compactReplicatedConfig, compactDramConfig, mesh.get());
    auto bufferCompactImag = tt::tt_metal::distributed::MeshBuffer::create(compactReplicatedConfig, compactDramConfig, mesh.get());

    auto bufferOut0Real = makeTileBuffer(inputBytes);
    auto bufferOut0Imag = makeTileBuffer(inputBytes);
    auto bufferOut1Real = makeTileBuffer(inputBytes);
    auto bufferOut1Imag = makeTileBuffer(inputBytes);

    create_cb(program, core, 0,  numTiles + 1, TILE_BYTES);
    create_cb(program, core, 1,  numTiles + 1, TILE_BYTES);
    create_cb(program, core, 2,  numTiles + 1, TILE_BYTES);
    create_cb(program, core, 3,  numTiles + 1, TILE_BYTES);
    create_cb(program, core, 4,  numTiles,     TILE_BYTES);
    create_cb(program, core, 5,  numTiles,     TILE_BYTES);
    create_cb(program, core, 16, numTiles,     TILE_BYTES);
    create_cb(program, core, 17, numTiles,     TILE_BYTES);
    create_cb(program, core, 18, numTiles,     TILE_BYTES);
    create_cb(program, core, 19, numTiles,     TILE_BYTES);
    create_cb(program, core, 20, 1,            TILE_BYTES);
    create_cb(program, core, 21, 1,            TILE_BYTES);
    create_cb(program, core, 22, 1,            TILE_BYTES);
    create_cb(program, core, 23, 1,            TILE_BYTES);
    create_cb(program, core, 10, 1,            TILE_BYTES);
    create_cb(program, core, 11, 1,            TILE_BYTES);

    auto readerKernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    auto writerKernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );

    auto computeKernel = CreateKernel(
        program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false
        }
    );

    std::vector<uint32_t> readerArgs = {
        bufferEvenReal->address(), bufferEvenImag->address(),
        bufferOddReal->address(),  bufferOddImag->address(),
        bufferCompactReal->address(), bufferCompactImag->address(),
        numTiles, log2N, half_N
    };

    std::vector<uint32_t> writerArgs = {
        bufferOut0Real->address(), bufferOut0Imag->address(),
        bufferOut1Real->address(), bufferOut1Imag->address(),
        numTiles, log2N, half_N
    };

    std::vector<uint32_t> computeArgs = {log2N, numTiles};

    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange meshRange =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh->shape());
    workload.add_program(meshRange, std::move(program));

    auto& workloadProgram = workload.get_programs().begin()->second;
    SetRuntimeArgs(workloadProgram, readerKernel, core, readerArgs);
    SetRuntimeArgs(workloadProgram, writerKernel, core, writerArgs);
    SetRuntimeArgs(workloadProgram, computeKernel, core, computeArgs);

    using namespace tt::tt_metal::distributed;

    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(commandQueue, bufferEvenReal, evenRealTiles, false);
    EnqueueWriteMeshBuffer(commandQueue, bufferEvenImag, evenImagTiles, false);
    EnqueueWriteMeshBuffer(commandQueue, bufferOddReal,  oddRealTiles,  false);
    EnqueueWriteMeshBuffer(commandQueue, bufferOddImag,  oddImagTiles,  false);
    EnqueueWriteMeshBuffer(commandQueue, bufferCompactReal, compactTwiddleReal, false);
    EnqueueWriteMeshBuffer(commandQueue, bufferCompactImag, compactTwiddleImag, false);
    Finish(commandQueue);

    std::cout << "Launching FFT kernel (" << log2N << " stages on device)...\n";
    EnqueueMeshWorkload(commandQueue, workload, true);
    std::cout << "Kernel complete.\n";

    std::vector<uint32_t> out0RealRaw(numTiles * TILE_SIZE), out0ImagRaw(numTiles * TILE_SIZE);
    std::vector<uint32_t> out1RealRaw(numTiles * TILE_SIZE), out1ImagRaw(numTiles * TILE_SIZE);

    EnqueueReadMeshBuffer(commandQueue, out0RealRaw, bufferOut0Real, true);
    EnqueueReadMeshBuffer(commandQueue, out0ImagRaw, bufferOut0Imag, true);
    EnqueueReadMeshBuffer(commandQueue, out1RealRaw, bufferOut1Real, true);
    EnqueueReadMeshBuffer(commandQueue, out1ImagRaw, bufferOut1Imag, true);

    auto out0Real = unpack_tiles(out0RealRaw, half_N);
    auto out0Imag = unpack_tiles(out0ImagRaw, half_N);
    auto out1Real = unpack_tiles(out1RealRaw, half_N);
    auto out1Imag = unpack_tiles(out1ImagRaw, half_N);

    std::vector<float> resultReal(N), resultImag(N);
    for (uint32_t i = 0; i < half_N; i++) {
        resultReal[i] = out0Real[i];
        resultImag[i] = out0Imag[i];
        resultReal[i + half_N] = out1Real[i];
        resultImag[i + half_N] = out1Imag[i];
    }

    if (direction == 1) {
        for (uint32_t i = 0; i < N; i++) {
            resultReal[i] /= static_cast<float>(N);
            resultImag[i] /= static_cast<float>(N);
        }
    }

    std::cout << "\n════════════════════════════════════════\n";
    std::cout << " VALIDATION\n";
    std::cout << "════════════════════════════════════════\n";

    float maxErrorReal = 0.0f;
    float maxErrorImag = 0.0f;
    float meanError = 0.0f;

    for (uint32_t i = 0; i < N; i++) {
        float errorReal = std::abs(resultReal[i] - referenceReal[i]);
        float errorImag = std::abs(resultImag[i] - referenceImag[i]);
        maxErrorReal = std::max(maxErrorReal, errorReal);
        maxErrorImag = std::max(maxErrorImag, errorImag);
        meanError += errorReal + errorImag;
    }
    meanError /= (2 * N);

    std::cout << " Max error (real): " << maxErrorReal << "\n";
    std::cout << " Max error (imag): " << maxErrorImag << "\n";
    std::cout << " Mean error      : " << meanError << "\n";

    bool passed = (maxErrorReal < 0.5f) && (maxErrorImag < 0.5f);
    std::cout << " Result: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";

    std::cout << "\n════════════════════════════════════════\n";
    std::cout << " FIRST 16 RESULTS\n";
    std::cout << "════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);

    for (uint32_t i = 0; i < 16 && i < N; i++) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(12) << resultReal[i]
                  << (resultImag[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(resultImag[i]) << "j"
                  << "   ref: " << std::setw(12) << referenceReal[i]
                  << (referenceImag[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(referenceImag[i]) << "j\n";
    }

    mesh->close();
    std::cout << "\n════════════════════════════════════════\n Done\n";
    std::cout << "════════════════════════════════════════\n";

    return passed ? 0 : 1;
}
