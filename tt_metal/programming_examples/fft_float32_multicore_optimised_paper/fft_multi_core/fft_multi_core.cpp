// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_multi_core.cpp
//
// Host code for these kernels:
//   - kernels/dataflow/reader_fft_f32_prod.cpp
//   - kernels/dataflow/writer_fft_f32_prod.cpp
//   - kernels/compute/fft_compute_f32_prod.cpp

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

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

inline uint32_t ceilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

inline bool isPowerOfTwo(uint32_t x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

inline uint32_t log2u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) {
        ++r;
    }
    return r;
}

inline uint32_t floatToU32(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
}

inline float u32ToFloat(uint32_t v) {
    float out;
    std::memcpy(&out, &v, sizeof(uint32_t));
    return out;
}

std::shared_ptr<distributed::MeshBuffer> createDramMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes) {

    distributed::DeviceLocalBufferConfig localConfig{
        .page_size = TILE_BYTES,
        .buffer_type = BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig replicatedConfig{
        .size = sizeBytes
    };

    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

void packStage0EvenOdd(
    const std::vector<float>& inputReal,
    const std::vector<float>& inputImag,
    uint32_t batchSize,
    uint32_t nRow,
    uint32_t numCores,
    uint32_t rowsPerCore,
    std::vector<uint32_t>& evenRealPacked,
    std::vector<uint32_t>& evenImagPacked,
    std::vector<uint32_t>& oddRealPacked,
    std::vector<uint32_t>& oddImagPacked) {

    const uint32_t batchPadded = numCores * rowsPerCore;
    const uint32_t halfN = nRow / 2;
    const uint32_t tilesPerRow = ceilDiv(halfN, TILE_ELEMS);
    const uint32_t elemsPerRowPadded = tilesPerRow * TILE_ELEMS;

    evenRealPacked.assign(batchPadded * elemsPerRowPadded, 0);
    evenImagPacked.assign(batchPadded * elemsPerRowPadded, 0);
    oddRealPacked.assign(batchPadded * elemsPerRowPadded, 0);
    oddImagPacked.assign(batchPadded * elemsPerRowPadded, 0);

    for (uint32_t row = 0; row < batchSize; ++row) {
        const uint32_t dstBase = row * elemsPerRowPadded;
        const uint32_t srcBase = row * nRow;

        for (uint32_t k = 0; k < halfN; ++k) {
            evenRealPacked[dstBase + k] = floatToU32(inputReal[srcBase + 2 * k]);
            evenImagPacked[dstBase + k] = floatToU32(inputImag[srcBase + 2 * k]);
            oddRealPacked[dstBase + k]  = floatToU32(inputReal[srcBase + 2 * k + 1]);
            oddImagPacked[dstBase + k]  = floatToU32(inputImag[srcBase + 2 * k + 1]);
        }
    }
}

void buildTwiddleTiles(
    uint32_t nRow,
    uint32_t numStages,
    uint32_t numCores,
    uint32_t rowsPerCore,
    uint32_t direction,
    std::vector<uint32_t>& twiddleRealPacked,
    std::vector<uint32_t>& twiddleImagPacked) {

    const uint32_t batchPadded = numCores * rowsPerCore;
    const uint32_t halfN = nRow / 2;
    const uint32_t tilesPerRow = ceilDiv(halfN, TILE_ELEMS);
    const uint32_t elemsPerRowPadded = tilesPerRow * TILE_ELEMS;
    const float sign = (direction == 1) ? 1.0f : -1.0f;

    twiddleRealPacked.assign(numStages * batchPadded * elemsPerRowPadded, 0);
    twiddleImagPacked.assign(numStages * batchPadded * elemsPerRowPadded, 0);

    for (uint32_t stage = 0; stage < numStages; ++stage) {
        const uint32_t m = 1u << (stage + 1);
        const uint32_t halfM = m >> 1;

        for (uint32_t row = 0; row < batchPadded; ++row) {
            const uint32_t base = (stage * batchPadded + row) * elemsPerRowPadded;

            for (uint32_t b = 0; b < halfN; ++b) {
                const uint32_t j = b % halfM;
                const uint32_t k = j * (nRow / m);
                const float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(nRow);

                twiddleRealPacked[base + b] = floatToU32(std::cos(angle));
                twiddleImagPacked[base + b] = floatToU32(std::sin(angle));
            }
        }
    }
}

void makeTestInput(
    uint32_t batchSize,
    uint32_t nRow,
    std::vector<float>& inputReal,
    std::vector<float>& inputImag) {

    inputReal.resize(batchSize * nRow);
    inputImag.resize(batchSize * nRow);

    for (uint32_t row = 0; row < batchSize; ++row) {
        for (uint32_t i = 0; i < nRow; ++i) {
            float x = std::sin(2.0f * PI * static_cast<float>(i) / static_cast<float>(nRow))
                    + 0.25f * std::cos(6.0f * PI * static_cast<float>(i) / static_cast<float>(nRow));
            inputReal[row * nRow + i] = x + 0.01f * static_cast<float>(row);
            inputImag[row * nRow + i] = 0.0f;
        }
    }
}

void printFirstOutputs(
    const std::vector<float>& outputReal,
    const std::vector<float>& outputImag,
    uint32_t batchSize,
    uint32_t nRow,
    uint32_t count = 16) {

    const uint32_t n = std::min(count, nRow);
    for (uint32_t row = 0; row < std::min(batchSize, 2u); ++row) {
        std::cout << "row " << row << ":\n";
        for (uint32_t i = 0; i < n; ++i) {
            std::cout << "  [" << i << "] = (" << outputReal[row * nRow + i]
                      << ", " << outputImag[row * nRow + i] << ")\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int deviceId = (argc > 1) ? std::stoi(argv[1]) : 0;
        const uint32_t nRow = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 256;
        const uint32_t numCores = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow)) {
            throw std::runtime_error("nRow must be a power of two");
        }
        if (nRow < 2) {
            throw std::runtime_error("nRow must be >= 2");
        }
        if (numCores == 0 || numCores > 64) {
            throw std::runtime_error("numCores must be in [1, 64]");
        }

        std::shared_ptr<distributed::MeshDevice> meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        distributed::MeshCommandQueue& cq = meshDevice->mesh_command_queue();

        const uint32_t numStages = log2u32(nRow);
        const uint32_t halfN = nRow / 2;
        const uint32_t tilesPerRow = ceilDiv(halfN, TILE_ELEMS);
        const uint32_t elemsPerRowPadded = tilesPerRow * TILE_ELEMS;

        const uint32_t rowsPerCore = ceilDiv(batchSize, numCores);
        const uint32_t batchPadded = numCores * rowsPerCore;
        const uint32_t totalTileCount = batchPadded * tilesPerRow;
        const uint32_t totalBytes = totalTileCount * TILE_BYTES;

        std::cout << "[fft_paper_host]\n";
        std::cout << "  nRow        = " << nRow << "\n";
        std::cout << "  batchSize   = " << batchSize << "\n";
        std::cout << "  numCores    = " << numCores << "\n";
        std::cout << "  rowsPerCore = " << rowsPerCore << "\n";
        std::cout << "  numStages   = " << numStages << "\n";
        std::cout << "  tilesPerRow = " << tilesPerRow << "\n";

        std::vector<float> inputReal;
        std::vector<float> inputImag;
        makeTestInput(batchSize, nRow, inputReal, inputImag);

        std::vector<uint32_t> evenRealPacked, evenImagPacked, oddRealPacked, oddImagPacked;
        packStage0EvenOdd(
            inputReal, inputImag, batchSize, nRow, numCores, rowsPerCore,
            evenRealPacked, evenImagPacked, oddRealPacked, oddImagPacked);

        std::vector<uint32_t> twiddleRealPacked, twiddleImagPacked;
        buildTwiddleTiles(
            nRow, numStages, numCores, rowsPerCore, direction,
            twiddleRealPacked, twiddleImagPacked);

        auto evenRealBuffer = createDramMeshBuffer(meshDevice, totalBytes);
        auto evenImagBuffer = createDramMeshBuffer(meshDevice, totalBytes);
        auto oddRealBuffer  = createDramMeshBuffer(meshDevice, totalBytes);
        auto oddImagBuffer  = createDramMeshBuffer(meshDevice, totalBytes);

        auto out0RealBuffer = createDramMeshBuffer(meshDevice, totalBytes);
        auto out0ImagBuffer = createDramMeshBuffer(meshDevice, totalBytes);
        auto out1RealBuffer = createDramMeshBuffer(meshDevice, totalBytes);
        auto out1ImagBuffer = createDramMeshBuffer(meshDevice, totalBytes);

        const uint32_t twiddleTotalBytes = static_cast<uint32_t>(twiddleRealPacked.size() * sizeof(uint32_t));
        auto twiddleRealBuffer = createDramMeshBuffer(meshDevice, twiddleTotalBytes);
        auto twiddleImagBuffer = createDramMeshBuffer(meshDevice, twiddleTotalBytes);

        Program program = CreateProgram();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange deviceRange(meshDevice->shape());

        CoreRange coreRange({0, 0}, {numCores - 1, 0});

        auto makeCb = [&](uint32_t cbId, uint32_t numPages) {
            CircularBufferConfig cbConfig =
                CircularBufferConfig(numPages * TILE_BYTES, {{cbId, tt::DataFormat::Float32}})
                    .set_page_size(cbId, TILE_BYTES);
            CreateCircularBuffer(program, coreRange, cbConfig);
        };

        makeCb(CB_EVEN_R, 2 * tilesPerRow);
        makeCb(CB_EVEN_I, 2 * tilesPerRow);
        makeCb(CB_ODD_R,  2 * tilesPerRow);
        makeCb(CB_ODD_I,  2 * tilesPerRow);
        makeCb(CB_TW_R,   2 * tilesPerRow);
        makeCb(CB_TW_I,   2 * tilesPerRow);

        makeCb(CB_OUT0_R, 2 * tilesPerRow);
        makeCb(CB_OUT0_I, 2 * tilesPerRow);
        makeCb(CB_OUT1_R, 2 * tilesPerRow);
        makeCb(CB_OUT1_I, 2 * tilesPerRow);

        makeCb(CB_TMP_R, 1);
        makeCb(CB_TMP_I, 1);

        KernelHandle readerKernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader_fft_f32_prod.cpp",
            coreRange,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default
            });

        KernelHandle writerKernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer_fft_f32_prod.cpp",
            coreRange,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default
            });

        KernelHandle computeKernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/fft_compute_f32_prod.cpp",
            coreRange,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true
            });

        const uint32_t tilesPerCore = rowsPerCore * tilesPerRow;

        for (uint32_t c = 0; c < numCores; ++c) {
            CoreCoord coreCoord{c, 0};
            const uint32_t tileOffset = c * tilesPerCore;

            SetRuntimeArgs(
                program,
                readerKernel,
                coreCoord,
                {
                    evenRealBuffer->address(),
                    evenImagBuffer->address(),
                    oddRealBuffer->address(),
                    oddImagBuffer->address(),
                    twiddleRealBuffer->address(),
                    twiddleImagBuffer->address(),
                    tilesPerRow,
                    tileOffset,
                    numStages,
                    rowsPerCore
                });

            SetRuntimeArgs(
                program,
                computeKernel,
                coreCoord,
                {
                    numStages,
                    tilesPerRow
                });

            SetRuntimeArgs(
                program,
                writerKernel,
                coreCoord,
                {
                    out0RealBuffer->address(),
                    out0ImagBuffer->address(),
                    out1RealBuffer->address(),
                    out1ImagBuffer->address(),
                    tilesPerRow,
                    numStages,
                    tileOffset,
                    rowsPerCore
                });
        }

        distributed::EnqueueWriteMeshBuffer(cq, evenRealBuffer, evenRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, evenImagBuffer, evenImagPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, oddRealBuffer, oddRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, oddImagBuffer, oddImagPacked, false);

        distributed::EnqueueWriteMeshBuffer(cq, twiddleRealBuffer, twiddleRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, twiddleImagBuffer, twiddleImagPacked, false);

        workload.add_program(deviceRange, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<uint32_t> out0RealPacked;
        std::vector<uint32_t> out0ImagPacked;
        std::vector<uint32_t> out1RealPacked;
        std::vector<uint32_t> out1ImagPacked;

        distributed::EnqueueReadMeshBuffer(cq, out0RealPacked, out0RealBuffer, true);
        distributed::EnqueueReadMeshBuffer(cq, out0ImagPacked, out0ImagBuffer, true);
        distributed::EnqueueReadMeshBuffer(cq, out1RealPacked, out1RealBuffer, true);
        distributed::EnqueueReadMeshBuffer(cq, out1ImagPacked, out1ImagBuffer, true);

        std::vector<float> outputReal(batchSize * nRow, 0.0f);
        std::vector<float> outputImag(batchSize * nRow, 0.0f);

        for (uint32_t row = 0; row < batchSize; ++row) {
            const uint32_t packedBase = row * elemsPerRowPadded;
            const uint32_t dstBase = row * nRow;

            for (uint32_t k = 0; k < halfN; ++k) {
                outputReal[dstBase + k]          = u32ToFloat(out0RealPacked[packedBase + k]);
                outputImag[dstBase + k]          = u32ToFloat(out0ImagPacked[packedBase + k]);
                outputReal[dstBase + halfN + k]  = u32ToFloat(out1RealPacked[packedBase + k]);
                outputImag[dstBase + halfN + k]  = u32ToFloat(out1ImagPacked[packedBase + k]);
            }
        }

        printFirstOutputs(outputReal, outputImag, batchSize, nRow);

        if (!meshDevice->close()) {
            throw std::runtime_error("meshDevice->close() failed");
        }

        std::cout << "FFT host run finished.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FFT host failed: " << e.what() << "\n";
        return 1;
    }
}