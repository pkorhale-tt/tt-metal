// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_multi_core.cpp  –  CORRECTED HOST
//
// Fixes applied vs. original:
//   1. Device  → IDevice   (API rename in newer TT-Metalium)
//   2. CommandQueue removed as a declared type; cq obtained via
//      device->command_queue() which returns ICommandQueue& directly.
//   3. stage0/stage1 are FULL-ROW ping-pong DRAM buffers.
//   4. Separate output DRAM buffer for final results.
//   5. CB sizes fixed (pair-sized CBs double-buffered; row scratch CBs
//      use separate indices for reader/writer to avoid race).
//   6. Compute kernel arg[1] = rowsPerCore * pair_tiles.
//   7. Reader args = 13, Writer args = 12 (matching kernels exactly).

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

constexpr float    PI         = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = 32;
constexpr uint32_t TILE_W     = 32;
constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

// CB indices – must match kernels.
constexpr uint32_t CB_EVEN_R       = 0;
constexpr uint32_t CB_EVEN_I       = 1;
constexpr uint32_t CB_ODD_R        = 2;
constexpr uint32_t CB_ODD_I        = 3;
constexpr uint32_t CB_TW_R         = 4;
constexpr uint32_t CB_TW_I         = 5;
constexpr uint32_t CB_OUT0_R       = 16;
constexpr uint32_t CB_OUT0_I       = 17;
constexpr uint32_t CB_OUT1_R       = 18;
constexpr uint32_t CB_OUT1_I       = 19;
constexpr uint32_t CB_TMP0         = 20;
constexpr uint32_t CB_TMP1         = 21;
constexpr uint32_t CB_TW_ODD_R     = 22;
constexpr uint32_t CB_TW_ODD_I     = 23;
constexpr uint32_t CB_READER_ROW_R = 24;
constexpr uint32_t CB_READER_ROW_I = 25;
constexpr uint32_t CB_WRITER_ROW_R = 26;
constexpr uint32_t CB_WRITER_ROW_I = 27;

inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline bool isPowerOfTwo(uint32_t x) { return x > 0 && ((x & (x - 1)) == 0); }
inline uint32_t log2u32(uint32_t x) { uint32_t r = 0; while ((1u << r) < x) ++r; return r; }
inline uint32_t floatToU32(float v)  { uint32_t u; std::memcpy(&u, &v, 4); return u; }
inline float    u32ToFloat(uint32_t u){ float v;   std::memcpy(&v, &u, 4); return v; }

std::shared_ptr<distributed::MeshBuffer> createDramMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes)
{
    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig replicatedConfig{.size = sizeBytes};
    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

void packFullRows(
    const std::vector<float>& inputReal,
    const std::vector<float>& inputImag,
    uint32_t batchSize, uint32_t batchPadded, uint32_t nRow, uint32_t rowTiles,
    std::vector<uint32_t>& stage0Real, std::vector<uint32_t>& stage0Imag)
{
    const uint32_t elemsPerRowPadded = rowTiles * TILE_ELEMS;
    stage0Real.assign(batchPadded * elemsPerRowPadded, 0u);
    stage0Imag.assign(batchPadded * elemsPerRowPadded, 0u);
    for (uint32_t row = 0; row < batchSize; ++row) {
        const uint32_t dst = row * elemsPerRowPadded;
        const uint32_t src = row * nRow;
        for (uint32_t i = 0; i < nRow; ++i) {
            stage0Real[dst + i] = floatToU32(inputReal[src + i]);
            stage0Imag[dst + i] = floatToU32(inputImag[src + i]);
        }
    }
}

void buildTwiddleTiles(
    uint32_t nRow, uint32_t numStages, uint32_t totalRows, uint32_t pairTiles,
    uint32_t direction,
    std::vector<uint32_t>& twReal, std::vector<uint32_t>& twImag)
{
    const uint32_t halfN             = nRow / 2;
    const uint32_t elemsPerRowPadded = pairTiles * TILE_ELEMS;
    const float    sign              = (direction == 1) ? 1.0f : -1.0f;

    twReal.assign(numStages * totalRows * elemsPerRowPadded, 0u);
    twImag.assign(numStages * totalRows * elemsPerRowPadded, 0u);

    for (uint32_t stage = 0; stage < numStages; ++stage) {
        const uint32_t halfM = 1u << stage;
        const uint32_t m     = halfM << 1u;
        for (uint32_t row = 0; row < totalRows; ++row) {
            const uint32_t base = (stage * totalRows + row) * elemsPerRowPadded;
            for (uint32_t b = 0; b < halfN; ++b) {
                const uint32_t j     = b % halfM;
                const uint32_t k     = j * (nRow / m);
                const float    angle = sign * 2.0f * PI * static_cast<float>(k)
                                              / static_cast<float>(nRow);
                twReal[base + b] = floatToU32(std::cos(angle));
                twImag[base + b] = floatToU32(std::sin(angle));
            }
        }
    }
}

void makeTestInput(uint32_t batchSize, uint32_t nRow,
                   std::vector<float>& inputReal, std::vector<float>& inputImag)
{
    inputReal.resize(batchSize * nRow);
    inputImag.resize(batchSize * nRow);
    for (uint32_t row = 0; row < batchSize; ++row)
        for (uint32_t i = 0; i < nRow; ++i) {
            float x = std::sin(2.0f * PI * i / nRow)
                    + 0.25f * std::cos(6.0f * PI * i / nRow);
            inputReal[row * nRow + i] = x + 0.01f * static_cast<float>(row);
            inputImag[row * nRow + i] = 0.0f;
        }
}

void printFirstOutputs(const std::vector<float>& outputReal,
                       const std::vector<float>& outputImag,
                       uint32_t batchSize, uint32_t nRow, uint32_t count = 16)
{
    const uint32_t n = std::min(count, nRow);
    for (uint32_t row = 0; row < std::min(batchSize, 2u); ++row) {
        std::cout << "row " << row << ":\n";
        for (uint32_t i = 0; i < n; ++i)
            std::cout << "  [" << i << "] = ("
                      << outputReal[row * nRow + i] << ", "
                      << outputImag[row * nRow + i] << ")\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int      deviceId  = (argc > 1) ? std::stoi(argv[1])                         : 0;
        const uint32_t nRow      = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 256;
        const uint32_t numCores  = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow)) throw std::runtime_error("nRow must be power of 2");
        if (nRow < 2)            throw std::runtime_error("nRow must be >= 2");
        if (numCores == 0 || numCores > 64)
            throw std::runtime_error("numCores must be in [1, 64]");

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        // FIX: use mesh_command_queue() which returns the correct queue type.
        auto& cq = meshDevice->mesh_command_queue();

        const uint32_t numStages   = log2u32(nRow);
        const uint32_t halfN       = nRow / 2;
        const uint32_t rowTiles    = ceilDiv(nRow,  TILE_ELEMS);
        const uint32_t pairTiles   = ceilDiv(halfN, TILE_ELEMS);
        const uint32_t rowsPerCore = ceilDiv(batchSize, numCores);
        const uint32_t batchPadded = numCores * rowsPerCore;
        const uint32_t totalRows   = batchPadded;

        const uint32_t tilesPerStageCompute = rowsPerCore * pairTiles;

        const uint32_t rowBufBytes = totalRows * rowTiles  * TILE_BYTES;
        const uint32_t twBufBytes  = numStages * totalRows * pairTiles * TILE_BYTES;

        std::cout << "[fft_paper_host]\n"
                  << "  nRow               = " << nRow      << "\n"
                  << "  batchSize          = " << batchSize << "\n"
                  << "  numCores           = " << numCores  << "\n"
                  << "  rowsPerCore        = " << rowsPerCore << "\n"
                  << "  numStages          = " << numStages << "\n"
                  << "  rowTiles           = " << rowTiles  << "\n"
                  << "  pairTiles          = " << pairTiles << "\n"
                  << "  tilesPerStageComp  = " << tilesPerStageCompute << "\n";

        std::vector<float> inputReal, inputImag;
        makeTestInput(batchSize, nRow, inputReal, inputImag);

        std::vector<uint32_t> stage0RealPacked, stage0ImagPacked;
        packFullRows(inputReal, inputImag, batchSize, batchPadded, nRow,
                     rowTiles, stage0RealPacked, stage0ImagPacked);

        std::vector<uint32_t> twRealPacked, twImagPacked;
        buildTwiddleTiles(nRow, numStages, totalRows, pairTiles, direction,
                          twRealPacked, twImagPacked);

        auto stage0RealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto stage0ImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto stage1RealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto stage1ImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputRealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto twRealBuf     = createDramMeshBuffer(meshDevice, twBufBytes);
        auto twImagBuf     = createDramMeshBuffer(meshDevice, twBufBytes);

        Program program = CreateProgram();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange deviceRange(meshDevice->shape());

        CoreRange coreRange({0, 0}, {numCores - 1, 0});

        auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
            CircularBufferConfig cfg =
                CircularBufferConfig(depthTiles * TILE_BYTES,
                                     {{cbId, tt::DataFormat::Float32}})
                    .set_page_size(cbId, TILE_BYTES);
            CreateCircularBuffer(program, coreRange, cfg);
        };

        makeCb(CB_EVEN_R,       2 * pairTiles);
        makeCb(CB_EVEN_I,       2 * pairTiles);
        makeCb(CB_ODD_R,        2 * pairTiles);
        makeCb(CB_ODD_I,        2 * pairTiles);
        makeCb(CB_TW_R,         2 * pairTiles);
        makeCb(CB_TW_I,         2 * pairTiles);
        makeCb(CB_OUT0_R,       2 * pairTiles);
        makeCb(CB_OUT0_I,       2 * pairTiles);
        makeCb(CB_OUT1_R,       2 * pairTiles);
        makeCb(CB_OUT1_I,       2 * pairTiles);
        makeCb(CB_TMP0,         1);
        makeCb(CB_TMP1,         1);
        makeCb(CB_TW_ODD_R,     1);
        makeCb(CB_TW_ODD_I,     1);
        makeCb(CB_READER_ROW_R, 2 * rowTiles);
        makeCb(CB_READER_ROW_I, 2 * rowTiles);
        makeCb(CB_WRITER_ROW_R, 2 * rowTiles);
        makeCb(CB_WRITER_ROW_I, 2 * rowTiles);

        KernelHandle readerKernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX
            "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader_fft_f32.cpp",
            coreRange,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default});

        KernelHandle writerKernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX
            "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer_fft_f32.cpp",
            coreRange,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default});

        KernelHandle computeKernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX
            "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/fft_compute_f32.cpp",
            coreRange,
            ComputeConfig{
                .math_fidelity    = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true});

        for (uint32_t c = 0; c < numCores; ++c) {
            CoreCoord coreCoord{c, 0};
            const uint32_t rowStart = c * rowsPerCore;

            SetRuntimeArgs(program, readerKernel, coreCoord,
                {
                    stage0RealBuf->address(),  // 0
                    stage0ImagBuf->address(),  // 1
                    stage1RealBuf->address(),  // 2
                    stage1ImagBuf->address(),  // 3
                    twRealBuf->address(),      // 4
                    twImagBuf->address(),      // 5
                    rowTiles,                  // 6
                    pairTiles,                 // 7
                    nRow,                      // 8
                    numStages,                 // 9
                    totalRows,                 // 10
                    rowStart,                  // 11
                    rowsPerCore                // 12
                });

            SetRuntimeArgs(program, computeKernel, coreCoord,
                {
                    numStages,             // 0
                    tilesPerStageCompute   // 1
                });

            SetRuntimeArgs(program, writerKernel, coreCoord,
                {
                    stage0RealBuf->address(),  // 0
                    stage0ImagBuf->address(),  // 1
                    stage1RealBuf->address(),  // 2
                    stage1ImagBuf->address(),  // 3
                    outputRealBuf->address(),  // 4
                    outputImagBuf->address(),  // 5
                    rowTiles,                  // 6
                    pairTiles,                 // 7
                    nRow,                      // 8
                    numStages,                 // 9
                    rowStart,                  // 10
                    rowsPerCore                // 11
                });
        }

        distributed::EnqueueWriteMeshBuffer(cq, stage0RealBuf, stage0RealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, stage0ImagBuf, stage0ImagPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, twRealBuf,     twRealPacked,     false);
        distributed::EnqueueWriteMeshBuffer(cq, twImagBuf,     twImagPacked,     false);

        workload.add_program(deviceRange, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::cout << "Kernel execution finished.\n";

        std::vector<uint32_t> outRawReal, outRawImag;
        distributed::EnqueueReadMeshBuffer(cq, outRawReal, outputRealBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, outRawImag, outputImagBuf, true);

        std::vector<float> outputReal(batchSize * nRow, 0.0f);
        std::vector<float> outputImag(batchSize * nRow, 0.0f);

        const uint32_t elemsPerRowPadded = rowTiles * TILE_ELEMS;
        for (uint32_t row = 0; row < batchSize; ++row) {
            const uint32_t src = row * elemsPerRowPadded;
            const uint32_t dst = row * nRow;
            for (uint32_t i = 0; i < nRow; ++i) {
                outputReal[dst + i] = u32ToFloat(outRawReal[src + i]);
                outputImag[dst + i] = u32ToFloat(outRawImag[src + i]);
            }
        }

        printFirstOutputs(outputReal, outputImag, batchSize, nRow);

        if (!meshDevice->close())
            throw std::runtime_error("meshDevice->close() failed");

        std::cout << "FFT host run finished.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FFT host failed: " << e.what() << "\n";
        return 1;
    }
}