// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

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

constexpr uint32_t CB_DATA0_R   = 0;
constexpr uint32_t CB_DATA0_I   = 1;
constexpr uint32_t CB_DATA1_R   = 2;
constexpr uint32_t CB_DATA1_I   = 3;
constexpr uint32_t CB_TW_R      = 4;
constexpr uint32_t CB_TW_I      = 5;
constexpr uint32_t CB_OUT0_R    = 16;
constexpr uint32_t CB_OUT0_I    = 17;
constexpr uint32_t CB_OUT1_R    = 18;
constexpr uint32_t CB_OUT1_I    = 19;
constexpr uint32_t CB_INT0      = 20;
constexpr uint32_t CB_INT1      = 21;
constexpr uint32_t CB_F0        = 22;
constexpr uint32_t CB_F1        = 23;

// SRAM base above CB region
constexpr uint32_t SRAM_DATA_BASE = 0x40000;

inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline bool isPowerOfTwo(uint32_t x) { return x > 0 && ((x & (x - 1)) == 0); }
inline uint32_t log2u32(uint32_t x) { uint32_t r = 0; while ((1u << r) < x) ++r; return r; }
inline uint32_t floatToU32(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }
inline float u32ToFloat(uint32_t u) { float v; std::memcpy(&v, &u, 4); return v; }

std::shared_ptr<distributed::MeshBuffer> createDramMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes)
{
    const uint32_t rounded = ceilDiv(sizeBytes, TILE_BYTES) * TILE_BYTES;
    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig replicatedConfig{.size = rounded};
    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

void buildTwiddles(
    uint32_t nRow,
    uint32_t numStages,
    uint32_t direction,
    std::vector<uint32_t>& twR,
    std::vector<uint32_t>& twI)
{
    const uint32_t halfN = nRow / 2;
    const float sign = (direction == 1) ? 1.0f : -1.0f;
    twR.assign(numStages * halfN, 0u);
    twI.assign(numStages * halfN, 0u);

    for (uint32_t step = 0; step < numStages; ++step) {
        const uint32_t halfM = 1u << step;
        const uint32_t m     = halfM << 1u;
        for (uint32_t p = 0; p < halfN; ++p) {
            const uint32_t j = p % halfM;
            const uint32_t k = j * (nRow / m);
            const float angle = sign * 2.0f * PI * k / nRow;
            twR[step * halfN + p] = floatToU32(std::cos(angle));
            twI[step * halfN + p] = floatToU32(std::sin(angle));
        }
    }
}

void makeTestInput(uint32_t nRow, std::vector<float>& real, std::vector<float>& imag) {
    real.resize(nRow);
    imag.resize(nRow, 0.0f);
    for (uint32_t i = 0; i < nRow; ++i) {
        real[i] = std::sin(2.0f * PI * i / nRow)
                + 0.25f * std::cos(6.0f * PI * i / nRow);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int deviceId       = (argc > 1) ? std::stoi(argv[1]) : 0;
        const uint32_t nRow      = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 2048;
        const uint32_t batchSize = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint32_t numCores  = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow))    throw std::runtime_error("nRow must be power of 2");
        if (nRow < 2 * TILE_ELEMS)  throw std::runtime_error("nRow must be >= 2048 (2 x TILE_ELEMS)");
        if (nRow > 16384)           throw std::runtime_error("nRow > 16384 exceeds SRAM budget");
        if (numCores == 0 || numCores > 64) throw std::runtime_error("numCores must be in [1,64]");
        if (batchSize < numCores)   throw std::runtime_error("batchSize must be >= numCores");

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        auto& cq = meshDevice->mesh_command_queue();

        const uint32_t numStages = log2u32(nRow);
        const uint32_t halfN     = nRow / 2;

        // chunkSize = one full tile worth of butterfly pairs
        // numChunks = how many tiles needed to cover all halfN pairs
        const uint32_t chunkSize = TILE_ELEMS;
        const uint32_t numChunks = halfN / chunkSize;

        const uint32_t rowTiles      = ceilDiv(nRow, TILE_ELEMS);
        const uint32_t rowsThisLaunch = std::min(batchSize, numCores);

        const uint32_t sramDataBytes = nRow * sizeof(float);
        const uint32_t sramTwBytes   = numStages * halfN * sizeof(float);
        const uint32_t sramTotal     = 2 * sramDataBytes + 2 * sramTwBytes;

        std::cout << "[fft_paper_host]\n"
                  << "  nRow           = " << nRow << "\n"
                  << "  numStages      = " << numStages << "\n"
                  << "  halfN          = " << halfN << "\n"
                  << "  chunkSize      = " << chunkSize << "\n"
                  << "  numChunks      = " << numChunks << "\n"
                  << "  rowTiles       = " << rowTiles << "\n"
                  << "  rowsThisLaunch = " << rowsThisLaunch << "\n"
                  << "  SRAM per core  = " << sramTotal << " bytes\n";

        if (SRAM_DATA_BASE + sramTotal > 1300000)
            throw std::runtime_error("SRAM layout exceeds 1.3MB");

        // ── DRAM buffers ──────────────────────────────────────────────────────
        const uint32_t rowBufBytes = rowsThisLaunch * rowTiles * TILE_BYTES;
        auto inputRealBuf  = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto inputImagBuf  = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputRealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);

        const uint32_t elemsPerRow = rowTiles * TILE_ELEMS;
        std::vector<uint32_t> inputRealPacked(rowsThisLaunch * elemsPerRow, 0u);
        std::vector<uint32_t> inputImagPacked(rowsThisLaunch * elemsPerRow, 0u);

        for (uint32_t r = 0; r < rowsThisLaunch; ++r) {
            std::vector<float> rowR, rowI;
            makeTestInput(nRow, rowR, rowI);
            for (uint32_t i = 0; i < nRow; ++i) rowR[i] += 0.01f * r;
            for (uint32_t i = 0; i < nRow; ++i) {
                inputRealPacked[r * elemsPerRow + i] = floatToU32(rowR[i]);
                inputImagPacked[r * elemsPerRow + i] = floatToU32(rowI[i]);
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, inputRealBuf, inputRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, inputImagBuf, inputImagPacked, false);

        // ── Twiddle factors ───────────────────────────────────────────────────
        std::vector<uint32_t> twR, twI;
        buildTwiddles(nRow, numStages, direction, twR, twI);

        const uint32_t twBufBytes = numStages * halfN * sizeof(float);
        auto twRealDramBuf = createDramMeshBuffer(meshDevice, twBufBytes);
        auto twImagDramBuf = createDramMeshBuffer(meshDevice, twBufBytes);
        distributed::EnqueueWriteMeshBuffer(cq, twRealDramBuf, twR, false);
        distributed::EnqueueWriteMeshBuffer(cq, twImagDramBuf, twI, false);

        // ── Twiddle init kernel ───────────────────────────────────────────────
        {
            Program twInitProg = CreateProgram();
            CoreRange coreRange({0, 0}, {rowsThisLaunch - 1, 0});

            KernelHandle twInitKernel = CreateKernel(
                twInitProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/twiddle_init_f32.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc       = NOC::RISCV_0_default});

            const uint32_t sramTwRAddr = SRAM_DATA_BASE + 2 * sramDataBytes;
            const uint32_t sramTwIAddr = sramTwRAddr + twBufBytes;

            for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
                SetRuntimeArgs(twInitProg, twInitKernel, CoreCoord{c, 0},
                    { twRealDramBuf->address(), twImagDramBuf->address(),
                      sramTwRAddr, sramTwIAddr, twBufBytes });
            }

            distributed::MeshWorkload twWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            twWorkload.add_program(deviceRange, std::move(twInitProg));
            distributed::EnqueueMeshWorkload(cq, twWorkload, false);
            distributed::Finish(cq);
            std::cout << "Twiddle init finished.\n";
        }

        // ── FFT kernel ────────────────────────────────────────────────────────
        {
            Program fftProg = CreateProgram();
            CoreRange coreRange({0, 0}, {rowsThisLaunch - 1, 0});

            auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
                CircularBufferConfig cfg =
                    CircularBufferConfig(depthTiles * TILE_BYTES,
                                        {{cbId, tt::DataFormat::Float32}})
                        .set_page_size(cbId, TILE_BYTES);
                CreateCircularBuffer(fftProg, coreRange, cfg);
            };

            makeCb(CB_DATA0_R, 2);
            makeCb(CB_DATA0_I, 2);
            makeCb(CB_DATA1_R, 2);
            makeCb(CB_DATA1_I, 2);
            makeCb(CB_TW_R,    2);
            makeCb(CB_TW_I,    2);
            makeCb(CB_OUT0_R,  2);
            makeCb(CB_OUT0_I,  2);
            makeCb(CB_OUT1_R,  2);
            makeCb(CB_OUT1_I,  2);
            makeCb(CB_INT0,    2);
            makeCb(CB_INT1,    2);
            makeCb(CB_F0,      2);
            makeCb(CB_F1,      2);
            // NOTE: CB_STEP_SYNC removed — was causing deadlock

            KernelHandle readerKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader_fft_f32.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc       = NOC::RISCV_0_default});

            KernelHandle writerKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer_fft_f32.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc       = NOC::RISCV_1_default});

            KernelHandle computeKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/fft_compute_f32.cpp",
                coreRange,
                ComputeConfig{
                    .math_fidelity    = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true});

            for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
                CoreCoord cc{c, 0};
                const uint32_t rowByteOffset = c * rowTiles * TILE_BYTES;

                SetRuntimeArgs(fftProg, readerKernel, cc,
                    { inputRealBuf->address() + rowByteOffset,
                      inputImagBuf->address() + rowByteOffset,
                      nRow, numStages, numChunks, chunkSize,
                      SRAM_DATA_BASE });

                SetRuntimeArgs(fftProg, computeKernel, cc,
                    { numStages, numChunks });

                SetRuntimeArgs(fftProg, writerKernel, cc,
                    { outputRealBuf->address() + rowByteOffset,
                      outputImagBuf->address() + rowByteOffset,
                      nRow, numStages, numChunks, chunkSize,
                      SRAM_DATA_BASE });
            }

            distributed::MeshWorkload fftWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            fftWorkload.add_program(deviceRange, std::move(fftProg));
            distributed::EnqueueMeshWorkload(cq, fftWorkload, false);
            distributed::Finish(cq);
            std::cout << "FFT kernel execution finished.\n";
        }

        // ── Read and print results ────────────────────────────────────────────
        std::vector<uint32_t> outRawReal, outRawImag;
        distributed::EnqueueReadMeshBuffer(cq, outRawReal, outputRealBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, outRawImag, outputImagBuf, true);

        std::cout << "Results (first 8 elements of first 2 rows):\n";
        for (uint32_t r = 0; r < std::min(rowsThisLaunch, 2u); ++r) {
            std::cout << "row " << r << ":\n";
            for (uint32_t i = 0; i < std::min(nRow, 8u); ++i) {
                const uint32_t idx = r * elemsPerRow + i;
                std::cout << "  [" << i << "] = ("
                          << u32ToFloat(outRawReal[idx]) << ", "
                          << u32ToFloat(outRawImag[idx]) << ")\n";
            }
        }

        if (!meshDevice->close())
            throw std::runtime_error("meshDevice->close() failed");

        std::cout << "FFT host run finished.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FFT host failed: " << e.what() << "\n";
        return 1;
    }
}