// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

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
constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;   // 1024
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

constexpr uint32_t CB_DATA0_R = 0;
constexpr uint32_t CB_DATA0_I = 1;
constexpr uint32_t CB_DATA1_R = 2;
constexpr uint32_t CB_DATA1_I = 3;
constexpr uint32_t CB_TW_R    = 4;
constexpr uint32_t CB_TW_I    = 5;

constexpr uint32_t CB_OUT0_R  = 16;
constexpr uint32_t CB_OUT0_I  = 17;
constexpr uint32_t CB_OUT1_R  = 18;
constexpr uint32_t CB_OUT1_I  = 19;

constexpr uint32_t CB_INT0    = 20;
constexpr uint32_t CB_INT1    = 21;
constexpr uint32_t CB_F0      = 22;
constexpr uint32_t CB_F1      = 23;

constexpr uint32_t SRAM_DATA_BASE = 0x40000;
constexpr uint32_t SYNC_FLAG_ADDR = SRAM_DATA_BASE - sizeof(uint32_t);

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
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

inline float u32ToFloat(uint32_t u) {
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// Raw DRAM buffer: one page = entire buffer.
// This matches the current paper-Figure-3 path where reader/writer use
// noc_async_read / noc_async_write on plain row-major float arrays in DRAM.
std::shared_ptr<distributed::MeshBuffer> createRawDramBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes)
{
    const uint32_t rounded = (sizeBytes + 3u) & ~3u;

    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = rounded,
        .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig replicatedConfig{
        .size = rounded};

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
        const uint32_t m = halfM << 1u;

        for (uint32_t p = 0; p < halfN; ++p) {
            const uint32_t j = p % halfM;
            const uint32_t k = j * (nRow / m);
            const float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(nRow);

            twR[step * halfN + p] = floatToU32(std::cos(angle));
            twI[step * halfN + p] = floatToU32(std::sin(angle));
        }
    }
}

void makeTestInput(uint32_t nRow, std::vector<float>& real, std::vector<float>& imag) {
    real.resize(nRow);
    imag.assign(nRow, 0.0f);

    for (uint32_t i = 0; i < nRow; ++i) {
        real[i] = std::sin(2.0f * PI * static_cast<float>(i) / static_cast<float>(nRow))
                + 0.25f * std::cos(6.0f * PI * static_cast<float>(i) / static_cast<float>(nRow));
    }
}

void cpuFft(std::vector<float>& re, std::vector<float>& im) {
    const uint32_t n = re.size();

    for (uint32_t i = 1, j = 0; i < n; ++i) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (uint32_t len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * PI / static_cast<float>(len);
        const float wr = std::cos(ang);
        const float wi = std::sin(ang);

        for (uint32_t i = 0; i < n; i += len) {
            float cr = 1.0f;
            float ci = 0.0f;

            for (uint32_t j = 0; j < len / 2; ++j) {
                const float ur = re[i + j];
                const float ui = im[i + j];

                const float vr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                const float vi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;

                re[i + j] = ur + vr;
                im[i + j] = ui + vi;
                re[i + j + len / 2] = ur - vr;
                im[i + j + len / 2] = ui - vi;

                const float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int deviceId = (argc > 1) ? std::stoi(argv[1]) : 0;
        const uint32_t nRow = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint32_t numCores = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow)) {
            throw std::runtime_error("nRow must be power of 2");
        }
        if (nRow < 2) {
            throw std::runtime_error("nRow must be >= 2");
        }
        if (nRow > 2048) {
            throw std::runtime_error("nRow > 2048: halfN would exceed one tile");
        }
        if (numCores == 0 || numCores > 64) {
            throw std::runtime_error("numCores must be in [1,64]");
        }
        if (batchSize < numCores) {
            throw std::runtime_error("batchSize must be >= numCores");
        }

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        auto& cq = meshDevice->mesh_command_queue();

        const uint32_t numStages = log2u32(nRow);
        const uint32_t halfN = nRow / 2;
        const uint32_t numChunks = (halfN * 2 <= TILE_ELEMS) ? 1u : 2u;
        const uint32_t chunkSize = halfN / numChunks;
        const uint32_t rowsThisLaunch = std::min(batchSize, numCores);

        if (chunkSize == 0) {
            throw std::runtime_error("chunkSize == 0");
        }

        const uint32_t rowBytes = nRow * sizeof(float);
        const uint32_t rowBufBytes = rowsThisLaunch * rowBytes;

        const uint32_t twBufBytes = numStages * halfN * sizeof(float);

        const uint32_t sramDataBytes = rowBytes;
        const uint32_t sramTwBytes = twBufBytes;
        const uint32_t sramTotal = 2 * sramDataBytes + 2 * sramTwBytes + sizeof(uint32_t);

        std::cout << "[fft_paper_host]\n"
                  << "  nRow           = " << nRow << "\n"
                  << "  numStages      = " << numStages << "\n"
                  << "  halfN          = " << halfN << "\n"
                  << "  chunkSize      = " << chunkSize << "\n"
                  << "  numChunks      = " << numChunks << "\n"
                  << "  rowTiles       = 1\n"
                  << "  rowsThisLaunch = " << rowsThisLaunch << "\n"
                  << "  SRAM per core  = " << sramTotal << " bytes\n"
                  << "  sync_flag_addr = 0x" << std::hex << SYNC_FLAG_ADDR << std::dec << "\n";

        if (SRAM_DATA_BASE + sramTotal > 1300000) {
            throw std::runtime_error("SRAM layout exceeds 1.3MB");
        }

        auto inputRealBuf  = createRawDramBuffer(meshDevice, rowBufBytes);
        auto inputImagBuf  = createRawDramBuffer(meshDevice, rowBufBytes);
        auto outputRealBuf = createRawDramBuffer(meshDevice, rowBufBytes);
        auto outputImagBuf = createRawDramBuffer(meshDevice, rowBufBytes);

        std::vector<uint32_t> inputRealPacked(rowsThisLaunch * nRow, 0u);
        std::vector<uint32_t> inputImagPacked(rowsThisLaunch * nRow, 0u);

        std::vector<float> refRe;
        std::vector<float> refIm;
        makeTestInput(nRow, refRe, refIm);

        for (uint32_t r = 0; r < rowsThisLaunch; ++r) {
            std::vector<float> rowR;
            std::vector<float> rowI;
            makeTestInput(nRow, rowR, rowI);

            for (uint32_t i = 0; i < nRow; ++i) {
                rowR[i] += 0.01f * static_cast<float>(r);
            }

            for (uint32_t i = 0; i < nRow; ++i) {
                inputRealPacked[r * nRow + i] = floatToU32(rowR[i]);
                inputImagPacked[r * nRow + i] = floatToU32(rowI[i]);
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, inputRealBuf, inputRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, inputImagBuf, inputImagPacked, false);

        std::vector<uint32_t> twR;
        std::vector<uint32_t> twI;
        buildTwiddles(nRow, numStages, direction, twR, twI);

        auto twRealDramBuf = createRawDramBuffer(meshDevice, twBufBytes);
        auto twImagDramBuf = createRawDramBuffer(meshDevice, twBufBytes);

        distributed::EnqueueWriteMeshBuffer(cq, twRealDramBuf, twR, false);
        distributed::EnqueueWriteMeshBuffer(cq, twImagDramBuf, twI, false);

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
                    .noc = NOC::RISCV_0_default});

            const uint32_t sramTwRAddr = SRAM_DATA_BASE + 2 * sramDataBytes;
            const uint32_t sramTwIAddr = sramTwRAddr + twBufBytes;

            for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
                SetRuntimeArgs(
                    twInitProg,
                    twInitKernel,
                    CoreCoord{c, 0},
                    {
                        twRealDramBuf->address(),
                        twImagDramBuf->address(),
                        sramTwRAddr,
                        sramTwIAddr,
                        twBufBytes
                    });
            }

            distributed::MeshWorkload twWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            twWorkload.add_program(deviceRange, std::move(twInitProg));
            distributed::EnqueueMeshWorkload(cq, twWorkload, false);
            distributed::Finish(cq);

            std::cout << "Twiddle init finished.\n";
        }

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

            makeCb(CB_DATA0_R, 2); makeCb(CB_DATA0_I, 2);
            makeCb(CB_DATA1_R, 2); makeCb(CB_DATA1_I, 2);
            makeCb(CB_TW_R,    2); makeCb(CB_TW_I,    2);
            makeCb(CB_OUT0_R,  2); makeCb(CB_OUT0_I,  2);
            makeCb(CB_OUT1_R,  2); makeCb(CB_OUT1_I,  2);
            makeCb(CB_INT0,    2); makeCb(CB_INT1,    2);
            makeCb(CB_F0,      2); makeCb(CB_F1,      2);

            KernelHandle readerKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default});
                    
            #error "WRITER DEBUG ACTIVE"        
            KernelHandle writerKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default});

            KernelHandle computeKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/compute.cpp",
                coreRange,
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true});

            for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
                CoreCoord cc{c, 0};
                const uint32_t rowByteOffset = c * rowBytes;

                SetRuntimeArgs(
                    fftProg,
                    readerKernel,
                    cc,
                    {
                        inputRealBuf->address() + rowByteOffset,
                        inputImagBuf->address() + rowByteOffset,
                        nRow,
                        numStages,
                        numChunks,
                        chunkSize,
                        SRAM_DATA_BASE,
                        SYNC_FLAG_ADDR
                    });

                SetRuntimeArgs(
                    fftProg,
                    computeKernel,
                    cc,
                    {
                        numStages,
                        numChunks
                    });

                SetRuntimeArgs(
                    fftProg,
                    writerKernel,
                    cc,
                    {
                        outputRealBuf->address() + rowByteOffset,
                        outputImagBuf->address() + rowByteOffset,
                        nRow,
                        numStages,
                        numChunks,
                        chunkSize,
                        SRAM_DATA_BASE,
                        SYNC_FLAG_ADDR
                    });
            }

            distributed::MeshWorkload fftWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            fftWorkload.add_program(deviceRange, std::move(fftProg));
            distributed::EnqueueMeshWorkload(cq, fftWorkload, false);
            distributed::Finish(cq);

            std::cout << "FFT kernel execution finished ok ok ok .\n";
        }

        std::vector<uint32_t> outRawReal;
        std::vector<uint32_t> outRawImag;
        distributed::EnqueueReadMeshBuffer(cq, outRawReal, outputRealBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, outRawImag, outputImagBuf, true);

        cpuFft(refRe, refIm);

        std::cout << "\n=== Row 0: Key frequency bins ===\n";
        std::cout << "bin  |  WH_re        |  WH_im        |  WH_mag       ||  CPU_re       |  CPU_im       |  CPU_mag\n";
        std::cout << std::string(110, '-') << "\n";

        for (uint32_t bin : {0u, 1u, 2u, 3u, 4u, nRow / 2, nRow - 4, nRow - 3, nRow - 2, nRow - 1}) {
            const float wre = u32ToFloat(outRawReal[bin]);
            const float wim = u32ToFloat(outRawImag[bin]);
            const float wmag = std::sqrt(wre * wre + wim * wim);

            const float cre = refRe[bin];
            const float cim = refIm[bin];
            const float cmag = std::sqrt(cre * cre + cim * cim);

            std::printf(
                "%-5u| %13.3f | %13.3f | %13.3f || %13.3f | %13.3f | %13.3f\n",
                bin, wre, wim, wmag, cre, cim, cmag);
        }

        float maxErr = 0.0f;
        float maxMag = 0.0f;
        for (uint32_t i = 0; i < nRow; ++i) {
            const float wre = u32ToFloat(outRawReal[i]);
            const float wim = u32ToFloat(outRawImag[i]);
            const float cre = refRe[i];
            const float cim = refIm[i];

            const float err = std::sqrt((wre - cre) * (wre - cre) + (wim - cim) * (wim - cim));
            const float mag = std::sqrt(cre * cre + cim * cim);

            if (std::isfinite(err)) {
                maxErr = std::max(maxErr, err);
            }
            maxMag = std::max(maxMag, mag);
        }

        std::cout << "\nRow 0 max absolute error vs CPU: " << maxErr << "\n";
        std::cout << "Row 0 max bin magnitude (CPU):   " << maxMag << "\n";

        if (maxMag > 0.0f) {
            std::cout << "Row 0 relative error:            " << (maxErr / maxMag * 100.0f) << " %\n";
        }

        if (!meshDevice->close()) {
            throw std::runtime_error("meshDevice->close() failed");
        }

        std::cout << "\nFFT host run finished.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FFT host failed: " << e.what() << "\n";
        return 1;
    }
}