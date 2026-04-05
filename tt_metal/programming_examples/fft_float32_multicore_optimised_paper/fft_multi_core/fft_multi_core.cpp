// SPDX-FileCopyrightText: © 2026
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

constexpr float PI = 3.14159265358979323846f;

constexpr uint32_t TILE_H     = 32;
constexpr uint32_t TILE_W     = 32;
constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;   // 1024
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

constexpr uint32_t CB_EVEN_R    = 0;
constexpr uint32_t CB_EVEN_I    = 1;
constexpr uint32_t CB_ODD_R     = 2;
constexpr uint32_t CB_ODD_I     = 3;
constexpr uint32_t CB_TW_R      = 4;
constexpr uint32_t CB_TW_I      = 5;
constexpr uint32_t CB_COMPACT_R = 10;
constexpr uint32_t CB_COMPACT_I = 11;

constexpr uint32_t CB_OUT0_R    = 16;
constexpr uint32_t CB_OUT0_I    = 17;
constexpr uint32_t CB_OUT1_R    = 18;
constexpr uint32_t CB_OUT1_I    = 19;
constexpr uint32_t CB_TMP0      = 20;
constexpr uint32_t CB_TMP1      = 21;
constexpr uint32_t CB_TW_ODD_R  = 22;
constexpr uint32_t CB_TW_ODD_I  = 23;

constexpr uint32_t WORKER_GRID_X = 8;
constexpr uint32_t WORKER_GRID_Y = 8;
constexpr uint32_t MAX_WORKERS   = WORKER_GRID_X * WORKER_GRID_Y;

inline uint32_t alignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
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
    uint32_t u;
    std::memcpy(&u, &v, sizeof(uint32_t));
    return u;
}

inline float u32ToFloat(uint32_t u) {
    float v;
    std::memcpy(&v, &u, sizeof(float));
    return v;
}

std::shared_ptr<distributed::MeshBuffer> createPagedDramBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t pageSizeBytes,
    uint32_t numPages)
{
    const uint32_t roundedPage = alignUp(pageSizeBytes, 4u);
    const uint32_t totalSize   = roundedPage * numPages;

    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = roundedPage,
        .buffer_type = BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig replicatedConfig{
        .size = totalSize
    };

    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

uint32_t bitReverseIndex(uint32_t value, uint32_t numBits) {
    uint32_t reversed = 0;
    for (uint32_t b = 0; b < numBits; ++b) {
        reversed = (reversed << 1u) | ((value >> b) & 1u);
    }
    return reversed;
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
    const uint32_t n = static_cast<uint32_t>(re.size());

    for (uint32_t i = 1, j = 0; i < n; ++i) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1u) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (uint32_t len = 2; len <= n; len <<= 1u) {
        const float ang = -2.0f * PI / static_cast<float>(len);
        const float wr  = std::cos(ang);
        const float wi  = std::sin(ang);

        for (uint32_t i = 0; i < n; i += len) {
            float cr = 1.0f;
            float ci = 0.0f;
            for (uint32_t j = 0; j < len / 2u; ++j) {
                const float ur = re[i + j];
                const float ui = im[i + j];
                const float vr = re[i + j + len / 2u] * cr - im[i + j + len / 2u] * ci;
                const float vi = re[i + j + len / 2u] * ci + im[i + j + len / 2u] * cr;

                re[i + j]            = ur + vr;
                im[i + j]            = ui + vi;
                re[i + j + len / 2u] = ur - vr;
                im[i + j + len / 2u] = ui - vi;

                const float nextCr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nextCr;
            }
        }
    }
}

void buildCompactTwiddles(
    uint32_t nRow,
    uint32_t direction,
    std::vector<uint32_t>& twR,
    std::vector<uint32_t>& twI)
{
    const uint32_t halfN = nRow / 2u;
    const float sign = (direction == 1u) ? 1.0f : -1.0f;

    twR.assign(TILE_ELEMS, 0u);
    twI.assign(TILE_ELEMS, 0u);

    for (uint32_t k = 0; k < halfN; ++k) {
        const float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(nRow);
        twR[k] = floatToU32(std::cos(angle));
        twI[k] = floatToU32(std::sin(angle));
    }
}

void prepareStage0Split(
    const std::vector<float>& inputReal,
    const std::vector<float>& inputImag,
    uint32_t nRow,
    std::vector<float>& evenReal,
    std::vector<float>& evenImag,
    std::vector<float>& oddReal,
    std::vector<float>& oddImag)
{
    const uint32_t numStages = log2u32(nRow);
    const uint32_t halfN     = nRow / 2u;

    std::vector<float> bitrevReal(nRow, 0.0f);
    std::vector<float> bitrevImag(nRow, 0.0f);

    for (uint32_t i = 0; i < nRow; ++i) {
        const uint32_t j = bitReverseIndex(i, numStages);
        bitrevReal[j] = inputReal[i];
        bitrevImag[j] = inputImag[i];
    }

    evenReal.assign(halfN, 0.0f);
    evenImag.assign(halfN, 0.0f);
    oddReal.assign(halfN, 0.0f);
    oddImag.assign(halfN, 0.0f);

    for (uint32_t p = 0; p < halfN; ++p) {
        evenReal[p] = bitrevReal[2u * p];
        evenImag[p] = bitrevImag[2u * p];
        oddReal[p]  = bitrevReal[2u * p + 1u];
        oddImag[p]  = bitrevImag[2u * p + 1u];
    }
}

std::vector<uint32_t> buildPrintBins(uint32_t nRow) {
    std::vector<uint32_t> raw = {0u, 1u, 2u, 3u, 4u, nRow / 2u, nRow - 4u, nRow - 3u, nRow - 2u, nRow - 1u};
    std::vector<uint32_t> filtered;
    for (uint32_t bin : raw) {
        if (bin < nRow && std::find(filtered.begin(), filtered.end(), bin) == filtered.end()) {
            filtered.push_back(bin);
        }
    }
    return filtered;
}

inline CoreCoord coordForIndex(uint32_t idx) {
    return CoreCoord{idx % WORKER_GRID_X, idx / WORKER_GRID_X};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int      deviceId   = (argc > 1) ? std::stoi(argv[1]) : 0;
        const uint32_t nRow       = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize  = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint32_t numCores   = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction  = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow)) {
            throw std::runtime_error("nRow must be a power of 2");
        }
        if (nRow < 2u) {
            throw std::runtime_error("nRow must be >= 2");
        }
        if (nRow > 2048u) {
            throw std::runtime_error("nRow > 2048 is not supported in this host");
        }
        if (numCores == 0u || numCores > MAX_WORKERS) {
            throw std::runtime_error("numCores must be in [1,64]");
        }
        if (batchSize < numCores) {
            throw std::runtime_error("batchSize must be >= numCores");
        }

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        auto& cq        = meshDevice->mesh_command_queue();

        const uint32_t numStages      = log2u32(nRow);
        const uint32_t halfN          = nRow / 2u;
        const uint32_t localHalf      = halfN;  // row decomposition: one full 1D FFT per core
        const uint32_t localTiles     = (localHalf + TILE_ELEMS - 1u) / TILE_ELEMS;
        const uint32_t rowsThisLaunch = std::min(batchSize, numCores);
        const uint32_t pagesPerBuffer = rowsThisLaunch * localTiles;
        const uint32_t elemsPerRowBuf = localTiles * TILE_ELEMS;
        const uint32_t bytesPerRowBuf = localTiles * TILE_BYTES;

        std::cout << "[fft_tiled_host]\n"
                  << "  nRow           = " << nRow << "\n"
                  << "  numStages      = " << numStages << "\n"
                  << "  halfN          = " << halfN << "\n"
                  << "  localHalf      = " << localHalf << "\n"
                  << "  localTiles     = " << localTiles << "\n"
                  << "  rowsThisLaunch = " << rowsThisLaunch << "\n"
                  << "  bytesPerRowBuf = " << bytesPerRowBuf << "\n";

        auto evenRealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
        auto evenImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
        auto oddRealBuf  = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
        auto oddImagBuf  = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);

        auto out0RealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
        auto out0ImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
        auto out1RealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
        auto out1ImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);

        auto compactTwRealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, 1);
        auto compactTwImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, 1);

        std::vector<uint32_t> evenRealPacked(pagesPerBuffer * TILE_ELEMS, 0u);
        std::vector<uint32_t> evenImagPacked(pagesPerBuffer * TILE_ELEMS, 0u);
        std::vector<uint32_t> oddRealPacked (pagesPerBuffer * TILE_ELEMS, 0u);
        std::vector<uint32_t> oddImagPacked (pagesPerBuffer * TILE_ELEMS, 0u);

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

            std::vector<float> evenR, evenI, oddR, oddI;
            prepareStage0Split(rowR, rowI, nRow, evenR, evenI, oddR, oddI);

            const uint32_t rowBase = r * elemsPerRowBuf;
            for (uint32_t i = 0; i < localHalf; ++i) {
                evenRealPacked[rowBase + i] = floatToU32(evenR[i]);
                evenImagPacked[rowBase + i] = floatToU32(evenI[i]);
                oddRealPacked [rowBase + i] = floatToU32(oddR[i]);
                oddImagPacked [rowBase + i] = floatToU32(oddI[i]);
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, evenRealBuf, evenRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, evenImagBuf, evenImagPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, oddRealBuf,  oddRealPacked,  false);
        distributed::EnqueueWriteMeshBuffer(cq, oddImagBuf,  oddImagPacked,  false);

        std::vector<uint32_t> compactTwR, compactTwI;
        buildCompactTwiddles(nRow, direction, compactTwR, compactTwI);

        distributed::EnqueueWriteMeshBuffer(cq, compactTwRealBuf, compactTwR, false);
        distributed::EnqueueWriteMeshBuffer(cq, compactTwImagBuf, compactTwI, false);

        Program fftProg = CreateProgram();

        // Full 8x8 worker rectangle. Unused cores get localTiles=0 and numStages=0.
        CoreRange coreRange({0, 0}, {WORKER_GRID_X - 1, WORKER_GRID_Y - 1});

        auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
            CircularBufferConfig cfg =
                CircularBufferConfig(depthTiles * TILE_BYTES, {{cbId, tt::DataFormat::Float32}})
                    .set_page_size(cbId, TILE_BYTES);
            CreateCircularBuffer(fftProg, coreRange, cfg);
        };

        makeCb(CB_EVEN_R, std::max(1u, localTiles));
        makeCb(CB_EVEN_I, std::max(1u, localTiles));
        makeCb(CB_ODD_R,  std::max(1u, localTiles));
        makeCb(CB_ODD_I,  std::max(1u, localTiles));

        makeCb(CB_TW_R, std::max(1u, numStages * std::max(1u, localTiles)));
        makeCb(CB_TW_I, std::max(1u, numStages * std::max(1u, localTiles)));

        makeCb(CB_COMPACT_R, 1);
        makeCb(CB_COMPACT_I, 1);

        makeCb(CB_OUT0_R, std::max(1u, localTiles));
        makeCb(CB_OUT0_I, std::max(1u, localTiles));
        makeCb(CB_OUT1_R, std::max(1u, localTiles));
        makeCb(CB_OUT1_I, std::max(1u, localTiles));

        makeCb(CB_TMP0, 1);
        makeCb(CB_TMP1, 1);
        makeCb(CB_TW_ODD_R, 1);
        makeCb(CB_TW_ODD_I, 1);

        KernelHandle readerKernel = CreateKernel(
            fftProg,
            OVERRIDE_KERNEL_PREFIX
            "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader.cpp",
            coreRange,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc       = NOC::RISCV_0_default
            });

        KernelHandle writerKernel = CreateKernel(
            fftProg,
            OVERRIDE_KERNEL_PREFIX
            "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer.cpp",
            coreRange,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc       = NOC::RISCV_1_default
            });

        KernelHandle computeKernel = CreateKernel(
            fftProg,
            OVERRIDE_KERNEL_PREFIX
            "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/compute.cpp",
            coreRange,
            ComputeConfig{
                .math_fidelity    = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true
            });

        for (uint32_t c = 0; c < MAX_WORKERS; ++c) {
            const bool active = (c < rowsThisLaunch);
            const CoreCoord cc = coordForIndex(c);

            const uint32_t rowByteOffset = active ? (c * bytesPerRowBuf) : 0u;
            const uint32_t argLocalTiles = active ? localTiles : 0u;
            const uint32_t argNumStages  = active ? numStages : 0u;

            SetRuntimeArgs(fftProg, readerKernel, cc, {
                evenRealBuf->address() + rowByteOffset,   // 0 even_r_addr
                evenImagBuf->address() + rowByteOffset,   // 1 even_i_addr
                oddRealBuf->address()  + rowByteOffset,   // 2 odd_r_addr
                oddImagBuf->address()  + rowByteOffset,   // 3 odd_i_addr
                compactTwRealBuf->address(),              // 4 compact_r_addr
                compactTwImagBuf->address(),              // 5 compact_i_addr
                argLocalTiles,                            // 6 local_tiles
                0u,                                       // 7 tile_offset
                argNumStages,                             // 8 num_stages
                halfN,                                    // 9 half_N
                localHalf,                                // 10 local_half
                0u                                        // 11 core_elem_base
            });

            SetRuntimeArgs(fftProg, computeKernel, cc, {
                argNumStages,                             // 0 num_stages
                argLocalTiles                             // 1 tiles_per_stage
            });

            SetRuntimeArgs(fftProg, writerKernel, cc, {
                out0RealBuf->address() + rowByteOffset,   // 0 out0_r_addr
                out0ImagBuf->address() + rowByteOffset,   // 1 out0_i_addr
                out1RealBuf->address() + rowByteOffset,   // 2 out1_r_addr
                out1ImagBuf->address() + rowByteOffset,   // 3 out1_i_addr
                argLocalTiles,                            // 4 local_tiles
                argNumStages,                             // 5 num_stages
                localHalf,                                // 6 local_half
                halfN,                                    // 7 half_N
                1u,                                       // 8 num_cores (row decomposition per active core)
                0u,                                       // 9 core_id
                0u,                                       // 10 log2_cores
                0u,                                       // 11 tile_offset
                0u                                        // 12 core_elem_base
            });
        }

        distributed::MeshWorkload fftWorkload;
        distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
        fftWorkload.add_program(deviceRange, std::move(fftProg));
        distributed::EnqueueMeshWorkload(cq, fftWorkload, false);
        distributed::Finish(cq);

        std::cout << "FFT tiled kernel execution finished.\n";

        std::vector<uint32_t> out0RealRaw, out0ImagRaw, out1RealRaw, out1ImagRaw;
        distributed::EnqueueReadMeshBuffer(cq, out0RealRaw, out0RealBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, out0ImagRaw, out0ImagBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, out1RealRaw, out1RealBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, out1ImagRaw, out1ImagBuf, true);

        std::vector<uint32_t> outRawReal(nRow, 0u);
        std::vector<uint32_t> outRawImag(nRow, 0u);

        const uint32_t row0Base = 0;
        for (uint32_t i = 0; i < halfN; ++i) {
            outRawReal[i]         = out0RealRaw[row0Base + i];
            outRawImag[i]         = out0ImagRaw[row0Base + i];
            outRawReal[i + halfN] = out1RealRaw[row0Base + i];
            outRawImag[i + halfN] = out1ImagRaw[row0Base + i];
        }

        cpuFft(refRe, refIm);

        std::cout << "\n=== Row 0: Key frequency bins ===\n";
        std::cout << "bin  |  WH_re        |  WH_im        |  WH_mag       ||  CPU_re       |  CPU_im       |  CPU_mag\n";
        std::cout << std::string(110, '-') << "\n";

        const std::vector<uint32_t> binsToPrint = buildPrintBins(nRow);
        for (uint32_t bin : binsToPrint) {
            const float wre  = u32ToFloat(outRawReal[bin]);
            const float wim  = u32ToFloat(outRawImag[bin]);
            const float wmag = std::sqrt(wre * wre + wim * wim);

            const float cre  = refRe[bin];
            const float cim  = refIm[bin];
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
            const float err = std::sqrt(
                (wre - refRe[i]) * (wre - refRe[i]) +
                (wim - refIm[i]) * (wim - refIm[i]));
            const float mag = std::sqrt(refRe[i] * refRe[i] + refIm[i] * refIm[i]);
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