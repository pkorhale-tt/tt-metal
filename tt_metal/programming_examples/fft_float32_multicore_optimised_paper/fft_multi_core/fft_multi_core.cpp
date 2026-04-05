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

constexpr uint32_t TILE_H    = 32;
constexpr uint32_t TILE_W    = 32;
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

// Two uint32_t words are needed below SRAM_DATA_BASE:
//   [SYNC_FLAG_ADDR+0]  rdy_flag  – writer → reader
//   [SYNC_FLAG_ADDR+4]  ack_flag  – reader → writer
constexpr uint32_t SRAM_DATA_BASE = 0x40000;
constexpr uint32_t SYNC_FLAG_ADDR = SRAM_DATA_BASE - 2 * sizeof(uint32_t);  // 0x3FFF8

inline bool isPowerOfTwo(uint32_t x) { return x > 0 && ((x & (x - 1)) == 0); }

inline uint32_t log2u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) ++r;
    return r;
}

inline uint32_t floatToU32(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }
inline float    u32ToFloat(uint32_t u) { float v; std::memcpy(&v, &u, 4); return v; }

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
    const uint32_t halfN  = nRow / 2;
    const float    sign   = (direction == 1) ? 1.0f : -1.0f;

    twR.assign(numStages * halfN, 0u);
    twI.assign(numStages * halfN, 0u);

    for (uint32_t step = 0; step < numStages; ++step) {
        const uint32_t halfM = 1u << step;
        const uint32_t m     = halfM << 1u;

        for (uint32_t p = 0; p < halfN; ++p) {
            const uint32_t j     = p % halfM;
            const uint32_t k     = j * (nRow / m);
            const float    angle = sign * 2.0f * PI * static_cast<float>(k)
                                                     / static_cast<float>(nRow);
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
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (uint32_t len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * PI / static_cast<float>(len);
        const float wr  = std::cos(ang);
        const float wi  = std::sin(ang);
        for (uint32_t i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (uint32_t j = 0; j < len / 2; ++j) {
                const float ur = re[i+j],          ui = im[i+j];
                const float vr = re[i+j+len/2]*cr - im[i+j+len/2]*ci;
                const float vi = re[i+j+len/2]*ci + im[i+j+len/2]*cr;
                re[i+j]        = ur + vr;  im[i+j]        = ui + vi;
                re[i+j+len/2]  = ur - vr;  im[i+j+len/2]  = ui - vi;
                const float ncr = cr*wr - ci*wi;  ci = cr*wi + ci*wr;  cr = ncr;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int      deviceId   = (argc > 1) ? std::stoi(argv[1])                      : 0;
        const uint32_t nRow       = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize  = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint32_t numCores   = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction  = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow))    throw std::runtime_error("nRow must be power of 2");
        if (nRow < 2)               throw std::runtime_error("nRow must be >= 2");
        if (nRow > 2048)            throw std::runtime_error("nRow > 2048: halfN would exceed one tile");
        if (numCores == 0 || numCores > 64) throw std::runtime_error("numCores must be in [1,64]");
        if (batchSize < numCores)   throw std::runtime_error("batchSize must be >= numCores");

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        auto& cq        = meshDevice->mesh_command_queue();

        const uint32_t numStages      = log2u32(nRow);
        const uint32_t halfN          = nRow / 2;
        // chunkSize = halfN when the entire half-row fits in one tile (nRow <= 2048)
        const uint32_t numChunks      = (halfN * 2 <= TILE_ELEMS) ? 1u : 2u;
        const uint32_t chunkSize      = halfN / numChunks;
        const uint32_t rowsThisLaunch = std::min(batchSize, numCores);

        if (chunkSize == 0) throw std::runtime_error("chunkSize == 0");

        const uint32_t rowBytes    = nRow * sizeof(float);
        const uint32_t rowBufBytes = rowsThisLaunch * rowBytes;
        const uint32_t twBufBytes  = numStages * halfN * sizeof(float);

        // SRAM layout per core (growing upward from SRAM_DATA_BASE):
        //   [SRAM_DATA_BASE + 0*rowBytes]           real data row
        //   [SRAM_DATA_BASE + 1*rowBytes]           imag data row
        //   [SRAM_DATA_BASE + 2*rowBytes]           twiddle real table
        //   [SRAM_DATA_BASE + 2*rowBytes+twBufBytes] twiddle imag table
        // Below SRAM_DATA_BASE (growing downward):
        //   [SYNC_FLAG_ADDR + 0]  rdy_flag (4 bytes)
        //   [SYNC_FLAG_ADDR + 4]  ack_flag (4 bytes)
        const uint32_t sramDataBytes  = rowBytes;
        const uint32_t sramTwBytes    = twBufBytes;
        const uint32_t sramTotal      = 2 * sramDataBytes + 2 * sramTwBytes + 2 * sizeof(uint32_t);

        std::cout << "[fft_paper_host]\n"
                  << "  nRow           = " << nRow          << "\n"
                  << "  numStages      = " << numStages     << "\n"
                  << "  halfN          = " << halfN         << "\n"
                  << "  chunkSize      = " << chunkSize     << "\n"
                  << "  numChunks      = " << numChunks     << "\n"
                  << "  rowTiles       = 1\n"
                  << "  rowsThisLaunch = " << rowsThisLaunch << "\n"
                  << "  SRAM per core  = " << sramTotal     << " bytes\n"
                  << "  sync_flag_addr = 0x" << std::hex << SYNC_FLAG_ADDR << std::dec << "\n";

        if (SRAM_DATA_BASE + sramTotal > 1300000)
            throw std::runtime_error("SRAM layout exceeds 1.3MB");

        // ── DRAM buffers ────────────────────────────────────────────────────
        auto inputRealBuf  = createRawDramBuffer(meshDevice, rowBufBytes);
        auto inputImagBuf  = createRawDramBuffer(meshDevice, rowBufBytes);
        auto outputRealBuf = createRawDramBuffer(meshDevice, rowBufBytes);
        auto outputImagBuf = createRawDramBuffer(meshDevice, rowBufBytes);

        // ── Pack input data ─────────────────────────────────────────────────
        std::vector<uint32_t> inputRealPacked(rowsThisLaunch * nRow, 0u);
        std::vector<uint32_t> inputImagPacked(rowsThisLaunch * nRow, 0u);

        std::vector<float> refRe, refIm;
        makeTestInput(nRow, refRe, refIm);

        for (uint32_t r = 0; r < rowsThisLaunch; ++r) {
            std::vector<float> rowR, rowI;
            makeTestInput(nRow, rowR, rowI);
            for (uint32_t i = 0; i < nRow; ++i) rowR[i] += 0.01f * static_cast<float>(r);
            for (uint32_t i = 0; i < nRow; ++i) {
                inputRealPacked[r * nRow + i] = floatToU32(rowR[i]);
                inputImagPacked[r * nRow + i] = floatToU32(rowI[i]);
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, inputRealBuf, inputRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, inputImagBuf, inputImagPacked, false);

        // ── Build and upload twiddle tables ─────────────────────────────────
        std::vector<uint32_t> twR, twI;
        buildTwiddles(nRow, numStages, direction, twR, twI);

        auto twRealDramBuf = createRawDramBuffer(meshDevice, twBufBytes);
        auto twImagDramBuf = createRawDramBuffer(meshDevice, twBufBytes);
        distributed::EnqueueWriteMeshBuffer(cq, twRealDramBuf, twR, false);
        distributed::EnqueueWriteMeshBuffer(cq, twImagDramBuf, twI, false);

        // ── Twiddle init program (copies twiddle tables from DRAM → SRAM) ──
        {
            Program      twInitProg = CreateProgram();
            CoreRange    coreRange({0, 0}, {rowsThisLaunch - 1, 0});

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
                    { twRealDramBuf->address(),
                      twImagDramBuf->address(),
                      sramTwRAddr,
                      sramTwIAddr,
                      twBufBytes });
            }

            distributed::MeshWorkload      twWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            twWorkload.add_program(deviceRange, std::move(twInitProg));
            distributed::EnqueueMeshWorkload(cq, twWorkload, false);
            distributed::Finish(cq);
            std::cout << "Twiddle init finished.\n";
        }

        // ── FFT program ──────────────────────────────────────────────────────
        {
            Program   fftProg  = CreateProgram();
            CoreRange coreRange({0, 0}, {rowsThisLaunch - 1, 0});

            // Helper: create a CB with the given depth (in tiles)
            auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
                CircularBufferConfig cfg =
                    CircularBufferConfig(depthTiles * TILE_BYTES,
                                         {{cbId, tt::DataFormat::Float32}})
                        .set_page_size(cbId, TILE_BYTES);
                CreateCircularBuffer(fftProg, coreRange, cfg);
            };

            // Input CBs: depth 2 for double-buffering
            makeCb(CB_DATA0_R, 2);  makeCb(CB_DATA0_I, 2);
            makeCb(CB_DATA1_R, 2);  makeCb(CB_DATA1_I, 2);
            makeCb(CB_TW_R,    2);  makeCb(CB_TW_I,    2);

            // Output CBs: depth 2
            makeCb(CB_OUT0_R,  2);  makeCb(CB_OUT0_I,  2);
            makeCb(CB_OUT1_R,  2);  makeCb(CB_OUT1_I,  2);

            // Intermediate CBs: depth 2
            makeCb(CB_INT0,    2);  makeCb(CB_INT1,    2);
            makeCb(CB_F0,      2);  makeCb(CB_F1,      2);

            // ── Kernels ──────────────────────────────────────────────────────
            KernelHandle readerKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc       = NOC::RISCV_0_default});

            KernelHandle writerKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer.cpp",
                coreRange,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc       = NOC::RISCV_1_default});

            KernelHandle computeKernel = CreateKernel(
                fftProg,
                OVERRIDE_KERNEL_PREFIX
                "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/compute.cpp",
                coreRange,
                ComputeConfig{
                    .math_fidelity  = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = true});

            // ── Runtime args ─────────────────────────────────────────────────
            for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
                CoreCoord    cc{c, 0};
                const uint32_t rowByteOffset = c * rowBytes;

                // Reader: 8 args
                SetRuntimeArgs(fftProg, readerKernel, cc,
                    { inputRealBuf->address() + rowByteOffset,  // 0: dram_input_r_addr
                      inputImagBuf->address() + rowByteOffset,  // 1: dram_input_i_addr
                      nRow,                                      // 2: n
                      numStages,                                 // 3: num_steps
                      numChunks,                                 // 4: num_chunks
                      chunkSize,                                 // 5: chunk_size
                      SRAM_DATA_BASE,                            // 6: sram_buf_r_addr
                      SYNC_FLAG_ADDR });                         // 7: sync_flag_addr (rdy@+0, ack@+4)

                // Compute: 2 args
                SetRuntimeArgs(fftProg, computeKernel, cc,
                    { numStages,   // 0: num_steps
                      numChunks }); // 1: num_chunks

                // Writer: 8 args  (same layout as reader)
                SetRuntimeArgs(fftProg, writerKernel, cc,
                    { outputRealBuf->address() + rowByteOffset, // 0: dram_output_r_addr
                      outputImagBuf->address() + rowByteOffset, // 1: dram_output_i_addr
                      nRow,                                      // 2: n
                      numStages,                                 // 3: num_steps
                      numChunks,                                 // 4: num_chunks
                      chunkSize,                                 // 5: chunk_size
                      SRAM_DATA_BASE,                            // 6: sram_buf_r_addr
                      SYNC_FLAG_ADDR });                         // 7: sync_flag_addr (rdy@+0, ack@+4)
            }

            distributed::MeshWorkload      fftWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            fftWorkload.add_program(deviceRange, std::move(fftProg));
            distributed::EnqueueMeshWorkload(cq, fftWorkload, false);
            distributed::Finish(cq);
            std::cout << "FFT kernel execution finished....\n";
        }

        // ── Read back results ────────────────────────────────────────────────
        std::vector<uint32_t> outRawReal, outRawImag;
        distributed::EnqueueReadMeshBuffer(cq, outRawReal, outputRealBuf, true);
        distributed::EnqueueReadMeshBuffer(cq, outRawImag, outputImagBuf, true);

        // ── CPU reference ────────────────────────────────────────────────────
        cpuFft(refRe, refIm);

        std::cout << "\n=== Row 0: Key frequency bins ===\n";
        std::cout << "bin  |  WH_re        |  WH_im        |  WH_mag       ||  CPU_re       |  CPU_im       |  CPU_mag\n";
        std::cout << std::string(110, '-') << "\n";

        for (uint32_t bin : {0u, 1u, 2u, 3u, 4u, nRow/2, nRow-4, nRow-3, nRow-2, nRow-1}) {
            const float wre = u32ToFloat(outRawReal[bin]);
            const float wim = u32ToFloat(outRawImag[bin]);
            const float wmag = std::sqrt(wre*wre + wim*wim);
            const float cre  = refRe[bin];
            const float cim  = refIm[bin];
            const float cmag = std::sqrt(cre*cre + cim*cim);
            std::printf("%-5u| %13.3f | %13.3f | %13.3f || %13.3f | %13.3f | %13.3f\n",
                        bin, wre, wim, wmag, cre, cim, cmag);
        }

        float maxErr = 0.0f, maxMag = 0.0f;
        for (uint32_t i = 0; i < nRow; ++i) {
            const float wre = u32ToFloat(outRawReal[i]);
            const float wim = u32ToFloat(outRawImag[i]);
            const float err = std::sqrt((wre-refRe[i])*(wre-refRe[i]) + (wim-refIm[i])*(wim-refIm[i]));
            const float mag = std::sqrt(refRe[i]*refRe[i] + refIm[i]*refIm[i]);
            if (std::isfinite(err)) maxErr = std::max(maxErr, err);
            maxMag = std::max(maxMag, mag);
        }

        std::cout << "\nRow 0 max absolute error vs CPU: " << maxErr << "\n";
        std::cout << "Row 0 max bin magnitude (CPU):   " << maxMag  << "\n";
        if (maxMag > 0.0f)
            std::cout << "Row 0 relative error:            "
                      << (maxErr / maxMag * 100.0f) << " %\n";

        if (!meshDevice->close()) throw std::runtime_error("meshDevice->close() failed");

        std::cout << "\nFFT host run finished.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FFT host failed: " << e.what() << "\n";
        return 1;
    }
}



// //====
// // SPDX-FileCopyrightText: © 2026
// // SPDX-License-Identifier: Apache-2.0

// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <cstring>
// #include <iostream>
// #include <memory>
// #include <stdexcept>
// #include <vector>

// #include "tt_metal/api/tt-metalium/host_api.hpp"
// #include <tt-metalium/device.hpp>
// #include <tt-metalium/distributed.hpp>

// using namespace tt;
// using namespace tt::tt_metal;

// #ifndef OVERRIDE_KERNEL_PREFIX
// #define OVERRIDE_KERNEL_PREFIX ""
// #endif

// namespace {

// constexpr float PI = 3.14159265358979323846f;

// constexpr uint32_t TILE_H     = 32;
// constexpr uint32_t TILE_W     = 32;
// constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;          // 1024
// constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

// constexpr uint32_t CB_EVEN_R    = 0;
// constexpr uint32_t CB_EVEN_I    = 1;
// constexpr uint32_t CB_ODD_R     = 2;
// constexpr uint32_t CB_ODD_I     = 3;
// constexpr uint32_t CB_TW_R      = 4;
// constexpr uint32_t CB_TW_I      = 5;
// constexpr uint32_t CB_COMPACT_R = 10;
// constexpr uint32_t CB_COMPACT_I = 11;

// constexpr uint32_t CB_OUT0_R    = 16;
// constexpr uint32_t CB_OUT0_I    = 17;
// constexpr uint32_t CB_OUT1_R    = 18;
// constexpr uint32_t CB_OUT1_I    = 19;
// constexpr uint32_t CB_TMP0      = 20;
// constexpr uint32_t CB_TMP1      = 21;
// constexpr uint32_t CB_TW_ODD_R  = 22;
// constexpr uint32_t CB_TW_ODD_I  = 23;

// inline uint32_t alignUp(uint32_t value, uint32_t alignment) {
//     return (value + alignment - 1u) / alignment * alignment;
// }

// inline bool isPowerOfTwo(uint32_t x) {
//     return x > 0 && ((x & (x - 1)) == 0);
// }

// inline uint32_t log2u32(uint32_t x) {
//     uint32_t r = 0;
//     while ((1u << r) < x) {
//         ++r;
//     }
//     return r;
// }

// inline uint32_t floatToU32(float v) {
//     uint32_t u;
//     std::memcpy(&u, &v, sizeof(uint32_t));
//     return u;
// }

// inline float u32ToFloat(uint32_t u) {
//     float v;
//     std::memcpy(&v, &u, sizeof(float));
//     return v;
// }

// std::shared_ptr<distributed::MeshBuffer> createPagedDramBuffer(
//     const std::shared_ptr<distributed::MeshDevice>& meshDevice,
//     uint32_t pageSizeBytes,
//     uint32_t numPages)
// {
//     const uint32_t roundedPage = alignUp(pageSizeBytes, 4u);
//     const uint32_t totalSize   = roundedPage * numPages;

//     distributed::DeviceLocalBufferConfig localConfig{
//         .page_size   = roundedPage,
//         .buffer_type = BufferType::DRAM
//     };

//     distributed::ReplicatedBufferConfig replicatedConfig{
//         .size = totalSize
//     };

//     return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
// }

// uint32_t bitReverseIndex(uint32_t value, uint32_t numBits) {
//     uint32_t reversed = 0;
//     for (uint32_t b = 0; b < numBits; ++b) {
//         reversed = (reversed << 1u) | ((value >> b) & 1u);
//     }
//     return reversed;
// }

// void makeTestInput(uint32_t nRow, std::vector<float>& real, std::vector<float>& imag) {
//     real.resize(nRow);
//     imag.assign(nRow, 0.0f);

//     for (uint32_t i = 0; i < nRow; ++i) {
//         real[i] = std::sin(2.0f * PI * static_cast<float>(i) / static_cast<float>(nRow))
//                 + 0.25f * std::cos(6.0f * PI * static_cast<float>(i) / static_cast<float>(nRow));
//     }
// }

// void cpuFft(std::vector<float>& re, std::vector<float>& im) {
//     const uint32_t n = static_cast<uint32_t>(re.size());

//     for (uint32_t i = 1, j = 0; i < n; ++i) {
//         uint32_t bit = n >> 1;
//         for (; j & bit; bit >>= 1u) {
//             j ^= bit;
//         }
//         j ^= bit;
//         if (i < j) {
//             std::swap(re[i], re[j]);
//             std::swap(im[i], im[j]);
//         }
//     }

//     for (uint32_t len = 2; len <= n; len <<= 1u) {
//         const float ang = -2.0f * PI / static_cast<float>(len);
//         const float wr  = std::cos(ang);
//         const float wi  = std::sin(ang);

//         for (uint32_t i = 0; i < n; i += len) {
//             float cr = 1.0f;
//             float ci = 0.0f;
//             for (uint32_t j = 0; j < len / 2u; ++j) {
//                 const float ur = re[i + j];
//                 const float ui = im[i + j];
//                 const float vr = re[i + j + len / 2u] * cr - im[i + j + len / 2u] * ci;
//                 const float vi = re[i + j + len / 2u] * ci + im[i + j + len / 2u] * cr;

//                 re[i + j]             = ur + vr;
//                 im[i + j]             = ui + vi;
//                 re[i + j + len / 2u]  = ur - vr;
//                 im[i + j + len / 2u]  = ui - vi;

//                 const float nextCr = cr * wr - ci * wi;
//                 ci = cr * wi + ci * wr;
//                 cr = nextCr;
//             }
//         }
//     }
// }

// void buildCompactTwiddles(
//     uint32_t nRow,
//     uint32_t direction,
//     std::vector<uint32_t>& twR,
//     std::vector<uint32_t>& twI)
// {
//     const uint32_t halfN = nRow / 2u;
//     const float sign = (direction == 1u) ? 1.0f : -1.0f;

//     twR.assign(TILE_ELEMS, 0u);
//     twI.assign(TILE_ELEMS, 0u);

//     for (uint32_t k = 0; k < halfN; ++k) {
//         const float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(nRow);
//         twR[k] = floatToU32(std::cos(angle));
//         twI[k] = floatToU32(std::sin(angle));
//     }
// }

// void prepareStage0Split(
//     const std::vector<float>& inputReal,
//     const std::vector<float>& inputImag,
//     uint32_t nRow,
//     std::vector<float>& evenReal,
//     std::vector<float>& evenImag,
//     std::vector<float>& oddReal,
//     std::vector<float>& oddImag)
// {
//     const uint32_t numStages = log2u32(nRow);
//     const uint32_t halfN     = nRow / 2u;

//     std::vector<float> bitrevReal(nRow, 0.0f);
//     std::vector<float> bitrevImag(nRow, 0.0f);

//     for (uint32_t i = 0; i < nRow; ++i) {
//         const uint32_t j = bitReverseIndex(i, numStages);
//         bitrevReal[j] = inputReal[i];
//         bitrevImag[j] = inputImag[i];
//     }

//     evenReal.assign(halfN, 0.0f);
//     evenImag.assign(halfN, 0.0f);
//     oddReal.assign(halfN, 0.0f);
//     oddImag.assign(halfN, 0.0f);

//     for (uint32_t p = 0; p < halfN; ++p) {
//         evenReal[p] = bitrevReal[2u * p];
//         evenImag[p] = bitrevImag[2u * p];
//         oddReal[p]  = bitrevReal[2u * p + 1u];
//         oddImag[p]  = bitrevImag[2u * p + 1u];
//     }
// }

// std::vector<uint32_t> buildPrintBins(uint32_t nRow) {
//     std::vector<uint32_t> raw = {0u, 1u, 2u, 3u, 4u, nRow / 2u, nRow - 4u, nRow - 3u, nRow - 2u, nRow - 1u};
//     std::vector<uint32_t> filtered;
//     for (uint32_t bin : raw) {
//         if (bin < nRow && std::find(filtered.begin(), filtered.end(), bin) == filtered.end()) {
//             filtered.push_back(bin);
//         }
//     }
//     return filtered;
// }

// } // namespace

// int main(int argc, char** argv) {
//     try {
//         const int deviceId      = (argc > 1) ? std::stoi(argv[1]) : 0;
//         const uint32_t nRow     = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
//         const uint32_t batchSize= (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
//         const uint32_t numCores = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
//         const uint32_t direction= (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

//         if (!isPowerOfTwo(nRow)) {
//             throw std::runtime_error("nRow must be a power of 2");
//         }
//         if (nRow < 2u) {
//             throw std::runtime_error("nRow must be >= 2");
//         }
//         if (nRow > 2048u) {
//             throw std::runtime_error("nRow > 2048 is not supported in this host");
//         }
//         if (numCores == 0u || numCores > 8u) {
//             throw std::runtime_error("This host maps one row per core on x-axis only; use numCores in [1,8]");
//         }
//         if (batchSize < numCores) {
//             throw std::runtime_error("batchSize must be >= numCores");
//         }

//         auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
//         auto& cq        = meshDevice->mesh_command_queue();

//         const uint32_t numStages      = log2u32(nRow);
//         const uint32_t halfN          = nRow / 2u;
//         const uint32_t localHalf      = halfN; // row decomposition: each core runs one full row FFT
//         const uint32_t localTiles     = (localHalf + TILE_ELEMS - 1u) / TILE_ELEMS;
//         const uint32_t rowsThisLaunch = std::min(batchSize, numCores);
//         const uint32_t pagesPerBuffer = rowsThisLaunch * localTiles;
//         const uint32_t elemsPerRowBuf = localTiles * TILE_ELEMS;
//         const uint32_t bytesPerRowBuf = localTiles * TILE_BYTES;

//         std::cout << "[fft_tiled_host]\n"
//                   << "  nRow           = " << nRow << "\n"
//                   << "  numStages      = " << numStages << "\n"
//                   << "  halfN          = " << halfN << "\n"
//                   << "  localHalf      = " << localHalf << "\n"
//                   << "  localTiles     = " << localTiles << "\n"
//                   << "  rowsThisLaunch = " << rowsThisLaunch << "\n"
//                   << "  bytesPerRowBuf = " << bytesPerRowBuf << "\n";

//         // ── DRAM buffers: stage-0 tiled even/odd inputs ────────────────────
//         auto evenRealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
//         auto evenImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
//         auto oddRealBuf  = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
//         auto oddImagBuf  = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);

//         // ── DRAM buffers: final out0/out1 tiles ─────────────────────────────
//         auto out0RealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
//         auto out0ImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
//         auto out1RealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);
//         auto out1ImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, pagesPerBuffer);

//         // ── DRAM buffers: compact twiddle table (single tiled page) ─────────
//         auto compactTwRealBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, 1);
//         auto compactTwImagBuf = createPagedDramBuffer(meshDevice, TILE_BYTES, 1);

//         // ── Build padded tiled input buffers ────────────────────────────────
//         std::vector<uint32_t> evenRealPacked(pagesPerBuffer * TILE_ELEMS, 0u);
//         std::vector<uint32_t> evenImagPacked(pagesPerBuffer * TILE_ELEMS, 0u);
//         std::vector<uint32_t> oddRealPacked (pagesPerBuffer * TILE_ELEMS, 0u);
//         std::vector<uint32_t> oddImagPacked (pagesPerBuffer * TILE_ELEMS, 0u);

//         std::vector<float> refRe;
//         std::vector<float> refIm;
//         makeTestInput(nRow, refRe, refIm);

//         for (uint32_t r = 0; r < rowsThisLaunch; ++r) {
//             std::vector<float> rowR;
//             std::vector<float> rowI;
//             makeTestInput(nRow, rowR, rowI);

//             for (uint32_t i = 0; i < nRow; ++i) {
//                 rowR[i] += 0.01f * static_cast<float>(r);
//             }

//             std::vector<float> evenR;
//             std::vector<float> evenI;
//             std::vector<float> oddR;
//             std::vector<float> oddI;
//             prepareStage0Split(rowR, rowI, nRow, evenR, evenI, oddR, oddI);

//             const uint32_t rowBase = r * elemsPerRowBuf;
//             for (uint32_t i = 0; i < localHalf; ++i) {
//                 evenRealPacked[rowBase + i] = floatToU32(evenR[i]);
//                 evenImagPacked[rowBase + i] = floatToU32(evenI[i]);
//                 oddRealPacked [rowBase + i] = floatToU32(oddR[i]);
//                 oddImagPacked [rowBase + i] = floatToU32(oddI[i]);
//             }
//         }

//         distributed::EnqueueWriteMeshBuffer(cq, evenRealBuf, evenRealPacked, false);
//         distributed::EnqueueWriteMeshBuffer(cq, evenImagBuf, evenImagPacked, false);
//         distributed::EnqueueWriteMeshBuffer(cq, oddRealBuf,  oddRealPacked,  false);
//         distributed::EnqueueWriteMeshBuffer(cq, oddImagBuf,  oddImagPacked,  false);

//         // ── Build compact twiddles ──────────────────────────────────────────
//         std::vector<uint32_t> compactTwR;
//         std::vector<uint32_t> compactTwI;
//         buildCompactTwiddles(nRow, direction, compactTwR, compactTwI);

//         distributed::EnqueueWriteMeshBuffer(cq, compactTwRealBuf, compactTwR, false);
//         distributed::EnqueueWriteMeshBuffer(cq, compactTwImagBuf, compactTwI, false);

//         // ── FFT program ─────────────────────────────────────────────────────
//         Program fftProg = CreateProgram();
//         CoreRange coreRange({0, 0}, {rowsThisLaunch - 1, 0});

//         auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
//             CircularBufferConfig cfg =
//                 CircularBufferConfig(depthTiles * TILE_BYTES, {{cbId, tt::DataFormat::Float32}})
//                     .set_page_size(cbId, TILE_BYTES);
//             CreateCircularBuffer(fftProg, coreRange, cfg);
//         };

//         // Stage input/output CBs
//         makeCb(CB_EVEN_R, localTiles);
//         makeCb(CB_EVEN_I, localTiles);
//         makeCb(CB_ODD_R,  localTiles);
//         makeCb(CB_ODD_I,  localTiles);

//         // Reader pushes all stage twiddle pages up front
//         makeCb(CB_TW_R, std::max(1u, numStages * localTiles));
//         makeCb(CB_TW_I, std::max(1u, numStages * localTiles));

//         // Compact twiddles
//         makeCb(CB_COMPACT_R, 1);
//         makeCb(CB_COMPACT_I, 1);

//         // Outputs
//         makeCb(CB_OUT0_R, localTiles);
//         makeCb(CB_OUT0_I, localTiles);
//         makeCb(CB_OUT1_R, localTiles);
//         makeCb(CB_OUT1_I, localTiles);

//         // Scratch / intermediates
//         makeCb(CB_TMP0, 1);
//         makeCb(CB_TMP1, 1);
//         makeCb(CB_TW_ODD_R, 1);
//         makeCb(CB_TW_ODD_I, 1);

//         KernelHandle readerKernel = CreateKernel(
//             fftProg,
//             OVERRIDE_KERNEL_PREFIX
//             "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/reader.cpp",
//             coreRange,
//             DataMovementConfig{
//                 .processor = DataMovementProcessor::RISCV_0,
//                 .noc       = NOC::RISCV_0_default
//             });

//         KernelHandle writerKernel = CreateKernel(
//             fftProg,
//             OVERRIDE_KERNEL_PREFIX
//             "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/dataflow/writer.cpp",
//             coreRange,
//             DataMovementConfig{
//                 .processor = DataMovementProcessor::RISCV_1,
//                 .noc       = NOC::RISCV_1_default
//             });

//         KernelHandle computeKernel = CreateKernel(
//             fftProg,
//             OVERRIDE_KERNEL_PREFIX
//             "fft_float32_multicore_optimised_paper/fft_multi_core/kernels/compute/compute.cpp",
//             coreRange,
//             ComputeConfig{
//                 .math_fidelity    = MathFidelity::HiFi4,
//                 .fp32_dest_acc_en = true
//             });

//         for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
//             CoreCoord cc{c, 0};
//             const uint32_t rowByteOffset = c * bytesPerRowBuf;

//             SetRuntimeArgs(fftProg, readerKernel, cc, {
//                 evenRealBuf->address() + rowByteOffset,   // 0 even_r_addr
//                 evenImagBuf->address() + rowByteOffset,   // 1 even_i_addr
//                 oddRealBuf->address()  + rowByteOffset,   // 2 odd_r_addr
//                 oddImagBuf->address()  + rowByteOffset,   // 3 odd_i_addr
//                 compactTwRealBuf->address(),              // 4 compact_r_addr
//                 compactTwImagBuf->address(),              // 5 compact_i_addr
//                 localTiles,                               // 6 local_tiles
//                 0u,                                       // 7 tile_offset
//                 numStages,                                // 8 num_stages
//                 halfN,                                    // 9 half_N
//                 localHalf,                                // 10 local_half
//                 0u                                        // 11 core_elem_base
//             });

//             SetRuntimeArgs(fftProg, computeKernel, cc, {
//                 numStages,                                // 0 num_stages
//                 localTiles                                // 1 tiles_per_stage
//             });

//             SetRuntimeArgs(fftProg, writerKernel, cc, {
//                 out0RealBuf->address() + rowByteOffset,   // 0 out0_r_addr
//                 out0ImagBuf->address() + rowByteOffset,   // 1 out0_i_addr
//                 out1RealBuf->address() + rowByteOffset,   // 2 out1_r_addr
//                 out1ImagBuf->address() + rowByteOffset,   // 3 out1_i_addr
//                 localTiles,                               // 4 local_tiles
//                 numStages,                                // 5 num_stages
//                 localHalf,                                // 6 local_half
//                 halfN,                                    // 7 half_N
//                 1u,                                       // 8 num_cores (row decomposition)
//                 0u,                                       // 9 core_id
//                 0u,                                       // 10 log2_cores
//                 0u,                                       // 11 tile_offset
//                 0u                                        // 12 core_elem_base
//             });
//         }

//         distributed::MeshWorkload fftWorkload;
//         distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
//         fftWorkload.add_program(deviceRange, std::move(fftProg));
//         distributed::EnqueueMeshWorkload(cq, fftWorkload, false);
//         distributed::Finish(cq);

//         std::cout << "FFT tiled kernel execution finished.\n";

//         // ── Read back out0 / out1 ───────────────────────────────────────────
//         std::vector<uint32_t> out0RealRaw;
//         std::vector<uint32_t> out0ImagRaw;
//         std::vector<uint32_t> out1RealRaw;
//         std::vector<uint32_t> out1ImagRaw;

//         distributed::EnqueueReadMeshBuffer(cq, out0RealRaw, out0RealBuf, true);
//         distributed::EnqueueReadMeshBuffer(cq, out0ImagRaw, out0ImagBuf, true);
//         distributed::EnqueueReadMeshBuffer(cq, out1RealRaw, out1RealBuf, true);
//         distributed::EnqueueReadMeshBuffer(cq, out1ImagRaw, out1ImagBuf, true);

//         // ── Reconstruct full row-0 FFT output: [out0 half | out1 half] ─────
//         std::vector<uint32_t> outRawReal(nRow, 0u);
//         std::vector<uint32_t> outRawImag(nRow, 0u);

//         const uint32_t row0Base = 0;
//         for (uint32_t i = 0; i < halfN; ++i) {
//             outRawReal[i]         = out0RealRaw[row0Base + i];
//             outRawImag[i]         = out0ImagRaw[row0Base + i];
//             outRawReal[i + halfN] = out1RealRaw[row0Base + i];
//             outRawImag[i + halfN] = out1ImagRaw[row0Base + i];
//         }

//         // ── CPU reference for row 0 ─────────────────────────────────────────
//         cpuFft(refRe, refIm);

//         std::cout << "\n=== Row 0: Key frequency bins ===\n";
//         std::cout << "bin  |  WH_re        |  WH_im        |  WH_mag       ||  CPU_re       |  CPU_im       |  CPU_mag\n";
//         std::cout << std::string(110, '-') << "\n";

//         const std::vector<uint32_t> binsToPrint = buildPrintBins(nRow);
//         for (uint32_t bin : binsToPrint) {
//             const float wre  = u32ToFloat(outRawReal[bin]);
//             const float wim  = u32ToFloat(outRawImag[bin]);
//             const float wmag = std::sqrt(wre * wre + wim * wim);

//             const float cre  = refRe[bin];
//             const float cim  = refIm[bin];
//             const float cmag = std::sqrt(cre * cre + cim * cim);

//             std::printf(
//                 "%-5u| %13.3f | %13.3f | %13.3f || %13.3f | %13.3f | %13.3f\n",
//                 bin, wre, wim, wmag, cre, cim, cmag);
//         }

//         float maxErr = 0.0f;
//         float maxMag = 0.0f;
//         for (uint32_t i = 0; i < nRow; ++i) {
//             const float wre = u32ToFloat(outRawReal[i]);
//             const float wim = u32ToFloat(outRawImag[i]);
//             const float err = std::sqrt(
//                 (wre - refRe[i]) * (wre - refRe[i]) +
//                 (wim - refIm[i]) * (wim - refIm[i]));
//             const float mag = std::sqrt(refRe[i] * refRe[i] + refIm[i] * refIm[i]);
//             if (std::isfinite(err)) {
//                 maxErr = std::max(maxErr, err);
//             }
//             maxMag = std::max(maxMag, mag);
//         }

//         std::cout << "\nRow 0 max absolute error vs CPU: " << maxErr << "\n";
//         std::cout << "Row 0 max bin magnitude (CPU):   " << maxMag << "\n";
//         if (maxMag > 0.0f) {
//             std::cout << "Row 0 relative error:            " << (maxErr / maxMag * 100.0f) << " %\n";
//         }

//         if (!meshDevice->close()) {
//             throw std::runtime_error("meshDevice->close() failed");
//         }

//         std::cout << "\nFFT host run finished.\n";
//         return 0;

//     } catch (const std::exception& e) {
//         std::cerr << "FFT host failed: " << e.what() << "\n";
//         return 1;
//     }
// }