// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_multi_core.cpp  –  rewritten to match reader/writer/compute kernel interfaces
//
// Kernel interfaces (all 3 match):
//   reader:  args[7] = dram_r, dram_i, n, num_steps, num_chunks, chunk_size, sram_buf_r
//   compute: args[2] = num_steps, num_chunks
//   writer:  args[7] = dram_r, dram_i, n, num_steps, num_chunks, chunk_size, sram_buf_r
//
// SRAM layout per core (starting at SRAM_DATA_BASE):
//   [0          .. n*4)          real ping buffer
//   [n*4        .. 2*n*4)        imag ping buffer
//   [2*n*4      .. 2*n*4 + ns*(n/2)*4)   twiddle real  (ns = num_steps)
//   [2*n*4+..   .. 2*n*4 + 2*ns*(n/2)*4) twiddle imag
//
// The host writes twiddles into L1 via a temporary L1 MeshBuffer before
// launching the FFT workload. Input/output use DRAM buffers, one row per core.

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

// CB indices – must match all three kernels exactly.
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

// SRAM base: place data above CB region. CBs typically occupy the first
// ~200KB; we start at 256KB to be safe. Adjust if CB allocation grows.
// For N=1024 FP32: 2*1024*4 + 2*10*512*4 = 8192 + 40960 = 49152 bytes.
// Well within 1.3MB per core.
constexpr uint32_t SRAM_DATA_BASE = 0x40000;  // 256 KB

inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline bool isPowerOfTwo(uint32_t x) { return x > 0 && ((x & (x - 1)) == 0); }
inline uint32_t log2u32(uint32_t x) { uint32_t r = 0; while ((1u << r) < x) ++r; return r; }
inline uint32_t floatToU32(float v)  { uint32_t u; std::memcpy(&u, &v, 4); return u; }
inline float    u32ToFloat(uint32_t u){ float v;   std::memcpy(&v, &u, 4); return v; }

// Create a DRAM MeshBuffer, page-aligned to one tile.
std::shared_ptr<distributed::MeshBuffer> createDramMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes)
{
    // Round up to tile boundary.
    const uint32_t rounded = ceilDiv(sizeBytes, TILE_BYTES) * TILE_BYTES;
    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig replicatedConfig{.size = rounded};
    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

// Create an L1 (SRAM) MeshBuffer for twiddle pre-loading.
std::shared_ptr<distributed::MeshBuffer> createL1MeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes)
{
    const uint32_t rounded = ceilDiv(sizeBytes, TILE_BYTES) * TILE_BYTES;
    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig replicatedConfig{.size = rounded};
    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

// Pack one row of floats into a flat uint32 tile-padded buffer.
std::vector<uint32_t> packRow(const std::vector<float>& data,
                               uint32_t nRow, uint32_t rowTiles) {
    const uint32_t padded = rowTiles * TILE_ELEMS;
    std::vector<uint32_t> out(padded, 0u);
    for (uint32_t i = 0; i < nRow; ++i)
        out[i] = floatToU32(data[i]);
    return out;
}

// Build twiddle factors for all stages for one row.
// Layout: tw_r[step * halfN + p], tw_i[step * halfN + p]
// (flat array, step-major, matching reader's sram_tw_r layout)
void buildTwiddles(uint32_t nRow, uint32_t numStages, uint32_t direction,
                   std::vector<uint32_t>& twR, std::vector<uint32_t>& twI) {
    const uint32_t halfN = nRow / 2;
    const float sign = (direction == 1) ? 1.0f : -1.0f;
    twR.assign(numStages * halfN, 0u);
    twI.assign(numStages * halfN, 0u);
    for (uint32_t step = 0; step < numStages; ++step) {
        const uint32_t halfM = 1u << step;
        const uint32_t m     = halfM << 1u;
        for (uint32_t p = 0; p < halfN; ++p) {
            const uint32_t j     = p % halfM;
            const uint32_t k     = j * (nRow / m);
            const float angle    = sign * 2.0f * PI * k / nRow;
            twR[step * halfN + p] = floatToU32(std::cos(angle));
            twI[step * halfN + p] = floatToU32(std::sin(angle));
        }
    }
}

void makeTestInput(uint32_t nRow, std::vector<float>& real, std::vector<float>& imag) {
    real.resize(nRow);
    imag.resize(nRow, 0.0f);
    for (uint32_t i = 0; i < nRow; ++i)
        real[i] = std::sin(2.0f * PI * i / nRow)
                + 0.25f * std::cos(6.0f * PI * i / nRow);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int      deviceId  = (argc > 1) ? std::stoi(argv[1])                         : 0;
        const uint32_t nRow      = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint32_t numCores  = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow)) throw std::runtime_error("nRow must be power of 2");
        if (nRow < 2)            throw std::runtime_error("nRow must be >= 2");
        if (nRow > 16384)        throw std::runtime_error("nRow > 16384 exceeds SRAM budget");
        if (numCores == 0 || numCores > 64)
            throw std::runtime_error("numCores must be in [1, 64]");
        if (batchSize < numCores)
            throw std::runtime_error("batchSize must be >= numCores");

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        auto& cq = meshDevice->mesh_command_queue();

        const uint32_t numStages  = log2u32(nRow);
        const uint32_t halfN      = nRow / 2;
        const uint32_t rowTiles   = ceilDiv(nRow,  TILE_ELEMS);
        const uint32_t pairTiles  = ceilDiv(halfN, TILE_ELEMS);

        // Each core processes exactly one row per launch.
        // We launch batchSize/numCores rounds, or simply assign row 0..numCores-1
        // for this design (extend later for full batching).
        // For now: numCores rows total, one per core.
        const uint32_t rowsThisLaunch = std::min(batchSize, numCores);

        // SRAM layout sizes (per core, in bytes):
        const uint32_t sramDataBytes = nRow   * sizeof(float);  // one component
        const uint32_t sramTwBytes   = numStages * halfN * sizeof(float);  // one component
        const uint32_t sramTotal     = 2 * sramDataBytes + 2 * sramTwBytes;

        std::cout << "[fft_paper_host]\n"
                  << "  nRow             = " << nRow           << "\n"
                  << "  batchSize        = " << batchSize       << "\n"
                  << "  numCores         = " << numCores        << "\n"
                  << "  rowsThisLaunch   = " << rowsThisLaunch  << "\n"
                  << "  numStages        = " << numStages        << "\n"
                  << "  halfN            = " << halfN            << "\n"
                  << "  rowTiles         = " << rowTiles         << "\n"
                  << "  pairTiles        = " << pairTiles        << "\n"
                  << "  SRAM per core    = " << sramTotal << " bytes\n";

        if (SRAM_DATA_BASE + sramTotal > 1300000)
            throw std::runtime_error("SRAM layout exceeds 1.3MB");

        // ---------------------------------------------------------------
        // DRAM buffers: one tile per row, replicated across all devices.
        // We allocate numCores rows worth.
        // ---------------------------------------------------------------
        const uint32_t rowBufBytes = rowsThisLaunch * rowTiles * TILE_BYTES;

        auto inputRealBuf  = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto inputImagBuf  = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputRealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);

        // ---------------------------------------------------------------
        // Build and write input data (rowsThisLaunch rows).
        // ---------------------------------------------------------------
        const uint32_t elemsPerRow = rowTiles * TILE_ELEMS;
        std::vector<uint32_t> inputRealPacked(rowsThisLaunch * elemsPerRow, 0u);
        std::vector<uint32_t> inputImagPacked(rowsThisLaunch * elemsPerRow, 0u);

        for (uint32_t r = 0; r < rowsThisLaunch; ++r) {
            std::vector<float> rowR, rowI;
            makeTestInput(nRow, rowR, rowI);
            // Slightly vary each row so we can distinguish them.
            for (uint32_t i = 0; i < nRow; ++i)
                rowR[i] += 0.01f * r;
            for (uint32_t i = 0; i < nRow; ++i) {
                inputRealPacked[r * elemsPerRow + i] = floatToU32(rowR[i]);
                inputImagPacked[r * elemsPerRow + i] = floatToU32(rowI[i]);
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, inputRealBuf, inputRealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, inputImagBuf, inputImagPacked, false);

        // ---------------------------------------------------------------
        // Build twiddle factors and write into L1 on each core.
        // Reader expects them at SRAM_DATA_BASE + 2*n*4 (tw_r) and
        //                         SRAM_DATA_BASE + 2*n*4 + ns*(n/2)*4 (tw_i)
        // We write the same twiddles to all cores (same N, same direction).
        // ---------------------------------------------------------------
        std::vector<uint32_t> twR, twI;
        buildTwiddles(nRow, numStages, direction, twR, twI);

        // twR and twI are each numStages * halfN uint32s.
        // We need to write them into L1 at the right offset.
        // Strategy: create an L1 buffer large enough for twiddles only,
        // placed at the correct SRAM address via a fixed-address L1 buffer.
        // TT-Metalium doesn't support fixed-address L1 buffers directly from
        // the host API, so we instead embed twiddles into the DRAM input
        // and have the reader fetch from DRAM, OR we pass the DRAM twiddle
        // address as an extra arg.
        //
        // Simplest correct fix: pass a DRAM twiddle buffer address to the
        // reader as arg[6] alongside the SRAM buffer.
        // BUT the reader kernel computes sram_tw from sram_buf_r_addr
        // internally and reads from SRAM — it cannot read from DRAM.
        //
        // Therefore we must rewrite the reader to accept a DRAM twiddle addr,
        // OR pre-populate the SRAM via a twiddle-init kernel that runs first.
        //
        // We use a twiddle-init dataflow kernel: a simple reader that copies
        // from DRAM twiddle buffers into SRAM at the correct offsets.
        // This runs as a separate program before the FFT program.

        const uint32_t twBufBytes = numStages * halfN * sizeof(float);
        // Round up to tile boundary for DRAM buffer.
        auto twRealDramBuf = createDramMeshBuffer(meshDevice, twBufBytes);
        auto twImagDramBuf = createDramMeshBuffer(meshDevice, twBufBytes);

        distributed::EnqueueWriteMeshBuffer(cq, twRealDramBuf, twR, false);
        distributed::EnqueueWriteMeshBuffer(cq, twImagDramBuf, twI, false);

        // ---------------------------------------------------------------
        // Program 1: twiddle init — copy twiddles from DRAM into SRAM on
        // each core using a simple dataflow kernel.
        // ---------------------------------------------------------------
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

            // sram_tw_r_addr = SRAM_DATA_BASE + 2*n*4
            const uint32_t sramTwRAddr = SRAM_DATA_BASE + 2 * nRow * sizeof(float);
            const uint32_t sramTwIAddr = sramTwRAddr + twBufBytes;

            for (uint32_t c = 0; c < rowsThisLaunch; ++c) {
                SetRuntimeArgs(twInitProg, twInitKernel, CoreCoord{c, 0},
                    {
                        twRealDramBuf->address(),  // 0 dram_tw_r
                        twImagDramBuf->address(),  // 1 dram_tw_i
                        sramTwRAddr,               // 2 sram_tw_r destination
                        sramTwIAddr,               // 3 sram_tw_i destination
                        twBufBytes                 // 4 bytes to copy
                    });
            }

            distributed::MeshWorkload twWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            twWorkload.add_program(deviceRange, std::move(twInitProg));
            distributed::EnqueueMeshWorkload(cq, twWorkload, false);
            distributed::Finish(cq);
            std::cout << "Twiddle init finished.\n";
        }

        // ---------------------------------------------------------------
        // Program 2: FFT — reader / compute / writer.
        // num_chunks = 1, chunk_size = halfN (entire row in one chunk).
        // ---------------------------------------------------------------
        {
            Program fftProg = CreateProgram();
            CoreRange coreRange({0, 0}, {rowsThisLaunch - 1, 0});

            const uint32_t numChunks = 1u;
            const uint32_t chunkSize = halfN;  // all pairs in one chunk

            auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
                CircularBufferConfig cfg =
                    CircularBufferConfig(depthTiles * TILE_BYTES,
                                         {{cbId, tt::DataFormat::Float32}})
                        .set_page_size(cbId, TILE_BYTES);
                CreateCircularBuffer(fftProg, coreRange, cfg);
            };

            // CBs matching compute kernel indices exactly.
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
            makeCb(CB_INT0,    1);
            makeCb(CB_INT1,    1);
            makeCb(CB_F0,      1);
            makeCb(CB_F1,      1);

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
                CoreCoord coreCoord{c, 0};

                // Each core reads its own row from DRAM at tile offset c.
                // The reader reads tile 'c' (page c) from the DRAM buffer.
                // reader_fft_f32.cpp reads tile 0 from DRAM for step 0 —
                // we pass per-core DRAM addresses offset by row.
                // Since MeshBuffer is replicated and row-major, core c's row
                // starts at byte offset c * rowTiles * TILE_BYTES.
                // We pass the base address; the reader always reads tile 0.
                // For multi-row support we'd need to pass a row offset arg,
                // but the current reader kernel always uses tile index 0.
                // So we create per-core virtual buffers by offsetting the address.
                const uint32_t rowByteOffset = c * rowTiles * TILE_BYTES;

                SetRuntimeArgs(fftProg, readerKernel, coreCoord,
                    {
                        inputRealBuf->address() + rowByteOffset,  // 0 dram_input_r
                        inputImagBuf->address() + rowByteOffset,  // 1 dram_input_i
                        nRow,                                      // 2 n
                        numStages,                                 // 3 num_steps
                        numChunks,                                 // 4 num_chunks
                        chunkSize,                                 // 5 chunk_size
                        SRAM_DATA_BASE                             // 6 sram_buf_r_addr
                    });

                SetRuntimeArgs(fftProg, computeKernel, coreCoord,
                    {
                        numStages,   // 0 num_steps
                        numChunks    // 1 num_chunks
                    });

                SetRuntimeArgs(fftProg, writerKernel, coreCoord,
                    {
                        outputRealBuf->address() + rowByteOffset,  // 0 dram_output_r
                        outputImagBuf->address() + rowByteOffset,  // 1 dram_output_i
                        nRow,                                       // 2 n
                        numStages,                                  // 3 num_steps
                        numChunks,                                  // 4 num_chunks
                        chunkSize,                                  // 5 chunk_size
                        SRAM_DATA_BASE                              // 6 sram_buf_r_addr
                    });
            }

            distributed::MeshWorkload fftWorkload;
            distributed::MeshCoordinateRange deviceRange(meshDevice->shape());
            fftWorkload.add_program(deviceRange, std::move(fftProg));
            distributed::EnqueueMeshWorkload(cq, fftWorkload, false);
            distributed::Finish(cq);
            std::cout << "FFT kernel execution finished.\n";
        }

        // ---------------------------------------------------------------
        // Read back results.
        // ---------------------------------------------------------------
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