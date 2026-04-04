// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_multi_core.cpp  –  CORRECTED HOST
//
// Key structural changes vs. original:
//   1. stage0/stage1 are FULL-ROW ping-pong DRAM buffers (not pre-split
//      even/odd halves).  The reader kernel does the per-stage even/odd
//      split itself, which is what the reader/writer kernels expect.
//   2. A separate output DRAM buffer receives the last-stage results.
//   3. CB sizes fixed:
//        CBs  0-5   (even/odd/tw)   : 2 * pair_tiles pages
//        CBs 16-19  (out0/out1)     : 2 * pair_tiles pages
//        CBs 20-23  (tmp/tw_odd)    :     1          page
//        CBs 24-25  (reader scratch): 2 * row_tiles  pages
//        CBs 26-27  (writer scratch): 2 * row_tiles  pages
//      Note: reader and writer now use different scratch CB indices to
//      avoid the race condition that existed when both used 24/25.
//   4. Compute kernel arg[1] is now rowsPerCore * pair_tiles (not tilesPerRow),
//      so the inner loop count matches the reader/writer inner loop count.
//   5. Reader args corrected to 13 values matching the kernel's get_arg_val.
//   6. Writer args corrected to 12 values matching the kernel's get_arg_val.
//
// Kernel arg layout (must match kernels exactly):
//   Reader  (13): s0r,s0i, s1r,s1i, twr,twi, row_tiles, pair_tiles,
//                 n_row, num_stages, total_rows, row_start, rows_this_core
//   Compute  (2): num_stages, tiles_per_stage (= rowsPerCore * pair_tiles)
//   Writer  (12): s0r,s0i, s1r,s1i, outr,outi, row_tiles, pair_tiles,
//                 n_row, num_stages, row_start, rows_this_core

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
constexpr uint32_t TILE_ELEMS = TILE_H * TILE_W;          // 1024 elements
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float); // 4096 bytes

// CB indices – must match kernels
constexpr uint32_t CB_EVEN_R   = 0;
constexpr uint32_t CB_EVEN_I   = 1;
constexpr uint32_t CB_ODD_R    = 2;
constexpr uint32_t CB_ODD_I    = 3;
constexpr uint32_t CB_TW_R     = 4;
constexpr uint32_t CB_TW_I     = 5;
constexpr uint32_t CB_OUT0_R   = 16;
constexpr uint32_t CB_OUT0_I   = 17;
constexpr uint32_t CB_OUT1_R   = 18;
constexpr uint32_t CB_OUT1_I   = 19;
constexpr uint32_t CB_TMP0     = 20;
constexpr uint32_t CB_TMP1     = 21;
constexpr uint32_t CB_TW_ODD_R = 22;
constexpr uint32_t CB_TW_ODD_I = 23;
// Reader scratch (RISCV_0 data mover only)
constexpr uint32_t CB_READER_ROW_R = 24;
constexpr uint32_t CB_READER_ROW_I = 25;
// Writer scratch (RISCV_1 data mover only) – separate to avoid race
constexpr uint32_t CB_WRITER_ROW_R = 26;
constexpr uint32_t CB_WRITER_ROW_I = 27;

inline uint32_t ceilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

inline bool isPowerOfTwo(uint32_t x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

inline uint32_t log2u32(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) ++r;
    return r;
}

inline uint32_t floatToU32(float v) {
    uint32_t out; std::memcpy(&out, &v, sizeof(float)); return out;
}

inline float u32ToFloat(uint32_t v) {
    float out; std::memcpy(&out, &v, sizeof(uint32_t)); return out;
}

// ---------------------------------------------------------------------------
// Create a replicated DRAM buffer across the mesh, paged at TILE_BYTES.
// ---------------------------------------------------------------------------
std::shared_ptr<distributed::MeshBuffer> createDramMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& meshDevice,
    uint32_t sizeBytes)
{
    distributed::DeviceLocalBufferConfig localConfig{
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig replicatedConfig{ .size = sizeBytes };
    return distributed::MeshBuffer::create(replicatedConfig, localConfig, meshDevice.get());
}

// ---------------------------------------------------------------------------
// Pack the full input signal (batchSize rows, nRow elements each) into flat
// tile-sized arrays to upload to stage0 DRAM.
// Rows beyond batchSize (up to batchPadded) are zero-padded.
// ---------------------------------------------------------------------------
void packFullRows(
    const std::vector<float>& inputReal,
    const std::vector<float>& inputImag,
    uint32_t batchSize,
    uint32_t batchPadded,
    uint32_t nRow,
    uint32_t rowTiles,                    // = ceilDiv(nRow, TILE_ELEMS) = 1
    std::vector<uint32_t>& stage0Real,
    std::vector<uint32_t>& stage0Imag)
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

// ---------------------------------------------------------------------------
// Build twiddle factor tiles for all stages.
// Layout in DRAM: [stage][global_row][pair] in tile-sized pages.
// Reader accesses tile ID = (stage * total_rows + global_row) * pair_tiles + t
// ---------------------------------------------------------------------------
void buildTwiddleTiles(
    uint32_t nRow,
    uint32_t numStages,
    uint32_t totalRows,    // = numCores * rowsPerCore (= batchPadded)
    uint32_t pairTiles,    // = ceilDiv(nRow/2, TILE_ELEMS)
    uint32_t direction,    // 0 = forward, 1 = inverse
    std::vector<uint32_t>& twReal,
    std::vector<uint32_t>& twImag)
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

void makeTestInput(
    uint32_t batchSize, uint32_t nRow,
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

void printFirstOutputs(
    const std::vector<float>& outputReal,
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
        const int      deviceId  = (argc > 1) ? std::stoi(argv[1])                    : 0;
        const uint32_t nRow      = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 1024;
        const uint32_t batchSize = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 256;
        const uint32_t numCores  = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const uint32_t direction = (argc > 5) ? static_cast<uint32_t>(std::stoul(argv[5])) : 0;

        if (!isPowerOfTwo(nRow)) throw std::runtime_error("nRow must be power of 2");
        if (nRow < 2)            throw std::runtime_error("nRow must be >= 2");
        if (numCores == 0 || numCores > 64)
            throw std::runtime_error("numCores must be in [1, 64]");

        auto meshDevice = distributed::MeshDevice::create_unit_mesh(deviceId);
        auto& cq        = meshDevice->mesh_command_queue();

        // -----------------------------------------------------------------
        // Derived constants
        // -----------------------------------------------------------------
        const uint32_t numStages    = log2u32(nRow);
        const uint32_t halfN        = nRow / 2;

        // Tiles needed for a full row vs. a pair (half-row)
        const uint32_t rowTiles   = ceilDiv(nRow,  TILE_ELEMS);   // full row
        const uint32_t pairTiles  = ceilDiv(halfN, TILE_ELEMS);   // half row (pairs)

        const uint32_t rowsPerCore  = ceilDiv(batchSize, numCores);
        const uint32_t batchPadded  = numCores * rowsPerCore;      // = totalRows
        const uint32_t totalRows    = batchPadded;

        // Tiles per core in stage0/stage1 buffers
        const uint32_t tilesPerCore  = rowsPerCore * rowTiles;

        // Inner-loop tile count for compute: must match reader/writer row count
        const uint32_t tilesPerStageCompute = rowsPerCore * pairTiles;

        // Buffer byte sizes
        const uint32_t rowBufBytes    = totalRows * rowTiles  * TILE_BYTES; // stage0/1/output
        const uint32_t twBufBytes     = numStages * totalRows * pairTiles * TILE_BYTES;

        std::cout << "[fft_paper_host]\n"
                  << "  nRow               = " << nRow      << "\n"
                  << "  batchSize          = " << batchSize << "\n"
                  << "  numCores           = " << numCores  << "\n"
                  << "  rowsPerCore        = " << rowsPerCore << "\n"
                  << "  numStages          = " << numStages << "\n"
                  << "  rowTiles           = " << rowTiles  << "\n"
                  << "  pairTiles          = " << pairTiles << "\n"
                  << "  tilesPerStageComp  = " << tilesPerStageCompute << "\n";

        // -----------------------------------------------------------------
        // Generate input and twiddle data
        // -----------------------------------------------------------------
        std::vector<float> inputReal, inputImag;
        makeTestInput(batchSize, nRow, inputReal, inputImag);

        // Pack full rows into stage0 (no pre-splitting – reader handles that)
        std::vector<uint32_t> stage0RealPacked, stage0ImagPacked;
        packFullRows(inputReal, inputImag, batchSize, batchPadded, nRow,
                     rowTiles, stage0RealPacked, stage0ImagPacked);

        std::vector<uint32_t> twRealPacked, twImagPacked;
        buildTwiddleTiles(nRow, numStages, totalRows, pairTiles, direction,
                          twRealPacked, twImagPacked);

        // -----------------------------------------------------------------
        // DRAM buffers
        //   stage0 / stage1 : full-row ping-pong buffers
        //   output          : receives final-stage results
        //   twiddle         : precomputed, read-only during execution
        // -----------------------------------------------------------------
        auto stage0RealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto stage0ImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto stage1RealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto stage1ImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputRealBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto outputImagBuf = createDramMeshBuffer(meshDevice, rowBufBytes);
        auto twRealBuf     = createDramMeshBuffer(meshDevice, twBufBytes);
        auto twImagBuf     = createDramMeshBuffer(meshDevice, twBufBytes);

        // -----------------------------------------------------------------
        // Program: CBs and kernels
        // -----------------------------------------------------------------
        Program program = CreateProgram();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange deviceRange(meshDevice->shape());

        CoreRange coreRange({0, 0}, {numCores - 1, 0});

        // Helper: create a CB with a given id and depth (in whole tiles)
        auto makeCb = [&](uint32_t cbId, uint32_t depthTiles) {
            CircularBufferConfig cfg =
                CircularBufferConfig(depthTiles * TILE_BYTES,
                                     {{cbId, tt::DataFormat::Float32}})
                    .set_page_size(cbId, TILE_BYTES);
            CreateCircularBuffer(program, coreRange, cfg);
        };

        // Pair-sized CBs (double-buffered so reader/compute/writer can overlap)
        makeCb(CB_EVEN_R,   2 * pairTiles);
        makeCb(CB_EVEN_I,   2 * pairTiles);
        makeCb(CB_ODD_R,    2 * pairTiles);
        makeCb(CB_ODD_I,    2 * pairTiles);
        makeCb(CB_TW_R,     2 * pairTiles);
        makeCb(CB_TW_I,     2 * pairTiles);
        makeCb(CB_OUT0_R,   2 * pairTiles);
        makeCb(CB_OUT0_I,   2 * pairTiles);
        makeCb(CB_OUT1_R,   2 * pairTiles);
        makeCb(CB_OUT1_I,   2 * pairTiles);

        // Intermediate compute CBs (depth 1; produced and consumed within
        // a single butterfly iteration so no double-buffering needed)
        makeCb(CB_TMP0,     1);
        makeCb(CB_TMP1,     1);
        makeCb(CB_TW_ODD_R, 1);
        makeCb(CB_TW_ODD_I, 1);

        // Reader scratch (CBs 24-25): holds one full row while the reader
        // scatters it into even/odd pair CBs.
        makeCb(CB_READER_ROW_R, 2 * rowTiles);
        makeCb(CB_READER_ROW_I, 2 * rowTiles);

        // Writer scratch (CBs 26-27): holds the reassembled full-row output
        // before the writer NOC-writes it back to DRAM.
        // MUST be different from reader scratch to avoid the race condition
        // that existed when both data movers shared CBs 24-25.
        makeCb(CB_WRITER_ROW_R, 2 * rowTiles);
        makeCb(CB_WRITER_ROW_I, 2 * rowTiles);

        // -----------------------------------------------------------------
        // Kernels
        // -----------------------------------------------------------------
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
                .math_fidelity  = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true});

        // -----------------------------------------------------------------
        // Per-core runtime args
        // -----------------------------------------------------------------
        for (uint32_t c = 0; c < numCores; ++c) {
            CoreCoord coreCoord{c, 0};

            const uint32_t rowStart = c * rowsPerCore;

            // Reader: 13 args
            // [s0r, s0i, s1r, s1i, twr, twi,
            //  row_tiles, pair_tiles, n_row, num_stages,
            //  total_rows, row_start, rows_this_core]
            SetRuntimeArgs(program, readerKernel, coreCoord,
                {
                    stage0RealBuf->address(),  // 0:  stage0_r_addr
                    stage0ImagBuf->address(),  // 1:  stage0_i_addr
                    stage1RealBuf->address(),  // 2:  stage1_r_addr
                    stage1ImagBuf->address(),  // 3:  stage1_i_addr
                    twRealBuf->address(),      // 4:  twiddle_r_addr
                    twImagBuf->address(),      // 5:  twiddle_i_addr
                    rowTiles,                  // 6:  row_tiles
                    pairTiles,                 // 7:  pair_tiles
                    nRow,                      // 8:  n_row
                    numStages,                 // 9:  num_stages
                    totalRows,                 // 10: total_rows
                    rowStart,                  // 11: row_start
                    rowsPerCore                // 12: rows_this_core
                });

            // Compute: 2 args
            // [num_stages, tiles_per_stage]
            // tiles_per_stage MUST equal rowsPerCore * pairTiles so the
            // compute inner loop count matches the reader/writer row count.
            SetRuntimeArgs(program, computeKernel, coreCoord,
                {
                    numStages,              // 0: num_stages
                    tilesPerStageCompute    // 1: tiles_per_stage  ← was tilesPerRow (wrong)
                });

            // Writer: 12 args
            // [s0r, s0i, s1r, s1i, outr, outi,
            //  row_tiles, pair_tiles, n_row, num_stages,
            //  row_start, rows_this_core]
            SetRuntimeArgs(program, writerKernel, coreCoord,
                {
                    stage0RealBuf->address(),  // 0:  stage0_r_addr
                    stage0ImagBuf->address(),  // 1:  stage0_i_addr
                    stage1RealBuf->address(),  // 2:  stage1_r_addr
                    stage1ImagBuf->address(),  // 3:  stage1_i_addr
                    outputRealBuf->address(),  // 4:  output_r_addr
                    outputImagBuf->address(),  // 5:  output_i_addr
                    rowTiles,                  // 6:  row_tiles
                    pairTiles,                 // 7:  pair_tiles
                    nRow,                      // 8:  n_row
                    numStages,                 // 9:  num_stages
                    rowStart,                  // 10: row_start
                    rowsPerCore                // 11: rows_this_core
                });
        }

        // -----------------------------------------------------------------
        // Upload input data and twiddles
        // -----------------------------------------------------------------
        distributed::EnqueueWriteMeshBuffer(cq, stage0RealBuf, stage0RealPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, stage0ImagBuf, stage0ImagPacked, false);
        distributed::EnqueueWriteMeshBuffer(cq, twRealBuf,     twRealPacked,     false);
        distributed::EnqueueWriteMeshBuffer(cq, twImagBuf,     twImagPacked,     false);

        // -----------------------------------------------------------------
        // Run
        // -----------------------------------------------------------------
        workload.add_program(deviceRange, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::cout << "Kernel execution finished.\n";

        // -----------------------------------------------------------------
        // Read output – the writer sends last-stage results to outputReal/Imag.
        // Each row is stored as rowTiles tiles of TILE_ELEMS floats in order.
        // -----------------------------------------------------------------
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