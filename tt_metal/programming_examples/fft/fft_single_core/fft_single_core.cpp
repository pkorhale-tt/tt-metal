// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <cmath>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;

#include <fmt/core.h>

// ═══════════════════════════════════════════════════════════════════════════
// VERIFICATION FUNCTION
// ═══════════════════════════════════════════════════════════════════════════
bool verify_fft(
    std::vector<bfloat16>& lhs_r, std::vector<bfloat16>& lhs_i,
    std::vector<bfloat16>& rhs_r, std::vector<bfloat16>& rhs_i,
    std::vector<bfloat16>& tw_r, std::vector<bfloat16>& tw_i,
    std::vector<bfloat16>& out_lhs_r, std::vector<bfloat16>& out_lhs_i,
    std::vector<bfloat16>& out_rhs_r, std::vector<bfloat16>& out_rhs_i)
{
    bool pass = true;
    float max_rtol = 0.0f;
    float max_diff = 0.0f;
    
    for (size_t i = 0; i < lhs_r.size(); i++) {
        // Convert inputs to float32 for reference calculation
        float r1 = static_cast<float>(rhs_r[i]);
        float i1 = static_cast<float>(rhs_i[i]);
        float wr = static_cast<float>(tw_r[i]);
        float wi = static_cast<float>(tw_i[i]);
        
        // --- ADDED: Print inputs for the first 5 elements ---
        if (i < 5) {
            fmt::print("Input [{}]:\n", i);
            fmt::print("  LHS: (R: {:.4f}, I: {:.4f})\n", static_cast<float>(lhs_r[i]), static_cast<float>(lhs_i[i]));
            fmt::print("  RHS: (R: {:.4f}, I: {:.4f})\n", r1, i1);
            fmt::print("  Twiddle: (R: {:.4f}, I: {:.4f})\n", wr, wi);
            fmt::print("  -------------------------\n");
        }
        
        // Butterfly math (same as hardware does, but in float32)
        float f0 = r1 * wr - i1 * wi;
        float f1 = r1 * wi + i1 * wr;
        
        float expected_lhs_r = static_cast<float>(lhs_r[i]) + f0;
        float expected_lhs_i = static_cast<float>(lhs_i[i]) + f1;
        float expected_rhs_r = static_cast<float>(lhs_r[i]) - f0;
        float expected_rhs_i = static_cast<float>(lhs_i[i]) - f1;
        
        // Get hardware outputs
        float out_lr = static_cast<float>(out_lhs_r[i]);
        float out_li = static_cast<float>(out_lhs_i[i]);
        float out_rr = static_cast<float>(out_rhs_r[i]);
        float out_ri = static_cast<float>(out_rhs_i[i]);
        
        // Tolerance check
        // With fp32 internal computation, we expect much better precision!
        auto check_tol = [&](float expected, float actual) {
            float diff = std::abs(expected - actual);
            float rtol = (expected != 0.0f) ? (diff / std::abs(expected)) : 0.0f;
            
            if (diff > max_diff) max_diff = diff;
            if (rtol > max_rtol) max_rtol = rtol;
            
            // Tighter tolerance: 1% relative + 0.5 absolute
            return diff <= (0.5f + 0.01f * std::abs(expected));
        };
        
        if (!check_tol(expected_lhs_r, out_lr) || 
            !check_tol(expected_lhs_i, out_li) ||
            !check_tol(expected_rhs_r, out_rr) || 
            !check_tol(expected_rhs_i, out_ri)) 
        {
            if (pass) {
                fmt::print("Mismatch at index {}:\n", i);
                fmt::print("  LHS R: expected {}, got {}\n", expected_lhs_r, out_lr);
                fmt::print("  LHS I: expected {}, got {}\n", expected_lhs_i, out_li);
                fmt::print("  RHS R: expected {}, got {}\n", expected_rhs_r, out_rr);
                fmt::print("  RHS I: expected {}, got {}\n", expected_rhs_i, out_ri);
            }
            pass = false;
        }
    }
    
    fmt::print("Verification summary: Max Diff = {}, Max RTol = {:.2f}%\n", 
               max_diff, max_rtol * 100.0f);
    
    return pass;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    bool pass = true;
    
    try {
        // ═══════════════════════════════════════════════════════════════════
        // DEVICE SETUP
        // ═══════════════════════════════════════════════════════════════════
        int device_id = 0;
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device = 
            tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        tt::tt_metal::distributed::MeshCommandQueue& cq = 
            mesh_device->mesh_command_queue();
        
        tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
        CoreCoord core = {0, 0};
        
        // ═══════════════════════════════════════════════════════════════════
        // BUFFER CONFIGURATION
        // ═══════════════════════════════════════════════════════════════════
        uint32_t num_tiles = 1;
        uint32_t tile_elems = 32 * 32;  // 1024 elements per tile
        
        // Tile sizes for different formats
        uint32_t bf16_tile_size = tile_elems * sizeof(bfloat16);  // 2048 bytes
        uint32_t fp32_tile_size = tile_elems * sizeof(float);     // 4096 bytes
        
        uint32_t bf16_buffer_size = bf16_tile_size * num_tiles;
        
        // DRAM buffer config (bfloat16 for input/output)
        tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
            .page_size = bf16_tile_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        
        tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config{
            .size = bf16_buffer_size
        };
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE DRAM BUFFERS (all bfloat16)
        // ═══════════════════════════════════════════════════════════════════
        auto src0_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto src0_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto src1_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto src1_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto tw_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto tw_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        
        auto dst0_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto dst0_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto dst1_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        auto dst1_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            buffer_config, dram_config, mesh_device.get());
        
        // ═══════════════════════════════════════════════════════════════════
        // CIRCULAR BUFFER INDICES
        // ═══════════════════════════════════════════════════════════════════
        
        // Input CBs (bfloat16 from DRAM via reader)
        uint32_t cb_in0_r_bf16 = tt::CBIndex::c_0;
        uint32_t cb_in0_i_bf16 = tt::CBIndex::c_1;
        uint32_t cb_in1_r_bf16 = tt::CBIndex::c_2;
        uint32_t cb_in1_i_bf16 = tt::CBIndex::c_3;
        uint32_t cb_tw_r_bf16 = tt::CBIndex::c_4;
        uint32_t cb_tw_i_bf16 = tt::CBIndex::c_5;
        
        // Converted input CBs (float32 after bf16→fp32 conversion)
        uint32_t cb_in0_r_fp32 = tt::CBIndex::c_6;
        uint32_t cb_in0_i_fp32 = tt::CBIndex::c_7;
        uint32_t cb_in1_r_fp32 = tt::CBIndex::c_8;
        uint32_t cb_in1_i_fp32 = tt::CBIndex::c_9;
        uint32_t cb_tw_r_fp32 = tt::CBIndex::c_10;
        uint32_t cb_tw_i_fp32 = tt::CBIndex::c_11;
        
        // FFT output CBs (float32 from FFT compute)
        uint32_t cb_out0_r_fp32 = tt::CBIndex::c_12;
        uint32_t cb_out0_i_fp32 = tt::CBIndex::c_13;
        uint32_t cb_out1_r_fp32 = tt::CBIndex::c_14;
        uint32_t cb_out1_i_fp32 = tt::CBIndex::c_15;
        
        // Final output CBs (bfloat16 after fp32→bf16 conversion)
        uint32_t cb_out0_r_bf16 = tt::CBIndex::c_16;
        uint32_t cb_out0_i_bf16 = tt::CBIndex::c_17;
        uint32_t cb_out1_r_bf16 = tt::CBIndex::c_18;
        uint32_t cb_out1_i_bf16 = tt::CBIndex::c_19;
        
        // Temporary CBs (float32 for intermediate computation)
        uint32_t cb_tmp0 = tt::CBIndex::c_24;
        uint32_t cb_tmp1 = tt::CBIndex::c_25;
        uint32_t cb_f0 = tt::CBIndex::c_26;
        uint32_t cb_f1 = tt::CBIndex::c_27;
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE CIRCULAR BUFFERS
        // ═══════════════════════════════════════════════════════════════════
        uint32_t tiles_in_cb = 2;  // Double buffering
        
        // BFloat16 CBs (input from DRAM, output to DRAM)
        std::vector<uint32_t> bf16_cbs = {
            cb_in0_r_bf16, cb_in0_i_bf16,
            cb_in1_r_bf16, cb_in1_i_bf16,
            cb_tw_r_bf16, cb_tw_i_bf16,
            cb_out0_r_bf16, cb_out0_i_bf16,
            cb_out1_r_bf16, cb_out1_i_bf16
        };
        
        for (auto cb_idx : bf16_cbs) {
            tt_metal::CircularBufferConfig cb_config = 
                tt_metal::CircularBufferConfig(
                    bf16_tile_size * tiles_in_cb,
                    {{cb_idx, tt::DataFormat::Float16_b}}
                ).set_page_size(cb_idx, bf16_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_config);
        }
        
        // Float32 CBs (intermediate computation)
        std::vector<uint32_t> fp32_cbs = {
            cb_in0_r_fp32, cb_in0_i_fp32,
            cb_in1_r_fp32, cb_in1_i_fp32,
            cb_tw_r_fp32, cb_tw_i_fp32,
            cb_out0_r_fp32, cb_out0_i_fp32,
            cb_out1_r_fp32, cb_out1_i_fp32,
            cb_tmp0, cb_tmp1,
            cb_f0, cb_f1
        };
        
        for (auto cb_idx : fp32_cbs) {
            tt_metal::CircularBufferConfig cb_config = 
                tt_metal::CircularBufferConfig(
                    fp32_tile_size * tiles_in_cb,
                    {{cb_idx, tt::DataFormat::Float32}}
                ).set_page_size(cb_idx, fp32_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_config);
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE KERNELS
        // ═══════════════════════════════════════════════════════════════════
        
        // Reader kernel (your existing code - unchanged)
        auto reader_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/dataflow/reader_fft.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default
            }
        );
        
        // Writer kernel (your existing code - unchanged)
        auto writer_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/dataflow/writer_fft.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default
            }
        );
        
                // NEW: bf16 → fp32 conversion kernel
        auto convert_to_fp32_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/compute/convert_bf16_to_fp32.cpp",
            core,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false
            }
        );
        
        // NEW: FFT compute kernel (operates in fp32)
        auto fft_compute_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/compute/fft_compute_fp32.cpp",
            core,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false
            }
        );
        
        // NEW: fp32 → bf16 conversion kernel
        auto convert_to_bf16_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/compute/convert_fp32_to_bf16.cpp",
            core,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false
            }
        );
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE HOST DATA
        // ═══════════════════════════════════════════════════════════════════
        
        // Using smaller range for better precision
        float max_val = 5.0f;
        
        std::vector<bfloat16> src0_r_vec = create_random_vector_of_bfloat16_native(
            bf16_buffer_size, max_val, 
            std::chrono::system_clock::now().time_since_epoch().count());
        
        std::vector<bfloat16> src0_i_vec = create_random_vector_of_bfloat16_native(
            bf16_buffer_size, max_val,
            std::chrono::system_clock::now().time_since_epoch().count() + 1);
        
        std::vector<bfloat16> src1_r_vec = create_random_vector_of_bfloat16_native(
            bf16_buffer_size, max_val,
            std::chrono::system_clock::now().time_since_epoch().count() + 2);
        
        std::vector<bfloat16> src1_i_vec = create_random_vector_of_bfloat16_native(
            bf16_buffer_size, max_val,
            std::chrono::system_clock::now().time_since_epoch().count() + 3);
        
        std::vector<bfloat16> tw_r_vec = create_random_vector_of_bfloat16_native(
            bf16_buffer_size, max_val,
            std::chrono::system_clock::now().time_since_epoch().count() + 4);
        
        std::vector<bfloat16> tw_i_vec = create_random_vector_of_bfloat16_native(
            bf16_buffer_size, max_val,
            std::chrono::system_clock::now().time_since_epoch().count() + 5);
        
        // ═══════════════════════════════════════════════════════════════════
        // WRITE INPUT DATA TO DEVICE
        // ═══════════════════════════════════════════════════════════════════
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            cq, src0_r_buffer, src0_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            cq, src0_i_buffer, src0_i_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            cq, src1_r_buffer, src1_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            cq, src1_i_buffer, src1_i_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            cq, tw_r_buffer, tw_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
            cq, tw_i_buffer, tw_i_vec, false);
        
        // ═══════════════════════════════════════════════════════════════════
        // SET KERNEL RUNTIME ARGUMENTS
        // ═══════════════════════════════════════════════════════════════════
        
        // Reader kernel args (your existing args - unchanged)
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {
                src0_r_buffer->address(),
                src0_i_buffer->address(),
                src1_r_buffer->address(),
                src1_i_buffer->address(),
                tw_r_buffer->address(),
                tw_i_buffer->address(),
                num_tiles
            }
        );
        
        // Writer kernel args (your existing args - unchanged)
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                dst0_r_buffer->address(),
                dst0_i_buffer->address(),
                dst1_r_buffer->address(),
                dst1_i_buffer->address(),
                num_tiles
            }
        );
        
        // bf16 → fp32 conversion kernel args
        tt::tt_metal::SetRuntimeArgs(
            program,
            convert_to_fp32_id,
            core,
            {num_tiles}
        );
        
        // FFT compute kernel args
        tt::tt_metal::SetRuntimeArgs(
            program,
            fft_compute_id,
            core,
            {num_tiles}
        );
        
        // fp32 → bf16 conversion kernel args
        tt::tt_metal::SetRuntimeArgs(
            program,
            convert_to_bf16_id,
            core,
            {num_tiles}
        );
        
        // ═══════════════════════════════════════════════════════════════════
        // RUN PROGRAM
        // ═══════════════════════════════════════════════════════════════════
        tt::tt_metal::distributed::MeshWorkload workload;
        tt::tt_metal::distributed::MeshCoordinateRange device_range = 
            tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
        tt::tt_metal::distributed::Finish(cq);
        
        // ═══════════════════════════════════════════════════════════════════
        // READ RESULTS FROM DEVICE
        // ═══════════════════════════════════════════════════════════════════
        std::vector<bfloat16> out0_r_vec;
        std::vector<bfloat16> out0_i_vec;
        std::vector<bfloat16> out1_r_vec;
        std::vector<bfloat16> out1_i_vec;
        
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            cq, out0_r_vec, dst0_r_buffer, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            cq, out0_i_vec, dst0_i_buffer, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            cq, out1_r_vec, dst1_r_buffer, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(
            cq, out1_i_vec, dst1_i_buffer, true);
        
        // ═══════════════════════════════════════════════════════════════════
        // VERIFY RESULTS
        // ═══════════════════════════════════════════════════════════════════
        pass &= verify_fft(
            src0_r_vec, src0_i_vec,
            src1_r_vec, src1_i_vec,
            tw_r_vec, tw_i_vec,
            out0_r_vec, out0_i_vec,
            out1_r_vec, out1_i_vec
        );
        
        // ═══════════════════════════════════════════════════════════════════
        // CLEANUP
        // ═══════════════════════════════════════════════════════════════════
        pass &= mesh_device->close();
        
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }
    
    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }
    
    return 0;
}