// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <cmath>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;

#include <fmt/core.h>

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Create random float32 vector
// ═══════════════════════════════════════════════════════════════════════════
std::vector<float> create_random_float32_vector(uint32_t num_elements, float max_val, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-max_val, max_val);
    
    std::vector<float> vec(num_elements);
    for (uint32_t i = 0; i < num_elements; i++) {
        vec[i] = dist(gen);
    }
    return vec;
}

// ═══════════════════════════════════════════════════════════════════════════
// VERIFICATION FUNCTION (Both CPU and Hardware use float32)
// ═══════════════════════════════════════════════════════════════════════════
bool verify_fft_f32(
    std::vector<float>& lhs_r, std::vector<float>& lhs_i,
    std::vector<float>& rhs_r, std::vector<float>& rhs_i,
    std::vector<float>& tw_r, std::vector<float>& tw_i,
    std::vector<float>& out_lhs_r, std::vector<float>& out_lhs_i,
    std::vector<float>& out_rhs_r, std::vector<float>& out_rhs_i)
{
    bool pass = true;
    float max_diff = 0.0f;
    float max_rtol = 0.0f;
    
    for (size_t i = 0; i < lhs_r.size(); i++) {
        // CPU reference calculation (float32)
        float r1 = rhs_r[i];
        float i1 = rhs_i[i];
        float wr = tw_r[i];
        float wi = tw_i[i];
        
        // Complex multiplication: f = twiddle * rhs
        float f0 = r1 * wr - i1 * wi;  // Real part
        float f1 = r1 * wi + i1 * wr;  // Imag part
        
        // Butterfly
        float expected_lhs_r = lhs_r[i] + f0;
        float expected_lhs_i = lhs_i[i] + f1;
        float expected_rhs_r = lhs_r[i] - f0;
        float expected_rhs_i = lhs_i[i] - f1;
        
        // Hardware output
        float hw_lhs_r = out_lhs_r[i];
        float hw_lhs_i = out_lhs_i[i];
        float hw_rhs_r = out_rhs_r[i];
        float hw_rhs_i = out_rhs_i[i];
        
        // Check tolerance (float32 should be very close!)
        auto check_tol = [&](float expected, float actual, const char* name) {
            float diff = std::abs(expected - actual);
            float rtol = (expected != 0.0f) ? (diff / std::abs(expected)) : 0.0f;
            
            if (diff > max_diff) max_diff = diff;
            if (rtol > max_rtol) max_rtol = rtol;
            
            // Very tight tolerance for float32: 0.001% relative or 1e-5 absolute
            bool ok = diff <= (1e-5f + 0.00001f * std::abs(expected));
            
            if (!ok && pass) {
                fmt::print("Mismatch at index {} ({}):\n", i, name);
                fmt::print("  Expected: {}\n", expected);
                fmt::print("  Got:      {}\n", actual);
                fmt::print("  Diff:     {}\n", diff);
            }
            
            return ok;
        };
        
        if (!check_tol(expected_lhs_r, hw_lhs_r, "LHS_R") ||
            !check_tol(expected_lhs_i, hw_lhs_i, "LHS_I") ||
            !check_tol(expected_rhs_r, hw_rhs_r, "RHS_R") ||
            !check_tol(expected_rhs_i, hw_rhs_i, "RHS_I")) {
            pass = false;
        }
    }
    
    fmt::print("Verification: Max Diff = {}, Max RTol = {:.6f}%\n", 
               max_diff, max_rtol * 100.0f);
    
    return pass;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    bool pass = true;
    
    try {
        // Device setup
        int device_id = 0;
        auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();
        
        tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
        CoreCoord core = {0, 0};
        
        // ═══════════════════════════════════════════════════════════════════
        // BUFFER CONFIGURATION (All Float32!)
        // ═══════════════════════════════════════════════════════════════════
        uint32_t num_tiles = 1;
        uint32_t tile_elems = 32 * 32;  // 1024 elements per tile
        uint32_t f32_tile_size = tile_elems * sizeof(float);  // 4096 bytes
        uint32_t buffer_size = f32_tile_size * num_tiles;
        
        tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
            .page_size = f32_tile_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        
        tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config{
            .size = buffer_size
        };
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE DRAM BUFFERS (All Float32)
        // ═══════════════════════════════════════════════════════════════════
        auto src0_r_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src0_i_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src1_r_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src1_i_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto tw_r_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto tw_i_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        
        auto dst0_r_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst0_i_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst1_r_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst1_i_buf = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE CIRCULAR BUFFERS (All Float32)
        // ═══════════════════════════════════════════════════════════════════
        uint32_t tiles_in_cb = 2;
        
        std::vector<uint32_t> all_cbs = {
            tt::CBIndex::c_0,  tt::CBIndex::c_1,  tt::CBIndex::c_2,
            tt::CBIndex::c_3,  tt::CBIndex::c_4,  tt::CBIndex::c_5,
            tt::CBIndex::c_16, tt::CBIndex::c_17, tt::CBIndex::c_18,
            tt::CBIndex::c_19, tt::CBIndex::c_24, tt::CBIndex::c_25,
            tt::CBIndex::c_26, tt::CBIndex::c_27
        };
        
        for (auto cb_idx : all_cbs) {
            tt_metal::CircularBufferConfig cb_config = 
                tt_metal::CircularBufferConfig(
                    f32_tile_size * tiles_in_cb,
                    {{cb_idx, tt::DataFormat::Float32}}
                ).set_page_size(cb_idx, f32_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_config);
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE KERNELS
        // ═══════════════════════════════════════════════════════════════════
        auto reader_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default
            }
        );
        
        auto writer_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default
            }
        );
        
        auto compute_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft_float32/fft_single_core/kernels/compute/fft_compute_f32.cpp",
            core,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .math_approx_mode = false
            }
        );
        
        // ═══════════════════════════════════════════════════════════════════
        // CREATE HOST DATA (Float32)
                // ═══════════════════════════════════════════════════════════════════
        // CREATE HOST DATA (Float32)
        // ═══════════════════════════════════════════════════════════════════
        float max_val = 5.0f;
        uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
        
        std::vector<float> src0_r_vec = create_random_float32_vector(tile_elems, max_val, seed);
        std::vector<float> src0_i_vec = create_random_float32_vector(tile_elems, max_val, seed + 1);
        std::vector<float> src1_r_vec = create_random_float32_vector(tile_elems, max_val, seed + 2);
        std::vector<float> src1_i_vec = create_random_float32_vector(tile_elems, max_val, seed + 3);
        std::vector<float> tw_r_vec = create_random_float32_vector(tile_elems, max_val, seed + 4);
        std::vector<float> tw_i_vec = create_random_float32_vector(tile_elems, max_val, seed + 5);
        
        // ═══════════════════════════════════════════════════════════════════
        // WRITE INPUT DATA TO DEVICE
        // ═══════════════════════════════════════════════════════════════════
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src0_r_buf, src0_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src0_i_buf, src0_i_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src1_r_buf, src1_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src1_i_buf, src1_i_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, tw_r_buf, tw_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, tw_i_buf, tw_i_vec, false);
        
        // ═══════════════════════════════════════════════════════════════════
        // SET KERNEL RUNTIME ARGUMENTS
        // ═══════════════════════════════════════════════════════════════════
        tt::tt_metal::SetRuntimeArgs(
            program, reader_id, core,
            {
                src0_r_buf->address(),
                src0_i_buf->address(),
                src1_r_buf->address(),
                src1_i_buf->address(),
                tw_r_buf->address(),
                tw_i_buf->address(),
                num_tiles
            }
        );
        
        tt::tt_metal::SetRuntimeArgs(
            program, writer_id, core,
            {
                dst0_r_buf->address(),
                dst0_i_buf->address(),
                dst1_r_buf->address(),
                dst1_i_buf->address(),
                num_tiles
            }
        );
        
        tt::tt_metal::SetRuntimeArgs(
            program, compute_id, core,
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
        std::vector<float> out0_r_vec;
        std::vector<float> out0_i_vec;
        std::vector<float> out1_r_vec;
        std::vector<float> out1_i_vec;
        
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_r_vec, dst0_r_buf, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_i_vec, dst0_i_buf, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_r_vec, dst1_r_buf, true);
        tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_i_vec, dst1_i_buf, true);
        
        // ═══════════════════════════════════════════════════════════════════
        // VERIFY RESULTS (CPU float32 vs Hardware float32)
        // ═══════════════════════════════════════════════════════════════════
        pass &= verify_fft_f32(
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