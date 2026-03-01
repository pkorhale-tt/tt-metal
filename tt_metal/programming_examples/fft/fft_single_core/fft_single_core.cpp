// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
#include <fmt/core.h>

bool verify_fft(std::vector<bfloat16>& lhs_r, std::vector<bfloat16>& lhs_i,
                std::vector<bfloat16>& rhs_r, std::vector<bfloat16>& rhs_i,
                std::vector<bfloat16>& tw_r, std::vector<bfloat16>& tw_i,
                std::vector<bfloat16>& out_lhs_r, std::vector<bfloat16>& out_lhs_i,
                std::vector<bfloat16>& out_rhs_r, std::vector<bfloat16>& out_rhs_i) {
    bool pass = true;
    for (size_t i = 0; i < lhs_r.size(); i++) {
        float r1 = static_cast<float>(rhs_r[i]);
        float i1 = static_cast<float>(rhs_i[i]);
        float wr = static_cast<float>(tw_r[i]);
        float wi = static_cast<float>(tw_i[i]);
        
        // Butterfly math
        float f0 = r1 * wr - i1 * wi;
        float f1 = r1 * wi + i1 * wr;

        float expected_lhs_r = static_cast<float>(lhs_r[i]) + f0;
        float expected_lhs_i = static_cast<float>(lhs_i[i]) + f1;

        float expected_rhs_r = static_cast<float>(lhs_r[i]) - f0;
        float expected_rhs_i = static_cast<float>(lhs_i[i]) - f1;

        float out_lr = static_cast<float>(out_lhs_r[i]);
        float out_li = static_cast<float>(out_lhs_i[i]);
        float out_rr = static_cast<float>(out_rhs_r[i]);
        float out_ri = static_cast<float>(out_rhs_i[i]);

        // Tolerance check - bfloat16 has 7 bits of mantissa, precision drops rapidly as magnitude increases
        auto check_tol = [](float expected, float actual, float rtol=0.05f, float atol=1.0f) {
            return std::abs(expected - actual) <= (atol + rtol * std::abs(expected));
        };

        if (!check_tol(expected_lhs_r, out_lr) || !check_tol(expected_lhs_i, out_li) ||
            !check_tol(expected_rhs_r, out_rr) || !check_tol(expected_rhs_i, out_ri)) {
            pass = false;
            fmt::print("Mismatch at index {}:\n", i);
            fmt::print("  LHS R: expected {}, got {}\n", expected_lhs_r, out_lr);
            fmt::print("  LHS I: expected {}, got {}\n", expected_lhs_i, out_li);
            fmt::print("  RHS R: expected {}, got {}\n", expected_rhs_r, out_rr);
            fmt::print("  RHS I: expected {}, got {}\n", expected_rhs_i, out_ri);
            break;
        }
    }
    return pass;
}

int main() {
    bool pass = true;

    try {
        int device_id = 0;
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

        tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        uint32_t num_tiles = 1;
        uint32_t tile_elems = 32 * 32;                // 1024 elements per bfloat16 tile
        uint32_t single_tile_size = tile_elems * sizeof(bfloat16); // 2048 bytes
        uint32_t buffer_size = single_tile_size * num_tiles;

        tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
            .page_size = single_tile_size,
            .buffer_type = tt::tt_metal::BufferType::DRAM
        };

        tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config{
            .size = buffer_size
        };

        // Create DRAM buffers
        auto src0_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src0_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src1_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto src1_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto tw_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto tw_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst0_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst0_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst1_r_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst1_i_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        // Circular Buffers (L1) Setup
        uint32_t src0_r_cb_index = tt::CBIndex::c_0;
        uint32_t src0_i_cb_index = tt::CBIndex::c_1;
        uint32_t src1_r_cb_index = tt::CBIndex::c_2;
        uint32_t src1_i_cb_index = tt::CBIndex::c_3;
        uint32_t tw_r_cb_index = tt::CBIndex::c_4;
        uint32_t tw_i_cb_index = tt::CBIndex::c_5;
        uint32_t out0_r_cb_index = tt::CBIndex::c_16;
        uint32_t out0_i_cb_index = tt::CBIndex::c_17;
        uint32_t out1_r_cb_index = tt::CBIndex::c_18;
        uint32_t out1_i_cb_index = tt::CBIndex::c_19;
        
        // Temp cb's used by compute kernel
        uint32_t tmp0_cb_index = tt::CBIndex::c_24;
        uint32_t tmp1_cb_index = tt::CBIndex::c_25;
        uint32_t tmp2_f0_cb_index = tt::CBIndex::c_26;
        uint32_t tmp3_f1_cb_index = tt::CBIndex::c_27;

        std::vector<uint32_t> all_cbs = {
            src0_r_cb_index, src0_i_cb_index,
            src1_r_cb_index, src1_i_cb_index,
            tw_r_cb_index, tw_i_cb_index,
            out0_r_cb_index, out0_i_cb_index,
            out1_r_cb_index, out1_i_cb_index,
            tmp0_cb_index, tmp1_cb_index,
            tmp2_f0_cb_index, tmp3_f1_cb_index
        };

        uint32_t tiles_in_cb = 2; // double buffering capacity
        for (auto cb_idx : all_cbs) {
            tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(single_tile_size * tiles_in_cb, {{cb_idx, tt::DataFormat::Float16_b}})
                .set_page_size(cb_idx, single_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_config);
        }

        // Kernels
        auto reader_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/dataflow/reader_fft.cpp",
            core,
            tt::tt_metal::DataMovementConfig{.processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default}
        );

        auto writer_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/dataflow/writer_fft.cpp",
            core,
            tt::tt_metal::DataMovementConfig{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default}
        );

        auto compute_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/fft/fft_single_core/kernels/compute/fft_compute.cpp",
            core,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4, 
                .fp32_dest_acc_en = true, 
                .math_approx_mode = false
            }
        );

        // Host data
        std::vector<bfloat16> src0_r_vec = create_random_vector_of_bfloat16_native(
            buffer_size, 100.0f, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<bfloat16> src0_i_vec = create_random_vector_of_bfloat16_native(
            buffer_size, 100.0f, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<bfloat16> src1_r_vec = create_random_vector_of_bfloat16_native(
            buffer_size, 100.0f, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<bfloat16> src1_i_vec = create_random_vector_of_bfloat16_native(
            buffer_size, 100.0f, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<bfloat16> tw_r_vec = create_random_vector_of_bfloat16_native(
            buffer_size, 100.0f, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<bfloat16> tw_i_vec = create_random_vector_of_bfloat16_native(
            buffer_size, 100.0f, std::chrono::system_clock::now().time_since_epoch().count());

        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src0_r_buffer, src0_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src0_i_buffer, src0_i_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src1_r_buffer, src1_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, src1_i_buffer, src1_i_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, tw_r_buffer, tw_r_vec, false);
        tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, tw_i_buffer, tw_i_vec, false);

        // Set kernel args
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src0_r_buffer->address(), src0_i_buffer->address(),
             src1_r_buffer->address(), src1_i_buffer->address(),
             tw_r_buffer->address(), tw_i_buffer->address(),
             num_tiles}
        );

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {dst0_r_buffer->address(), dst0_i_buffer->address(),
             dst1_r_buffer->address(), dst1_i_buffer->address(),
             num_tiles}
        );

        tt::tt_metal::SetRuntimeArgs(
            program,
            compute_id,
            core,
            {num_tiles}
        );

        // Run
        tt::tt_metal::distributed::MeshWorkload workload;
        tt::tt_metal::distributed::MeshCoordinateRange device_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
        tt::tt_metal::distributed::Finish(cq);

        // Read results
        std::vector<bfloat16> out0_r_vec; tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_r_vec, dst0_r_buffer, true);
        std::vector<bfloat16> out0_i_vec; tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out0_i_vec, dst0_i_buffer, true);
        std::vector<bfloat16> out1_r_vec; tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_r_vec, dst1_r_buffer, true);
        std::vector<bfloat16> out1_i_vec; tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, out1_i_vec, dst1_i_buffer, true);

        pass &= verify_fft(src0_r_vec, src0_i_vec, src1_r_vec, src1_i_vec, tw_r_vec, tw_i_vec,
                           out0_r_vec, out0_i_vec, out1_r_vec, out1_i_vec);

        pass &= mesh_device->close();

    } catch (const std::exception &e) {
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
