// ============================================================
// fft_host.cpp – host-side program setup (TT-Metalium layout)
// ============================================================

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"
#include "tt_metal/api/tt-metalium/allocator.hpp"
#include "tt_metal/api/tt-metalium/hal.hpp"

#include <cmath>
#include <vector>
#include <cassert>
#include <stdexcept>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// ── CB IDs shared with device kernels ───────────────────────
// TODO: ensure these match what your kernels expect.
enum CbId : uint32_t {
    CB_LHS_R      = 0,
    CB_LHS_I      = 1,
    CB_RHS_R      = 2,
    CB_RHS_I      = 3,
    CB_TWIDDLE_R  = 4,
    CB_TWIDDLE_I  = 5,
    CB_OUT_R      = 6,
    CB_OUT_I      = 7,
    CB_SCRATCH_R  = 8,
    CB_SCRATCH_I  = 9,
    CB_SYNC       = 10,
    CB_TMP_R      = 11,
    CB_TMP_I      = 12,
    CB_WR_R       = 13,
    CB_WR_I       = 14,
};

// ── Twiddle factor precomputation ───────────────────────────
std::vector<float> precompute_twiddles(
    uint32_t N, uint32_t num_stages, bool is_ifft)
{
    std::vector<float> tw;
    tw.reserve(num_stages * (N / 2) * 2);

    for (uint32_t s = 0; s < num_stages; s++) {
        uint32_t stride = 1u << s;
        uint32_t M = 2 * stride;

        for (uint32_t k = 0; k < N / 2; k++) {
            uint32_t kk = k % stride;
            double angle = -2.0 * M_PI * kk / M;
            if (is_ifft) angle = -angle;

            tw.push_back(static_cast<float>(std::cos(angle))); // real
            tw.push_back(static_cast<float>(std::sin(angle))); // imag
        }
    }
    return tw;
}

// ── Core grid helpers ───────────────────────────────────────
CoreCoord linear_to_core(uint32_t id, uint32_t grid_cols = 8) {
    return {static_cast<int>(id % grid_cols), static_cast<int>(id / grid_cols)};
}

uint32_t core_to_linear(CoreCoord c, uint32_t grid_cols = 8) {
    return static_cast<uint32_t>(c.y) * grid_cols + static_cast<uint32_t>(c.x);
}

// ── Main FFT program builder ─────────────────────────────────
struct FFTConfig {
    uint32_t N;
    uint32_t num_cores;
    bool     is_ifft;
};

void run_fft(
    Device*       device,
    CommandQueue& cq,
    const FFTConfig& cfg,
    Buffer&       input_buf,
    Buffer&       output_buf)
{
    assert((cfg.N & (cfg.N - 1)) == 0 && "N must be power of 2");
    assert((cfg.num_cores & (cfg.num_cores - 1)) == 0 && "num_cores must be power of 2");
    assert(cfg.N % cfg.num_cores == 0 && "N must be divisible by num_cores");

    uint32_t local_N       = cfg.N / cfg.num_cores;
    uint32_t num_stages    = static_cast<uint32_t>(std::log2(cfg.N));
    uint32_t num_local_stg = static_cast<uint32_t>(std::log2(local_N));
    uint32_t num_noc_stg   = num_stages - num_local_stg;
    uint32_t elem_bytes    = 4;

    // ── Twiddle DRAM buffer ──────────────────────────────────
    auto tw_floats = precompute_twiddles(cfg.N, num_stages, cfg.is_ifft);
    uint32_t tw_bytes = static_cast<uint32_t>(tw_floats.size()) * elem_bytes;

    auto twiddle_buf = CreateBuffer(device, {
        .size        = tw_bytes,
        .page_size   = elem_bytes,
        .buffer_type = BufferType::DRAM,
    });

    EnqueueWriteBuffer(cq, twiddle_buf,
                       reinterpret_cast<const void*>(tw_floats.data()),
                       /*blocking*/ false);

    // ── Build program ────────────────────────────────────────
    Program program = CreateProgram();

    // 1D strip of num_cores logical cores starting at (0,0)
    std::vector<CoreCoord> cores;
    cores.reserve(cfg.num_cores);
    for (uint32_t i = 0; i < cfg.num_cores; i++) {
        cores.push_back(linear_to_core(i));
    }
    CoreRange core_range(cores.front(), cores.back());

    // ── Semaphore ────────────────────────────────────────────
    auto sem_id = CreateSemaphore(program, core_range, 0);

    // ── Circular buffers ─────────────────────────────────────
    // TILE_SIZE_FP32 should come from tt_metal/api/tt-metalium/constants.hpp
    uint32_t tile_size = TILE_SIZE_FP32;  // if this fails, open that header and use the right constant

    auto make_cb = [&](uint32_t cb_id, uint32_t num_tiles = 1) {
        CircularBufferConfig cb_cfg(num_tiles * tile_size,
                                    {{cb_id, tt::DataFormat::Float32}});
        cb_cfg.set_page_size(cb_id, tile_size);
        return CreateCircularBuffer(program, core_range, cb_cfg);
    };

    make_cb(CB_LHS_R,     2);
    make_cb(CB_LHS_I,     2);
    make_cb(CB_RHS_R,     2);
    make_cb(CB_RHS_I,     2);
    make_cb(CB_TWIDDLE_R, 2);
    make_cb(CB_TWIDDLE_I, 2);
    make_cb(CB_OUT_R,     2);
    make_cb(CB_OUT_I,     2);
    make_cb(CB_SCRATCH_R, 1);
    make_cb(CB_SCRATCH_I, 1);
    make_cb(CB_SYNC,      1);
    make_cb(CB_TMP_R,     1);
    make_cb(CB_TMP_I,     1);
    make_cb(CB_WR_R,      1);
    make_cb(CB_WR_I,      1);

    // ── Kernel compilation ───────────────────────────────────
    std::vector<uint32_t> ct_reader  = { local_N, cfg.num_cores, num_stages, 0u };
    std::vector<uint32_t> ct_writer  = ct_reader;
    std::vector<uint32_t> ct_compute = {
        num_local_stg,
        num_noc_stg,
        cfg.is_ifft ? 1u : 0u,
        cfg.N
    };

    auto reader_kernel = CreateKernel(
        program,
        "kernels/fft_reader.cpp",
        core_range,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = ct_reader
        });

    auto writer_kernel = CreateKernel(
        program,
        "kernels/fft_writer.cpp",
        core_range,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_1,
            .noc          = NOC::RISCV_1_default,
            .compile_args = ct_writer
        });

    auto compute_kernel = CreateKernel(
        program,
        "kernels/fft_compute.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args     = ct_compute
        });

    // ── Per-core runtime args ────────────────────────────────
    for (uint32_t my_id = 0; my_id < cfg.num_cores; my_id++) {
        CoreCoord my_core = linear_to_core(my_id);

        uint32_t scratch_r_addr = GetCircularBufferL1Address(
            program, my_core, CB_SCRATCH_R);
        uint32_t scratch_i_addr = GetCircularBufferL1Address(
            program, my_core, CB_SCRATCH_I);
        uint32_t sem_addr = GetSemaphoreAddr(program, my_core, sem_id);

        // Reader args
        std::vector<uint32_t> reader_args = {
            input_buf.address(),
            input_buf.bank_id(0),      // matches your earlier codegen style
            twiddle_buf.address(),
            twiddle_buf.bank_id(0),
            local_N,
            my_id,
            cfg.N,
            num_local_stg,
            num_stages,
            0u
        };
        SetRuntimeArgs(program, reader_kernel, my_core, reader_args);

        // Writer args
        std::vector<uint32_t> writer_args = {
            GetCircularBufferL1Address(program, my_core, CB_LHS_R),
            GetCircularBufferL1Address(program, my_core, CB_LHS_I),
            scratch_r_addr,
            scratch_i_addr,
            output_buf.address(),
            output_buf.bank_id(0),
            cfg.num_cores,
            my_id,
            num_local_stg,
            sem_id,
        };

        // Peer table
        for (uint32_t dst = 0; dst < cfg.num_cores; dst++) {
            if (dst == my_id) continue;
            CoreCoord peer = linear_to_core(dst);

            auto peer_noc = device->worker_core_from_logical_core(peer);

            uint32_t peer_scratch_r = GetCircularBufferL1Address(
                program, peer, CB_SCRATCH_R);
            uint32_t peer_scratch_i = GetCircularBufferL1Address(
                program, peer, CB_SCRATCH_I);
            uint32_t peer_sem = GetSemaphoreAddr(program, peer, sem_id);

            writer_args.push_back(static_cast<uint32_t>(peer_noc.x));
            writer_args.push_back(static_cast<uint32_t>(peer_noc.y));
            writer_args.push_back(peer_scratch_r);
            writer_args.push_back(peer_scratch_i);
            writer_args.push_back(peer_sem);
        }

        SetRuntimeArgs(program, writer_kernel, my_core, writer_args);
        // compute kernel has only compile-time args
    }

    EnqueueProgram(cq, program, /*blocking*/ false);
    Finish(cq);
}

// ── Convenience wrappers ─────────────────────────────────────
void fft(Device* device, CommandQueue& cq,
         uint32_t N, uint32_t num_cores,
         Buffer& in, Buffer& out)
{
    run_fft(device, cq,
            {.N = N, .num_cores = num_cores,
             .is_ifft = false},
            in, out);
}

void ifft(Device* device, CommandQueue& cq,
          uint32_t N, uint32_t num_cores,
          Buffer& in, Buffer& out)
{
    run_fft(device, cq,
            {.N = N, .num_cores = num_cores,
             .is_ifft = true},
            in, out);
}