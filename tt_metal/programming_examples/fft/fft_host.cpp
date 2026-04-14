// ============================================================
//  fft_host.cpp  –  host-side program setup
//
//  Creates the TT-Metal program for multi-core 1D FFT.
//  Handles:
//    - Twiddle factor precomputation and DRAM upload
//    - CB allocation per core (local + scratch + tmp)
//    - Semaphore allocation
//    - Kernel compilation with compile-time args
//    - Per-core runtime arg construction (including peer tables)
//    - Program dispatch and result readback
//
//  Supports both fp32 and bfloat16.
//  Forward FFT and inverse FFT (is_ifft=true).
// ============================================================

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/common/util.hpp"
#include "tt_metal/common/bfloat16.hpp"

#include <cmath>
#include <vector>
#include <cassert>
#include <stdexcept>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// ── Twiddle factor precomputation ───────────────────────────
// Returns a flat vector of [real_0, imag_0, real_1, imag_1, ...]
// for all stages.  Layout: stage s, index k → offset (s*(N/2) + k)*2
//
// Forward FFT: W_N^k = exp(-j*2*pi*k/N) = cos(2*pi*k/N) - j*sin(2*pi*k/N)
// Inverse FFT: W_N^k = exp(+j*2*pi*k/N)  (conjugate)
std::vector<float> precompute_twiddles(
    uint32_t N, uint32_t num_stages, bool is_ifft)
{
    std::vector<float> tw;
    tw.reserve(num_stages * (N / 2) * 2);

    for (uint32_t s = 0; s < num_stages; s++) {
        uint32_t stride = 1u << s;
        uint32_t M      = 2 * stride;  // butterfly group size at stage s

        for (uint32_t k = 0; k < N / 2; k++) {
            uint32_t kk = k % stride;   // twiddle index within group
            double angle = -2.0 * M_PI * kk / M;
            if (is_ifft) angle = -angle;

            tw.push_back(static_cast<float>(std::cos(angle)));  // real
            tw.push_back(static_cast<float>(std::sin(angle)));  // imag
        }
    }
    return tw;
}

// ── Convert float vector to bf16 for upload ─────────────────
std::vector<bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<bfloat16> out;
    out.reserve(v.size());
    for (float f : v) out.push_back(bfloat16(f));
    return out;
}

// ── Core grid helpers ────────────────────────────────────────
// Map linear core id → (col, row) on Wormhole n300 grid.
// Wormhole has usable Tensix cores starting at (1,1) in NOC coords.
// For simplicity we use a 1D row of cores here.
CoreCoord linear_to_core(uint32_t id, uint32_t grid_cols = 8) {
    return {id % grid_cols, id / grid_cols};
}

uint32_t core_to_linear(CoreCoord c, uint32_t grid_cols = 8) {
    return c.y * grid_cols + c.x;
}

// ── Main FFT program builder ─────────────────────────────────
struct FFTConfig {
    uint32_t N;           // total FFT size (power of 2)
    uint32_t num_cores;   // must divide N evenly, power of 2
    bool     is_ifft;     // true = inverse FFT
    bool     use_bf16;    // true = bfloat16, false = fp32
};

void run_fft(
    Device* device,
    CommandQueue& cq,
    const FFTConfig& cfg,
    Buffer& input_buf,    // DRAM: interleaved [r0,i0,r1,i1,...] fp32 or bf16
    Buffer& output_buf)   // DRAM: same layout
{
    assert((cfg.N & (cfg.N - 1)) == 0 && "N must be power of 2");
    assert((cfg.num_cores & (cfg.num_cores - 1)) == 0);
    assert(cfg.N % cfg.num_cores == 0);

    uint32_t local_N        = cfg.N / cfg.num_cores;
    uint32_t num_stages     = static_cast<uint32_t>(std::log2(cfg.N));
    uint32_t num_local_stg  = static_cast<uint32_t>(std::log2(local_N));
    uint32_t num_noc_stg    = num_stages - num_local_stg;
    uint32_t elem_bytes     = cfg.use_bf16 ? 2 : 4;
    uint32_t buf_bytes      = local_N * elem_bytes;  // per CB, real or imag

    // ── Twiddle DRAM buffer ──────────────────────────────────
    auto tw_floats = precompute_twiddles(cfg.N, num_stages, cfg.is_ifft);
    uint32_t tw_bytes = tw_floats.size() * elem_bytes;

    auto twiddle_buf = CreateBuffer(device, {
        .size       = tw_bytes,
        .page_size  = elem_bytes,
        .buffer_type = BufferType::DRAM,
    });

    if (cfg.use_bf16) {
        auto tw_bf16 = to_bf16(tw_floats);
        EnqueueWriteBuffer(cq, twiddle_buf,
            reinterpret_cast<const void*>(tw_bf16.data()), false);
    } else {
        EnqueueWriteBuffer(cq, twiddle_buf,
            reinterpret_cast<const void*>(tw_floats.data()), false);
    }

    // ── Build program ────────────────────────────────────────
    Program program = CreateProgram();

    // Core range: 1D strip of num_cores cores starting at (1,1)
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < cfg.num_cores; i++)
        cores.push_back(linear_to_core(i));
    CoreRange core_range(cores.front(), cores.back());

    // ── Semaphore (one per core, shared address space) ───────
    // Each core's semaphore counts incoming NOC signals.
    // Initialized to 0; reset to 0 after each NOC stage by writer.
    auto sem_id = CreateSemaphore(program, core_range, 0);

    // ── Circular buffers ─────────────────────────────────────
    // We need to be careful about sizes:
    //   - Local data CBs: 1 tile = local_N elements real or imag
    //   - Scratch CBs: same size (receives from one partner per stage)
    //   - Sync CB: 1 element (just a signal)

    uint32_t tile_size = cfg.use_bf16 ? TILE_SIZE_BF16 : TILE_SIZE_FP32;

    auto make_cb = [&](uint32_t cb_id, uint32_t num_tiles = 1) {
        CircularBufferConfig cb_cfg(num_tiles * tile_size,
            {{cb_id, cfg.use_bf16 ? tt::DataFormat::BFloat16
                                  : tt::DataFormat::Float32}});
        cb_cfg.set_page_size(cb_id, tile_size);
        return CreateCircularBuffer(program, core_range, cb_cfg);
    };

    // Data CBs (double-buffered for pipelining: 2 tiles each)
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
    // Temporaries for butterfly intermediate results
    make_cb(11 /*CB_TMP_R*/, 1);
    make_cb(12 /*CB_TMP_I*/, 1);
    make_cb(13 /*CB_WR_R*/,  1);
    make_cb(14 /*CB_WR_I*/,  1);

    // ── Kernel compilation ───────────────────────────────────
    // Compile-time args are baked in at JIT compile time.
    std::vector<uint32_t> ct_reader = {
        local_N, cfg.num_cores, num_stages, cfg.use_bf16 ? 1u : 0u
    };
    std::vector<uint32_t> ct_writer = ct_reader;
    std::vector<uint32_t> ct_compute = {
        num_local_stg, num_noc_stg,
        cfg.is_ifft ? 1u : 0u,
        cfg.N
    };

    auto reader_kernel = CreateKernel(
        program,
        "kernels/fft_reader.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default,
            .compile_args = ct_reader
        });

    auto writer_kernel = CreateKernel(
        program,
        "kernels/fft_writer.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc       = NOC::RISCV_1_default,
            .compile_args = ct_writer
        });

    auto compute_kernel = CreateKernel(
        program,
        "kernels/fft_compute.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = !cfg.use_bf16,
            .compile_args = ct_compute
        });

    // ── Per-core runtime args ────────────────────────────────
    for (uint32_t my_id = 0; my_id < cfg.num_cores; my_id++) {
        CoreCoord my_core = linear_to_core(my_id);

        // Get L1 addresses of this core's CBs
        uint32_t scratch_r_addr = GetCircularBufferL1Address(
            program, my_core, CB_SCRATCH_R);
        uint32_t scratch_i_addr = GetCircularBufferL1Address(
            program, my_core, CB_SCRATCH_I);
        uint32_t sem_addr = GetSemaphoreAddr(program, my_core, sem_id);

        // ── Reader args ──────────────────────────────────────
        std::vector<uint32_t> reader_args = {
            input_buf.address(),
            input_buf.bank_id(0),
            twiddle_buf.address(),
            twiddle_buf.bank_id(0),
            local_N,
            my_id,
            cfg.N,
            num_local_stg,
            num_stages,
            cfg.use_bf16 ? 1u : 0u
        };
        SetRuntimeArgs(program, reader_kernel, my_core, reader_args);

        // ── Writer args ──────────────────────────────────────
        std::vector<uint32_t> writer_args = {
            GetCircularBufferL1Address(program, my_core, CB_LHS_R),
            GetCircularBufferL1Address(program, my_core, CB_LHS_I),
            scratch_r_addr,
            scratch_i_addr,
            output_buf.address(),      // reused RT_TWIDDLE_DRAM slot
            output_buf.bank_id(0),
            cfg.num_cores,
            my_id,
            num_local_stg,             // first_noc_stage
            sem_id,
        };

        // Build peer table: all cores except myself
        // Order: 0,1,...,my_id-1, my_id+1,...,num_cores-1
        for (uint32_t dst = 0; dst < cfg.num_cores; dst++) {
            if (dst == my_id) continue;
            CoreCoord peer = linear_to_core(dst);

            // Get peer's NOC coordinates (physical, not logical)
            auto peer_noc = device->worker_core_from_logical_core(peer);

            uint32_t peer_scratch_r = GetCircularBufferL1Address(
                program, peer, CB_SCRATCH_R);
            uint32_t peer_scratch_i = GetCircularBufferL1Address(
                program, peer, CB_SCRATCH_I);
            uint32_t peer_sem = GetSemaphoreAddr(program, peer, sem_id);

            writer_args.push_back(peer_noc.x);
            writer_args.push_back(peer_noc.y);
            writer_args.push_back(peer_scratch_r);
            writer_args.push_back(peer_scratch_i);
            writer_args.push_back(peer_sem);
        }

        SetRuntimeArgs(program, writer_kernel, my_core, writer_args);

        // ── Compute args ─────────────────────────────────────
        // Compute kernel uses only compile-time args for this design.
        // (No per-core runtime args needed for compute.)
    }

    // ── Dispatch ─────────────────────────────────────────────
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Output is now in output_buf on DRAM.
    // Caller reads it back with EnqueueReadBuffer.
}

// ── Convenience wrapper ──────────────────────────────────────
void fft(Device* device, CommandQueue& cq,
         uint32_t N, uint32_t num_cores,
         Buffer& in, Buffer& out, bool use_bf16 = false)
{
    run_fft(device, cq,
        {.N = N, .num_cores = num_cores,
         .is_ifft = false, .use_bf16 = use_bf16},
        in, out);
}

void ifft(Device* device, CommandQueue& cq,
          uint32_t N, uint32_t num_cores,
          Buffer& in, Buffer& out, bool use_bf16 = false)
{
    run_fft(device, cq,
        {.N = N, .num_cores = num_cores,
         .is_ifft = true, .use_bf16 = use_bf16},
        in, out);
}
