// ============================================================
// fft_host.cpp – corrected for current TT-Metalium API
// All fixes derived from actual compiler error messages.
// ============================================================

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
// CommandQueue lives here in current TT-Metal:
#include "tt_metal/api/tt-metalium/command_queue.hpp"

#include <cmath>
#include <vector>
#include <cassert>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// ── CB IDs ───────────────────────────────────────────────────
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

static constexpr uint32_t kTileSizeFp32 =
    tt::constants::TILE_HW * tt::constants::TILE_HW * sizeof(float);

// ── Twiddle precomputation ────────────────────────────────────
std::vector<float> precompute_twiddles(
    uint32_t N, uint32_t num_stages, bool is_ifft)
{
    std::vector<float> tw;
    tw.reserve(num_stages * (N / 2) * 2);
    for (uint32_t s = 0; s < num_stages; s++) {
        uint32_t stride = 1u << s;
        uint32_t M      = 2 * stride;
        for (uint32_t k = 0; k < N / 2; k++) {
            uint32_t kk    = k % stride;
            double   angle = -2.0 * M_PI * kk / M;
            if (is_ifft) angle = -angle;
            tw.push_back(static_cast<float>(std::cos(angle)));
            tw.push_back(static_cast<float>(std::sin(angle)));
        }
    }
    return tw;
}

CoreCoord linear_to_core(uint32_t id, uint32_t grid_cols = 8) {
    return {static_cast<int>(id % grid_cols),
            static_cast<int>(id / grid_cols)};
}

struct FFTConfig {
    uint32_t N;
    uint32_t num_cores;
    bool     is_ifft;
};

void run_fft(
    IDevice*      device,
    CommandQueue& cq,
    const FFTConfig& cfg,
    std::shared_ptr<Buffer> input_buf,
    std::shared_ptr<Buffer> output_buf)
{
    assert((cfg.N & (cfg.N - 1)) == 0);
    assert((cfg.num_cores & (cfg.num_cores - 1)) == 0);
    assert(cfg.N % cfg.num_cores == 0);

    uint32_t local_N       = cfg.N / cfg.num_cores;
    uint32_t num_stages    = static_cast<uint32_t>(std::log2(cfg.N));
    uint32_t num_local_stg = static_cast<uint32_t>(std::log2(local_N));
    uint32_t num_noc_stg   = num_stages - num_local_stg;

    // ── Twiddle DRAM buffer ───────────────────────────────────
    // FIX: CreateBuffer takes InterleavedBufferConfig{device, size, page_size, type}
    // NOT CreateBuffer(device*, {.size=...})
    auto tw_floats = precompute_twiddles(cfg.N, num_stages, cfg.is_ifft);
    uint32_t tw_bytes = static_cast<uint32_t>(tw_floats.size()) * sizeof(float);

    auto twiddle_buf = CreateBuffer(InterleavedBufferConfig{
        .device      = device,
        .size        = tw_bytes,
        .page_size   = static_cast<uint32_t>(sizeof(float)),
        .buffer_type = BufferType::DRAM,
    });
    EnqueueWriteBuffer(cq, twiddle_buf,
                       reinterpret_cast<const void*>(tw_floats.data()),
                       false);

    // ── Program + core range ──────────────────────────────────
    Program program = CreateProgram();

    std::vector<CoreCoord> cores;
    cores.reserve(cfg.num_cores);
    for (uint32_t i = 0; i < cfg.num_cores; i++)
        cores.push_back(linear_to_core(i));
    CoreRange core_range(cores.front(), cores.back());

    // ── Semaphore ─────────────────────────────────────────────
    // CreateSemaphore returns uint32_t semaphore ID.
    // To get the L1 address we use the HAL:
    //   hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::SEMAPHORE)
    //   + sem_id * hal.get_alignment(HalMemType::L1)
    // But for passing to kernels as runtime args, we pass sem_id directly
    // and let the kernel resolve via get_semaphore(sem_id) on-device.
    uint32_t sem_id = CreateSemaphore(program, core_range, 0u);

    // ── Circular buffers ──────────────────────────────────────
    auto make_cb = [&](uint32_t cb_id, uint32_t num_tiles = 1) -> CBHandle {
        CircularBufferConfig cb_cfg(
            num_tiles * kTileSizeFp32,
            {{cb_id, tt::DataFormat::Float32}});
        cb_cfg.set_page_size(cb_id, kTileSizeFp32);
        return CreateCircularBuffer(program, core_range, cb_cfg);
    };

    CBHandle h_lhs_r     = make_cb(CB_LHS_R,     2);
    CBHandle h_lhs_i     = make_cb(CB_LHS_I,     2);
    make_cb(CB_RHS_R,     2);  make_cb(CB_RHS_I,     2);
    make_cb(CB_TWIDDLE_R, 2);  make_cb(CB_TWIDDLE_I, 2);
    make_cb(CB_OUT_R,     2);  make_cb(CB_OUT_I,     2);
    CBHandle h_scratch_r = make_cb(CB_SCRATCH_R, 1);
    CBHandle h_scratch_i = make_cb(CB_SCRATCH_I, 1);
    make_cb(CB_SYNC,  1);
    make_cb(CB_TMP_R, 1);  make_cb(CB_TMP_I, 1);
    make_cb(CB_WR_R,  1);  make_cb(CB_WR_I,  1);

    // FIX: .locally_allocated_address → .globally_allocated_address()
    // (it's a method, not a field, and the name changed)
    auto cb_l1_addr = [&](CBHandle h) -> uint32_t {
        auto opt = GetCircularBufferConfig(program, h).globally_allocated_address();
        TT_FATAL(opt.has_value(), "CB not allocated");
        return opt.value();
    };

    uint32_t scratch_r_addr = cb_l1_addr(h_scratch_r);
    uint32_t scratch_i_addr = cb_l1_addr(h_scratch_i);
    uint32_t lhs_r_addr     = cb_l1_addr(h_lhs_r);
    uint32_t lhs_i_addr     = cb_l1_addr(h_lhs_i);

    // ── Kernels ───────────────────────────────────────────────
    std::vector<uint32_t> ct_reader  = { local_N, cfg.num_cores, num_stages, 0u };
    std::vector<uint32_t> ct_writer  = ct_reader;
    std::vector<uint32_t> ct_compute = {
        num_local_stg, num_noc_stg, cfg.is_ifft ? 1u : 0u, cfg.N
    };

    auto reader_kernel = CreateKernel(
        program, "kernels/fft_reader.cpp", core_range,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_0,
            .noc          = NOC::RISCV_0_default,
            .compile_args = ct_reader });

    auto writer_kernel = CreateKernel(
        program, "kernels/fft_writer.cpp", core_range,
        DataMovementConfig{
            .processor    = DataMovementProcessor::RISCV_1,
            .noc          = NOC::RISCV_1_default,
            .compile_args = ct_writer });

    CreateKernel(
        program, "kernels/fft_compute.cpp", core_range,
        ComputeConfig{
            .math_fidelity    = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args     = ct_compute });

    // ── Per-core runtime args ─────────────────────────────────
    for (uint32_t my_id = 0; my_id < cfg.num_cores; my_id++) {
        CoreCoord my_core = linear_to_core(my_id);

        // FIX: GetSemaphoreAddress doesn't exist.
        // Pass sem_id as runtime arg; kernel calls get_semaphore(sem_id)
        // which resolves to the correct L1 address on-device.
        // This is the standard pattern used by all TT-Metal examples.

        std::vector<uint32_t> reader_args = {
            input_buf->address(),
            0u,                     // padding (bank_id removed)
            twiddle_buf->address(),
            0u,                     // padding
            local_N,
            my_id,
            cfg.N,
            num_local_stg,
            num_stages,
            0u                      // use_bf16 = false
        };
        SetRuntimeArgs(program, reader_kernel, my_core, reader_args);

        std::vector<uint32_t> writer_args = {
            lhs_r_addr,
            lhs_i_addr,
            scratch_r_addr,
            scratch_i_addr,
            output_buf->address(),
            0u,                     // padding
            cfg.num_cores,
            my_id,
            num_local_stg,
            sem_id,                 // kernel resolves via get_semaphore(sem_id)
        };

        // Peer table: [noc_x, noc_y, scratch_r, scratch_i, sem_id]
        for (uint32_t dst = 0; dst < cfg.num_cores; dst++) {
            if (dst == my_id) continue;
            CoreCoord peer_logical = linear_to_core(dst);
            CoreCoord peer_noc     =
                device->worker_core_from_logical_core(peer_logical);

            writer_args.push_back(static_cast<uint32_t>(peer_noc.x));
            writer_args.push_back(static_cast<uint32_t>(peer_noc.y));
            writer_args.push_back(scratch_r_addr);
            writer_args.push_back(scratch_i_addr);
            writer_args.push_back(sem_id);  // each peer resolves its own address
        }

        SetRuntimeArgs(program, writer_kernel, my_core, writer_args);
    }

    EnqueueProgram(cq, program, false);
    Finish(cq);
}

// ── Convenience wrappers ──────────────────────────────────────
void fft(IDevice* device, CommandQueue& cq,
         uint32_t N, uint32_t num_cores,
         std::shared_ptr<Buffer> in,
         std::shared_ptr<Buffer> out)
{
    run_fft(device, cq, {N, num_cores, false}, in, out);
}

void ifft(IDevice* device, CommandQueue& cq,
          uint32_t N, uint32_t num_cores,
          std::shared_ptr<Buffer> in,
          std::shared_ptr<Buffer> out)
{
    run_fft(device, cq, {N, num_cores, true}, in, out);
}