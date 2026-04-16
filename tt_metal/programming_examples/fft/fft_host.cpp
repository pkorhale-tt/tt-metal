// ============================================================
// fft_host.cpp — TT-Metalium confirmed API
// Fix: Don't allocate L1 buffers explicitly.
// CB addresses are deterministic: they start at L1_UNRESERVED_BASE
// and are laid out sequentially. We read the base from the HAL
// and compute offsets matching our CB allocation order.
// ============================================================

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include <cmath>
#include <vector>
#include <cassert>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt::constants;

enum CbId : uint32_t {
    CB_LHS_R=0, CB_LHS_I=1, CB_RHS_R=2, CB_RHS_I=3,
    CB_TWIDDLE_R=4, CB_TWIDDLE_I=5, CB_OUT_R=6, CB_OUT_I=7,
    CB_SCRATCH_R=8, CB_SCRATCH_I=9, CB_SYNC=10,
    CB_TMP_R=11, CB_TMP_I=12, CB_WR_R=13, CB_WR_I=14,
};

static constexpr uint32_t kTileSizeFp32 =
    tt::constants::TILE_HW * tt::constants::TILE_HW * sizeof(float);

// CB sizes in tiles matching allocation order below
// CBs 0-1 (LHS_R/I):     2 tiles each
// CBs 2-3 (RHS_R/I):     2 tiles each
// CBs 4-5 (TWIDDLE_R/I): 2 tiles each
// CBs 6-7 (OUT_R/I):     2 tiles each
// CBs 8-9 (SCRATCH_R/I): 1 tile each
// CBs 10-14:             1 tile each
static constexpr uint32_t kCbTiles[] = {2,2,2,2,2,2,2,2,1,1,1,1,1,1,1};

std::vector<float> precompute_twiddles(uint32_t N, uint32_t S, bool inv) {
    std::vector<float> tw; tw.reserve(S*(N/2)*2);
    for (uint32_t s=0; s<S; s++) {
        uint32_t stride=1u<<s, M=2*stride;
        for (uint32_t k=0; k<N/2; k++) {
            double a=-2.0*M_PI*(k%stride)/M; if(inv) a=-a;
            tw.push_back(float(std::cos(a)));
            tw.push_back(float(std::sin(a)));
        }
    }
    return tw;
}

CoreCoord linear_to_core(uint32_t id, uint32_t cols=8)
    { return {int(id%cols), int(id/cols)}; }

struct FFTConfig { uint32_t N, num_cores; bool is_ifft; };

static std::shared_ptr<MeshBuffer> make_mesh_buf(
    std::shared_ptr<MeshDevice> md, uint32_t size, uint32_t page_size=4)
{
    ReplicatedBufferConfig rep_cfg{.size=size};
    DeviceLocalBufferConfig dev_cfg{.page_size=page_size, .buffer_type=BufferType::DRAM};
    return MeshBuffer::create(rep_cfg, dev_cfg, md.get());
}

static uint32_t buf_addr(const std::shared_ptr<MeshBuffer>& mb) {
    return mb->get_device_buffer(MeshCoordinate(0,0))->address();
}

void run_fft(
    std::shared_ptr<MeshDevice> md,
    const FFTConfig& cfg,
    std::shared_ptr<MeshBuffer> input_buf,
    std::shared_ptr<MeshBuffer> output_buf)
{
    assert((cfg.N&(cfg.N-1))==0 && (cfg.num_cores&(cfg.num_cores-1))==0);
    assert(cfg.N % cfg.num_cores == 0);

    IDevice* device = md->get_device(0, 0);
    MeshCommandQueue& cq = md->mesh_command_queue();

    uint32_t local_N = cfg.N/cfg.num_cores;
    uint32_t S       = uint32_t(std::log2(cfg.N));
    uint32_t S_loc   = uint32_t(std::log2(local_N));
    uint32_t S_noc   = S - S_loc;

    // Twiddle DRAM buffer
    auto tw_floats = precompute_twiddles(cfg.N, S, cfg.is_ifft);
    auto tw_buf = make_mesh_buf(md, uint32_t(tw_floats.size()*sizeof(float)), sizeof(float));
    WriteShard(cq, tw_buf, tw_floats, MeshCoordinate(0,0), false);

    // Program
    Program prog = CreateProgram();
    std::vector<CoreCoord> cores;
    for (uint32_t i=0; i<cfg.num_cores; i++) cores.push_back(linear_to_core(i));
    CoreRange cr(cores.front(), cores.back());

    uint32_t sem_id = CreateSemaphore(prog, cr, 0u);

    // ── CB L1 address computation ─────────────────────────────
    // CBs are allocated sequentially from L1_UNRESERVED_BASE.
    // We compute each CB's address as cumulative sum of previous CB sizes.
    // This matches what the runtime does internally.
    uint32_t cb_base = hal.get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);

    // Compute cumulative offsets
    uint32_t cb_offsets[15];
    cb_offsets[0] = 0;
    for (int i=1; i<15; i++)
        cb_offsets[i] = cb_offsets[i-1] + kCbTiles[i-1] * kTileSizeFp32;

    uint32_t lhs_r_addr = cb_base + cb_offsets[CB_LHS_R];
    uint32_t lhs_i_addr = cb_base + cb_offsets[CB_LHS_I];
    uint32_t scr_r_addr = cb_base + cb_offsets[CB_SCRATCH_R];
    uint32_t scr_i_addr = cb_base + cb_offsets[CB_SCRATCH_I];

    // ── Create CBs ────────────────────────────────────────────
    auto make_cb = [&](uint32_t id, uint32_t ntiles=1) {
        CircularBufferConfig c(ntiles*kTileSizeFp32,{{id,tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        return CreateCircularBuffer(prog, cr, c);
    };

    for (uint32_t id=0; id<15; id++) make_cb(id, kCbTiles[id]);

    // ── Kernels ───────────────────────────────────────────────
    std::vector<uint32_t> ct_rw={local_N,cfg.num_cores,S,0u};
    std::vector<uint32_t> ct_c={S_loc,S_noc,cfg.is_ifft?1u:0u,cfg.N};

    auto rk=CreateKernel(prog,"kernels/fft_reader.cpp",cr,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                           .noc=NOC::RISCV_0_default,.compile_args=ct_rw});
    auto wk=CreateKernel(prog,"kernels/fft_writer.cpp",cr,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,
                           .noc=NOC::RISCV_1_default,.compile_args=ct_rw});
    CreateKernel(prog,"kernels/fft_compute.cpp",cr,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                      .fp32_dest_acc_en=true,.compile_args=ct_c});

    // ── Runtime args ──────────────────────────────────────────
    for (uint32_t my=0; my<cfg.num_cores; my++) {
        CoreCoord mc=linear_to_core(my);
        SetRuntimeArgs(prog, rk, mc, {
            buf_addr(input_buf),0u,buf_addr(tw_buf),0u,
            local_N,my,cfg.N,S_loc,S,0u});
        std::vector<uint32_t> wa={
            lhs_r_addr,lhs_i_addr,scr_r_addr,scr_i_addr,
            buf_addr(output_buf),0u,
            cfg.num_cores,my,S_loc,sem_id};
        for (uint32_t dst=0; dst<cfg.num_cores; dst++) {
            if (dst==my) continue;
            CoreCoord pn=device->worker_core_from_logical_core(linear_to_core(dst));
            wa.push_back(uint32_t(pn.x)); wa.push_back(uint32_t(pn.y));
            wa.push_back(scr_r_addr); wa.push_back(scr_i_addr);
            wa.push_back(sem_id);
        }
        SetRuntimeArgs(prog, wk, mc, wa);
    }

    // ── Dispatch ──────────────────────────────────────────────
    MeshWorkload workload;
    workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0,0), MeshCoordinate(0,0)),
        std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    cq.finish();
}

void fft(std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t nc,
         std::shared_ptr<MeshBuffer> in, std::shared_ptr<MeshBuffer> out)
{ run_fft(md,{N,nc,false},in,out); }

void ifft(std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t nc,
          std::shared_ptr<MeshBuffer> in, std::shared_ptr<MeshBuffer> out)
{ run_fft(md,{N,nc,true},in,out); }