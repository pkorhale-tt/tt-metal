// ============================================================
// fft_host.cpp — TT-Metalium confirmed API
//
// Confirmed from headers:
//   MeshDevice::create_unit_mesh(device_id) → shared_ptr<MeshDevice>
//   mesh_device->mesh_command_queue()       → MeshCommandQueue&
//   mesh_device->get_device(0,0)            → IDevice*
//   Synchronize(MeshDevice*, optional<cq_id>) → blocks until done
//   EnqueueWriteMeshBuffer/EnqueueReadMeshBuffer → MeshBuffer only
//
//   For plain Buffer I/O: use PushCurrentCommandQueueIdForThread +
//   the IDevice-level enqueue functions via host_api.hpp
//   (EnqueueWriteBuffer/ReadBuffer on IDevice take thread-local cq_id)
//
//   EnqueueProgram: not in distributed.hpp for plain Program.
//   Use PushCurrentCommandQueueIdForThread + IDevice path.
// ============================================================

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"

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

static std::shared_ptr<Buffer> make_dram(IDevice* dev, uint32_t sz, uint32_t pg=4) {
    return CreateBuffer(InterleavedBufferConfig{
        .device=dev, .size=sz, .page_size=pg, .buffer_type=BufferType::DRAM});
}

// ── Write/Read helpers using thread-local CQ ─────────────────
// PushCurrentCommandQueueIdForThread sets the cq used by
// IDevice-level Enqueue* calls in host_api.hpp.
static void write_buf(IDevice* dev, std::shared_ptr<Buffer> buf,
                      const void* data, uint8_t cq_id=0) {
    PushCurrentCommandQueueIdForThread(cq_id);
    EnqueueWriteBuffer(*dev, buf, data, false);
    PopCurrentCommandQueueIdForThread();
}
static void read_buf(IDevice* dev, std::shared_ptr<Buffer> buf,
                     void* data, uint8_t cq_id=0) {
    PushCurrentCommandQueueIdForThread(cq_id);
    EnqueueReadBuffer(*dev, buf, data, true);   // blocking=true
    PopCurrentCommandQueueIdForThread();
}
static void enqueue_prog(IDevice* dev, Program& prog,
                         bool blocking, uint8_t cq_id=0) {
    PushCurrentCommandQueueIdForThread(cq_id);
    EnqueueProgram(*dev, prog, blocking);
    PopCurrentCommandQueueIdForThread();
}

// ── Main FFT builder ──────────────────────────────────────────
void run_fft(
    std::shared_ptr<MeshDevice> mesh_device,
    const FFTConfig& cfg,
    std::shared_ptr<Buffer> input_buf,
    std::shared_ptr<Buffer> output_buf,
    uint8_t cq_id = 0)
{
    assert((cfg.N&(cfg.N-1))==0 && (cfg.num_cores&(cfg.num_cores-1))==0);
    assert(cfg.N % cfg.num_cores == 0);

    IDevice* device = mesh_device->get_device(0, 0);

    uint32_t local_N = cfg.N/cfg.num_cores;
    uint32_t S       = uint32_t(std::log2(cfg.N));
    uint32_t S_loc   = uint32_t(std::log2(local_N));
    uint32_t S_noc   = S - S_loc;

    // Twiddle buffer
    auto tw = precompute_twiddles(cfg.N, S, cfg.is_ifft);
    auto tw_buf = make_dram(device, uint32_t(tw.size()*sizeof(float)), sizeof(float));
    write_buf(device, tw_buf, tw.data(), cq_id);

    // Program
    Program prog = CreateProgram();
    std::vector<CoreCoord> cores;
    for (uint32_t i=0; i<cfg.num_cores; i++) cores.push_back(linear_to_core(i));
    CoreRange cr(cores.front(), cores.back());

    uint32_t sem_id = CreateSemaphore(prog, cr, 0u);

    auto make_cb = [&](uint32_t id, uint32_t n=1) -> CBHandle {
        CircularBufferConfig c(n*kTileSizeFp32,{{id,tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        return CreateCircularBuffer(prog, cr, c);
    };
    CBHandle h_lr=make_cb(CB_LHS_R,2), h_li=make_cb(CB_LHS_I,2);
    make_cb(CB_RHS_R,2); make_cb(CB_RHS_I,2);
    make_cb(CB_TWIDDLE_R,2); make_cb(CB_TWIDDLE_I,2);
    make_cb(CB_OUT_R,2); make_cb(CB_OUT_I,2);
    CBHandle h_sr=make_cb(CB_SCRATCH_R), h_si=make_cb(CB_SCRATCH_I);
    make_cb(CB_SYNC); make_cb(CB_TMP_R); make_cb(CB_TMP_I);
    make_cb(CB_WR_R); make_cb(CB_WR_I);

    auto cb_addr = [&](CBHandle h) -> uint32_t {
        auto opt=GetCircularBufferConfig(prog,h).globally_allocated_address();
        TT_FATAL(opt.has_value(),"CB not allocated"); return opt.value();
    };
    uint32_t scr_r=cb_addr(h_sr), scr_i=cb_addr(h_si);
    uint32_t lhs_r=cb_addr(h_lr), lhs_i=cb_addr(h_li);

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

    for (uint32_t my=0; my<cfg.num_cores; my++) {
        CoreCoord mc=linear_to_core(my);
        SetRuntimeArgs(prog,rk,mc,{
            input_buf->address(),0u,tw_buf->address(),0u,
            local_N,my,cfg.N,S_loc,S,0u});
        std::vector<uint32_t> wa={
            lhs_r,lhs_i,scr_r,scr_i,
            output_buf->address(),0u,
            cfg.num_cores,my,S_loc,sem_id};
        for (uint32_t dst=0; dst<cfg.num_cores; dst++) {
            if (dst==my) continue;
            CoreCoord pn=device->worker_core_from_logical_core(linear_to_core(dst));
            wa.push_back(uint32_t(pn.x)); wa.push_back(uint32_t(pn.y));
            wa.push_back(scr_r); wa.push_back(scr_i); wa.push_back(sem_id);
        }
        SetRuntimeArgs(prog,wk,mc,wa);
    }

    enqueue_prog(device, prog, false, cq_id);
    // Synchronize(MeshDevice*, optional<cq_id>) is the confirmed Finish equivalent
    Synchronize(mesh_device.get(), cq_id);
}

void fft(std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t nc,
         std::shared_ptr<Buffer> in, std::shared_ptr<Buffer> out)
{ run_fft(md,{N,nc,false},in,out); }

void ifft(std::shared_ptr<MeshDevice> md, uint32_t N, uint32_t nc,
          std::shared_ptr<Buffer> in, std::shared_ptr<Buffer> out)
{ run_fft(md,{N,nc,true},in,out); }