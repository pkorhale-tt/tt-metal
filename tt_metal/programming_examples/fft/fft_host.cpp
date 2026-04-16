// ============================================================
// fft_host.cpp — TT-Metalium API (MeshDevice path)
// Buffer layout matching kernel expectations:
//   Input:   tiles [0..C-1] = real per core, [C..2C-1] = imag per core
//   Output:  same layout
//   Twiddle: tiles [0..S*C-1] = real, [S*C..2*S*C-1] = imag
//            where S=num_stages, C=num_cores
// ============================================================

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"

#include <cmath>
#include <vector>
#include <cassert>
#include <complex>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace tt::constants;

// CB IDs
enum CbId : uint32_t {
    CB_LHS_R=0, CB_LHS_I=1, CB_RHS_R=2, CB_RHS_I=3,
    CB_TWIDDLE_R=4, CB_TWIDDLE_I=5, CB_OUT_R=6, CB_OUT_I=7,
    CB_SCRATCH_R=8, CB_SCRATCH_I=9, CB_SYNC=10,
    CB_TMP_R=11, CB_TMP_I=12, CB_WR_R=13, CB_WR_I=14,
};

// RT arg indices (must match fft_common.h)
enum RtArg : uint32_t {
    RT_CB_R=0, RT_CB_I=1, RT_SCRATCH_R=2, RT_SCRATCH_I=3,
    RT_TWIDDLE_DRAM=4, RT_TWIDDLE_BANK=5,
    RT_NUM_CORES=6, RT_MY_CORE_ID=7, RT_FIRST_NOC_STG=8, RT_SEM_ID=9,
    RT_PEER_BASE=10,
};

// CT arg indices
enum CtArg : uint32_t { CT_LOCAL_N=0, CT_NUM_CORES=1, CT_NUM_STAGES=2, CT_USE_BF16=3 };

static constexpr uint32_t kTileHW      = tt::constants::TILE_HW;
static constexpr uint32_t kTileSizeFp32 = kTileHW * kTileHW * sizeof(float);
static constexpr uint32_t kCbTiles[]   = {2,2,2,2,2,2,2,2,1,1,1,1,1,1,1};

// ── Twiddle precomputation ────────────────────────────────────
// Returns tiles for all stages, real then imag.
// Layout: real[stage*C + core_id], imag[S*C + stage*C + core_id]
// Each tile holds the twiddle factors for that core's elements at that stage.
std::vector<float> precompute_twiddles_tiled(
    uint32_t N, uint32_t num_cores, uint32_t num_stages, bool is_ifft)
{
    uint32_t local_N   = N / num_cores;
    uint32_t tile_elems = kTileHW * kTileHW; // elements per tile = 1024
    // Each core gets one tile per stage
    uint32_t total_tiles = num_stages * num_cores * 2; // *2 for real+imag
    std::vector<float> tw(total_tiles * tile_elems, 0.0f);

    for (uint32_t s = 0; s < num_stages; s++) {
        uint32_t stride = 1u << s;
        uint32_t M      = 2 * stride;
        for (uint32_t c = 0; c < num_cores; c++) {
            uint32_t global_offset = c * local_N;
            uint32_t tile_r_idx    = s * num_cores + c;
            uint32_t tile_i_idx    = num_stages * num_cores + tile_r_idx;
            float* tile_r = tw.data() + tile_r_idx * tile_elems;
            float* tile_i = tw.data() + tile_i_idx * tile_elems;
            for (uint32_t i = 0; i < local_N / 2 && i < tile_elems; i++) {
                uint32_t lo    = i; // simplified: twiddle for element i
                uint32_t k     = (global_offset + lo) % stride;
                double   angle = -2.0 * M_PI * k / M;
                if (is_ifft) angle = -angle;
                tile_r[i] = float(std::cos(angle));
                tile_i[i] = float(std::sin(angle));
            }
        }
    }
    return tw;
}

CoreCoord linear_to_core(uint32_t id, uint32_t cols=8)
    { return {int(id%cols), int(id/cols)}; }

struct FFTConfig { uint32_t N, num_cores; bool is_ifft; };

static std::shared_ptr<MeshBuffer> make_mesh_buf(
    std::shared_ptr<MeshDevice> md, uint32_t size, uint32_t page_size)
{
    ReplicatedBufferConfig rep{.size=size};
    DeviceLocalBufferConfig dev{.page_size=page_size, .buffer_type=BufferType::DRAM};
    return MeshBuffer::create(rep, dev, md.get());
}

static uint32_t buf_addr(const std::shared_ptr<MeshBuffer>& mb)
    { return mb->get_device_buffer(MeshCoordinate(0,0))->address(); }

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

    uint32_t local_N  = cfg.N / cfg.num_cores;
    uint32_t S        = uint32_t(std::log2(cfg.N));
    uint32_t S_loc    = uint32_t(std::log2(local_N));
    uint32_t S_noc    = S - S_loc;

    // Twiddle buffer
    auto tw_data = precompute_twiddles_tiled(cfg.N, cfg.num_cores, S, cfg.is_ifft);
    uint32_t tw_tile_size = kTileSizeFp32;
    uint32_t tw_bytes     = uint32_t(tw_data.size() * sizeof(float));
    auto tw_buf = make_mesh_buf(md, tw_bytes, tw_tile_size);
    WriteShard(cq, tw_buf, tw_data, MeshCoordinate(0,0), false);

    // Program
    Program prog = CreateProgram();
    std::vector<CoreCoord> cores;
    for (uint32_t i=0; i<cfg.num_cores; i++) cores.push_back(linear_to_core(i));
    CoreRange cr(cores.front(), cores.back());

    uint32_t sem_id = CreateSemaphore(prog, cr, 0u);

    // CBs
    auto make_cb = [&](uint32_t id, uint32_t n=1) {
        CircularBufferConfig c(n*kTileSizeFp32,{{id,tt::DataFormat::Float32}});
        c.set_page_size(id, kTileSizeFp32);
        return CreateCircularBuffer(prog, cr, c);
    };
    for (uint32_t id=0; id<15; id++) make_cb(id, kCbTiles[id]);

    // CB base address for NOC scratch passing
    CoreCoord vc0 = device->worker_core_from_logical_core(linear_to_core(0));
    uint32_t cb_base = uint32_t(
        device->get_dev_addr(vc0, HalL1MemAddrType::DEFAULT_UNRESERVED));
    uint32_t cb_offsets[15];
    cb_offsets[0] = 0;
    for (int i=1; i<15; i++)
        cb_offsets[i] = cb_offsets[i-1] + kCbTiles[i-1]*kTileSizeFp32;

    uint32_t lhs_r_addr = cb_base + cb_offsets[CB_LHS_R];
    uint32_t lhs_i_addr = cb_base + cb_offsets[CB_LHS_I];
    uint32_t scr_r_addr = cb_base + cb_offsets[CB_SCRATCH_R];
    uint32_t scr_i_addr = cb_base + cb_offsets[CB_SCRATCH_I];

    // Kernels
    std::vector<uint32_t> ct_rw = {local_N, cfg.num_cores, S, 0u};
    std::vector<uint32_t> ct_c  = {S_loc, S_noc, cfg.is_ifft?1u:0u, cfg.N};

    auto rk = CreateKernel(prog,
        "tt_metal/programming_examples/fft/kernel/fft_reader.cpp", cr,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                           .noc=NOC::RISCV_0_default, .compile_args=ct_rw});
    auto wk = CreateKernel(prog,
        "tt_metal/programming_examples/fft/kernel/fft_writer.cpp", cr,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,
                           .noc=NOC::RISCV_1_default, .compile_args=ct_rw});
    CreateKernel(prog,
        "tt_metal/programming_examples/fft/kernel/fft_compute.cpp", cr,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                      .fp32_dest_acc_en=true, .compile_args=ct_c});

    // Runtime args
    for (uint32_t my=0; my<cfg.num_cores; my++) {
        CoreCoord mc = linear_to_core(my);

        // Reader: input_addr, unused, twiddle_addr, unused,
        //         local_N, my_id, total_N, num_local_stg, num_stages, use_bf16
        SetRuntimeArgs(prog, rk, mc, {
            buf_addr(input_buf), 0u,
            buf_addr(tw_buf),    0u,
            local_N, my, cfg.N, S_loc, S, 0u
        });

        // Writer
        std::vector<uint32_t> wa = {
            lhs_r_addr, lhs_i_addr,
            scr_r_addr, scr_i_addr,
            buf_addr(output_buf), 0u,
            cfg.num_cores, my, S_loc, sem_id
        };
        for (uint32_t dst=0; dst<cfg.num_cores; dst++) {
            if (dst==my) continue;
            CoreCoord pn = device->worker_core_from_logical_core(linear_to_core(dst));
            wa.push_back(uint32_t(pn.x)); wa.push_back(uint32_t(pn.y));
            wa.push_back(scr_r_addr);     wa.push_back(scr_i_addr);
            wa.push_back(sem_id);
        }
        SetRuntimeArgs(prog, wk, mc, wa);
    }

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

// Helper for test: make a mesh buffer of given tile count
std::shared_ptr<MeshBuffer> make_fft_buf(
    std::shared_ptr<MeshDevice> md, uint32_t num_tiles)
{
    return make_mesh_buf(md, num_tiles * kTileSizeFp32, kTileSizeFp32);
}