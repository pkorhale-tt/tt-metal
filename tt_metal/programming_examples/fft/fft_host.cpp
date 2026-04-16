// fft_host.cpp

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

static constexpr uint32_t kTileHW       = tt::constants::TILE_HW;
static constexpr uint32_t kTileElems    = kTileHW * kTileHW;
static constexpr uint32_t kTileSizeFp32 = kTileElems * sizeof(float);

// 17 CBs, each 1 tile (double-buffer even/odd/out with 2 tiles)
// CB sizes: even/odd/out get 2 tiles for pipelining, rest 1 tile
static constexpr uint32_t kCbTiles[17] = {2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1};
static constexpr uint32_t NUM_CBS = 17;

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

// Precompute twiddle tiles.
// Layout: num_stages * num_cores tiles real, then same imag.
// Tile [s*C + c] holds twiddle factors for core c at stage s.
// Each tile: local_N/2 valid entries (twiddle for each butterfly pair).
std::vector<float> precompute_twiddles(
    uint32_t N, uint32_t num_cores, uint32_t num_stages, bool is_ifft)
{
    uint32_t local_N  = N / num_cores;
    uint32_t total_tiles = 2 * num_stages * num_cores;
    std::vector<float> tw(total_tiles * kTileElems, 0.0f);

    for (uint32_t s=0; s<num_stages; s++) {
        uint32_t stride = 1u << s;
        uint32_t M      = 2 * stride;
        for (uint32_t c=0; c<num_cores; c++) {
            uint32_t global_base = c * local_N;
            float* tile_r = tw.data() + (s*num_cores + c) * kTileElems;
            float* tile_i = tw.data() + (num_stages*num_cores + s*num_cores + c) * kTileElems;
            for (uint32_t i=0; i<local_N/2 && i<kTileElems; i++) {
                // butterfly pair (lo, hi=lo+stride) where lo = grp*2*stride + pos
                uint32_t grp = i / stride;
                uint32_t pos = i % stride;
                uint32_t lo  = grp*(2*stride) + pos;
                uint32_t k   = (global_base + lo) % M;
                double angle = -2.0 * M_PI * k / M;
                if (is_ifft) angle = -angle;
                tile_r[i] = float(std::cos(angle));
                tile_i[i] = float(std::sin(angle));
            }
        }
    }
    return tw;
}

// Bit-reverse permutation of indices
static uint32_t bit_rev(uint32_t x, uint32_t bits) {
    uint32_t r=0;
    for (uint32_t i=0;i<bits;i++){r=(r<<1)|(x&1);x>>=1;}
    return r;
}

// Pack input into DRAM buffer layout:
// 4 separate buffers: even_r, even_i, odd_r, odd_i
// Each has num_cores tiles, tile c = elements for core c
// Even = lower half of each butterfly pair, Odd = upper half
// For stage 0: even[i] = x[bit_rev(2i, log2N)], odd[i] = x[bit_rev(2i+1, log2N)]
// Returns flat vector: [even_r tiles | even_i tiles | odd_r tiles | odd_i tiles]
std::vector<float> pack_input_even_odd(
    const std::vector<std::complex<float>>& x, uint32_t num_cores)
{
    uint32_t N       = x.size();
    uint32_t local_N = N / num_cores;
    uint32_t log2N   = 0; { uint32_t n=N; while(n>1){log2N++;n>>=1;} }

    // 4 channels * num_cores tiles
    std::vector<float> buf(4 * num_cores * kTileElems, 0.0f);
    float* even_r = buf.data() + 0 * num_cores * kTileElems;
    float* even_i = buf.data() + 1 * num_cores * kTileElems;
    float* odd_r  = buf.data() + 2 * num_cores * kTileElems;
    float* odd_i  = buf.data() + 3 * num_cores * kTileElems;

    for (uint32_t c=0; c<num_cores; c++) {
        uint32_t base = c * local_N;
        for (uint32_t i=0; i<local_N/2; i++) {
            uint32_t lo_rev = bit_rev(base + 2*i,   log2N);
            uint32_t hi_rev = bit_rev(base + 2*i+1, log2N);
            even_r[c*kTileElems + i] = x[lo_rev].real();
            even_i[c*kTileElems + i] = x[lo_rev].imag();
            odd_r [c*kTileElems + i] = x[hi_rev].real();
            odd_i [c*kTileElems + i] = x[hi_rev].imag();
        }
    }
    return buf;
}

// Unpack output: num_cores tiles of out_r, then out_i
std::vector<std::complex<float>> unpack_output(
    const std::vector<float>& r_buf,
    const std::vector<float>& i_buf,
    uint32_t N, uint32_t num_cores)
{
    uint32_t local_N = N / num_cores;
    std::vector<std::complex<float>> out(N);
    for (uint32_t c=0; c<num_cores; c++) {
        const float* tr = r_buf.data() + c * kTileElems;
        const float* ti = i_buf.data() + c * kTileElems;
        for (uint32_t i=0; i<local_N; i++)
            out[c*local_N + i] = {tr[i], ti[i]};
    }
    return out;
}

void run_fft(
    std::shared_ptr<MeshDevice> md,
    const FFTConfig& cfg,
    // Input: flat [even_r|even_i|odd_r|odd_i] tiles, 4*num_cores tiles total
    std::shared_ptr<MeshBuffer> input_buf,
    // Output: [out_r tiles | out_i tiles], 2*num_cores tiles total
    std::shared_ptr<MeshBuffer> out_r_buf,
    std::shared_ptr<MeshBuffer> out_i_buf)
{
    assert((cfg.N&(cfg.N-1))==0 && (cfg.num_cores&(cfg.num_cores-1))==0);
    assert(cfg.N % cfg.num_cores == 0);

    IDevice* device = md->get_device(0,0);
    MeshCommandQueue& cq = md->mesh_command_queue();

    uint32_t local_N   = cfg.N / cfg.num_cores;
    uint32_t S         = 0; { uint32_t n=cfg.N; while(n>1){S++;n>>=1;} }
    uint32_t S_loc     = 0; { uint32_t n=local_N; while(n>1){S_loc++;n>>=1;} }
    uint32_t S_noc     = S - S_loc;
    uint32_t num_tiles = cfg.num_cores; // one tile per core per channel

    // Twiddle buffer
    auto tw_data = precompute_twiddles(cfg.N, cfg.num_cores, S, cfg.is_ifft);
    uint32_t tw_total_tiles = 2 * S * cfg.num_cores;
    auto tw_buf = make_mesh_buf(md, tw_total_tiles*kTileSizeFp32, kTileSizeFp32);
    WriteShard(cq, tw_buf, tw_data, MeshCoordinate(0,0), false);

    // Split twiddle into separate r/i buffers for reader args
    // tw_r = first S*C tiles, tw_i = next S*C tiles — same buffer, different offsets
    uint32_t tw_r_addr = buf_addr(tw_buf);
    // tw_i starts at offset S*num_cores tiles into the buffer
    // We pass this as a separate DRAM address to the reader
    // Since it's interleaved, we use the same buffer base but offset by tile count
    // Actually we need separate buffers for clean addressing:
    uint32_t tw_half = S * cfg.num_cores;
    auto tw_r_buf = make_mesh_buf(md, tw_half*kTileSizeFp32, kTileSizeFp32);
    auto tw_i_buf = make_mesh_buf(md, tw_half*kTileSizeFp32, kTileSizeFp32);
    std::vector<float> tw_r_data(tw_data.begin(), tw_data.begin() + tw_half*kTileElems);
    std::vector<float> tw_i_data(tw_data.begin() + tw_half*kTileElems, tw_data.end());
    WriteShard(cq, tw_r_buf, tw_r_data, MeshCoordinate(0,0), false);
    WriteShard(cq, tw_i_buf, tw_i_data, MeshCoordinate(0,0), false);

    // Separate even/odd input buffers from packed input
    // input_buf contains [even_r|even_i|odd_r|odd_i] each num_cores tiles
    // We need separate DRAM addresses for each channel
    uint32_t ch_bytes = num_tiles * kTileSizeFp32;
    auto even_r_buf = make_mesh_buf(md, ch_bytes, kTileSizeFp32);
    auto even_i_buf = make_mesh_buf(md, ch_bytes, kTileSizeFp32);
    auto odd_r_buf  = make_mesh_buf(md, ch_bytes, kTileSizeFp32);
    auto odd_i_buf  = make_mesh_buf(md, ch_bytes, kTileSizeFp32);
    // Read input and split (host-side)
    std::vector<float> input_flat;
    ReadShard(cq, input_flat, input_buf, MeshCoordinate(0,0), true);
    std::vector<float> ev_r(input_flat.begin(),                         input_flat.begin()+num_tiles*kTileElems);
    std::vector<float> ev_i(input_flat.begin()+  num_tiles*kTileElems,  input_flat.begin()+2*num_tiles*kTileElems);
    std::vector<float> od_r(input_flat.begin()+2*num_tiles*kTileElems,  input_flat.begin()+3*num_tiles*kTileElems);
    std::vector<float> od_i(input_flat.begin()+3*num_tiles*kTileElems,  input_flat.end());
    WriteShard(cq, even_r_buf, ev_r, MeshCoordinate(0,0), false);
    WriteShard(cq, even_i_buf, ev_i, MeshCoordinate(0,0), false);
    WriteShard(cq, odd_r_buf,  od_r, MeshCoordinate(0,0), false);
    WriteShard(cq, odd_i_buf,  od_i, MeshCoordinate(0,0), false);

    // Program
    Program prog = CreateProgram();
    std::vector<CoreCoord> cores;
    for (uint32_t i=0;i<cfg.num_cores;i++) cores.push_back(linear_to_core(i));
    CoreRange cr(cores.front(), cores.back());

    uint32_t sem_id = CreateSemaphore(prog, cr, 0u);

    // CBs — sized to local_N elements per channel
    uint32_t cb_elem_bytes = local_N * sizeof(float);
    for (uint32_t id=0; id<NUM_CBS; id++) {
        uint32_t cb_bytes = kCbTiles[id] * cb_elem_bytes;
        CircularBufferConfig c(cb_bytes, {{id, tt::DataFormat::Float32}});
        c.set_page_size(id, cb_elem_bytes);
        CreateCircularBuffer(prog, cr, c);
    }

    // CB base for scratch NOC addresses
    CoreCoord vc0 = device->worker_core_from_logical_core(linear_to_core(0));
    uint32_t cb_base = uint32_t(device->get_dev_addr(vc0, HalL1MemAddrType::DEFAULT_UNRESERVED));
    uint32_t offsets[NUM_CBS]; offsets[0]=0;
    for (int i=1;i<(int)NUM_CBS;i++)
        offsets[i]=offsets[i-1]+kCbTiles[i-1]*cb_elem_bytes;
    uint32_t scratch_r_addr = cb_base + offsets[11]; // CB_SCRATCH_R
    uint32_t scratch_i_addr = cb_base + offsets[12]; // CB_SCRATCH_I

    // Compile-time args: writer=[local_N, num_stages], compute=[S_loc,S_noc,is_ifft,N]
    std::vector<uint32_t> ct_w = {local_N, S};
    std::vector<uint32_t> ct_c = {S_loc, S_noc, cfg.is_ifft?1u:0u, cfg.N};

    auto rk = CreateKernel(prog,
        "tt_metal/programming_examples/fft/kernel/fft_reader.cpp", cr,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                           .noc=NOC::RISCV_0_default});
    auto wk = CreateKernel(prog,
        "tt_metal/programming_examples/fft/kernel/fft_writer.cpp", cr,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,
                           .noc=NOC::RISCV_1_default,
                           .compile_args=ct_w});
    CreateKernel(prog,
        "tt_metal/programming_examples/fft/kernel/fft_compute.cpp", cr,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                      .fp32_dest_acc_en=true, .compile_args=ct_c});

    for (uint32_t my=0; my<cfg.num_cores; my++) {
        CoreCoord mc = linear_to_core(my);

        // Reader args: even_r, even_i, odd_r, odd_i, tw_r, tw_i,
        //              num_tiles, num_stages, my_id, num_local_stg
        SetRuntimeArgs(prog, rk, mc, {
            buf_addr(even_r_buf), buf_addr(even_i_buf),
            buf_addr(odd_r_buf),  buf_addr(odd_i_buf),
            buf_addr(tw_r_buf),   buf_addr(tw_i_buf),
            num_tiles, S, my, S_loc
        });

        // Writer args: dram_out_r, dram_out_i, scratch_r, scratch_i,
        //              num_cores, my_id, first_noc_stg, sem_id, num_tiles,
        //              [peer table]
        std::vector<uint32_t> wa = {
            buf_addr(out_r_buf), buf_addr(out_i_buf),
            scratch_r_addr, scratch_i_addr,
            cfg.num_cores, my, S_loc, sem_id, num_tiles
        };
        for (uint32_t dst=0; dst<cfg.num_cores; dst++) {
            if (dst==my) continue;
            CoreCoord pn = device->worker_core_from_logical_core(linear_to_core(dst));
            wa.push_back(uint32_t(pn.x)); wa.push_back(uint32_t(pn.y));
            wa.push_back(scratch_r_addr);  wa.push_back(scratch_i_addr);
            wa.push_back(sem_id);
        }
        SetRuntimeArgs(prog, wk, mc, wa);
    }

    MeshWorkload workload;
    workload.add_program(
        MeshCoordinateRange(MeshCoordinate(0,0),MeshCoordinate(0,0)),
        std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    cq.finish();
}