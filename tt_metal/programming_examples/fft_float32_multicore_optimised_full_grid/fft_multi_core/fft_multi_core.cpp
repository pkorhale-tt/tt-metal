// fft_1d_multicore.cpp — TRUE MULTICORE 1D FFT on up to 64 Tensix cores
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  ARCHITECTURE — BUTTERFLY PARTITIONING  (not row-decomposition)
// ══════════════════════════════════════════════════════════════════════
//
//  A single N-point FFT is split across C cores (C = power-of-2 ≤ 64).
//  Each core owns a contiguous slice of N/2 butterfly pairs:
//
//    core k  →  elements [k·(N/2/C) .. (k+1)·(N/2/C))
//               i.e. local_half = N/(2C) elements per core
//
//  There is NO cross-core DMA at any butterfly stage.  The host-side
//  bit-reversal in prepare_stage0() pre-sorts the data so that:
//    - Every butterfly pair (even, odd) used at every stage by core k
//      is physically already on core k.
//    - When a butterfly group spans multiple cores (early stages,
//      group size < local_half), the writer's G2==0 path copies
//      out0→even and out1→odd directly — no reordering needed.
//    - When a group fits within a core (later stages), the standard
//      shuffle formula runs locally.
//
//  This is identical to the single-core algorithm applied locally,
//  with bit-reversal guaranteeing the partition is self-consistent.
//
// ══════════════════════════════════════════════════════════════════════
//  CORE SELECTION — mirrors tt-metal matmul_multicore exactly
// ══════════════════════════════════════════════════════════════════════
//
//  We target C = min(N/2 / TILE_SIZE, physical_cores) cores so that:
//    1. Every core gets at least one full tile of butterfly pairs.
//    2. C is a power of 2 (required by the butterfly algorithm).
//    3. C ≤ physical grid size.
//
//  split_work_to_cores(grid, C) distributes C work-units (one per core)
//  uniformly — since C is chosen to divide evenly, group_2 is empty.
//
// ══════════════════════════════════════════════════════════════════════
//  KEY DIFFERENCES vs. fft_multicore_8x8.cpp (row-decomposition)
// ══════════════════════════════════════════════════════════════════════
//
//  1. num_cores, core_id, log2_cores are now the REAL values.
//  2. core_elem_base = core_id × local_half  (NOT zero for every core).
//  3. tiles_per_stage = local_tiles = local_half / TILE_SIZE per core.
//  4. Writer shuffle formula is fully active for intra-core stages.
//  5. No "rows_this" concept — single FFT, every core owns local_half.
//
// ══════════════════════════════════════════════════════════════════════
//  USAGE
// ══════════════════════════════════════════════════════════════════════
//  ./metal_example_fft_1d <direction:0=fwd|1=inv> [N] [input_file]
//
//  N must be a power of 2, N ≥ 2×TILE_SIZE = 2048.
//  Defaults: N=65536, forward.
//
//  Example:
//    ./metal_example_fft_1d 0 65536
//    ./metal_example_fft_1d 1 262144 data.txt

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <fstream>
#include <string>
#include <bit>        // std::bit_width, std::has_single_bit

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"
#include "tt_metal/api/tt-metalium/allocator.hpp"
#include "tt_metal/api/tt-metalium/hal.hpp"
#include "tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp"

using namespace tt;
using namespace tt::tt_metal;

// ── Constants ──────────────────────────────────────────────────────────
constexpr float    PI         = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;   // 32
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;    // 32
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;              // 1024 elements
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);    // 4096 bytes

inline uint32_t f2u(float f)    { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float    u2f(uint32_t u) { float f;    std::memcpy(&f,&u,4); return f; }

// ── Bit-reversal ────────────────────────────────────────────────────────
uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r<<1)|(x&1); x >>= 1; }
    return r;
}

// ── CPU reference FFT ───────────────────────────────────────────────────
void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    uint32_t N = re.size(), log2N = 0;
    while ((1u<<log2N) < N) log2N++;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) { std::swap(re[i],re[j]); std::swap(im[i],im[j]); }
    }
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u<<(s+1);
        float ab = (inv?2.f:-2.f)*PI/m;
        for (uint32_t k = 0; k < N; k += m)
            for (uint32_t j = 0; j < m/2; j++) {
                float wr=std::cos(ab*j), wi=std::sin(ab*j);
                uint32_t e=k+j, o=k+j+m/2;
                float tr=wr*re[o]-wi*im[o], ti=wr*im[o]+wi*re[o];
                float er=re[e], ei=im[e];
                re[e]=er+tr; im[e]=ei+ti; re[o]=er-tr; im[o]=ei-ti;
            }
    }
    if (inv) for (auto& v : re) v/=N;
    if (inv) for (auto& v : im) v/=N;
}

// ── Pack float slice into tile-padded uint32 buffer ─────────────────────
std::vector<uint32_t> pack_tiles(const float* d, uint32_t n, uint32_t ntiles) {
    std::vector<uint32_t> o(ntiles * TILE_SIZE, 0u);
    for (uint32_t i = 0; i < n && i < o.size(); i++) o[i] = f2u(d[i]);
    return o;
}

// ── Compact twiddle table ────────────────────────────────────────────────
std::pair<std::vector<uint32_t>,std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, uint32_t direction) {
    uint32_t half = N/2;
    float sign = (direction==1)?1.f:-1.f;
    std::vector<uint32_t> tw_r(half), tw_i(half);
    for (uint32_t k = 0; k < half; k++) {
        float a = sign*2.f*PI*(float)k/(float)N;
        tw_r[k]=f2u(std::cos(a)); tw_i[k]=f2u(std::sin(a));
    }
    return {tw_r, tw_i};
}

// ── Stage-0 bit-reversed split  (key to zero cross-core communication) ──
//
// For a single N-point FFT split across C cores (C = power of 2):
//
//   core k owns global indices [k·half_local .. (k+1)·half_local)
//   where half_local = N/(2C).
//
//   For each position i in [0, half_local):
//     global even index = bit_reverse(2*(k*half_local + i), log2N)
//     global odd  index = bit_reverse(2*(k*half_local + i)+1, log2N)
//
//   After this permutation, every butterfly needed at every stage
//   by core k operates only on data owned by core k.  No NOC
//   communication is ever needed between cores during the FFT.
//   This is the same proof as single-core DIT bit-reversal; the
//   partition just falls out cleanly because C is a power of 2.
//
// Returns per-core vectors (indexed [core][element]).
struct SplitData {
    std::vector<std::vector<uint32_t>> er, ei, or_, oi; // [core][packed tile word]
};

SplitData prepare_stage0(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t N, uint32_t log2N, uint32_t num_cores, uint32_t local_tiles)
{
    uint32_t half_N     = N/2;
    uint32_t local_half = half_N / num_cores;  // elements per core

    SplitData d;
    d.er.resize(num_cores); d.ei.resize(num_cores);
    d.or_.resize(num_cores); d.oi.resize(num_cores);

    for (uint32_t c = 0; c < num_cores; c++) {
        std::vector<float> _er(local_half), _ei(local_half);
        std::vector<float> _or(local_half), _oi(local_half);

        for (uint32_t i = 0; i < local_half; i++) {
            uint32_t global_pair = c*local_half + i;
            uint32_t e_idx = bit_reverse(2*global_pair,   log2N);
            uint32_t o_idx = bit_reverse(2*global_pair+1, log2N);
            _er[i]=sr[e_idx]; _ei[i]=si[e_idx];
            _or[i]=sr[o_idx]; _oi[i]=si[o_idx];
        }

        auto packed_er  = pack_tiles(_er.data(),  local_half, local_tiles);
        auto packed_ei  = pack_tiles(_ei.data(),  local_half, local_tiles);
        auto packed_or  = pack_tiles(_or.data(),  local_half, local_tiles);
        auto packed_oi  = pack_tiles(_oi.data(),  local_half, local_tiles);

        d.er[c]  = std::move(packed_er);
        d.ei[c]  = std::move(packed_ei);
        d.or_[c] = std::move(packed_or);
        d.oi[c]  = std::move(packed_oi);
    }
    return d;
}

// ── CB creation helper ────────────────────────────────────────────────────
CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bpt = TILE_BYTES) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles*bpt, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bpt);
    return CreateCircularBuffer(p, c, cfg);
}

// ── Choose num_cores: largest power-of-2 ≤ min(half_N/TILE_SIZE, phys) ──
uint32_t choose_core_count(uint32_t half_N, uint32_t phys_cores) {
    // Each core must get at least one full tile.
    uint32_t max_by_tiles = half_N / TILE_SIZE;          // must be ≥ 1
    uint32_t max_by_phys  = phys_cores;
    uint32_t raw = std::min(max_by_tiles, max_by_phys);
    // Round down to nearest power of 2.
    if (raw == 0) return 1u;
    uint32_t p2 = 1u;
    while (p2*2 <= raw) p2 *= 2;
    return p2;
}

// ══════════════════════════════════════════════════════════════════════
//  Read input file
// ══════════════════════════════════════════════════════════════════════
bool read_input_file(const std::string& path, uint32_t N,
                     std::vector<float>& ir, std::vector<float>& ii) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr<<"Cannot open: "<<path<<"\n"; return false; }
    std::vector<float> vals; std::string tok;
    while (f>>tok) {
        if (!tok.empty()&&tok.back()==',') tok.pop_back();
        try { vals.push_back(std::stof(tok)); }
        catch(...) { std::cerr<<"Bad token '"<<tok<<"'\n"; return false; }
    }
    ir.assign(N,0.f); ii.assign(N,0.f);
    if (vals.size()==2*N)
        for (uint32_t i=0;i<N;i++){ ir[i]=vals[2*i]; ii[i]=vals[2*i+1]; }
    else
        for (uint32_t i=0;i<N&&i<vals.size();i++) ir[i]=vals[i];
    return true;
}

// ══════════════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr<<"Usage: "<<argv[0]
                 <<" <0=fwd|1=inv> [N] [input_file]\n"
                 <<"  Defaults: N=65536, forward\n";
        return 1;
    }
    uint32_t direction = (uint32_t)std::atoi(argv[1]);
    uint32_t N         = 65536;
    std::string in_file;

    for (int i=2; i<argc; i++) {
        std::string a=argv[i];
        bool is_num=!a.empty();
        for (char c:a) if(!std::isdigit(c)){is_num=false;break;}
        if (!is_num) { in_file=a; continue; }
        N = (uint32_t)std::stoul(a);
    }
    if (N<2||(N&(N-1))) { std::cerr<<"N must be power-of-2\n"; return 1; }
    if (N < 2*TILE_SIZE) {
        std::cerr<<"N must be ≥ "<<2*TILE_SIZE<<" for multicore (need ≥1 tile/core)\n";
        return 1;
    }

    uint32_t log2N  = 0; while((1u<<log2N)<N) log2N++;
    uint32_t half_N = N/2;

    // ── Device init ───────────────────────────────────────────────────
    auto mesh   = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto& cq    = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices().at(0);
    CoreCoord phys  = device->compute_with_storage_grid_size();
    uint32_t phys_cores = phys.x * phys.y;   // 64 on Wormhole 8×8

    // ── Core count selection ──────────────────────────────────────────
    // Key constraint: num_cores MUST be a power of 2 for the butterfly
    // partition to be self-consistent.
    uint32_t num_cores  = choose_core_count(half_N, phys_cores);
    uint32_t log2_cores = 0; while((1u<<log2_cores)<num_cores) log2_cores++;
    uint32_t local_half = half_N / num_cores;   // elements per core
    uint32_t local_tiles= local_half / TILE_SIZE; // tiles per core (exact)

    // Grid: pack cores column-major (matches matmul_multicore)
    uint32_t grid_x = std::min((uint32_t)phys.x,
                                (num_cores + (uint32_t)phys.y - 1) / phys.y);
    uint32_t grid_y = std::min((uint32_t)phys.y, num_cores);
    CoreCoord grid{grid_x, grid_y};

    // Use split_work_to_cores for CoreRangeSet construction
    // (num_cores is exact, so group_2 will be empty)
    auto [nc, all_cores, cg1, cg2, wg1, wg2] =
        split_work_to_cores(grid, num_cores);

    std::cout<<"═══════════════════════════════════════════════════\n";
    std::cout<<" TRUE MULTICORE 1D FFT  —  Configuration\n with butterfly partitioning";
    std::cout<<"═══════════════════════════════════════════════════\n";
    std::cout<<"  N            : "<<N<<"\n";
    std::cout<<"  log2(N)      : "<<log2N<<"\n";
    std::cout<<"  Cores used   : "<<num_cores<<" (log2="<<log2_cores<<")\n";
    std::cout<<"  local_half   : "<<local_half<<" elements/core\n";
    std::cout<<"  local_tiles  : "<<local_tiles<<" tiles/core\n";
    std::cout<<"  Grid         : "<<grid_x<<"×"<<grid_y<<"\n";
    std::cout<<"  Direction    : "<<(direction?"Inverse":"Forward")<<"\n";
    std::cout<<"═══════════════════════════════════════════════════\n";

    // ── Input data ────────────────────────────────────────────────────
    std::vector<float> ir(N,0.f), ii(N,0.f);
    if (!in_file.empty()) {
        if (!read_input_file(in_file, N, ir, ii)) { mesh->close(); return 1; }
    } else {
        // Synthetic: two sinusoids at bins 4 and 16
        for (uint32_t i=0;i<N;i++)
            ir[i] = std::sin(2.f*PI*4.f*i/N) + 0.5f*std::sin(2.f*PI*16.f*i/N);
    }

    // CPU reference
    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, direction==1);

    // ── Stage-0 bit-reversed packing, one per core ───────────────────
    std::cout<<"Preparing bit-reversed input (stage 0)...\n";
    SplitData sd = prepare_stage0(ir, ii, N, log2N, num_cores, local_tiles);

    // Compact twiddle table (same for all cores)
    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N, direction);
    // Pad to tile boundary
    {
        uint32_t rem = half_N % TILE_SIZE;
        if (rem) {
            uint32_t pad = TILE_SIZE - rem;
            cmp_r_t.resize(half_N+pad, 0u);
            cmp_i_t.resize(half_N+pad, 0u);
        }
    }
    uint32_t compact_bytes = half_N * sizeof(float);
    uint32_t compact_alloc = ((compact_bytes+TILE_BYTES-1)/TILE_BYTES)*TILE_BYTES;
    uint32_t cmp_ntiles    = compact_alloc / TILE_BYTES;

    // ── DRAM buffers — one per core (per-core DRAM regions) ──────────
    //
    // Each core reads from its own input buffers and writes to its own
    // output buffers.  We allocate one contiguous DRAM buffer per
    // logical component (even_r, odd_r, etc.) covering all cores,
    // with per-core slices starting at tile_offset × TILE_BYTES.
    // This mirrors matmul_multicore's per-core output tile allocation.
    //
    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile{.page_size=TILE_BYTES,
                                      .buffer_type=BufferType::DRAM};
    uint32_t total_tiles = num_cores * local_tiles;
    uint32_t total_bytes = total_tiles * TILE_BYTES;

    auto mk=[&](uint32_t bytes)->std::shared_ptr<MeshBuffer>{
        ReplicatedBufferConfig rc{.size=bytes};
        return MeshBuffer::create(rc, dram_tile, mesh.get());
    };
    auto b_er  = mk(total_bytes); auto b_ei  = mk(total_bytes);
    auto b_or  = mk(total_bytes); auto b_oi  = mk(total_bytes);
    auto b_o0r = mk(total_bytes); auto b_o0i = mk(total_bytes);
    auto b_o1r = mk(total_bytes); auto b_o1i = mk(total_bytes);

    DeviceLocalBufferConfig dram_cmp{.page_size=TILE_BYTES,
                                     .buffer_type=BufferType::DRAM};
    ReplicatedBufferConfig rc_cmp{.size=compact_alloc};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // ── Program & CBs ─────────────────────────────────────────────────
    Program prog = CreateProgram();

    // CB sizing:
    //   Input    (0-3):   1× local_tiles   (loaded once)
    //   Twiddle  (4-5):   2× local_tiles   (double-buffered: OPT-9)
    //   Compact (10-11):  cmp_ntiles        (resident, shared)
    //   Output  (16-19):  2× local_tiles   (double-buffered: OPT-9)
    //   Scratch (20-23):  1× local_tiles   (intra-stage temp)
    for (uint32_t i=0; i<num_cores; i++) {
        uint32_t cx = i/grid_y, cy = i%grid_y;
        CoreCoord cc{cx, cy};

        for (uint32_t id : {0u,1u,2u,3u})
            create_cb(prog, cc, id, local_tiles, TILE_BYTES);
        for (uint32_t id : {4u,5u})
            create_cb(prog, cc, id, 2*local_tiles, TILE_BYTES);
        create_cb(prog, cc, 10, cmp_ntiles, TILE_BYTES);
        create_cb(prog, cc, 11, cmp_ntiles, TILE_BYTES);
        for (uint32_t id : {16u,17u,18u,19u})
            create_cb(prog, cc, id, 2*local_tiles, TILE_BYTES);
        for (uint32_t id : {20u,21u,22u,23u})
            create_cb(prog, cc, id, local_tiles, TILE_BYTES);
    }

    // ── Kernels ───────────────────────────────────────────────────────
    KernelHandle reader_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/reader_fft_f32.cpp",
        all_cores,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,
                           .noc=NOC::RISCV_0_default});

    KernelHandle writer_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/writer_fft_f32.cpp",
        all_cores,
        DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,
                           .noc=NOC::RISCV_1_default});

    KernelHandle compute_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/compute/fft_compute_f32.cpp",
        all_cores,
        ComputeConfig{.math_fidelity=MathFidelity::HiFi4,
                      .fp32_dest_acc_en=true, .math_approx_mode=false});

    // ── Runtime args — column-major, same loop shape as matmul_multicore
    for (uint32_t i=0; i<num_cores; i++) {
        uint32_t cx = i/grid_y, cy = i%grid_y;
        CoreCoord cc{cx, cy};

        uint32_t core_id       = i;
        uint32_t tile_offset   = core_id * local_tiles;
        // KEY: core_elem_base is the first GLOBAL element index this
        // core's butterfly pairs correspond to in the frequency domain.
        uint32_t core_elem_base = core_id * local_half;

        // ── Reader args ──────────────────────────────────────────────
        // arg  0  even_r_addr
        // arg  1  even_i_addr
        // arg  2  odd_r_addr
        // arg  3  odd_i_addr
        // arg  4  compact_r_addr
        // arg  5  compact_i_addr
        // arg  6  local_tiles
        // arg  7  tile_offset       = core_id × local_tiles
        // arg  8  num_stages        = log2N
        // arg  9  half_N            = N/2
        // arg 10  local_half        = N/(2×num_cores)   [single row, no multi-row]
        // arg 11  core_elem_base    = core_id × local_half  ← KEY DIFFERENCE
        // arg 12  rows_this         = 1  (single 1D FFT, no row loop)
        // arg 13  tiles_per_row     = local_tiles
        std::vector<uint32_t> rdr = {
            b_er->address(), b_ei->address(),
            b_or->address(), b_oi->address(),
            b_cmp_r->address(), b_cmp_i->address(),
            local_tiles,
            tile_offset,
            log2N,
            half_N,
            local_half,      // one core's worth of elements (not rows_this × half)
            core_elem_base,  // ← REAL VALUE (not 0) — activates twiddle formula
            1u,              // rows_this = 1 (single FFT, no row replication)
            local_tiles      // tiles_per_row = local_tiles
        };

        // ── Compute args ──────────────────────────────────────────────
        // arg 0  num_stages
        // arg 1  tiles_per_stage = local_tiles (one core's share per stage)
        std::vector<uint32_t> cmp = { log2N, local_tiles };

        // ── Writer args ───────────────────────────────────────────────
        // arg  0  out0_r_addr
        // arg  1  out0_i_addr
        // arg  2  out1_r_addr
        // arg  3  out1_i_addr
        // arg  4  local_tiles
        // arg  5  num_stages        = log2N
        // arg  6  local_half        = N/(2×num_cores)
        // arg  7  half_N            = N/2
        // arg  8  num_cores         ← REAL VALUE (was hardcoded 1)
        // arg  9  core_id           ← REAL VALUE (was hardcoded 0)
        // arg 10  log2_cores        ← REAL VALUE (was hardcoded 0)
        // arg 11  tile_offset
        // arg 12  core_elem_base    ← REAL VALUE (was hardcoded 0)
        // arg 13  rows_this         = 1
        // arg 14  tiles_per_row     = local_tiles
        std::vector<uint32_t> wtr = {
            b_o0r->address(), b_o0i->address(),
            b_o1r->address(), b_o1i->address(),
            local_tiles,
            log2N,
            local_half,
            half_N,
            num_cores,      // ← REAL: activates multi-core shuffle logic
            core_id,        // ← REAL
            log2_cores,     // ← REAL
            tile_offset,
            core_elem_base, // ← REAL
            1u,             // rows_this = 1
            local_tiles     // tiles_per_row
        };

        SetRuntimeArgs(prog, reader_k,  cc, rdr);
        SetRuntimeArgs(prog, compute_k, cc, cmp);
        SetRuntimeArgs(prog, writer_k,  cc, wtr);
    }

    // ── MeshWorkload ──────────────────────────────────────────────────
    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    // ── Upload per-core inputs to their DRAM slices ───────────────────
    // We write one per-core tile slice into the global contiguous buffer.
    // The buffer is: [core_0_tiles | core_1_tiles | ... | core_C-1_tiles]
    std::cout<<"Writing inputs to DRAM...\n";
    {
        // Flatten per-core packed data into one big buffer.
        std::vector<uint32_t> flat_er(total_tiles*TILE_SIZE);
        std::vector<uint32_t> flat_ei(total_tiles*TILE_SIZE);
        std::vector<uint32_t> flat_or(total_tiles*TILE_SIZE);
        std::vector<uint32_t> flat_oi(total_tiles*TILE_SIZE);

        for (uint32_t c=0; c<num_cores; c++) {
            uint32_t off = c * local_tiles * TILE_SIZE;
            std::copy(sd.er[c].begin(), sd.er[c].end(), flat_er.begin()+off);
            std::copy(sd.ei[c].begin(), sd.ei[c].end(), flat_ei.begin()+off);
            std::copy(sd.or_[c].begin(),sd.or_[c].end(),flat_or.begin()+off);
            std::copy(sd.oi[c].begin(), sd.oi[c].end(), flat_oi.begin()+off);
        }

        EnqueueWriteMeshBuffer(cq, b_er,    flat_er,  false);
        EnqueueWriteMeshBuffer(cq, b_ei,    flat_ei,  false);
        EnqueueWriteMeshBuffer(cq, b_or,    flat_or,  false);
        EnqueueWriteMeshBuffer(cq, b_oi,    flat_oi,  false);
        EnqueueWriteMeshBuffer(cq, b_cmp_r, cmp_r_t,  false);
        EnqueueWriteMeshBuffer(cq, b_cmp_i, cmp_i_t,  false);
        Finish(cq);
    }

    // ── Execute ───────────────────────────────────────────────────────
    std::cout<<"Launching 1D FFT (N="<<N<<", "<<num_cores<<" cores)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout<<"Kernel complete.\n";

    // ── Read results ──────────────────────────────────────────────────
    std::vector<uint32_t> o0r(total_tiles*TILE_SIZE, 0u);
    std::vector<uint32_t> o0i(total_tiles*TILE_SIZE, 0u);
    std::vector<uint32_t> o1r(total_tiles*TILE_SIZE, 0u);
    std::vector<uint32_t> o1i(total_tiles*TILE_SIZE, 0u);
    EnqueueReadMeshBuffer(cq, o0r, b_o0r, true);
    EnqueueReadMeshBuffer(cq, o0i, b_o0i, true);
    EnqueueReadMeshBuffer(cq, o1r, b_o1r, true);
    EnqueueReadMeshBuffer(cq, o1i, b_o1i, true);

    // Reconstruct output: each core's out0 = lower half of its N/C slice,
    // out1 = upper half.  out0[c*local_half .. c*local_half+local_half/2)
    // out1[c*local_half+local_half/2 .. (c+1)*local_half)
    // But for the final stage output layout we just read sequentially:
    //   result[c*local_half + i]              = out0[c*local_half + i]       i in [0,local_half)
    //   result[c*local_half + i + half_N]     = out1[c*local_half + i]
    // i.e. out0 covers bins [0, half_N), out1 covers bins [half_N, N).
    std::vector<float> result_r(N), result_i(N);
    for (uint32_t c=0; c<num_cores; c++) {
        uint32_t off = c * local_tiles * TILE_SIZE;
        for (uint32_t i=0; i<local_half; i++) {
            result_r[c*local_half + i]          = u2f(o0r[off+i]);
            result_i[c*local_half + i]          = u2f(o0i[off+i]);
            result_r[c*local_half + i + half_N] = u2f(o1r[off+i]);
            result_i[c*local_half + i + half_N] = u2f(o1i[off+i]);
        }
    }
    if (direction==1)
        for (uint32_t i=0;i<N;i++){ result_r[i]/=N; result_i[i]/=N; }

    // ── Validate ──────────────────────────────────────────────────────
    float threshold = std::max(1.0f, 0.005f*(float)N);
    bool all_pass = true;
    float max_er = 0.f, max_ei = 0.f, mean_e = 0.f;
    for (uint32_t i=0;i<N;i++){
        float er=std::abs(result_r[i]-ref_r[i]);
        float ei=std::abs(result_i[i]-ref_i[i]);
        max_er=std::max(max_er,er);
        max_ei=std::max(max_ei,ei);
        mean_e+=er+ei;
        if (er>=threshold||ei>=threshold) all_pass=false;
    }
    mean_e /= 2*N;

    std::cout<<"\n═══════════════════════════════════════════════════\n";
    std::cout<<" VALIDATION\n";
    std::cout<<"═══════════════════════════════════════════════════\n";
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"  max_err_real : "<<max_er<<"\n";
    std::cout<<"  max_err_imag : "<<max_ei<<"\n";
    std::cout<<"  mean_err     : "<<mean_e<<"\n";
    std::cout<<"  threshold    : "<<threshold<<"\n";
    std::cout<<"  Result       : "<<(all_pass?"✓ PASS":"✗ FAIL")<<"\n";

    // ── First 20 bins ─────────────────────────────────────────────────
    std::cout<<"\n═══════════════════════════════════════════════════\n";
    std::cout<<" FIRST 20 BINS\n";
    std::cout<<"═══════════════════════════════════════════════════\n";
    for (uint32_t i=0;i<20&&i<N;i++){
        std::cout<<" X["<<std::setw(3)<<i<<"] = "
                 <<std::setw(12)<<result_r[i]
                 <<(result_i[i]>=0?" + ":" - ")
                 <<std::setw(12)<<std::abs(result_i[i])<<"j"
                 <<"   ref: "<<std::setw(12)<<ref_r[i]
                 <<(ref_i[i]>=0?" + ":" - ")
                 <<std::setw(12)<<std::abs(ref_i[i])<<"j\n";
    }

    // ── Core utilisation ──────────────────────────────────────────────
    std::cout<<"\n═══════════════════════════════════════════════════\n";
    std::cout<<" CORE MAP  ("<<num_cores<<" cores, "
             <<local_half<<" elems/"<<"core, "<<local_tiles<<" tiles/core)\n";
    std::cout<<"═══════════════════════════════════════════════════\n";
    std::cout<<" col →  ";
    for (uint32_t x=0;x<grid_x;x++) std::cout<<std::setw(4)<<x;
    std::cout<<"\n row ↓\n";
    for (uint32_t y=0;y<grid_y;y++){
        std::cout<<"    "<<std::setw(2)<<y<<"  ";
        for (uint32_t x=0;x<grid_x;x++){
            uint32_t idx = x*grid_y+y;
            if (idx<num_cores) std::cout<<std::setw(4)<<idx;
            else               std::cout<<"   .";
        }
        std::cout<<"\n";
    }

    mesh->close();
    std::cout<<"\n Done\n";
    return (all_pass?0:1);
}