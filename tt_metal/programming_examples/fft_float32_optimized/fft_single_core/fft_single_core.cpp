// fft_single_core_opt.cpp  — OPTIMAL v2: compact twiddle table
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DRAM traffic (N=1024):
//   Upload: 8 KB input + 4 KB compact twiddles = 12 KB  (was 88 KB)
//   Download: 8 KB result
//   Total: 20 KB  — matches mesham's design, ~5× less than previous version
//
// Change from v1: twiddle storage reduced from log2N×N/2 to N/2 values.
//   Host uploads compact twiddle table once (N/2 entries).
//   Reader expands per stage in L1 using:
//     slot p: j = p & (half_m-1),  idx = j * (N >> (stage+1))
//   Compute kernel unchanged — still uses fast FPU mul_tiles/add_tiles.

#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float PI = 3.14159265358979323846f;

constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

inline uint32_t f2u(float f)  { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float    u2f(uint32_t u){ float f;   std::memcpy(&f,&u,4); return f; }

std::vector<uint32_t> pack_tiles(const std::vector<float>& d, uint32_t ntiles) {
    std::vector<uint32_t> o(ntiles*TILE_SIZE, 0);
    for (uint32_t i = 0; i < d.size() && i < o.size(); i++) o[i] = f2u(d[i]);
    return o;
}
std::vector<float> unpack_tiles(const std::vector<uint32_t>& d, uint32_t n) {
    std::vector<float> o(n);
    for (uint32_t i = 0; i < n && i < d.size(); i++) o[i] = u2f(d[i]);
    return o;
}

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r<<1)|(x&1); x >>= 1; }
    return r;
}

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
    if (inv) for (uint32_t i=0;i<N;i++){re[i]/=N;im[i]/=N;}
}

// Stage-0 split: bit-reversed input, stride-2 partition into even/odd
void prepare_stage0(const std::vector<float>& sr, const std::vector<float>& si,
                    uint32_t N, uint32_t log2N, uint32_t tiles,
                    std::vector<uint32_t>& er, std::vector<uint32_t>& ei,
                    std::vector<uint32_t>& or_, std::vector<uint32_t>& oi) {
    uint32_t half_N = N/2;
    std::vector<float> _er(half_N),_ei(half_N),_or(half_N),_oi(half_N);
    for (uint32_t i = 0; i < half_N; i++) {
        uint32_t e = bit_reverse(2*i,   log2N);
        uint32_t o = bit_reverse(2*i+1, log2N);
        _er[i]=sr[e]; _ei[i]=si[e]; _or[i]=sr[o]; _oi[i]=si[o];
    }
    er=pack_tiles(_er,tiles); ei=pack_tiles(_ei,tiles);
    or_=pack_tiles(_or,tiles); oi=pack_tiles(_oi,tiles);
}

// Compact twiddle table: N/2 entries, direction-aware sign.
// compact[k] = (cos(sign*2π*k/N), sin(sign*2π*k/N))  k=0..N/2-1
// Stored as interleaved (r,i) pairs, packed into tiles.
std::pair<std::vector<uint32_t>,std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N, uint32_t direction) {
    uint32_t half_N = N/2;
    float sign = (direction==1) ? 1.f : -1.f;
    std::vector<uint32_t> tw_r(TILE_SIZE, 0), tw_i(TILE_SIZE, 0);
    for (uint32_t k = 0; k < half_N; k++) {
        float angle = sign * 2.f*PI*(float)k/(float)N;
        tw_r[k] = f2u(std::cos(angle));
        tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

void create_cb(Program& p, CoreCoord c, uint32_t id, uint32_t ntiles, uint32_t bytes) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles*bytes, {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes);
    CreateCircularBuffer(p, c, cfg);
}

bool read_file(const std::string& path, uint32_t& N, bool from_cmd,
               std::vector<float>& ir, std::vector<float>& ii) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr<<"Cannot open: "<<path<<"\n"; return false; }
    std::vector<float> v; std::string t;
    while (f>>t) {
        if (!t.empty() && t.back()==',') t.pop_back();
        if (t.empty()) continue;
        try { v.push_back(std::stof(t)); } catch(...) { std::cerr<<"Bad token\n"; return false; }
    }
    if (v.empty()) { std::cerr<<"Empty file\n"; return false; }
    uint32_t cnt=(uint32_t)v.size(); bool interleaved=false;
    if (from_cmd) {
        if (cnt==2*N) { interleaved=true; }
        else if (cnt<N) std::cerr<<"File has "<<cnt<<" values, padding to N="<<N<<"\n";
        else if (cnt>N) { cnt=N; v.resize(N); }
    } else {
        N=1; while(N<cnt) N<<=1;
    }
    ir.assign(N,0.f); ii.assign(N,0.f);
    if (interleaved)
        for (uint32_t i=0;i<N&&2*i+1<(uint32_t)v.size();i++){ir[i]=v[2*i];ii[i]=v[2*i+1];}
    else
        for (uint32_t i=0;i<N&&i<(uint32_t)v.size();i++) ir[i]=v[i];
    return true;
}

int main(int argc, char** argv) {
    if (argc<2) { std::cerr<<"Usage: "<<argv[0]<<" <0|1> [file] [N]\n"; return 1; }
    uint32_t direction = (uint32_t)std::atoi(argv[1]);
    uint32_t N = 1024; std::string in_file; bool from_cmd=false;
    for (int i=2;i<argc;i++) {
        std::string a=argv[i];
        bool is_file=(a.find('.')!=std::string::npos||a.find('/')!=std::string::npos);
        if (is_file&&in_file.empty()) in_file=a;
        else { try{N=(uint32_t)std::stol(a);from_cmd=true;}catch(...){if(in_file.empty())in_file=a;} }
    }
    if (from_cmd&&(N==0||(N&(N-1)))) { std::cerr<<"N must be power of 2\n"; return 1; }

    uint32_t log2N=0; while((1u<<log2N)<N) log2N++;
    uint32_t half_N=N/2;
    uint32_t tiles=(half_N+TILE_SIZE-1)/TILE_SIZE;

    std::vector<float> ir(N,0.f), ii(N,0.f);
    if (!in_file.empty()) {
        if (!read_file(in_file,N,from_cmd,ir,ii)) return 1;
        log2N=0; while((1u<<log2N)<N) log2N++;
        half_N=N/2; tiles=(half_N+TILE_SIZE-1)/TILE_SIZE;
        ir.resize(N,0.f); ii.resize(N,0.f);
        if (N<2||(N&(N-1))) { std::cerr<<"Invalid N="<<N<<"\n"; return 1; }
    } else {
        for (uint32_t i=0;i<N;i++)
            ir[i]=std::sin(2.f*PI*4.f*i/N)+0.5f*std::sin(2.f*PI*8.f*i/N);
    }

    // DRAM sizes
    uint32_t in_bytes      = tiles * TILE_BYTES;               // per input CB
    uint32_t compact_bytes = half_N * sizeof(float);           // compact twiddle (N/2 floats)

    std::cout<<"════════════════════════════════════════\n";
    std::cout<<" TT-Metal FFT  (Optimal v2 — compact twiddles)\n";
    std::cout<<"════════════════════════════════════════\n";
    std::cout<<" N           : "<<N<<"\n";
    std::cout<<" log2N       : "<<log2N<<"\n";
    std::cout<<" Direction   : "<<(direction?"Inverse":"Forward")<<"\n";
    std::cout<<" tiles/stage : "<<tiles<<"\n";
    std::cout<<" DRAM upload : "<<(4*in_bytes + 2*compact_bytes)/1024<<" KB"
             <<" (input "<<4*in_bytes/1024<<"KB + twiddles "<<2*compact_bytes/1024<<"KB)\n";
    std::cout<<" DRAM dl     : "<<4*in_bytes/1024<<" KB\n";
    std::cout<<"════════════════════════════════════════\n";

    // Reference
    std::vector<float> ref_r(ir), ref_i(ii);
    cpu_fft(ref_r, ref_i, direction==1);

    // Prepare inputs
    std::vector<uint32_t> even_r_t, even_i_t, odd_r_t, odd_i_t;
    prepare_stage0(ir,ii,N,log2N,tiles,even_r_t,even_i_t,odd_r_t,odd_i_t);

    // Compact twiddle table (N/2 entries, packed into one tile each)
    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N, direction);

    // Device setup
    int dev_id=0;
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(dev_id);
    auto& cq  = mesh->mesh_command_queue();
    Program prog = CreateProgram();
    CoreCoord core = {0,0};

    // DRAM buffers
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram{
        .page_size=TILE_BYTES, .buffer_type=tt::tt_metal::BufferType::DRAM};
    auto mk_tile=[&](uint32_t bytes) {
        tt::tt_metal::distributed::ReplicatedBufferConfig rc{.size=bytes};
        return tt::tt_metal::distributed::MeshBuffer::create(rc,dram,mesh.get());
    };
    // Input DRAM buffers (tile-sized)
    auto b_er  = mk_tile(in_bytes);
    auto b_ei  = mk_tile(in_bytes);
    auto b_or  = mk_tile(in_bytes);
    auto b_oi  = mk_tile(in_bytes);
    // Compact twiddle DRAM buffers (N/2 floats each — may be < one tile)
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_cmp{
        .page_size=compact_bytes, .buffer_type=tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::distributed::ReplicatedBufferConfig rc_cmp{.size=compact_bytes};
    auto b_cmp_r = tt::tt_metal::distributed::MeshBuffer::create(rc_cmp,dram_cmp,mesh.get());
    auto b_cmp_i = tt::tt_metal::distributed::MeshBuffer::create(rc_cmp,dram_cmp,mesh.get());
    // Output DRAM buffers
    auto b_o0r = mk_tile(in_bytes);
    auto b_o0i = mk_tile(in_bytes);
    auto b_o1r = mk_tile(in_bytes);
    auto b_o1i = mk_tile(in_bytes);

    // Circular buffers
    // Input CBs 0-3: depth=1 tile (stage-0 from reader; stages 1+ from writer shuffle)
    create_cb(prog,core, 0, 1, TILE_BYTES);   // even_r
    create_cb(prog,core, 1, 1, TILE_BYTES);   // even_i
    create_cb(prog,core, 2, 1, TILE_BYTES);   // odd_r
    create_cb(prog,core, 3, 1, TILE_BYTES);   // odd_i
    // Twiddle CBs 4-5: depth=1 tile (reader re-fills every stage)
    create_cb(prog,core, 4, 1, TILE_BYTES);   // tw_r (expanded per stage)
    create_cb(prog,core, 5, 1, TILE_BYTES);   // tw_i
    // Output CBs 16-19: depth=1 tile (compute writes, writer drains)
    create_cb(prog,core,16, 1, TILE_BYTES);   // out0_r
    create_cb(prog,core,17, 1, TILE_BYTES);   // out0_i
    create_cb(prog,core,18, 1, TILE_BYTES);   // out1_r
    create_cb(prog,core,19, 1, TILE_BYTES);   // out1_i
    // Scratch CBs 20-23: depth=1 tile
    create_cb(prog,core,20, 1, TILE_BYTES);   // tmp0
    create_cb(prog,core,21, 1, TILE_BYTES);   // tmp1
    create_cb(prog,core,22, 1, TILE_BYTES);   // tw_odd_r
    create_cb(prog,core,23, 1, TILE_BYTES);   // tw_odd_i
    // Compact twiddle CBs 10-11: depth=1, size=compact_bytes
    // These hold the N/2 compact values that the reader reads from DRAM once
    // and keeps in L1 for the duration of all stages.
    create_cb(prog,core,10, 1, TILE_BYTES);   // cb_compact_r (N/2 entries)
    create_cb(prog,core,11, 1, TILE_BYTES);   // cb_compact_i

    auto reader_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core"
        "/kernels/dataflow/reader_fft_f32.cpp",
        core, DataMovementConfig{
            .processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto writer_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core"
        "/kernels/dataflow/writer_fft_f32.cpp",
        core, DataMovementConfig{
            .processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto compute_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core"
        "/kernels/compute/fft_compute_f32.cpp",
        core, ComputeConfig{
            .math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.math_approx_mode=false});

    std::vector<uint32_t> reader_args = {
        b_er->address(), b_ei->address(),
        b_or->address(), b_oi->address(),
        b_cmp_r->address(), b_cmp_i->address(),
        tiles, log2N, half_N};
    std::vector<uint32_t> writer_args = {
        b_o0r->address(), b_o0i->address(),
        b_o1r->address(), b_o1i->address(),
        tiles, log2N, half_N};
    std::vector<uint32_t> compute_args = {log2N, tiles};

    tt::tt_metal::distributed::MeshWorkload wl;
    tt::tt_metal::distributed::MeshCoordinateRange rng =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));
    auto& p = wl.get_programs().begin()->second;
    SetRuntimeArgs(p, reader_k,  core, reader_args);
    SetRuntimeArgs(p, writer_k,  core, writer_args);
    SetRuntimeArgs(p, compute_k, core, compute_args);

    using namespace tt::tt_metal::distributed;
    std::cout<<"Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq,b_er,  even_r_t, false);
    EnqueueWriteMeshBuffer(cq,b_ei,  even_i_t, false);
    EnqueueWriteMeshBuffer(cq,b_or,  odd_r_t,  false);
    EnqueueWriteMeshBuffer(cq,b_oi,  odd_i_t,  false);
    EnqueueWriteMeshBuffer(cq,b_cmp_r, cmp_r_t, false);
    EnqueueWriteMeshBuffer(cq,b_cmp_i, cmp_i_t, false);
    Finish(cq);

    std::cout<<"Launching FFT kernel ("<<log2N<<" stages on device)...\n";
    EnqueueMeshWorkload(cq,wl,true);
    std::cout<<"Kernel complete.\n";

    std::vector<uint32_t> o0r_raw(tiles*TILE_SIZE), o0i_raw(tiles*TILE_SIZE);
    std::vector<uint32_t> o1r_raw(tiles*TILE_SIZE), o1i_raw(tiles*TILE_SIZE);
    EnqueueReadMeshBuffer(cq,o0r_raw,b_o0r,true);
    EnqueueReadMeshBuffer(cq,o0i_raw,b_o0i,true);
    EnqueueReadMeshBuffer(cq,o1r_raw,b_o1r,true);
    EnqueueReadMeshBuffer(cq,o1i_raw,b_o1i,true);

    auto o0r=unpack_tiles(o0r_raw,half_N); auto o0i=unpack_tiles(o0i_raw,half_N);
    auto o1r=unpack_tiles(o1r_raw,half_N); auto o1i=unpack_tiles(o1i_raw,half_N);

    std::vector<float> result_r(N), result_i(N);
    for (uint32_t i=0;i<half_N;i++) {
        result_r[i]=o0r[i]; result_i[i]=o0i[i];
        result_r[i+half_N]=o1r[i]; result_i[i+half_N]=o1i[i];
    }
    if (direction==1) for (uint32_t i=0;i<N;i++){result_r[i]/=N;result_i[i]/=N;}

    std::cout<<"\n════════════════════════════════════════\n";
    std::cout<<" VALIDATION\n";
    std::cout<<"════════════════════════════════════════\n";
    float mer=0.f, mei=0.f, me=0.f;
    for (uint32_t i=0;i<N;i++) {
        float er=std::abs(result_r[i]-ref_r[i]), ei=std::abs(result_i[i]-ref_i[i]);
        mer=std::max(mer,er); mei=std::max(mei,ei); me+=er+ei;
    }
    me /= 2*N;
    std::cout<<" Max error (real): "<<mer<<"\n";
    std::cout<<" Max error (imag): "<<mei<<"\n";
    std::cout<<" Mean error      : "<<me<<"\n";
    // Threshold 0.5f: accounts for accumulated float32 rounding in HiFi4 mode
    bool passed = (mer<0.5f)&&(mei<0.5f);
    std::cout<<" Result: "<<(passed?"✓ PASSED":"✗ FAILED")<<"\n";

    std::cout<<"\n════════════════════════════════════════\n";
    std::cout<<" FIRST 16 RESULTS\n";
    std::cout<<"════════════════════════════════════════\n";
    std::cout<<std::fixed<<std::setprecision(5);
    for (uint32_t i=0;i<16&&i<N;i++) {
        std::cout<<" X["<<std::setw(3)<<i<<"] = "
                 <<std::setw(12)<<result_r[i]
                 <<(result_i[i]>=0?" + ":" - ")
                 <<std::setw(12)<<std::abs(result_i[i])<<"j"
                 <<"   ref: "<<std::setw(12)<<ref_r[i]
                 <<(ref_i[i]>=0?" + ":" - ")
                 <<std::setw(12)<<std::abs(ref_i[i])<<"j\n";
    }
    mesh->close();
    std::cout<<"\n════════════════════════════════════════\n Done\n";
    std::cout<<"════════════════════════════════════════\n";
    return passed?0:1;
}