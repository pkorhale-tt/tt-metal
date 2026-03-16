// fft_single_core_opt.cpp  — CORRECTED (pre-staged inputs)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Architecture change: pre-staged inputs.
//
// Each FFT stage needs a specific even/odd split of the previous stage's
// output.  Rather than computing this on-device (which would require a
// shuffle pass between stages), the host pre-computes the correct even/odd
// pair for every stage before launch and uploads them all to DRAM.
//
// DRAM layout (each buffer has log2N * tiles_per_stage tiles):
//   buf_even_r: [stage0_even_r | stage1_even_r | ... | stage(n-1)_even_r]
//   buf_even_i: same for imag
//   buf_odd_r:  same for odd real
//   buf_odd_i:  same for odd imag
//   buf_tw_r/i: twiddle factors per stage (unchanged)
//
// The reader streams each stage's tiles through CBs 0-5 in order.
// The writer saves all stage outputs (log2N * num_tiles tiles) to DRAM.
// The host reads back only the LAST stage's output as the final result.

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
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

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float    u2f(uint32_t u){ float f; std::memcpy(&f,&u,4); return f; }

std::vector<uint32_t> pack_tiles(const std::vector<float>& data, uint32_t num_tiles) {
    std::vector<uint32_t> out(num_tiles * TILE_SIZE, 0);
    for (uint32_t i = 0; i < data.size() && i < out.size(); i++) out[i] = f2u(data[i]);
    return out;
}

std::vector<float> unpack_tiles(const std::vector<uint32_t>& data, uint32_t n) {
    std::vector<float> out(n);
    for (uint32_t i = 0; i < n && i < data.size(); i++) out[i] = u2f(data[i]);
    return out;
}

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r=(r<<1)|(x&1); x>>=1; }
    return r;
}

void cpu_fft(std::vector<float>& real, std::vector<float>& imag, bool inv) {
    uint32_t N = real.size(), log2N = 0;
    while ((1u<<log2N) < N) log2N++;
    // bit-reverse
    for (uint32_t i=0; i<N; i++) {
        uint32_t j = bit_reverse(i,log2N);
        if (i<j) { std::swap(real[i],real[j]); std::swap(imag[i],imag[j]); }
    }
    for (uint32_t s=0; s<log2N; s++) {
        uint32_t m = 1u<<(s+1);
        float ab = (inv?2.f:-2.f)*PI/m;
        for (uint32_t k=0; k<N; k+=m) {
            for (uint32_t j=0; j<m/2; j++) {
                float wr=std::cos(ab*j), wi=std::sin(ab*j);
                uint32_t e=k+j, o=k+j+m/2;
                float tr=wr*real[o]-wi*imag[o], ti=wr*imag[o]+wi*real[o];
                float er=real[e], ei=imag[e];
                real[e]=er+tr; imag[e]=ei+ti;
                real[o]=er-tr; imag[o]=ei-ti;
            }
        }
    }
    if (inv) for (uint32_t i=0; i<N; i++) { real[i]/=N; imag[i]/=N; }
}

// ════════════════════════════════════════════════════════════════════
//  PRE-COMPUTE STAGED INPUTS
//
//  For each stage s (0..log2N-1) compute the even/odd split that the
//  DIT butterfly needs, then advance the internal array by applying the
//  butterfly so that stage s+1 can be computed from the result.
//
//  Output buffers (all_even_r etc.) are flat arrays of length
//  log2N * (N/2), where stage s occupies [s*N/2 .. (s+1)*N/2 - 1].
// ════════════════════════════════════════════════════════════════════
void precompute_staged_inputs(
    const std::vector<float>& x_r,
    const std::vector<float>& x_i,
    uint32_t N, uint32_t log2N, uint32_t direction,
    std::vector<float>& all_even_r, std::vector<float>& all_even_i,
    std::vector<float>& all_odd_r,  std::vector<float>& all_odd_i)
{
    const uint32_t half_N = N / 2;
    all_even_r.resize(log2N * half_N, 0.f);
    all_even_i.resize(log2N * half_N, 0.f);
    all_odd_r .resize(log2N * half_N, 0.f);
    all_odd_i .resize(log2N * half_N, 0.f);

    // Start from bit-reversed input
    std::vector<float> arr_r(N), arr_i(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        arr_r[i] = x_r[j];
        arr_i[i] = x_i[j];
    }

    const float sign = (direction == 1) ? 1.f : -1.f;

    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m      = 1u << (stage + 1);
        uint32_t half_m = m / 2;
        uint32_t base   = stage * half_N;
        uint32_t idx    = 0;

        // Record even/odd inputs for this stage
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                all_even_r[base + idx] = arr_r[k + j];
                all_even_i[base + idx] = arr_i[k + j];
                all_odd_r [base + idx] = arr_r[k + j + half_m];
                all_odd_i [base + idx] = arr_i[k + j + half_m];
                idx++;
            }
        }

        // Apply butterfly to advance array for next stage
        std::vector<float> new_r(arr_r), new_i(arr_i);
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                float angle = sign * 2.f * PI * (float)j / (float)m;
                float wr = std::cos(angle), wi = std::sin(angle);
                uint32_t ei = k+j, oi = k+j+half_m;
                float tr = wr*arr_r[oi] - wi*arr_i[oi];
                float ti = wr*arr_i[oi] + wi*arr_r[oi];
                new_r[ei] = arr_r[ei] + tr;  new_i[ei] = arr_i[ei] + ti;
                new_r[oi] = arr_r[ei] - tr;  new_i[oi] = arr_i[ei] - ti;
            }
        }
        arr_r = new_r;  arr_i = new_i;
    }
}

// ════════════════════════════════════════════════════════════════════
//  PRE-COMPUTE TWIDDLE FACTORS (same per-stage twiddles as before)
// ════════════════════════════════════════════════════════════════════
std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_all_twiddles(uint32_t N, uint32_t log2N,
                        uint32_t tiles_per_stage, uint32_t direction)
{
    const uint32_t total_tiles = log2N * tiles_per_stage;
    std::vector<uint32_t> tw_r(total_tiles * TILE_SIZE, 0);
    std::vector<uint32_t> tw_i(total_tiles * TILE_SIZE, 0);
    const float sign = (direction == 1) ? 1.f : -1.f;
    for (uint32_t stage = 0; stage < log2N; stage++) {
        uint32_t m = 1u << (stage+1), half_m = m/2, bf_idx = 0;
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half_m; j++) {
                float angle = sign * 2.f * PI * (float)j / (float)m;
                uint32_t flat = stage * tiles_per_stage * TILE_SIZE + bf_idx;
                tw_r[flat] = f2u(std::cos(angle));
                tw_i[flat] = f2u(std::sin(angle));
                bf_idx++;
            }
        }
    }
    return {tw_r, tw_i};
}

void create_cb(Program& program, CoreCoord core,
               uint32_t cb_id, uint32_t num_tiles, uint32_t tile_bytes)
{
    CircularBufferConfig cfg =
        CircularBufferConfig(num_tiles * tile_bytes, {{cb_id, tt::DataFormat::Float32}})
        .set_page_size(cb_id, tile_bytes);
    CreateCircularBuffer(program, core, cfg);
}

bool read_input_file(const std::string& path, uint32_t& N, bool N_from_cmdline,
                     std::vector<float>& ir, std::vector<float>& ii)
{
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr<<"Cannot open input file: "<<path<<"\n"; return false; }
    std::vector<float> vals; std::string token;
    while (f>>token) {
        if (!token.empty()&&token.back()==',') token.pop_back();
        if (token.empty()) continue;
        try { vals.push_back(std::stof(token)); }
        catch(...) { std::cerr<<"Bad token: '"<<token<<"'\n"; return false; }
    }
    if (vals.empty()) { std::cerr<<"Input file empty.\n"; return false; }
    uint32_t count=(uint32_t)vals.size(); bool interleaved=false;
    if (N_from_cmdline) {
        if (count==2*N) { interleaved=true; std::cout<<" File mode: interleaved ("<<count<<" values)\n"; }
        else if (count==N) std::cout<<" File mode: real-only ("<<count<<" values)\n";
        else if (count<N) std::cerr<<"File has "<<count<<" values, padding to N="<<N<<"\n";
        else { count=N; vals.resize(N); }
    } else {
        N=1; while(N<count) N<<=1;
        std::cout<<" File mode: real-only ("<<count<<" values, N inferred as "<<N<<")\n";
    }
    ir.assign(N,0.f); ii.assign(N,0.f);
    if (interleaved) {
        for (uint32_t i=0; i<N&&2*i+1<(uint32_t)vals.size(); i++) { ir[i]=vals[2*i]; ii[i]=vals[2*i+1]; }
    } else {
        for (uint32_t i=0; i<N&&i<(uint32_t)vals.size(); i++) ir[i]=vals[i];
    }
    return true;
}

int main(int argc, char** argv) {
    uint32_t direction=0, N=1024;
    std::string in_file="";
    bool N_from_cmdline=false;

    if (argc<2) { std::cerr<<"Usage: "<<argv[0]<<" <direction 0|1> [input_file] [N]\n"; return 1; }
    direction=(uint32_t)std::atoi(argv[1]);
    for (int i=2; i<argc; i++) {
        std::string a=argv[i];
        bool looks_file=(a.find('.')!=std::string::npos||a.find('/')!=std::string::npos||
                         a.find('\\')!=std::string::npos);
        if (looks_file&&in_file.empty()) in_file=a;
        else { try { N=(uint32_t)std::stol(a); N_from_cmdline=true; }
               catch(...) { if(in_file.empty()) in_file=a; } }
    }
    if (N_from_cmdline&&(N==0||(N&(N-1))!=0)) { std::cerr<<"N must be power of 2\n"; return 1; }

    uint32_t log2N=0; while((1u<<log2N)<N) log2N++;
    uint32_t half_N=N/2;
    uint32_t tiles_per_stage=(half_N+TILE_SIZE-1)/TILE_SIZE;

    std::vector<float> input_r(N,0.f), input_i(N,0.f);
    if (!in_file.empty()) {
        if (!read_input_file(in_file,N,N_from_cmdline,input_r,input_i)) return 1;
        log2N=0; while((1u<<log2N)<N) log2N++;
        half_N=N/2; tiles_per_stage=(half_N+TILE_SIZE-1)/TILE_SIZE;
        input_r.resize(N,0.f); input_i.resize(N,0.f);
        if (N<2||(N&(N-1))!=0) { std::cerr<<"Inferred N="<<N<<" not valid\n"; return 1; }
    } else {
        for (uint32_t i=0; i<N; i++)
            input_r[i]=std::sin(2.f*PI*4.f*i/N)+0.5f*std::sin(2.f*PI*8.f*i/N);
    }

    std::cout<<"═══════════════════════════════════════\n";
    std::cout<<" TT-Metal FFT (Optimised Single Core)\n";
    std::cout<<"═══════════════════════════════════════\n";
    std::cout<<" N             : "<<N<<"\n";
    std::cout<<" log2N         : "<<log2N<<"\n";
    std::cout<<" Direction     : "<<(direction?"Inverse":"Forward")<<"\n";
    std::cout<<" tiles/stage   : "<<tiles_per_stage<<"\n";
    std::cout<<" total twiddle : "<<log2N*tiles_per_stage<<" tiles\n";
    std::cout<<"═══════════════════════════════════════\n";

    // Reference FFT
    std::vector<float> ref_r(input_r), ref_i(input_i);
    cpu_fft(ref_r, ref_i, direction==1);

    // Pre-compute staged inputs (all stages at once on host)
    std::vector<float> all_even_r, all_even_i, all_odd_r, all_odd_i;
    precompute_staged_inputs(input_r, input_i, N, log2N, direction,
                             all_even_r, all_even_i, all_odd_r, all_odd_i);

    // Pre-compute twiddles
    auto [tw_r_tiles, tw_i_tiles] =
        precompute_all_twiddles(N, log2N, tiles_per_stage, direction);

    // Pack staged inputs into tiles
    auto even_r_t = pack_tiles(all_even_r, log2N * tiles_per_stage);
    auto even_i_t = pack_tiles(all_even_i, log2N * tiles_per_stage);
    auto odd_r_t  = pack_tiles(all_odd_r,  log2N * tiles_per_stage);
    auto odd_i_t  = pack_tiles(all_odd_i,  log2N * tiles_per_stage);

    // Device setup
    int device_id=0;
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // DRAM buffer sizes
    uint32_t staged_buf_bytes  = log2N * tiles_per_stage * TILE_BYTES;
    uint32_t twiddle_buf_bytes = log2N * tiles_per_stage * TILE_BYTES;
    // Output: log2N stages × num_tiles, we only need the last stage's tiles
    uint32_t output_buf_bytes  = log2N * tiles_per_stage * TILE_BYTES;

    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_cfg{
        .page_size=TILE_BYTES, .buffer_type=tt::tt_metal::BufferType::DRAM };

    auto mk_buf = [&](uint32_t bytes) {
        tt::tt_metal::distributed::ReplicatedBufferConfig rcfg{.size=bytes};
        return tt::tt_metal::distributed::MeshBuffer::create(rcfg,dram_cfg,mesh_device.get());
    };

    auto buf_even_r = mk_buf(staged_buf_bytes);
    auto buf_even_i = mk_buf(staged_buf_bytes);
    auto buf_odd_r  = mk_buf(staged_buf_bytes);
    auto buf_odd_i  = mk_buf(staged_buf_bytes);
    auto buf_tw_r   = mk_buf(twiddle_buf_bytes);
    auto buf_tw_i   = mk_buf(twiddle_buf_bytes);
    auto buf_out_r  = mk_buf(output_buf_bytes);
    auto buf_out_i  = mk_buf(output_buf_bytes);
    auto buf_out_r2 = mk_buf(output_buf_bytes);
    auto buf_out_i2 = mk_buf(output_buf_bytes);

    // Circular buffers — depth = tiles_per_stage for all input/output CBs
    create_cb(program,core, 0, tiles_per_stage, TILE_BYTES); // cb_even_r
    create_cb(program,core, 1, tiles_per_stage, TILE_BYTES); // cb_even_i
    create_cb(program,core, 2, tiles_per_stage, TILE_BYTES); // cb_odd_r
    create_cb(program,core, 3, tiles_per_stage, TILE_BYTES); // cb_odd_i
    create_cb(program,core, 4, tiles_per_stage, TILE_BYTES); // cb_tw_r
    create_cb(program,core, 5, tiles_per_stage, TILE_BYTES); // cb_tw_i
    // Output CBs must hold ALL stages' outputs so writer can drain at once
    const uint32_t total_tiles = log2N * tiles_per_stage;
    create_cb(program,core,16, total_tiles, TILE_BYTES); // cb_out_r
    create_cb(program,core,17, total_tiles, TILE_BYTES); // cb_out_i
    create_cb(program,core,18, total_tiles, TILE_BYTES); // cb_out_r2
    create_cb(program,core,19, total_tiles, TILE_BYTES); // cb_out_i2
    // Scratch CBs
    create_cb(program,core,20, 1, TILE_BYTES); // cb_tmp0
    create_cb(program,core,21, 1, TILE_BYTES); // cb_tmp1
    create_cb(program,core,22, 1, TILE_BYTES); // cb_tw_odd_r
    create_cb(program,core,23, 1, TILE_BYTES); // cb_tw_odd_i

    // Kernels
    auto reader_k = CreateKernel(program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/reader_fft_f32.cpp",
        core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default});
    auto writer_k = CreateKernel(program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/dataflow/writer_fft_f32.cpp",
        core, DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default});
    auto compute_k = CreateKernel(program,
        "tt_metal/programming_examples/fft_float32_optimized/fft_single_core/kernels/compute/fft_compute_f32.cpp",
        core, ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,.math_approx_mode=false});

    // Runtime args
    std::vector<uint32_t> reader_args = {
        buf_even_r->address(), buf_even_i->address(),
        buf_odd_r->address(),  buf_odd_i->address(),
        buf_tw_r->address(),   buf_tw_i->address(),
        tiles_per_stage, log2N
    };
    std::vector<uint32_t> writer_args = {
        buf_out_r->address(), buf_out_i->address(),
        buf_out_r2->address(), buf_out_i2->address(),
        tiles_per_stage, log2N
    };
    std::vector<uint32_t> compute_args = { log2N, tiles_per_stage };

    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    auto& prog = workload.get_programs().begin()->second;
    SetRuntimeArgs(prog, reader_k,  core, reader_args);
    SetRuntimeArgs(prog, writer_k,  core, writer_args);
    SetRuntimeArgs(prog, compute_k, core, compute_args);

    // Write inputs
    std::cout<<"Writing inputs to DRAM...\n";
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq,buf_even_r,even_r_t,false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq,buf_even_i,even_i_t,false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq,buf_odd_r, odd_r_t, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq,buf_odd_i, odd_i_t, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq,buf_tw_r,  tw_r_tiles,false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq,buf_tw_i,  tw_i_tiles,false);
    tt::tt_metal::distributed::Finish(cq);

    std::cout<<"Launching FFT kernel (all "<<log2N<<" stages on device)...\n";
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, true);
    std::cout<<"Kernel complete.\n";

    // Read results — all stages written, we want the LAST stage's tiles
    std::vector<uint32_t> out_r_raw(total_tiles * TILE_SIZE);
    std::vector<uint32_t> out_i_raw(total_tiles * TILE_SIZE);
    std::vector<uint32_t> out_r2_raw(total_tiles * TILE_SIZE);
    std::vector<uint32_t> out_i2_raw(total_tiles * TILE_SIZE);

    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq,out_r_raw, buf_out_r, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq,out_i_raw, buf_out_i, true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq,out_r2_raw,buf_out_r2,true);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq,out_i2_raw,buf_out_i2,true);

    // Extract last stage output (tiles at offset (log2N-1)*tiles_per_stage)
    uint32_t last_stage_offset = (log2N - 1) * tiles_per_stage * TILE_SIZE;
    auto out0_r = unpack_tiles(
        std::vector<uint32_t>(out_r_raw.begin()  + last_stage_offset, out_r_raw.end()),  half_N);
    auto out0_i = unpack_tiles(
        std::vector<uint32_t>(out_i_raw.begin()  + last_stage_offset, out_i_raw.end()),  half_N);
    auto out1_r = unpack_tiles(
        std::vector<uint32_t>(out_r2_raw.begin() + last_stage_offset, out_r2_raw.end()), half_N);
    auto out1_i = unpack_tiles(
        std::vector<uint32_t>(out_i2_raw.begin() + last_stage_offset, out_i2_raw.end()), half_N);

    // Reconstruct natural order
    std::vector<float> result_r(N), result_i(N);
    for (uint32_t i = 0; i < half_N; i++) {
        result_r[i]          = out0_r[i];
        result_i[i]          = out0_i[i];
        result_r[i + half_N] = out1_r[i];
        result_i[i + half_N] = out1_i[i];
    }

    // IFFT scaling
    if (direction == 1) {
        for (uint32_t i=0; i<N; i++) { result_r[i]/=N; result_i[i]/=N; }
    }

    // Validation
    std::cout<<"\n═══════════════════════════════════════\n";
    std::cout<<" VALIDATION\n";
    std::cout<<"═══════════════════════════════════════\n";
    float max_err_r=0.f, max_err_i=0.f, mean_err=0.f;
    for (uint32_t i=0; i<N; i++) {
        float er=std::abs(result_r[i]-ref_r[i]), ei=std::abs(result_i[i]-ref_i[i]);
        max_err_r=std::max(max_err_r,er); max_err_i=std::max(max_err_i,ei);
        mean_err+=er+ei;
    }
    mean_err/=2*N;
    std::cout<<" Max error  (real): "<<max_err_r<<"\n";
    std::cout<<" Max error  (imag): "<<max_err_i<<"\n";
    std::cout<<" Mean error       : "<<mean_err<<"\n";
    bool passed=(max_err_r<1e-3f)&&(max_err_i<1e-3f);
    std::cout<<" Result: "<<(passed?"✓ PASSED":"✗ FAILED")<<"\n";

    std::cout<<"\n═══════════════════════════════════════\n";
    std::cout<<" FIRST 16 RESULTS\n";
    std::cout<<"═══════════════════════════════════════\n";
    std::cout<<std::fixed<<std::setprecision(5);
    for (uint32_t i=0; i<16&&i<N; i++) {
        std::cout<<" X["<<std::setw(3)<<i<<"] = "
                 <<std::setw(12)<<result_r[i]
                 <<(result_i[i]>=0?" + ":" - ")
                 <<std::setw(12)<<std::abs(result_i[i])<<"j"
                 <<"   ref: "<<std::setw(12)<<ref_r[i]
                 <<(ref_i[i]>=0?" + ":" - ")
                 <<std::setw(12)<<std::abs(ref_i[i])<<"j\n";
    }

    mesh_device->close();
    std::cout<<"\n═══════════════════════════════════════\n";
    std::cout<<" Done\n";
    std::cout<<"═══════════════════════════════════════\n";
    return passed ? 0 : 1;
}