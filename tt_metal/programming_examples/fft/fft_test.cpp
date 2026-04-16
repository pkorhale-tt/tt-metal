// ============================================================
// fft_test.cpp — correctness tests for multi-core FFT
// ============================================================

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/constants.hpp"

#include "fft_host.cpp"

#include <cmath>
#include <complex>
#include <vector>
#include <cstdio>
#include <chrono>
#include <algorithm>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

// Note: kTileHW and kTileSizeFp32 defined in fft_host.cpp — don't redefine

// ── Reference DFT ────────────────────────────────────────────
std::vector<Complex> ref_dft(const std::vector<Complex>& x) {
    uint32_t N = x.size();
    std::vector<Complex> X(N, {0,0});
    for (uint32_t k=0; k<N; k++)
        for (uint32_t n=0; n<N; n++) {
            float a = -2.f*float(M_PI)*k*n/N;
            X[k] += x[n] * Complex(std::cos(a), std::sin(a));
        }
    return X;
}
std::vector<Complex> ref_idft(const std::vector<Complex>& X) {
    uint32_t N = X.size();
    std::vector<Complex> x(N, {0,0});
    for (uint32_t n=0; n<N; n++) {
        for (uint32_t k=0; k<N; k++) {
            float a = 2.f*float(M_PI)*k*n/N;
            x[n] += X[k] * Complex(std::cos(a), std::sin(a));
        }
        x[n] /= float(N);
    }
    return x;
}

float max_err(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    float e = 0;
    for (size_t i=0; i<a.size(); i++) e = std::max(e, std::abs(a[i]-b[i]));
    return e;
}

std::vector<Complex> make_impulse(uint32_t N)
    { std::vector<Complex> x(N,{0,0}); x[0]={1,0}; return x; }

std::vector<Complex> make_random(uint32_t N, uint32_t seed=42) {
    std::vector<Complex> x(N); std::srand(seed);
    for (auto& c : x)
        c = {(std::rand()/float(RAND_MAX))*2-1,
             (std::rand()/float(RAND_MAX))*2-1};
    return x;
}

// Buffer layout: 2*num_cores tiles, first half real, second half imag.
// Each tile stores local_N fp32 values in first local_N positions.
std::vector<float> pack_input(const std::vector<Complex>& x, uint32_t num_cores) {
    uint32_t N       = x.size();
    uint32_t local_N = N / num_cores;
    std::vector<float> buf(2 * num_cores * kTileElems, 0.0f);
    for (uint32_t c=0; c<num_cores; c++) {
        float* tile_r = buf.data() + c * kTileElems;
        float* tile_i = buf.data() + (num_cores + c) * kTileElems;
        for (uint32_t i=0; i<local_N; i++) {
            tile_r[i] = x[c * local_N + i].real();
            tile_i[i] = x[c * local_N + i].imag();
        }
    }
    return buf;
}

std::vector<Complex> unpack_output(
    const std::vector<float>& buf, uint32_t N, uint32_t num_cores)
{
    uint32_t local_N = N / num_cores;
    std::vector<Complex> out(N);
    for (uint32_t c=0; c<num_cores; c++) {
        const float* tile_r = buf.data() + c * kTileElems;
        const float* tile_i = buf.data() + (num_cores + c) * kTileElems;
        for (uint32_t i=0; i<local_N; i++)
            out[c * local_N + i] = {tile_r[i], tile_i[i]};
    }
    return out;
}

bool run_test(std::shared_ptr<MeshDevice> md,
              const std::vector<Complex>& input,
              uint32_t num_cores, bool is_ifft, const char* name)
{
    uint32_t N       = input.size();
    auto ref         = is_ifft ? ref_idft(input) : ref_dft(input);

    MeshCommandQueue& cq = md->mesh_command_queue();
    uint32_t buf_bytes   = 2 * num_cores * kTileSizeFp32;

    auto in_buf  = make_mesh_buf(md, buf_bytes, kTileSizeFp32);
    auto out_buf = make_mesh_buf(md, buf_bytes, kTileSizeFp32);

    auto flat = pack_input(input, num_cores);
    WriteShard(cq, in_buf, flat, MeshCoordinate(0,0), false);

    auto t0 = std::chrono::high_resolution_clock::now();
    run_fft(md, {N, num_cores, is_ifft}, in_buf, out_buf);
    double ms = std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-t0).count();

    // FIX: ReadShard takes (cq, mesh_buf, vector&, coord, blocking)
    // i.e. mesh_buf is 2nd arg, vector is 3rd
    std::vector<float> out_flat;
    ReadShard(cq, out_flat, out_buf, MeshCoordinate(0,0), true);

    auto  got  = unpack_output(out_flat, N, num_cores);
    float err  = max_err(ref, got);
    bool  pass = err < 1e-3f;

    std::printf("[%s] N=%-5u cores=%-2u %s | err=%.2e | %.1f ms  %s\n",
        pass?"PASS":"FAIL", N, num_cores,
        is_ifft?"IFFT":"FFT ", err, ms, name);
    return pass;
}

int main() {
    auto md  = MeshDevice::create_unit_mesh(0);
    bool all = true;

    all &= run_test(md, make_impulse(64),  1, false, "impulse FFT");
    all &= run_test(md, make_impulse(64),  1, true,  "impulse IFFT");
    all &= run_test(md, make_random(256),  4, false, "random 4-core");
    all &= run_test(md, make_random(1024), 8, false, "random 8-core");
    all &= run_test(md, make_random(1024), 8, true,  "IFFT 8-core");

    // Round-trip
    {
        auto x         = make_random(1024);
        uint32_t N     = 1024, nc = 8;
        uint32_t bytes = 2 * nc * kTileSizeFp32;
        MeshCommandQueue& cq = md->mesh_command_queue();
        auto b0 = make_mesh_buf(md,bytes,kTileSizeFp32);
        auto b1 = make_mesh_buf(md,bytes,kTileSizeFp32);
        auto b2 = make_mesh_buf(md,bytes,kTileSizeFp32);
        auto flat = pack_input(x, nc);
        WriteShard(cq,b0,flat,MeshCoordinate(0,0),false);
        fft( md,N,nc,b0,b1);
        ifft(md,N,nc,b1,b2);
        std::vector<float> out;
        ReadShard(cq,out,b2,MeshCoordinate(0,0),true);
        float e = max_err(x, unpack_output(out,N,nc));
        bool  p = e < 1e-3f; all &= p;
        std::printf("[%s] round-trip N=1024 | err=%.2e\n", p?"PASS":"FAIL", e);
    }

    md.reset();
    std::printf("\n%s\n", all?"All tests PASSED.":"SOME TESTS FAILED.");
    return all ? 0 : 1;
}