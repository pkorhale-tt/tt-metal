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

using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

std::vector<Complex> ref_dft(const std::vector<Complex>& x) {
    uint32_t N=x.size(); std::vector<Complex> X(N,{0,0});
    for (uint32_t k=0;k<N;k++) for (uint32_t n=0;n<N;n++) {
        float a=-2.f*float(M_PI)*k*n/N;
        X[k]+=x[n]*Complex(std::cos(a),std::sin(a));
    }
    return X;
}
std::vector<Complex> ref_idft(const std::vector<Complex>& X) {
    uint32_t N=X.size(); std::vector<Complex> x(N,{0,0});
    for (uint32_t n=0;n<N;n++) {
        for (uint32_t k=0;k<N;k++) {
            float a=2.f*float(M_PI)*k*n/N;
            x[n]+=X[k]*Complex(std::cos(a),std::sin(a));
        }
        x[n]/=float(N);
    }
    return x;
}
float max_err(const std::vector<Complex>& a,const std::vector<Complex>& b) {
    float e=0; for (size_t i=0;i<a.size();i++) e=std::max(e,std::abs(a[i]-b[i])); return e;
}
std::vector<Complex> make_impulse(uint32_t N)
    { std::vector<Complex> x(N,{0,0}); x[0]={1,0}; return x; }
std::vector<Complex> make_random(uint32_t N, uint32_t seed=42) {
    std::vector<Complex> x(N); std::srand(seed);
    for (auto& c:x) c={(std::rand()/float(RAND_MAX))*2-1,(std::rand()/float(RAND_MAX))*2-1};
    return x;
}

bool run_test(std::shared_ptr<MeshDevice> md,
              const std::vector<Complex>& input,
              uint32_t num_cores, bool is_ifft, const char* name)
{
    uint32_t N=input.size();
    auto ref = is_ifft ? ref_idft(input) : ref_dft(input);
    MeshCommandQueue& cq = md->mesh_command_queue();

    // Pack input into even/odd split layout
    auto packed = pack_input_even_odd(input, num_cores);
    auto in_buf  = make_mesh_buf(md, uint32_t(packed.size()*sizeof(float)), kTileSizeFp32);
    auto outr_buf = make_mesh_buf(md, num_cores*kTileSizeFp32, kTileSizeFp32);
    auto outi_buf = make_mesh_buf(md, num_cores*kTileSizeFp32, kTileSizeFp32);
    WriteShard(cq, in_buf, packed, MeshCoordinate(0,0), false);

    auto t0=std::chrono::high_resolution_clock::now();
    run_fft(md, {N,num_cores,is_ifft}, in_buf, outr_buf, outi_buf);
    double ms=std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-t0).count();

    std::vector<float> out_r, out_i;
    ReadShard(cq, out_r, outr_buf, MeshCoordinate(0,0), true);
    ReadShard(cq, out_i, outi_buf, MeshCoordinate(0,0), true);

    auto got = unpack_output(out_r, out_i, N, num_cores);
    float err=max_err(ref,got);
    bool pass=err<1e-3f;
    std::printf("[%s] N=%-5u cores=%-2u %s | err=%.2e | %.1f ms  %s\n",
        pass?"PASS":"FAIL",N,num_cores,is_ifft?"IFFT":"FFT ",err,ms,name);
    return pass;
}

int main() {
    auto md=MeshDevice::create_unit_mesh(0);
    bool all=true;
    all &= run_test(md, make_impulse(64),  1, false, "impulse FFT");
    all &= run_test(md, make_impulse(64),  1, true,  "impulse IFFT");
    all &= run_test(md, make_random(256),  4, false, "random 4-core");
    all &= run_test(md, make_random(1024), 8, false, "random 8-core");
    all &= run_test(md, make_random(1024), 8, true,  "IFFT 8-core");
    md.reset();
    std::printf("\n%s\n", all?"All tests PASSED.":"SOME TESTS FAILED.");
    return all?0:1;
}