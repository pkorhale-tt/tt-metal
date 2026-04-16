// ============================================================
// fft_test.cpp — TT-Metalium current API (MeshDevice path)
// ============================================================

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include "tt-metalium/mesh_coord.hpp"

#include "fft_host.cpp"

#include <cmath>
#include <complex>
#include <vector>
#include <cstdio>
#include <chrono>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Complex = std::complex<float>;

std::vector<Complex> ref_dft(const std::vector<Complex>& x) {
    uint32_t N=x.size(); std::vector<Complex> X(N,{0,0});
    for (uint32_t k=0;k<N;k++) for (uint32_t n=0;n<N;n++) {
        float a=-2.f*float(M_PI)*k*n/N;
        X[k]+=x[n]*Complex(std::cos(a),std::sin(a));
    } return X;
}
std::vector<Complex> ref_idft(const std::vector<Complex>& X) {
    uint32_t N=X.size(); std::vector<Complex> x(N,{0,0});
    for (uint32_t n=0;n<N;n++) {
        for (uint32_t k=0;k<N;k++) {
            float a=2.f*float(M_PI)*k*n/N;
            x[n]+=X[k]*Complex(std::cos(a),std::sin(a));
        } x[n]/=float(N);
    } return x;
}
std::vector<float> to_flat(const std::vector<Complex>& v) {
    std::vector<float> o; for (auto& c:v){o.push_back(c.real());o.push_back(c.imag());} return o;
}
std::vector<Complex> from_flat(const std::vector<float>& v) {
    std::vector<Complex> o; for (size_t i=0;i<v.size();i+=2) o.push_back({v[i],v[i+1]}); return o;
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
    uint32_t N=input.size(), bytes=N*2*sizeof(float);
    auto ref = is_ifft ? ref_idft(input) : ref_dft(input);

    // IDevice* for buffer allocation
    IDevice* device = md->get_device(0, 0);
    MeshCommandQueue& cq = md->mesh_command_queue();

    auto in_buf = CreateBuffer(InterleavedBufferConfig{
        .device=device,.size=bytes,.page_size=4u,.buffer_type=BufferType::DRAM});
    auto out_buf = CreateBuffer(InterleavedBufferConfig{
        .device=device,.size=bytes,.page_size=4u,.buffer_type=BufferType::DRAM});

    auto flat = to_flat(input);
    EnqueueWriteBuffer(cq, in_buf, flat.data(), false);

    auto t0=std::chrono::high_resolution_clock::now();
    run_fft(md, {N,num_cores,is_ifft}, in_buf, out_buf);
    double ms=std::chrono::duration<double,std::milli>(
        std::chrono::high_resolution_clock::now()-t0).count();

    std::vector<float> out_flat(N*2);
    EnqueueReadBuffer(cq, out_buf, out_flat.data(), true);

    float err=max_err(ref,from_flat(out_flat));
    bool pass=err<1e-4f;
    std::printf("[%s] N=%-5u cores=%-2u %s | err=%.2e | %.1f ms  %s\n",
        pass?"PASS":"FAIL",N,num_cores,is_ifft?"IFFT":"FFT ",err,ms,name);
    return pass;
}

int main() {
    // Open single device as 1x1 MeshDevice
    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig{.mesh_shape=MeshShape{1,1}});

    bool all=true;
    all &= run_test(mesh_device, make_impulse(64),  1, false, "impulse FFT");
    all &= run_test(mesh_device, make_impulse(64),  1, true,  "impulse IFFT");
    all &= run_test(mesh_device, make_random(256),  4, false, "random 4-core");
    all &= run_test(mesh_device, make_random(1024), 8, false, "random 8-core");
    all &= run_test(mesh_device, make_random(1024), 8, true,  "IFFT 8-core");

    // Round-trip
    {
        uint32_t N=1024, bytes=N*2*sizeof(float);
        IDevice* device=mesh_device->get_device(0,0);
        MeshCommandQueue& cq=mesh_device->mesh_command_queue();
        auto x=make_random(N); auto flat=to_flat(x);
        auto b0=CreateBuffer(InterleavedBufferConfig{.device=device,.size=bytes,.page_size=4u,.buffer_type=BufferType::DRAM});
        auto b1=CreateBuffer(InterleavedBufferConfig{.device=device,.size=bytes,.page_size=4u,.buffer_type=BufferType::DRAM});
        auto b2=CreateBuffer(InterleavedBufferConfig{.device=device,.size=bytes,.page_size=4u,.buffer_type=BufferType::DRAM});
        EnqueueWriteBuffer(cq,b0,flat.data(),false);
        fft( mesh_device,N,8,b0,b1);
        ifft(mesh_device,N,8,b1,b2);
        std::vector<float> out(N*2);
        EnqueueReadBuffer(cq,b2,out.data(),true);
        float e=max_err(x,from_flat(out)); bool p=e<1e-4f; all&=p;
        std::printf("[%s] round-trip N=1024 | err=%.2e\n",p?"PASS":"FAIL",e);
    }

    // MeshDevice closes on destruction (shared_ptr)
    mesh_device.reset();

    std::printf("\n%s\n",all?"All tests PASSED.":"SOME TESTS FAILED.");
    return all?0:1;
}