// fft_host.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "tt_metal_api.h"   // TT-Metal host API

using namespace std;

// Helper to initialize test data (sine + cosine)
void initFFTData(vector<float>& real, vector<float>& imag, size_t N) {
    for (size_t i = 0; i < N; i++) {
        real[i] = sin(2.0f * M_PI * i / N);
        imag[i] = 0.0f;
    }
}

int main() {
    // FFT parameters
    const size_t N = 16;              // FFT size (must be power of 2)
    uint32_t direction = 0;           // 0 = forward FFT, 1 = inverse FFT

    // Allocate CPU vectors
    vector<float> data0_r(N), data0_i(N);
    vector<float> data1_r(N), data1_i(N);
    vector<float> twiddle_r(N), twiddle_i(N);

    // Initialize input data
    initFFTData(data0_r, data0_i, N);
    initFFTData(data1_r, data1_i, N);
    initFFTData(twiddle_r, twiddle_i, N);

    // Create TT-Metal device
    tt::Device device;
    device.init();

    // Create Command Buffers
    auto cb_data0_r = device.createCommandBuffer(N);
    auto cb_data0_i = device.createCommandBuffer(N);
    auto cb_data1_r = device.createCommandBuffer(N);
    auto cb_data1_i = device.createCommandBuffer(N);
    auto cb_twiddle_r = device.createCommandBuffer(N);
    auto cb_twiddle_i = device.createCommandBuffer(N);
    auto cb_out_data0_r = device.createCommandBuffer(N);
    auto cb_out_data0_i = device.createCommandBuffer(N);
    auto cb_out_data1_r = device.createCommandBuffer(N);
    auto cb_out_data1_i = device.createCommandBuffer(N);

    // Copy input data to CBs
    cb_data0_r.write(data0_r.data(), N);
    cb_data0_i.write(data0_i.data(), N);
    cb_data1_r.write(data1_r.data(), N);
    cb_data1_i.write(data1_i.data(), N);
    cb_twiddle_r.write(twiddle_r.data(), N);
    cb_twiddle_i.write(twiddle_i.data(), N);

    // Launch FFT kernel
    fft_kernel::MAIN();   // CB indices are pre-assigned inside kernel

    // Retrieve results back to CPU
    cb_out_data0_r.read(data0_r.data(), N);
    cb_out_data0_i.read(data0_i.data(), N);
    cb_out_data1_r.read(data1_r.data(), N);
    cb_out_data1_i.read(data1_i.data(), N);

    // Print FFT output
    cout << "FFT Output (data0):" << endl;
    for (size_t i = 0; i < N; i++) {
        cout << "R: " << data0_r[i] << " I: " << data0_i[i] << endl;
    }

    cout << "\nFFT Output (data1):" << endl;
    for (size_t i = 0; i < N; i++) {
        cout << "R: " << data1_r[i] << " I: " << data1_i[i] << endl;
    }

    device.shutdown();
    return 0;
}