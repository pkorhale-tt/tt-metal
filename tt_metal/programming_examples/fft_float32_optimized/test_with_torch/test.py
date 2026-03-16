import torch

# Create input signal
signal = torch.tensor([22.10, 90.98, 88.12, 11.78], dtype=torch.float32)

# Compute Fourier Transform
fftResult = torch.fft.fft(signal)

print("Input Signal:", signal)
print("FFT Result:", fftResult)