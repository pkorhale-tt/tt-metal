import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Real and imaginary parts
real = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)

imag = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

print("real input :", real.tolist())
print("imag input :", imag.tolist())

# Torch complex tensor for reference
torchComplex = torch.complex(real, imag)

# TTNN tensors
ttReal = ttnn.from_torch(real, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

ttImag = ttnn.from_torch(imag, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

# Your FFT kernel
# Assumes your kernel API is:
# ttnn.fft(realTensor, imagTensor)

ttOutReal, ttOutImag = ttnn.fft(ttReal, ttImag)

# Convert back to torch
outReal = ttnn.to_torch(ttOutReal)
outImag = ttnn.to_torch(ttOutImag)

print("ttnn real :", outReal.tolist())
print("ttnn imag :", outImag.tolist())

# Torch reference
ref = torch.fft.fft(torchComplex)

print("torch real:", ref.real.tolist())
print("torch imag:", ref.imag.tolist())

# Numerical comparison
print("real match:", torch.allclose(outReal, ref.real, atol=1e-3, rtol=1e-3))

print("imag match:", torch.allclose(outImag, ref.imag, atol=1e-3, rtol=1e-3))

ttnn.close_device(device)
