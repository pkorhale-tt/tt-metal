import torch
import ttnn

device = ttnn.open_device(device_id=0)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
print("input :", x.tolist())

tt_x = ttnn.from_torch(
    x,
    dtype=ttnn.float32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)

# ttnn.fft returns a (real, imag) pair — unpack it.
tt_re, tt_im = ttnn.fft(tt_x)

# Bring back to host as plain Python lists for printing.
re = ttnn.to_torch(tt_re).reshape(-1).tolist()
im = ttnn.to_torch(tt_im).reshape(-1).tolist()

print("ttnn real :", [f"{v:+.4f}" for v in re])
print("ttnn imag :", [f"{v:+.4f}" for v in im])

# Reference — torch.fft.fft returns a complex tensor.
ref = torch.fft.fft(x)
print("torch real:", [f"{v:+.4f}" for v in ref.real.tolist()])
print("torch imag:", [f"{v:+.4f}" for v in ref.imag.tolist()])

# Quick sanity check.
got = torch.complex(
    ttnn.to_torch(tt_re).reshape(-1).to(torch.float32),
    ttnn.to_torch(tt_im).reshape(-1).to(torch.float32),
)
rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
print(f"\nrelative error vs torch.fft.fft: {rel:.2e}")

ttnn.close_device(device)
