import torch
import ttnn

device = ttnn.open_device(device_id=0)

x_torch = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
N = x_torch.numel()
print(f"input         : {x_torch.tolist()}  (N={N})")
print()


def run(label, tt_dtype, torch_dtype, precision="precise"):
    print(f"---- {label}  (precision={precision}) ----")

    x_tt = ttnn.from_torch(
        x_torch.to(torch_dtype),
        dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    re_tt, im_tt = ttnn.experimental.fft(x_tt, precision=precision)
    re = ttnn.to_torch(re_tt).reshape(-1).to(torch.float32)
    im = ttnn.to_torch(im_tt).reshape(-1).to(torch.float32)
    print("ttnn.experimental.fft  real:", [f"{v:+.4f}" for v in re.tolist()])
    print("ttnn.experimental.fft  imag:", [f"{v:+.4f}" for v in im.tolist()])

    rec_re_tt, rec_im_tt = ttnn.experimental.ifft(re_tt, im_tt, precision=precision)
    rec_re = ttnn.to_torch(rec_re_tt).reshape(-1).to(torch.float32)
    rec_im = ttnn.to_torch(rec_im_tt).reshape(-1).to(torch.float32)
    print("ttnn.experimental.ifft real:", [f"{v:+.4f}" for v in rec_re.tolist()])
    print("ttnn.experimental.ifft imag:", [f"{v:+.4f}" for v in rec_im.tolist()])

    err = (rec_re - x_torch).abs().max().item()
    print(f"round-trip err: {err:.2e}")
    print()


run("FP32 (precise)", ttnn.float32, torch.float32, precision="precise")
run("FP32 (fast)", ttnn.float32, torch.float32, precision="fast")
run("BFLOAT16", ttnn.bfloat16, torch.bfloat16)  # bf16 ignores precision


print("---- torch.fft (reference, complex64) ----")
ref = torch.fft.fft(x_torch.to(torch.complex64))
print("torch.fft real:", [f"{v:+.4f}" for v in ref.real.tolist()])
print("torch.fft imag:", [f"{v:+.4f}" for v in ref.imag.tolist()])

rec_torch = torch.fft.ifft(ref)
print("torch.ifft real:", [f"{v:+.4f}" for v in rec_torch.real.tolist()])
print("torch.ifft imag:", [f"{v:+.4f}" for v in rec_torch.imag.tolist()])

ttnn.close_device(device)
