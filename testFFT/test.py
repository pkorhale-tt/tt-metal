# fft_demo.py — exercises all 4 ttnn.fft backends end-to-end.
import torch
import ttnn


def run_one(device, N, dtype, label):
    """FFT one length-N signal, compare to torch.fft.fft, print rel err."""
    x_torch = torch.randn(N, dtype=torch.float32)
    ref = torch.fft.fft(x_torch)

    tt_x = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    re, im = ttnn.fft(tt_x)

    got = torch.complex(
        ttnn.to_torch(re).reshape(-1).to(torch.float32),
        ttnn.to_torch(im).reshape(-1).to(torch.float32),
    )
    rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
    print(f"  {label:30s}  N={N:>8d}  rel_err={rel:.2e}")


def run_roundtrip(device, N, dtype, label):
    """FFT → IFFT → check we recovered the input."""
    x_torch = torch.randn(N, dtype=torch.float32)
    tt_x = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    re, im = ttnn.fft(tt_x)
    rec_re, _ = ttnn.ifft(re, im)
    got = ttnn.to_torch(rec_re).reshape(-1).to(torch.float32)

    rel = (torch.linalg.norm(got - x_torch) / torch.linalg.norm(x_torch)).item()
    print(f"  {label:30s}  N={N:>8d}  rel_err={rel:.2e}  (roundtrip)")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        print("\n── Forward FFT — all 4 backends ──")
        run_one(device, 1024, ttnn.float32, "Stockham      (fp32 pow2)")
        run_one(device, 1000, ttnn.float32, "Universal     (fp32 composite)")
        run_one(device, 1009, ttnn.float32, "Universal     (fp32 prime/Bluestein)")
        run_one(device, 1024, ttnn.bfloat16, "UniversalBf16 (bf16 pow2)")
        run_one(device, 96, ttnn.bfloat16, "UniversalBf16 (bf16 composite)")

        print("\n── Roundtrip (FFT → IFFT) ──")
        run_roundtrip(device, 1024, ttnn.float32, "Stockham roundtrip")
        run_roundtrip(device, 256, ttnn.bfloat16, "Bf16 roundtrip")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
