# fft_1m_test.py — verify ttnn.fft on N = 1,048,576 (fp32, Stockham backend)
import time
import torch
import ttnn


def main():
    device = ttnn.open_device(device_id=0)
    try:
        N = 1024 * 1024  # 2^20 = 1,048,576

        # --- Build input ---------------------------------------------------
        torch.manual_seed(0)  # reproducible
        x = torch.randn(N, dtype=torch.float32)
        print(f"Input  : N={N:,d}  ({N*4 / 1024 / 1024:.1f} MB fp32)")

        # --- Reference on CPU ---------------------------------------------
        t0 = time.perf_counter()
        ref = torch.fft.fft(x)  # complex64
        t_cpu = time.perf_counter() - t0
        print(f"torch.fft.fft on host  : {t_cpu*1000:7.1f} ms")

        # --- Device round-trip --------------------------------------------
        tt_x = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        t0 = time.perf_counter()
        re, im = ttnn.fft(tt_x)  # → fft_stockham backend
        t_device = time.perf_counter() - t0
        print(f"ttnn.fft on Wormhole   : {t_device*1000:7.1f} ms")

        # --- Read result + correctness ------------------------------------
        got = torch.complex(
            ttnn.to_torch(re).reshape(-1),
            ttnn.to_torch(im).reshape(-1),
        )

        rel = (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()
        max_abs = (got - ref).abs().max().item()

        print(f"\nrel L2 error vs torch.fft : {rel:.3e}")
        print(f"max abs error            : {max_abs:.3e}")

        TOL = 5e-3  # ~10 stages of Stockham + bit-truncation
        if rel < TOL:
            print(f"PASS — within tolerance {TOL:.0e}")
        else:
            print(f"FAIL — exceeds tolerance {TOL:.0e}")

        # --- Spot-check first 4 bins (DC + low frequencies) --------------
        print("\nFirst 4 spectrum bins:")
        print(f"  ref [{0}] = {ref[0]:.4f}    got = {got[0]:.4f}")
        print(f"  ref [{1}] = {ref[1]:.4f}    got = {got[1]:.4f}")
        print(f"  ref [{2}] = {ref[2]:.4f}    got = {got[2]:.4f}")
        print(f"  ref [{3}] = {ref[3]:.4f}    got = {got[3]:.4f}")

        # --- IFFT round-trip (optional but cheap) -------------------------
        rec_re, rec_im = ttnn.ifft(re, im)
        rec = ttnn.to_torch(rec_re).reshape(-1)
        rt_rel = (torch.linalg.norm(rec - x) / torch.linalg.norm(x)).item()
        print(f"\nIFFT round-trip rel err : {rt_rel:.3e}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
