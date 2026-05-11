# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone reviewer-eyeball script for ttnn.experimental.fft / .ifft.
# Compares against torch.fft for a handful of representative sizes &
# precisions. Not a unit test — for the unit tests, see
# tests/ttnn/unit_tests/operations/experimental/fft/test_fft.py
#
# Run on a Wormhole box:
#     python tt_metal/programming_examples/fft_universal/fft_torch_compare.py

import math
import sys
import time

import torch
import ttnn


# ── helpers ─────────────────────────────────────────────────────────────────
def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return (torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item()


def to_complex(re_t, im_t) -> torch.Tensor:
    return torch.complex(
        ttnn.to_torch(re_t).reshape(-1).to(torch.float32),
        ttnn.to_torch(im_t).reshape(-1).to(torch.float32),
    )


def from_real(t: torch.Tensor, dtype, device):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def fmt(x: float) -> str:
    return f"{x:9.2e}"


def hr(width: int = 78) -> None:
    print("─" * width)


# ── individual checks ───────────────────────────────────────────────────────
def check_forward(N: int, dtype, precision: str, device) -> None:
    torch_in = torch.randn(N, dtype=torch.float32, generator=torch.Generator().manual_seed(N))
    tt_in = from_real(torch_in, dtype, device)

    t0 = time.perf_counter()
    if precision is None:
        tt_re, tt_im = ttnn.experimental.fft(tt_in)
    else:
        tt_re, tt_im = ttnn.experimental.fft(tt_in, precision=precision)
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    got = to_complex(tt_re, tt_im)
    ref = torch.fft.fft(torch_in.to(torch.complex64))
    err = rel_err(got, ref)

    label = f"fft  N={N:<8d} dtype={str(dtype).split('.')[-1]:<8} precision={precision or '(default)':<8}"
    print(f"  {label}  rel_err={fmt(err)}  time={elapsed_ms:7.2f} ms")


def check_roundtrip(N: int, dtype, device) -> None:
    torch_in = torch.randn(N, dtype=torch.float32, generator=torch.Generator().manual_seed(N + 1))
    tt_in = from_real(torch_in, dtype, device)

    spec_re, spec_im = ttnn.experimental.fft(tt_in)
    rec_re, rec_im = ttnn.experimental.ifft(spec_re, spec_im)

    got = ttnn.to_torch(rec_re).reshape(-1).to(torch.float32)
    err = rel_err(got, torch_in)

    label = f"ifft∘fft N={N:<8d} dtype={str(dtype).split('.')[-1]:<8}"
    print(f"  {label}                          rel_err={fmt(err)}")


def check_complex_input(N: int, device) -> None:
    re = torch.randn(N, dtype=torch.float32, generator=torch.Generator().manual_seed(N + 2))
    im = torch.randn(N, dtype=torch.float32, generator=torch.Generator().manual_seed(N + 3))
    tt_re = from_real(re, ttnn.float32, device)
    tt_im = from_real(im, ttnn.float32, device)

    out_re, out_im = ttnn.experimental.fft(tt_re, tt_im)
    got = to_complex(out_re, out_im)
    ref = torch.fft.fft(torch.complex(re, im))
    err = rel_err(got, ref)

    print(f"  complex input N={N:<8d} dtype=float32                          rel_err={fmt(err)}")


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    print("\nttnn.experimental.fft / .ifft  vs  torch.fft  parity sweep")
    hr()

    device = ttnn.open_device(device_id=0)
    try:
        # 1) small-N precision modes
        print("\n[1] Small-N fp32 — precision='precise' (default) vs 'fast'")
        for N in [6, 17, 24, 32]:
            check_forward(N, ttnn.float32, "precise", device)
            check_forward(N, ttnn.float32, "fast", device)

        # 2) medium / large pow2 fp32 (fft_stockham)
        print("\n[2] Stockham fp32 (pow2)")
        for N in [1024, 4096, 16384, 65536]:
            check_forward(N, ttnn.float32, None, device)

        # 3) non-pow2 fp32 (fft_universal — Bluestein / mixed radix)
        print("\n[3] Universal fp32 (non-pow2 / Bluestein)")
        for N in [96, 100, 1000, 4096 * 3]:  # last is 12288 = 2^12 * 3
            check_forward(N, ttnn.float32, None, device)

        # 4) bf16 (fft_universal_bf16)
        print("\n[4] bf16 (any N — precision kwarg ignored on bf16)")
        for N in [32, 256, 1024, 4096]:
            check_forward(N, ttnn.bfloat16, None, device)

        # 5) round-trip ifft(fft(x)) ≈ x
        print("\n[5] Round-trip ifft(fft(x)) ≈ x")
        for N, dt in [(96, ttnn.float32), (1024, ttnn.float32),
                      (4096, ttnn.float32), (256, ttnn.bfloat16)]:
            check_roundtrip(N, dt, device)

        # 6) complex-input forward FFT
        print("\n[6] Complex input — ttnn.experimental.fft(re, im)")
        for N in [128, 1024]:
            check_complex_input(N, device)

        hr()
        print("done\n")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
