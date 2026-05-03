#!/usr/bin/env python3
"""
demo_inputs.py — feed MANY kinds of input through both torch.fft.fft and
tt_fft.fft, print them side by side. Built for live presentation.

Run:
    python demo_inputs.py                # show all built-in examples
    python demo_inputs.py --N 1024       # change FFT length for the long ones
    python demo_inputs.py --only manual  # just the [1,2,3,4] example
    python demo_inputs.py --user 1,2,3,4,5,6,7,8   # your own values
    python demo_inputs.py --interactive  # type values at a prompt
    python demo_inputs.py --plot         # also save spectra plots to PNG

Each example prints:
    -> torch.fft.fft(input)
    -> tt_fft.fft(input)         (runs on Tenstorrent Wormhole)
    -> max abs diff
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

import tt_fft

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _short_complex(c, prec=4) -> str:
    """Format a complex number compactly: '+1.234+5.678j'."""
    return f"{c.real:+.{prec}f}{c.imag:+.{prec}f}j"


def _vec_preview(v, n=6) -> str:
    """Show first n entries of a complex vector as a one-liner."""
    v = np.asarray(v).ravel()
    if v.size <= 2 * n:
        return "[" + ", ".join(_short_complex(c) for c in v) + "]"
    head = ", ".join(_short_complex(c) for c in v[:n])
    tail = ", ".join(_short_complex(c) for c in v[-2:])
    return f"[{head},  ...({v.size - n - 2} more)...,  {tail}]"


def _print_compare(label: str, x, X_torch, X_tt, dt_torch_ms, dt_tt_ms) -> None:
    diff = np.max(np.abs(np.asarray(X_torch) - np.asarray(X_tt)))
    ref  = max(float(np.max(np.abs(X_torch))), 1e-30)
    rel  = diff / ref

    print()
    print("=" * 78)
    print(f"  {label}   (N = {len(x)})")
    print(f"  device path : {tt_fft.device_path(len(x))}")
    print("=" * 78)
    print(f"  input            : {_vec_preview(np.asarray(x).astype(np.complex128))}")
    print(f"  torch.fft.fft    : {_vec_preview(np.asarray(X_torch))}")
    print(f"                     wall = {dt_torch_ms:8.3f} ms  (CPU)")
    print(f"  tt_fft.fft       : {_vec_preview(np.asarray(X_tt))}")
    print(f"                     wall = {dt_tt_ms:8.3f} ms  (Wormhole, incl. dispatch)")
    print(f"  max abs diff     : {diff:.4e}")
    print(f"  max rel diff     : {rel:.4e}")


# ---------------------------------------------------------------------------
# The compare primitive — the only place that calls both FFT engines
# ---------------------------------------------------------------------------

def compare(label: str, x, plot_path: str | None = None) -> None:
    """Run both FFTs on the same input and print a comparison."""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    # torch reference
    t0 = time.time()
    X_torch = torch.fft.fft(torch.as_tensor(x_np)).numpy()
    dt_torch_ms = (time.time() - t0) * 1000.0

    # tt_fft on Wormhole
    t0 = time.time()
    X_tt = tt_fft.fft(x_np)
    dt_tt_ms = (time.time() - t0) * 1000.0

    _print_compare(label, x_np, X_torch, X_tt, dt_torch_ms, dt_tt_ms)

    if HAVE_PLT and plot_path is not None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].plot(np.abs(X_torch), label="torch.fft", lw=1.6, alpha=0.85)
        axs[0].plot(np.abs(X_tt),    label="tt_fft", lw=0.9, ls="--", alpha=0.85)
        axs[0].set_title(f"{label}    |FFT|"); axs[0].legend()
        axs[0].set_xlabel("bin"); axs[0].set_ylabel("magnitude")
        axs[1].semilogy(np.abs(X_torch - X_tt) + 1e-30, lw=0.8)
        axs[1].set_title("|tt_fft - torch.fft|")
        axs[1].set_xlabel("bin"); axs[1].set_ylabel("|err|")
        plt.tight_layout(); plt.savefig(plot_path, dpi=120); plt.close(fig)
        print(f"  plot saved       : {plot_path}")


# ---------------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------------

def example_manual() -> tuple[str, torch.Tensor]:
    """The simple [1, 2, 3, 4] case."""
    return ("MANUAL TENSOR  [1, 2, 3, 4]",
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))


def example_arange(N: int) -> tuple[str, torch.Tensor]:
    return (f"ARANGE  [0, 1, 2, ..., {N-1}]",
            torch.arange(N, dtype=torch.float32))


def example_random_real(N: int) -> tuple[str, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    return (f"RANDOM real torch.randn(N={N})",
            torch.randn(N, generator=g))


def example_random_complex(N: int) -> tuple[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    re = torch.randn(N, generator=g)
    im = torch.randn(N, generator=g)
    return (f"RANDOM complex  randn + j*randn  (N={N})",
            torch.complex(re, im))


def example_sine(N: int, k: int = 5) -> tuple[str, torch.Tensor]:
    n = torch.arange(N, dtype=torch.float32)
    return (f"SINE  sin(2*pi*{k}*n/N)  (N={N})  -> spike at bin {k}",
            torch.sin(2 * np.pi * k * n / N))


def example_cosine(N: int, k: int = 7) -> tuple[str, torch.Tensor]:
    n = torch.arange(N, dtype=torch.float32)
    return (f"COSINE  cos(2*pi*{k}*n/N)  (N={N})  -> spike at bin {k}",
            torch.cos(2 * np.pi * k * n / N))


def example_complex_tone(N: int, k: int = 17) -> tuple[str, torch.Tensor]:
    n = torch.arange(N, dtype=torch.float32)
    re = torch.cos(2 * np.pi * k * n / N)
    im = torch.sin(2 * np.pi * k * n / N)
    return (f"COMPLEX TONE  exp(2*pi*j*{k}*n/N)  -> single spike at bin {k}",
            torch.complex(re, im))


def example_square(N: int, period: int = 32) -> tuple[str, torch.Tensor]:
    n = torch.arange(N, dtype=torch.float32)
    x = torch.sign(torch.sin(2 * np.pi * n / period))
    return (f"SQUARE WAVE  period={period}  (N={N})  -> harmonics 1,3,5,...",
            x)


def example_chord(N: int) -> tuple[str, torch.Tensor]:
    """Three sines summed: should produce 3 visible spikes."""
    n = torch.arange(N, dtype=torch.float32)
    freqs, amps = (50, 120, 240), (1.0, 0.6, 0.3)
    x = sum(a * torch.sin(2 * np.pi * f * n / N) for f, a in zip(freqs, amps))
    return (f"CHORD  sin(50) + 0.6*sin(120) + 0.3*sin(240)  (N={N})",
            x)


def example_chirp(N: int) -> tuple[str, torch.Tensor]:
    """Linear chirp 5 -> 85 cycles per N."""
    n = torch.arange(N, dtype=torch.float32)
    inst_freq = 5 + 80 * n / N
    return (f"CHIRP  freq sweeps 5 -> 85 cycles/N  (N={N})",
            torch.sin(2 * np.pi * inst_freq * n / N))


def example_gaussian(N: int) -> tuple[str, torch.Tensor]:
    """Gaussian bump in time -> Gaussian bump in freq."""
    n = torch.arange(N, dtype=torch.float32)
    sigma = N / 40.0
    return (f"GAUSSIAN  bump centred at N/2, sigma=N/40  (N={N})",
            torch.exp(-((n - N / 2) ** 2) / (2 * sigma ** 2)))


def example_impulse(N: int) -> tuple[str, torch.Tensor]:
    """Delta at n=0 -> flat spectrum (all bins = 1)."""
    x = torch.zeros(N, dtype=torch.float32); x[0] = 1.0
    return (f"IMPULSE  delta[n]  (N={N})  -> flat spectrum (|X[k]|=1)", x)


def example_dc(N: int) -> tuple[str, torch.Tensor]:
    """Constant = N -> X[0]=N^2, rest=0."""
    x = torch.full((N,), 1.0, dtype=torch.float32)
    return (f"DC  x[n] = 1  (N={N})  -> X[0] = N, rest = 0", x)


def example_user(values_csv: str) -> tuple[str, torch.Tensor]:
    """Parse '1,2,3,4' into a tensor."""
    parts = [float(p.strip()) for p in values_csv.split(",") if p.strip()]
    if len(parts) < 2:
        sys.exit(f"--user needs at least 2 comma-separated numbers, got {parts}")
    return (f"USER INPUT  {parts}",
            torch.tensor(parts, dtype=torch.float32))


# Registry: name -> builder(N)
EXAMPLES: dict[str, callable] = {
    "manual":         lambda N: example_manual(),
    "arange":         lambda N: example_arange(min(N, 16)),
    "random_real":    example_random_real,
    "random_complex": example_random_complex,
    "sine":           example_sine,
    "cosine":         example_cosine,
    "tone":           example_complex_tone,
    "square":         example_square,
    "chord":          example_chord,
    "chirp":          example_chirp,
    "gaussian":       example_gaussian,
    "impulse":        example_impulse,
    "dc":             example_dc,
}


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_loop() -> None:
    print()
    print("Interactive mode — type a Python list/expression that evaluates to a")
    print("1-D sequence of numbers, then press Enter. Examples:")
    print("    [1, 2, 3, 4, 5, 6, 7, 8]")
    print("    list(range(16))")
    print("    [math.sin(2*math.pi*5*n/64) for n in range(64)]")
    print("Type 'quit' or Ctrl-D to exit.")
    print()
    import math  # noqa: F401  (made available in eval scope)
    g = {"__builtins__": __builtins__, "math": __import__("math"),
         "np": np, "torch": torch}
    while True:
        try:
            line = input("tt_fft> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line: continue
        if line in ("quit", "exit"): break
        try:
            x = eval(line, g)
        except Exception as e:
            print(f"  parse error: {e}"); continue
        try:
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim != 1 or arr.size < 2:
                print(f"  need 1-D length >= 2, got shape {arr.shape}"); continue
        except Exception as e:
            print(f"  shape error: {e}"); continue
        compare(f"INTERACTIVE  {line}", arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="torch.fft.fft vs tt_fft.fft on many input types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--N", type=int, default=64,
                    help="FFT length for the longer examples (default 64)")
    ap.add_argument("--only", default=None,
                    help=f"run only one example: {','.join(EXAMPLES)}")
    ap.add_argument("--user", default=None,
                    help="comma-separated numbers, e.g. '1,2,3,4,5,6,7,8'")
    ap.add_argument("--interactive", action="store_true",
                    help="REPL: type a Python expression, FFT both ways")
    ap.add_argument("--plot", action="store_true",
                    help="also save a PNG of |FFT| for each example")
    args = ap.parse_args()

    print("\nPyTorch-style FFT comparison: torch.fft.fft  vs  tt_fft.fft")
    print(f"  default FFT length for sweeps : N = {args.N}")
    print(f"  ttnn input gen                : "
          f"{'ON' if tt_fft._HAVE_TTNN else 'OFF (numpy host input; FFT still runs on Wormhole)'}")

    # 1. Always show the simple manual tensor first if no --only/--user given,
    #    because that's the slide they asked about.
    if args.user:
        label, x = example_user(args.user)
        compare(label, x, "tt_fft_user.png" if args.plot else None)
        return

    if args.interactive:
        interactive_loop()
        return

    if args.only:
        if args.only not in EXAMPLES:
            sys.exit(f"--only must be one of: {', '.join(EXAMPLES)}")
        label, x = EXAMPLES[args.only](args.N)
        png = f"tt_fft_{args.only}.png" if args.plot else None
        compare(label, x, png)
        return

    # Otherwise run the full suite in a presentation-friendly order.
    order = [
        "manual",          # [1,2,3,4]
        "arange",          # [0,1,2,...,15]
        "impulse",         # delta -> flat spectrum
        "dc",              # constant -> single bin
        "sine",            # sin -> single spike
        "cosine",          # cos -> single spike
        "tone",            # complex exp -> single spike
        "square",          # square -> odd harmonics
        "chord",           # 3 sines -> 3 spikes
        "chirp",           # sweep -> wide spectrum
        "gaussian",        # gauss -> gauss
        "random_real",     # generic random real
        "random_complex",  # generic random complex
    ]
    for name in order:
        label, x = EXAMPLES[name](args.N)
        png = f"tt_fft_{name}.png" if args.plot else None
        compare(label, x, png)

    print("\n", "=" * 78, sep="")
    print("All examples done. Both engines should agree to ~1e-3 (fp32 pipeline).")
    print("=" * 78)


if __name__ == "__main__":
    main()
