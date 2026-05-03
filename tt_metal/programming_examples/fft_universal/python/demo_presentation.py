#!/usr/bin/env python3
"""
demo_presentation.py — live presentation script for tt_fft.

PyTorch-style FFT on Wormhole, side-by-side with torch.fft / numpy.fft.

Run any of these from your shell:

    # 1. One-line "wow" — does PyTorch + Wormhole agree?
    python demo_presentation.py compare 1024

    # 2. Show the dispatch tree picks the right algorithm for ANY N.
    python demo_presentation.py compare 97          # prime  -> Bluestein
    python demo_presentation.py compare 60          # composite non-pow2
    python demo_presentation.py compare 65536       # big pow2
    python demo_presentation.py compare 1024 bf16   # TRUE bf16 path

    # 3. Pure-tone visual: Wormhole FFT recovers the spike at bin k.
    python demo_presentation.py spike 1024 --k 17

    # 4. Real audio-like signal: chord of sines + noise -> spectrogram bars.
    python demo_presentation.py chord 4096

    # 5. Round-trip sanity: x -> FFT -> IFFT -> x.
    python demo_presentation.py round_trip 1000

    # 6. Speed sweep across multiple N.
    python demo_presentation.py bench

The plotting steps require matplotlib; if it isn't available the script
falls back to a text summary so the demo still works.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # safe default; remove this line if you have a display
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

import tt_fft


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def _snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    diff = got.astype(np.complex128) - ref.astype(np.complex128)
    num = float(np.sum(np.abs(ref) ** 2))
    den = float(np.sum(np.abs(diff) ** 2)) or 1e-300
    return 10.0 * np.log10(num / den)


def _time_call(fn, *args, **kw):
    t0 = time.time()
    out = fn(*args, **kw)
    return out, (time.time() - t0) * 1000.0


def _save_or_show(path: str | None) -> None:
    if not HAVE_PLT: return
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        print(f"[plot] saved {path}")
    else:
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Demos
# ----------------------------------------------------------------------

def demo_compare(N: int, precision: str, plot: str | None) -> None:
    """Random signal: Wormhole FFT vs PyTorch FFT, side by side."""
    print()
    print("=" * 72)
    print(f"  PyTorch-style FFT on Wormhole  (N={N}, precision={precision})")
    print(f"  device dispatch path : {tt_fft.device_path(N)}")
    print("=" * 72)

    # Input ALSO comes from Tenstorrent (ttnn.rand on the device);
    # falls back to numpy if ttnn isn't available.
    x = tt_fft.randn(N, complex=True, seed=42)

    # Optional torch reference; numpy is always available
    X_np, ms_np = _time_call(np.fft.fft, x.astype(np.complex128))
    try:
        import torch
        X_th, ms_th = _time_call(
            lambda v: torch.fft.fft(torch.from_numpy(v)).numpy(),
            x.astype(np.complex128))
        have_torch = True
    except Exception:
        X_th, ms_th, have_torch = X_np, ms_np, False

    X_tt, ms_tt = _time_call(tt_fft.fft, x, precision=precision)

    snr_tt   = _snr_db(X_np, X_tt)
    rel_tt   = float(np.max(np.abs(X_tt - X_np)) / max(float(np.max(np.abs(X_np))), 1e-30))

    print(f"  numpy.fft.fft         : {ms_np:8.2f} ms   (CPU reference)")
    if have_torch:
        print(f"  torch.fft.fft         : {ms_th:8.2f} ms   (CPU reference)")
    print(f"  tt_fft.fft({precision:>4}) : {ms_tt:8.2f} ms   "
          f"(includes host file I/O + dispatch)")
    print()
    print(f"  Wormhole vs numpy SNR : {snr_tt:7.2f} dB")
    print(f"  Wormhole rel error    : {rel_tt:.3e}")
    print()

    if not HAVE_PLT:
        return

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    k = np.arange(N)
    axs[0].plot(k, np.abs(X_np), label="numpy.fft", lw=1.6, alpha=0.85)
    axs[0].plot(k, np.abs(X_tt), label=f"tt_fft ({precision})",
                lw=0.8, ls="--", alpha=0.85)
    axs[0].set_title(f"|FFT|  —  N={N}, dispatch={tt_fft.device_path(N)}")
    axs[0].set_xlabel("bin k"); axs[0].set_ylabel("magnitude")
    axs[0].legend()

    axs[1].semilogy(k, np.abs(X_tt - X_np) + 1e-30, lw=0.8)
    axs[1].set_title(f"|tt_fft - numpy|   SNR = {snr_tt:.1f} dB")
    axs[1].set_xlabel("bin k"); axs[1].set_ylabel("|err|")

    _save_or_show(plot)


def demo_spike(N: int, k: int, precision: str, plot: str | None) -> None:
    """Pure tone at bin k: spectrum should be a single spike of magnitude N."""
    if k <= 0 or k >= N:
        sys.exit(f"--k must be in (0, N); got k={k}, N={N}")
    x = tt_fft.tone(N, k=k)

    X, ms = _time_call(tt_fft.fft, x, precision=precision)
    top = np.argsort(np.abs(X))[-3:][::-1]

    print()
    print("=" * 72)
    print(f"  Pure tone at bin {k}, N={N}, precision={precision} "
          f"({tt_fft.device_path(N)})")
    print("=" * 72)
    print(f"  tt_fft.fft wall time : {ms:.2f} ms")
    print(f"  Top 3 output bins:")
    for kk in top:
        print(f"    k={int(kk):6d}  |X[k]|={float(np.abs(X[kk])):10.3f}  "
              f"X[k]=({X[kk].real:+.3e}{X[kk].imag:+.3e}j)")
    print(f"  Expected: |X[{k}]| = N = {N}")
    print()

    if not HAVE_PLT:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(np.abs(X), basefmt=" ", markerfmt=".", linefmt="-")
    ax.axvline(k, color="red", ls="--", alpha=0.4, label=f"input bin k={k}")
    ax.set_title(f"tt_fft pure-tone spike  (N={N}, expected |X[{k}]|={N})")
    ax.set_xlabel("bin"); ax.set_ylabel("magnitude"); ax.legend()
    _save_or_show(plot)


def demo_chord(N: int, precision: str, plot: str | None) -> None:
    """Sum of three sines (musical chord) + noise -> spectrum should show 3 peaks."""
    fs = N  # treat N samples as 1 second of audio at fs Hz
    t  = np.arange(N) / fs
    freqs = (50, 120, 240)   # arbitrary "notes" (in cycles per N samples)
    amps  = (1.0, 0.6, 0.3)
    x = tt_fft.chord(N, freqs=freqs, amps=amps, noise=0.02, seed=7)

    X, ms = _time_call(tt_fft.fft, x, precision=precision)
    mag = np.abs(X[: N // 2])

    print()
    print("=" * 72)
    print(f"  Musical chord, N={N}, precision={precision} "
          f"({tt_fft.device_path(N)})")
    print("=" * 72)
    print(f"  tt_fft.fft wall time : {ms:.2f} ms")
    peaks = np.argsort(mag)[-3:][::-1]
    print(f"  Three loudest bins (freq in cycles/N):")
    for p in sorted(peaks):
        print(f"    bin {int(p):5d}  |X|={float(mag[p]):.2f}")
    print(f"  Expected peaks at bins {freqs}")
    print()

    if not HAVE_PLT:
        return
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(t[:512], x[:512]); axs[0].set_title("input signal (first 512 samples)")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("amplitude")
    axs[1].plot(mag); axs[1].set_title(f"|tt_fft.fft(x)|  (one-sided)")
    axs[1].set_xlabel("bin"); axs[1].set_ylabel("magnitude")
    for f in freqs:
        axs[1].axvline(f, color="red", ls="--", alpha=0.4)
    _save_or_show(plot)


def demo_round_trip(N: int, precision: str) -> None:
    """x -> Wormhole FFT -> Wormhole IFFT -> should equal x."""
    x = tt_fft.randn(N, complex=True, seed=0)

    X, ms_fwd = _time_call(tt_fft.fft, x, precision=precision)
    y, ms_inv = _time_call(tt_fft.ifft, X, precision=precision)

    rel = float(np.max(np.abs(y - x)) / max(float(np.max(np.abs(x))), 1e-30))
    snr = _snr_db(x, y.astype(np.complex64))

    print()
    print("=" * 72)
    print(f"  Round-trip:   x -> tt_fft.fft -> tt_fft.ifft -> x_hat   "
          f"(N={N}, {precision})")
    print(f"  device path  : {tt_fft.device_path(N)}")
    print("=" * 72)
    print(f"  forward FFT wall   : {ms_fwd:.2f} ms")
    print(f"  inverse FFT wall   : {ms_inv:.2f} ms")
    print(f"  max rel error      : {rel:.3e}")
    print(f"  reconstruction SNR : {snr:.2f} dB")
    print()
    print("  First 5 input samples vs reconstructed:")
    for i in range(min(5, N)):
        print(f"    x[{i}] = {x[i].real:+.4f}{x[i].imag:+.4f}j   "
              f"x_hat[{i}] = {y[i].real:+.4f}{y[i].imag:+.4f}j")
    print()


def demo_bench(plot: str | None) -> None:
    """Speed sweep across a curated set of N values."""
    Ns = [32, 100, 512, 1024, 4096, 16384, 65536]
    print()
    print("=" * 72)
    print(f"  tt_fft warm-cached avg over 10 iters per N")
    print("=" * 72)
    rows = []
    for N in Ns:
        try:
            r = tt_fft.benchmark(N, iters=10, precision="fp32")
        except Exception as e:
            print(f"  N={N:>6d}  -> error: {e}")
            continue
        rows.append((N, r))
        print(f"  N={N:>6d}  warm avg {r['warm_avg_ms']:7.2f} ms   "
              f"(cold {r['cold_ms']:7.1f} ms)   SNR {r['snr_db']:6.1f} dB   "
              f"[{r['dispatch']}]")
    print()

    if not HAVE_PLT or not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r[0] for r in rows]
    ys = [r[1]["warm_avg_ms"] for r in rows]
    ax.loglog(xs, ys, "o-", lw=2)
    ax.set_xlabel("N"); ax.set_ylabel("end-to-end ms (warm avg)")
    ax.set_title("tt_fft.fft   end-to-end time vs N")
    ax.grid(True, which="both", alpha=0.3)
    _save_or_show(plot)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="PyTorch-style FFT demo for the Wormhole tt-metal pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("which",
                    choices=["compare", "spike", "chord", "round_trip", "bench"],
                    help="which demo to run")
    ap.add_argument("N", nargs="?", type=int, default=1024,
                    help="FFT length (any integer >= 2; default 1024)")
    ap.add_argument("precision", nargs="?", default="fp32",
                    choices=["fp32", "bf16"],
                    help="device pipeline precision (default fp32)")
    ap.add_argument("--k", type=int, default=17,
                    help="bin index for the 'spike' demo (default 17)")
    ap.add_argument("--plot", type=str, default=None,
                    help="save plot to this PNG (default: show interactively)")
    args = ap.parse_args()

    if args.which == "compare":     demo_compare(args.N, args.precision, args.plot)
    elif args.which == "spike":     demo_spike(args.N, args.k, args.precision, args.plot)
    elif args.which == "chord":     demo_chord(args.N, args.precision, args.plot)
    elif args.which == "round_trip": demo_round_trip(args.N, args.precision)
    elif args.which == "bench":     demo_bench(args.plot)


if __name__ == "__main__":
    main()
