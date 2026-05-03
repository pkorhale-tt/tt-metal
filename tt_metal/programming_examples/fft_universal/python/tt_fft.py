"""
tt_fft — PyTorch-style 1-D FFT API backed by Tenstorrent Wormhole.

    >>> import numpy as np, tt_fft
    >>> x = np.random.randn(1000) + 1j * np.random.randn(1000)
    >>> X = tt_fft.fft(x)              # forward FFT, fp32 pipeline
    >>> y = tt_fft.ifft(X)             # inverse FFT
    >>> X_bf16 = tt_fft.fft(x, precision='bf16')   # TRUE-bf16 pipeline

Accepts ANY length N >= 2 (pow2, prime, composite). The dispatcher on the
device side automatically picks the right algorithm:

    pow2          -> Stockham (fp32) / two-level Cooley-Tukey (bf16)
    prime         -> Bluestein chirp-Z transform
    composite N   -> mixed-radix Cooley-Tukey
    small N <= 32 -> packed direct-DFT (single tile)

Under the hood this writes the input to a tmp text file, invokes the
file-IO C++ binary built from fft_universal_run.cpp / fft_universal_bf16_run.cpp,
and reads back the output. No Python bindings needed — works on any machine
where the binaries are built.

Environment variables:
    TT_FFT_BIN_FP32 : path to metal_example_fft_universal_run
    TT_FFT_BIN_BF16 : path to metal_example_fft_universal_bf16_run
    TT_FFT_BUILD    : path to the build dir (default: ./build).
                      The two binaries are looked up at
                      $TT_FFT_BUILD/programming_examples/fft_universal{,_bf16}/...

Public API (mirrors torch.fft):
    fft(x, precision='fp32') -> np.ndarray (complex64)
    ifft(X, precision='fp32') -> np.ndarray (complex64)
    rfft(x, precision='fp32') -> np.ndarray (complex64)   # real input
    fft2(x, precision='fp32') -> np.ndarray               # 2-D via row+col
    benchmark(N, iters=20, precision='fp32') -> dict
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import numpy as np


__all__ = [
    "fft", "ifft", "rfft", "fft2",
    "rand", "randn", "tone", "chord",
    "benchmark", "set_binaries", "device_path",
    "from_ttnn", "to_ttnn",
]


# ----------------------------------------------------------------------
# Optional ttnn integration (so inputs can also live on the Tenstorrent
# device, not just on the host). Falls back to numpy if ttnn isn't
# installed / no device is reachable.
# ----------------------------------------------------------------------

# ttnn.rand has been observed to fail to JIT-compile on some tt-metal
# branches (chlkc_pack / chlkc_unpack undeclared in the rand kernel).
# To keep the demo log clean, ttnn-based input generation is OPT-IN.
# Set the env var TT_FFT_USE_TTNN=1 to enable it; otherwise we silently
# use numpy for the *input* (the FFT itself always runs on Tenstorrent).
_USE_TTNN_REQUESTED = os.environ.get("TT_FFT_USE_TTNN", "0") not in ("0", "", "false", "False")

try:
    if _USE_TTNN_REQUESTED:
        import ttnn  # type: ignore
        _HAVE_TTNN = True
    else:
        ttnn = None
        _HAVE_TTNN = False
except Exception:
    ttnn = None
    _HAVE_TTNN = False

_TTNN_DEVICE = None
_TTNN_PROBED_OK: Optional[bool] = None  # cached result of one-time probe


def _get_ttnn_device():
    """Lazily open device 0 the first time ttnn is needed."""
    global _TTNN_DEVICE
    if not _HAVE_TTNN:
        return None
    if _TTNN_DEVICE is None:
        _TTNN_DEVICE = ttnn.open_device(device_id=0)
    return _TTNN_DEVICE


def _probe_ttnn_rand_once() -> bool:
    """Try ttnn.rand on a tiny tensor exactly once. Cache pass/fail.
    If it fails, all later input generators silently use numpy.
    """
    global _TTNN_PROBED_OK, _HAVE_TTNN
    if _TTNN_PROBED_OK is not None:
        return _TTNN_PROBED_OK
    if not _HAVE_TTNN:
        _TTNN_PROBED_OK = False
        return False
    try:
        dev = _get_ttnn_device()
        _ = ttnn.rand((4,), dtype=ttnn.bfloat16,
                      layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        _TTNN_PROBED_OK = True
    except Exception:
        # Disable ttnn for the rest of the session, no spam.
        _HAVE_TTNN = False
        _TTNN_PROBED_OK = False
    return _TTNN_PROBED_OK


def to_ttnn(x: np.ndarray, dtype="bfloat16"):
    """Move a numpy array onto the Tenstorrent device as a ttnn.Tensor.
    For complex inputs this stacks (real, imag) along the last dim.
    """
    if not _HAVE_TTNN:
        raise RuntimeError("ttnn is not available in this Python environment")
    dev = _get_ttnn_device()
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        arr = np.stack([arr.real.astype(np.float32),
                        arr.imag.astype(np.float32)], axis=-1)
    else:
        arr = arr.astype(np.float32)
    dt = getattr(ttnn, dtype) if isinstance(dtype, str) else dtype
    return ttnn.from_torch(  # ttnn accepts torch -> let user import torch optionally
        __import__("torch").from_numpy(arr), dtype=dt, device=dev,
        layout=ttnn.ROW_MAJOR_LAYOUT)


def from_ttnn(t) -> np.ndarray:
    """Pull a ttnn.Tensor back to a numpy array."""
    if not _HAVE_TTNN:
        raise RuntimeError("ttnn is not available")
    return ttnn.to_torch(t).cpu().numpy()


# ----------------------------------------------------------------------
# Tenstorrent-native input generators
# ----------------------------------------------------------------------

def rand(N: int, complex: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """Uniform random complex signal of length N, sampled on the Tenstorrent
    device when ttnn is available (falls back to numpy otherwise).

    Returns the signal as a numpy complex64 array, ready to feed to
    `tt_fft.fft(...)`. The data was generated by `ttnn.rand(...)` on the
    Wormhole card, so the *whole pipeline* (random input + FFT) runs on TT.
    """
    if _probe_ttnn_rand_once():
        try:
            dev = _get_ttnn_device()
            shape = (N, 2) if complex else (N,)
            t = ttnn.rand(shape, dtype=ttnn.bfloat16,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            arr = from_ttnn(t).astype(np.float32)
            if complex:
                arr = arr * 2.0 - 1.0  # ttnn.rand is [0,1); shift to [-1, 1)
                return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
            return (arr * 2.0 - 1.0).astype(np.float32)
        except Exception:
            pass  # silently fall through to numpy on any per-call error
    rng = np.random.default_rng(seed)
    if complex:
        return (rng.uniform(-1, 1, N) + 1j * rng.uniform(-1, 1, N)).astype(np.complex64)
    return rng.uniform(-1, 1, N).astype(np.float32)


def randn(N: int, complex: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """Standard-normal random complex signal. Tries ttnn first (when
    TT_FFT_USE_TTNN=1 is set and the kernel actually JITs), otherwise uses
    numpy. Fallback is silent so the demo log stays clean.
    """
    if _probe_ttnn_rand_once():
        try:
            dev = _get_ttnn_device()
            shape = (N, 2) if complex else (N,)
            u1 = from_ttnn(ttnn.rand(shape, dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)).astype(np.float32)
            u2 = from_ttnn(ttnn.rand(shape, dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)).astype(np.float32)
            u1 = np.clip(u1, 1e-7, 1 - 1e-7)
            arr = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            if complex:
                return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
            return arr.astype(np.float32)
        except Exception:
            pass  # silent fallback to numpy
    rng = np.random.default_rng(seed)
    if complex:
        return (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    return rng.standard_normal(N).astype(np.float32)


def tone(N: int, k: int = 1) -> np.ndarray:
    """Pure complex tone exp(2*pi*i*k*n/N) — input for the 'spike' demo."""
    n = np.arange(N)
    return np.exp(2j * np.pi * k * n / N).astype(np.complex64)


def chord(N: int, freqs=(50, 120, 240), amps=(1.0, 0.6, 0.3),
          noise: float = 0.02, seed: int = 7) -> np.ndarray:
    """Sum of real sinusoids at given (cycles per N) frequencies, plus noise."""
    t = np.arange(N) / N
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    x += noise * np.random.default_rng(seed).standard_normal(N)
    return x.astype(np.float32)


# ----------------------------------------------------------------------
# Binary discovery
# ----------------------------------------------------------------------

_BUILD_DIR = os.environ.get("TT_FFT_BUILD", "./build")

_BIN_FP32 = os.environ.get(
    "TT_FFT_BIN_FP32",
    os.path.join(_BUILD_DIR,
                 "programming_examples/fft_universal/"
                 "metal_example_fft_universal_run"))

_BIN_BF16 = os.environ.get(
    "TT_FFT_BIN_BF16",
    os.path.join(_BUILD_DIR,
                 "programming_examples/fft_universal_bf16/"
                 "metal_example_fft_universal_bf16_run"))


def set_binaries(fp32: Optional[str] = None, bf16: Optional[str] = None) -> None:
    """Override the C++ binary locations at runtime."""
    global _BIN_FP32, _BIN_BF16
    if fp32 is not None: _BIN_FP32 = fp32
    if bf16 is not None: _BIN_BF16 = bf16


def _binary_for(precision: str) -> str:
    p = precision.lower()
    if p in ("fp32", "f32", "float32", "float"):
        bin_ = _BIN_FP32
    elif p in ("bf16", "bfloat16"):
        bin_ = _BIN_BF16
    else:
        raise ValueError(f"unknown precision {precision!r}; use 'fp32' or 'bf16'")
    if not os.path.exists(bin_):
        raise FileNotFoundError(
            f"tt_fft: cannot find {p} binary at {bin_!r}.\n"
            f"Build it with:\n"
            f"    cmake --build build --target "
            f"metal_example_fft_universal{'_bf16' if p == 'bf16' else ''}_run -j\n"
            f"or override TT_FFT_BIN_FP32 / TT_FFT_BIN_BF16.")
    return bin_


# ----------------------------------------------------------------------
# File I/O helpers
# ----------------------------------------------------------------------

def _write_complex(path: str, x: np.ndarray) -> None:
    """Write 1-D complex array as 'real imag\\n' lines."""
    arr = np.empty((x.size, 2), dtype=np.float64)
    arr[:, 0] = x.real
    arr[:, 1] = x.imag
    np.savetxt(path, arr, fmt="%.9e")


def _read_complex(path: str, n: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape != (n, 2):
        raise ValueError(
            f"{path}: expected ({n}, 2), got shape {data.shape}")
    return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)


# ----------------------------------------------------------------------
# Core dispatch
# ----------------------------------------------------------------------

def _normalize_input(x) -> np.ndarray:
    """Accept torch.Tensor / list / ndarray / real or complex; return 1-D complex64."""
    # torch tensor -> numpy
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"tt_fft expects a 1-D signal, got shape {x.shape}")
    if x.size < 2:
        raise ValueError(f"tt_fft requires N >= 2 (got N={x.size})")
    if not np.iscomplexobj(x):
        x = x.astype(np.float32) + 0j
    return x.astype(np.complex64)


def _run_device(x: np.ndarray, inverse: bool, precision: str,
                verbose: bool = False, return_ms: bool = False):
    bin_ = _binary_for(precision)
    n = x.size
    tmpdir = tempfile.mkdtemp(prefix="tt_fft_")
    in_path  = os.path.join(tmpdir, "in.txt")
    out_path = os.path.join(tmpdir, "out.txt")
    try:
        _write_complex(in_path, x)
        cmd = [bin_, in_path, out_path] + (["--inverse"] if inverse else [])
        env = os.environ.copy()
        env.setdefault("ARCH_NAME", "wormhole_b0")
        if verbose:
            print(f"[tt_fft] {' '.join(cmd)}")
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=not verbose, text=True)
        ms_wall = (time.time() - t0) * 1000.0
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").splitlines()[-20:]
            raise RuntimeError(
                f"tt_fft device call failed (exit={r.returncode}):\n"
                + "\n".join(tail))
        y = _read_complex(out_path, n)
        return (y, ms_wall) if return_ms else y
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------
# Public API — mirrors torch.fft
# ----------------------------------------------------------------------

def fft(x, precision: str = "fp32", verbose: bool = False) -> np.ndarray:
    """Forward 1-D FFT on Wormhole. Drop-in for torch.fft.fft / np.fft.fft.

    Args:
        x         : 1-D array-like (numpy / torch / list), real or complex.
        precision : 'fp32' (default) or 'bf16'.
        verbose   : print the device command.

    Returns:
        np.ndarray of dtype complex64, length N.
    """
    x = _normalize_input(x)
    return _run_device(x, inverse=False, precision=precision, verbose=verbose)


def ifft(X, precision: str = "fp32", verbose: bool = False) -> np.ndarray:
    """Inverse 1-D FFT on Wormhole. Drop-in for torch.fft.ifft / np.fft.ifft."""
    X = _normalize_input(X)
    return _run_device(X, inverse=True, precision=precision, verbose=verbose)


def rfft(x, precision: str = "fp32", verbose: bool = False) -> np.ndarray:
    """Real-input forward FFT. Same convention as torch.fft.rfft:
    returns the first N//2 + 1 bins (the rest are conjugate-symmetric).
    """
    X = fft(x, precision=precision, verbose=verbose)
    return X[: X.size // 2 + 1]


def fft2(x, precision: str = "fp32") -> np.ndarray:
    """Naive 2-D FFT via row/column 1-D FFTs. Useful for small images."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"fft2 expects 2-D input, got shape {x.shape}")
    R, C = x.shape
    out = np.empty((R, C), dtype=np.complex64)
    for r in range(R):
        out[r, :] = fft(x[r, :], precision=precision)
    for c in range(C):
        out[:, c] = fft(out[:, c], precision=precision)
    return out


def device_path(N: int) -> str:
    """Return a short string describing which algorithm the device will pick."""
    if N == 1: return "identity"
    if N >= 2 and (N & (N - 1)) == 0:
        return "pow2 (Stockham fp32 / two-level CT bf16)"
    if N <= 32: return "packed direct-DFT"
    p = 2
    while p * p <= N:
        if N % p == 0: return "Cooley-Tukey split (composite non-pow2)"
        p += 1
    return "Bluestein chirp-Z (prime)"


def benchmark(N: int, iters: int = 20, precision: str = "fp32",
              seed: int = 0) -> dict:
    """End-to-end (host + dispatch + device + readback) timing.

    Returns a dict with cold/avg/min/max ms, plus reference SNR vs numpy.
    """
    x = randn(N, complex=True, seed=seed)
    times = []
    last = None
    for i in range(iters + 1):  # iter 0 is cold (includes JIT)
        y, ms = _run_device(x, inverse=False, precision=precision,
                            verbose=False, return_ms=True)
        times.append(ms)
        last = y
    cold = times[0]
    warm = times[1:] or times
    X_ref = np.fft.fft(x.astype(np.complex128))
    diff = last.astype(np.complex128) - X_ref
    snr = 10 * np.log10(np.sum(np.abs(X_ref)**2) /
                        max(float(np.sum(np.abs(diff)**2)), 1e-300))
    return dict(
        N=N, precision=precision, dispatch=device_path(N),
        iters=iters, cold_ms=cold,
        warm_avg_ms=float(np.mean(warm)),
        warm_min_ms=float(np.min(warm)),
        warm_max_ms=float(np.max(warm)),
        snr_db=float(snr),
    )


if __name__ == "__main__":
    # quick smoke test:  python tt_fft.py [N]
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    x = randn(N, complex=True, seed=0)
    print(f"tt_fft smoke test, N={N}, dispatch={device_path(N)}")
    X = fft(x, verbose=True)
    X_ref = np.fft.fft(x)
    err = np.max(np.abs(X - X_ref)) / np.max(np.abs(X_ref))
    print(f"  max rel error vs numpy.fft: {err:.3e}")
    y = ifft(X)
    rt = np.max(np.abs(y - x)) / np.max(np.abs(x))
    print(f"  IFFT round-trip rel error : {rt:.3e}")
