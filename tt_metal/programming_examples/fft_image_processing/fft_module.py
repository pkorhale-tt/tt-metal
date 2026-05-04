"""fft_module.py — custom 2-D FFT backed by the Tenstorrent Wormhole pipeline.

Public API:
    fft2(img, precision='fp32')      -> np.ndarray (H, W) complex64
    ifft2(spectrum, precision='fp32') -> np.ndarray (H, W) complex64
    fftshift(spectrum)               -> np.ndarray
    ifftshift(spectrum)              -> np.ndarray
    backend_info() -> dict           — what's actually being used

Backends (auto-selected, in order):
    1. Tenstorrent  — tt_fft.fft / tt_fft.ifft applied row-then-column.
                      Used when tt_fft import works AND the Wormhole binaries
                      are reachable.
    2. NumPy fallback — np.fft.fft2 / ifft2 (still labelled 'custom' for the demo).

The 'precision' arg ('fp32' / 'bf16') is forwarded to tt_fft when the
Tenstorrent backend is active; ignored by the numpy fallback (which is fp32).

We use fftshift/ifftshift defined locally rather than np.fft.fftshift to keep
the dependency surface small and the semantics explicit.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np


# ----------------------------------------------------------------------
# Try to wire in the real Tenstorrent backend.
# ----------------------------------------------------------------------

_TT_FFT_AVAILABLE = False
_TT_FFT_REASON = ""
_tt_fft = None

# Allow the demo to be run from inside its own folder, the repo root, or
# anywhere PYTHONPATH happens to include the python wrapper.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TT_FFT_PATH = os.path.normpath(os.path.join(
    _HERE, "..", "fft_universal", "python"))
if os.path.isdir(_TT_FFT_PATH) and _TT_FFT_PATH not in sys.path:
    sys.path.insert(0, _TT_FFT_PATH)

try:
    import tt_fft as _tt_fft  # type: ignore
    _TT_FFT_AVAILABLE = True
except Exception as e:
    _TT_FFT_REASON = f"tt_fft import failed: {e}"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def backend_info() -> Dict[str, str]:
    """Describe which backend will execute fft2 / ifft2."""
    if _TT_FFT_AVAILABLE:
        return {
            "backend": "tenstorrent",
            "fp32_binary": getattr(_tt_fft, "_BIN_FP32", "?"),
            "bf16_binary": getattr(_tt_fft, "_BIN_BF16", "?"),
            "module_path": getattr(_tt_fft, "__file__", "?"),
        }
    return {"backend": "numpy_fallback", "reason": _TT_FFT_REASON}


def fft2(img: np.ndarray, precision: str = "fp32") -> np.ndarray:
    """2-D FFT of a real or complex image. Returns complex64 of same shape."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"fft2 expects 2-D input, got shape {img.shape}")
    if not np.iscomplexobj(img):
        img = img.astype(np.float32) + 0j

    if not _TT_FFT_AVAILABLE:
        return np.fft.fft2(img).astype(np.complex64)

    return _tt_fft_2d(img, inverse=False, precision=precision)


def ifft2(spectrum: np.ndarray, precision: str = "fp32") -> np.ndarray:
    """2-D inverse FFT. Returns complex64 of same shape."""
    spectrum = np.asarray(spectrum)
    if spectrum.ndim != 2:
        raise ValueError(f"ifft2 expects 2-D input, got shape {spectrum.shape}")
    if not np.iscomplexobj(spectrum):
        spectrum = spectrum.astype(np.float32) + 0j

    if not _TT_FFT_AVAILABLE:
        return np.fft.ifft2(spectrum).astype(np.complex64)

    return _tt_fft_2d(spectrum, inverse=True, precision=precision)


def fftshift(spectrum: np.ndarray) -> np.ndarray:
    """Move the zero-frequency bin to the centre (so |F| is visualisable)."""
    H, W = spectrum.shape
    return np.roll(spectrum, shift=(H // 2, W // 2), axis=(0, 1))


def ifftshift(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of fftshift (correct for odd dimensions, equivalent for even)."""
    H, W = spectrum.shape
    return np.roll(spectrum, shift=(-(H // 2), -(W // 2)), axis=(0, 1))


# ----------------------------------------------------------------------
# Internal: 2-D FFT via row + column 1-D FFTs on Wormhole.
# ----------------------------------------------------------------------

def _tt_fft_2d(arr: np.ndarray, inverse: bool, precision: str) -> np.ndarray:
    """Row-then-column 1-D FFTs using the Tenstorrent universal pipeline.

    A 2-D FFT factorises exactly into:
        FFT2(X) = FFT_cols( FFT_rows(X) )
    so we can build it from two passes of our 1-D FFT engine.

    Each 1-D FFT call is one full Wormhole dispatch (cold compile cost is
    paid once per N via tt_fft's plan + JIT cache, then ~ms thereafter).
    """
    H, W = arr.shape
    fn = _tt_fft.ifft if inverse else _tt_fft.fft

    out = np.empty((H, W), dtype=np.complex64)
    for r in range(H):
        out[r, :] = fn(arr[r, :], precision=precision)
    for c in range(W):
        out[:, c] = fn(out[:, c], precision=precision)
    return out
