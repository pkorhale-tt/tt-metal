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
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional

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
# Locate the single-process 2-D FFT binary (one device session for the
# whole image — avoids the 2N subprocess launches the per-row path needs).
# ----------------------------------------------------------------------

def _find_fft2_binary() -> Optional[str]:
    repo_root = (os.environ.get("TT_METAL_RUNTIME_ROOT")
                 or os.environ.get("TT_METAL_HOME"))
    if not repo_root:
        # Walk up from this file looking for "tt_metal/hw" — same trick tt_fft uses.
        cur = _HERE
        for _ in range(20):
            if (os.path.isdir(os.path.join(cur, "tt_metal"))
                    and os.path.isdir(os.path.join(cur, "tt_metal", "hw"))):
                repo_root = cur
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    if not repo_root:
        return None
    build = os.environ.get("TT_FFT_BUILD") or os.path.join(repo_root, "build")
    candidates = [
        os.path.join(build, "programming_examples", "fft_image_processing",
                     "metal_example_fft_image_processing_fft2_runner"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


_FFT2_BIN = _find_fft2_binary()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def backend_info() -> Dict[str, str]:
    """Describe which backend will execute fft2 / ifft2."""
    if _TT_FFT_AVAILABLE:
        info: Dict[str, str] = {
            "backend": "tenstorrent",
            "fp32_binary": getattr(_tt_fft, "_BIN_FP32", "?"),
            "bf16_binary": getattr(_tt_fft, "_BIN_BF16", "?"),
            "module_path": getattr(_tt_fft, "__file__", "?"),
            "fft2_binary": _FFT2_BIN or "(not built — using per-row fallback)",
            "fft2_path": "single-process 2D" if _FFT2_BIN else "per-row subprocess",
        }
        return info
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
    """2-D FFT on Wormhole.

    Two execution paths, picked automatically:

    1. **Single-process 2-D runner** (preferred). One C++ binary opens the
       Wormhole MeshDevice once, runs all H row + W column 1-D FFTs, returns.
       Used when ``_FFT2_BIN`` is found on disk. Scales cleanly to any image
       size (256x256, 512x512, ...).

    2. **Per-row subprocess fallback**. If the dedicated 2-D binary isn't
       built yet, fall back to launching ``tt_fft.fft`` per row + per column.
       Fine for tiny images (~32x32 max) but each call re-opens the device,
       so larger sizes will exhaust firmware-init resources.

    A 2-D FFT factorises exactly into ``FFT2(X) = FFT_cols(FFT_rows(X))``,
    so both paths give the same result.
    """
    if _FFT2_BIN is not None and precision == "fp32":
        return _run_fft2_binary(arr, inverse=inverse)

    # Per-row fallback (slow / size-limited).
    H, W = arr.shape
    fn = _tt_fft.ifft if inverse else _tt_fft.fft

    out = np.empty((H, W), dtype=np.complex64)
    for r in range(H):
        out[r, :] = fn(arr[r, :], precision=precision)
    for c in range(W):
        out[:, c] = fn(out[:, c], precision=precision)
    return out


def _run_fft2_binary(arr: np.ndarray, inverse: bool) -> np.ndarray:
    """Drive the single-process 2-D FFT C++ binary.

    File format (matches fft2_runner.cpp):
        line 1   : "H W"
        line 2.. : "real imag"  (one per line, row-major)
    """
    H, W = arr.shape
    tmpdir = tempfile.mkdtemp(prefix="tt_fft2_")
    in_path  = os.path.join(tmpdir, "in.txt")
    out_path = os.path.join(tmpdir, "out.txt")
    try:
        flat = arr.astype(np.complex64).reshape(-1)
        with open(in_path, "w") as f:
            f.write(f"{H} {W}\n")
            for c in flat:
                f.write(f"{float(c.real):.9e} {float(c.imag):.9e}\n")

        cmd = [_FFT2_BIN, in_path, out_path] + (["--inverse"] if inverse else [])
        env = os.environ.copy()
        env.setdefault("ARCH_NAME", "wormhole_b0")
        # Mirror tt_fft's env-passing so the C++ binary can find the repo root
        # regardless of which env-var name the local tt-metal version checks.
        repo_root = (env.get("TT_METAL_RUNTIME_ROOT")
                     or env.get("TT_METAL_HOME"))
        if repo_root:
            env["TT_METAL_HOME"]         = repo_root
            env["TT_METAL_RUNTIME_ROOT"] = repo_root

        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            so = (r.stdout or "").splitlines()
            se = (r.stderr or "").splitlines()
            tail = (so + se)[-30:]
            raise RuntimeError(
                f"fft2 device call failed (exit={r.returncode}):\n"
                + "\n".join(tail))

        # Read result back.
        with open(out_path, "r") as f:
            header = f.readline().split()
            if len(header) < 2:
                raise RuntimeError("fft2 binary returned malformed output (no header)")
            Hr, Wr = int(header[0]), int(header[1])
            if Hr != H or Wr != W:
                raise RuntimeError(
                    f"fft2 binary returned shape {Hr}x{Wr}, expected {H}x{W}")
            out_flat = np.empty(H * W, dtype=np.complex64)
            i = 0
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                out_flat[i] = complex(float(parts[0]), float(parts[1]))
                i += 1
            if i != H * W:
                raise RuntimeError(
                    f"fft2 binary returned {i} samples, expected {H*W}")
        return out_flat.reshape(H, W)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
