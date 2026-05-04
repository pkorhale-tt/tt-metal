"""torch_fft_module.py — PyTorch reference implementation.

Public API mirrors fft_module:
    torch_fft2(img)        -> np.ndarray complex64
    torch_ifft2(spectrum)  -> np.ndarray complex64
    torch_fftshift(s)
    torch_ifftshift(s)

We always return numpy arrays so callers don't need to touch torch tensors
directly. PyTorch is treated as the gold-standard 2-D FFT reference for
correctness checks and benchmarking.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False


def _require_torch():
    if not HAVE_TORCH:
        raise RuntimeError(
            "torch is not installed. Install with `pip install torch` "
            "or set --engines without 'torch' on the CLI.")


def torch_fft2(img: np.ndarray) -> np.ndarray:
    _require_torch()
    t = torch.as_tensor(img)
    if not torch.is_complex(t):
        t = t.to(torch.complex64)
    return torch.fft.fft2(t).cpu().numpy().astype(np.complex64)


def torch_ifft2(spectrum: np.ndarray) -> np.ndarray:
    _require_torch()
    t = torch.as_tensor(spectrum)
    if not torch.is_complex(t):
        t = t.to(torch.complex64)
    return torch.fft.ifft2(t).cpu().numpy().astype(np.complex64)


def torch_fftshift(spectrum: np.ndarray) -> np.ndarray:
    _require_torch()
    t = torch.as_tensor(spectrum)
    return torch.fft.fftshift(t, dim=(-2, -1)).cpu().numpy()


def torch_ifftshift(spectrum: np.ndarray) -> np.ndarray:
    _require_torch()
    t = torch.as_tensor(spectrum)
    return torch.fft.ifftshift(t, dim=(-2, -1)).cpu().numpy()
