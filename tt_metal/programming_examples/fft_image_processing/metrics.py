"""metrics.py — image-quality metrics.

    mse(a, b)               -> float
    psnr(a, b, data_range=1.0) -> float in dB (inf if perfect)
    snr_db(ref, got)        -> float in dB (signal-to-error ratio)
    summarise(reference, candidates) -> dict for pretty-printing
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"mse shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Inputs assumed in [0, data_range]."""
    e = mse(a, b)
    if e <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / e))


def snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    """Signal-to-error ratio in dB, defined as 10 log10(||ref||^2 / ||ref-got||^2).
    Useful for *complex* spectra (where 'data_range' isn't well-defined)."""
    ref = np.asarray(ref)
    got = np.asarray(got)
    diff = ref.astype(np.complex128) - got.astype(np.complex128)
    num = float(np.sum(np.abs(ref) ** 2))
    den = float(np.sum(np.abs(diff) ** 2)) or 1e-300
    return 10.0 * np.log10(num / den)


def summarise(
    reference: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    data_range: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Compute MSE / PSNR for each candidate vs the same reference."""
    out: Dict[str, Dict[str, float]] = {}
    for name, img in candidates.items():
        out[name] = {
            "mse": mse(reference, img),
            "psnr_db": psnr(reference, img, data_range=data_range),
        }
    return out
