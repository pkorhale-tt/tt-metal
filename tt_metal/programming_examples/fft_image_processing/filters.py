"""filters.py — frequency-domain masks (low-pass / high-pass / band-pass).

All masks live in **shifted** spectrum coordinates: the zero-frequency
(DC) bin is at the centre `(H/2, W/2)`. Apply them between `fftshift` and
`ifftshift`:

    F  = fftshift(fft2(img))
    Ff = F * mask          # use the masks below
    out = real(ifft2(ifftshift(Ff)))

Cutoffs are expressed as a fraction of `min(H, W) / 2` so they are
resolution-agnostic (`cutoff=0.25` covers ~ a quarter of the spectrum
radius, no matter the image size).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _radius_grid(shape: Tuple[int, int]) -> np.ndarray:
    """Distance of each pixel from the centre, in pixel units."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)


def _normalise_cutoff(shape: Tuple[int, int], cutoff: float) -> float:
    """Convert a relative cutoff in [0, 1] to absolute pixel radius."""
    H, W = shape
    max_r = min(H, W) / 2.0
    return float(cutoff) * max_r


def low_pass_filter(
    shape: Tuple[int, int],
    cutoff: float = 0.20,
    soft: bool = True,
    softness: float = 0.05,
) -> np.ndarray:
    """Low-pass mask: keep frequencies inside `cutoff`.

    Args:
        shape:    (H, W).
        cutoff:   fraction of (min(H,W)/2). 0 = keep DC only, 1 = keep all.
        soft:     if True, use a Butterworth-like smooth roll-off (avoids
                  ringing artefacts you'd get from a hard ideal cutoff).
        softness: roll-off width in the same units as `cutoff` (only used
                  when `soft=True`).
    """
    r = _radius_grid(shape)
    r0 = _normalise_cutoff(shape, cutoff)
    if not soft or softness <= 0.0:
        return (r <= r0).astype(np.float32)
    width = max(_normalise_cutoff(shape, softness), 1e-3)
    # smooth sigmoid roll-off
    return (1.0 / (1.0 + np.exp((r - r0) / width))).astype(np.float32)


def high_pass_filter(
    shape: Tuple[int, int],
    cutoff: float = 0.10,
    soft: bool = True,
    softness: float = 0.05,
) -> np.ndarray:
    """High-pass mask: keep frequencies outside `cutoff`. Edge enhancement."""
    return (1.0 - low_pass_filter(shape, cutoff, soft, softness)).astype(np.float32)


def band_pass_filter(
    shape: Tuple[int, int],
    low_cutoff: float = 0.05,
    high_cutoff: float = 0.30,
    soft: bool = True,
    softness: float = 0.03,
) -> np.ndarray:
    """Band-pass mask: keep frequencies between `low_cutoff` and `high_cutoff`."""
    if not (0.0 <= low_cutoff < high_cutoff <= 1.0):
        raise ValueError(
            f"need 0 <= low_cutoff < high_cutoff <= 1; "
            f"got low={low_cutoff}, high={high_cutoff}")
    lp_high = low_pass_filter(shape, high_cutoff, soft, softness)
    lp_low  = low_pass_filter(shape, low_cutoff,  soft, softness)
    return (lp_high - lp_low).astype(np.float32)


def make_filter(kind: str, shape: Tuple[int, int], **kwargs) -> np.ndarray:
    """Dispatcher: kind in {'low', 'high', 'band', 'lowpass', 'highpass', 'bandpass'}."""
    k = kind.lower().replace("-", "").replace("_", "")
    if k in ("low", "lowpass", "lp"):
        return low_pass_filter(shape, cutoff=kwargs.get("cutoff", 0.20),
                               soft=kwargs.get("soft", True),
                               softness=kwargs.get("softness", 0.05))
    if k in ("high", "highpass", "hp"):
        return high_pass_filter(shape, cutoff=kwargs.get("cutoff", 0.10),
                                soft=kwargs.get("soft", True),
                                softness=kwargs.get("softness", 0.05))
    if k in ("band", "bandpass", "bp"):
        return band_pass_filter(shape,
                                low_cutoff=kwargs.get("low_cutoff", 0.05),
                                high_cutoff=kwargs.get("high_cutoff", 0.30),
                                soft=kwargs.get("soft", True),
                                softness=kwargs.get("softness", 0.03))
    raise ValueError(f"unknown filter kind {kind!r}")


def apply_filter(spectrum_shifted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Element-wise multiply (the spectrum must already be fftshift-ed)."""
    if spectrum_shifted.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: spectrum {spectrum_shifted.shape} vs mask {mask.shape}")
    return (spectrum_shifted * mask).astype(np.complex64)
