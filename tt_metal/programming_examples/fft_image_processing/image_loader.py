"""image_loader.py — grayscale image input + optional synthetic noise.

Public API:
    load_image(path, size=None, grayscale=True)        -> np.ndarray (H, W) float32 in [0, 1]
    add_gaussian_noise(img, sigma=0.05, seed=0)        -> np.ndarray (H, W) float32
    make_synthetic_image(H=256, W=256, kind='checker') -> np.ndarray (H, W) float32

Supported kinds for `make_synthetic_image`:
    'checker' — high-frequency checkerboard
    'circles' — concentric ring pattern
    'gradient' — smooth gradient (low-frequency only)
    'mixed'    — gradient + circle + impulse spike (rich spectrum)

The loader has graceful fallbacks:
  * If pillow / opencv are not installed, falls back to scikit-image, then
    matplotlib.image.imread, then a synthetic image so the demo never breaks.
  * If `path` is None or 'synthetic', returns the synthetic image directly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Image I/O with graceful fallback
# ----------------------------------------------------------------------

def _read_with_pillow(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def _read_with_cv2(path: str) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img.astype(np.float32) / 255.0
    except Exception:
        return None


def _read_with_skimage(path: str) -> Optional[np.ndarray]:
    try:
        from skimage.io import imread  # type: ignore
        from skimage.color import rgb2gray  # type: ignore
        img = imread(path)
        if img.ndim == 3:
            img = rgb2gray(img)
        img = img.astype(np.float32)
        if img.max() > 1.5:  # likely uint8 range
            img /= 255.0
        return img
    except Exception:
        return None


def _read_with_matplotlib(path: str) -> Optional[np.ndarray]:
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(path)
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        return img
    except Exception:
        return None


def load_image(
    path: Optional[str] = None,
    size: Optional[Tuple[int, int]] = None,
    grayscale: bool = True,
) -> np.ndarray:
    """Load an image as a 2-D float32 numpy array in [0, 1].

    Args:
        path:       Path to image file. If None or 'synthetic', a synthetic
                    image is returned so the demo always works out of the box.
        size:       Optional (H, W) to resize to. Useful for FFT power-of-two.
        grayscale:  Always True in this demo (we operate on a single channel).

    Returns:
        np.ndarray of shape (H, W), dtype float32, range [0, 1].
    """
    if path is None or str(path).lower() == "synthetic":
        H, W = size if size is not None else (256, 256)
        return make_synthetic_image(H=H, W=W, kind="mixed")

    img = (_read_with_pillow(path)
           or _read_with_cv2(path)
           or _read_with_skimage(path)
           or _read_with_matplotlib(path))
    if img is None:
        print(f"[image_loader] could not read {path!r}; using synthetic image")
        H, W = size if size is not None else (256, 256)
        return make_synthetic_image(H=H, W=W, kind="mixed")

    if grayscale and img.ndim == 3:
        img = img.mean(axis=-1)

    if size is not None and img.shape != size:
        img = _resize(img, size)

    return img.astype(np.float32)


def _resize(img: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """Lightweight nearest-neighbour resize so we don't need cv2/PIL."""
    H, W = img.shape
    out_h, out_w = target
    rs = (np.arange(out_h) * H / out_h).astype(np.int64)
    cs = (np.arange(out_w) * W / out_w).astype(np.int64)
    return img[rs[:, None], cs[None, :]]


# ----------------------------------------------------------------------
# Synthetic noise
# ----------------------------------------------------------------------

def add_gaussian_noise(
    img: np.ndarray,
    sigma: float = 0.05,
    seed: int = 0,
    clip: bool = True,
) -> np.ndarray:
    """Add Gaussian noise with std `sigma` (in [0, 1] image scale).

    Set `clip=False` if you want the noisy image to keep negative / >1 values.
    """
    rng = np.random.default_rng(seed)
    noisy = img + rng.standard_normal(img.shape).astype(np.float32) * sigma
    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)


# ----------------------------------------------------------------------
# Synthetic test images
# ----------------------------------------------------------------------

def make_synthetic_image(H: int = 256, W: int = 256, kind: str = "mixed") -> np.ndarray:
    """Generate a deterministic test image. No external dependencies."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if kind == "checker":
        sq = 16
        img = ((((xx // sq) + (yy // sq)) % 2)).astype(np.float32)
    elif kind == "circles":
        img = 0.5 + 0.5 * np.cos(rr / 4.0)
    elif kind == "gradient":
        img = (xx / max(W - 1, 1)) * 0.6 + (yy / max(H - 1, 1)) * 0.4
    elif kind == "mixed":
        gradient = 0.5 * (xx / max(W - 1, 1)) + 0.3 * (yy / max(H - 1, 1))
        ring     = 0.4 * np.exp(-((rr - min(H, W) / 4.0) ** 2) / (2 * (min(H, W) / 30.0) ** 2))
        spike    = np.zeros_like(yy); spike[H // 4, W // 4] = 1.0
        sinusoid = 0.2 * np.cos(2 * np.pi * 8 * xx / W)
        img = gradient + ring + spike + sinusoid
    else:
        raise ValueError(f"unknown kind={kind!r}")

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img.astype(np.float32)
