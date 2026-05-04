"""visualization.py — matplotlib-based plot/save helpers.

    show_panel(images, titles, save_path=None, suptitle=None, cols=None)
    show_spectrum(spectrum, save_path=None, log=True, title='|F| (log)')
    save_image(img, path)

If matplotlib isn't available the helpers degrade to text printouts so the
pipeline still runs. Plots are saved at 120 dpi by default.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # safe for headless servers; comment out for live show
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def _grid_layout(n: int, cols: Optional[int]) -> tuple:
    if cols is None:
        cols = min(n, 3)
    rows = (n + cols - 1) // cols
    return rows, cols


def show_panel(
    images: Iterable[np.ndarray],
    titles: Iterable[str],
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    cols: Optional[int] = None,
    cmap: str = "gray",
    dpi: int = 120,
) -> None:
    """Plot multiple grayscale images in a grid."""
    images = list(images)
    titles = list(titles)
    if len(images) != len(titles):
        raise ValueError("images and titles must have same length")
    if not HAVE_PLT:
        for t, img in zip(titles, images):
            print(f"[viz] {t}: shape={img.shape}, "
                  f"min={float(img.min()):.4f}, max={float(img.max()):.4f}")
        return

    rows, cols = _grid_layout(len(images), cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.atleast_1d(axs).ravel()
    for i, (t, img) in enumerate(zip(titles, images)):
        axs[i].imshow(np.real(img), cmap=cmap)
        axs[i].set_title(t)
        axs[i].axis("off")
    for ax in axs[len(images):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"[viz] saved {save_path}")
        plt.close(fig)
    else:
        plt.show()


def show_spectrum(
    spectrum: np.ndarray,
    save_path: Optional[str] = None,
    log: bool = True,
    title: str = "|F| (log)",
    dpi: int = 120,
) -> None:
    """Show the magnitude of an fftshift-ed 2-D spectrum."""
    mag = np.abs(spectrum).astype(np.float64)
    disp = np.log1p(mag) if log else mag
    if not HAVE_PLT:
        print(f"[viz] {title}: shape={disp.shape}, max={disp.max():.3e}")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(disp, cmap="magma")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"[viz] saved {save_path}")
        plt.close(fig)
    else:
        plt.show()


def save_image(img: np.ndarray, path: str) -> None:
    """Save a single grayscale image to disk (no matplotlib chrome)."""
    arr = np.real(img).astype(np.float64)
    arr = np.clip(arr, 0.0, 1.0)
    if HAVE_PLT:
        plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)
        print(f"[viz] saved {path}")
    else:
        # very last resort: dump as text (rare)
        np.savetxt(path + ".txt", arr, fmt="%.4f")
        print(f"[viz] matplotlib unavailable; wrote {path}.txt")
