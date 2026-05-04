# `fft_image_processing` — modular FFT-based image processing on Wormhole

Drop-in PyTorch / NumPy-style pipeline that runs a real 2-D FFT on a
Tenstorrent Wormhole accelerator (via `tt_fft`), applies a frequency-domain
filter, and reconstructs the image — with built-in correctness metrics
and side-by-side benchmarking against `torch.fft` and `numpy.fft`.

## Layout

```
fft_image_processing/
├── __init__.py
├── image_loader.py      # load grayscale image; optional Gaussian noise; synthetic test images
├── fft_module.py        # custom 2-D FFT (Tenstorrent if available, numpy fallback)
├── torch_fft_module.py  # torch.fft.fft2 reference
├── filters.py           # low-pass / high-pass / band-pass masks (smooth or hard)
├── metrics.py           # MSE / PSNR / SNR (dB)
├── visualization.py     # matplotlib panel + spectrum + image savers
├── benchmarking.py      # head-to-head timing across engines
├── main.py              # CLI entry point that wires everything together
├── requirements.txt
└── outputs/             # generated PNGs land here
```

## Build & path setup (one time, on the Wormhole machine)

The custom FFT backend is the same `tt_fft` Python wrapper used by your
benchmarks. Make sure those two binaries exist:

```bash
cmake --build build --target \
    metal_example_fft_universal_run \
    metal_example_fft_universal_bf16_run -j 8

export TT_METAL_HOME=$PWD
export TT_FFT_BUILD=$PWD/build
```

`fft_image_processing/fft_module.py` automatically locates the `tt_fft`
module next to it (`../fft_universal/python/tt_fft.py`) — no extra
PYTHONPATH plumbing needed.

If `tt_fft` (or the binaries) aren't reachable, the `custom` engine
silently falls back to `np.fft.fft2` so the rest of the demo still runs.

## Quick start

```bash
cd tt_metal/programming_examples

# Default: synthetic 256x256 image, low-pass filter, fp32, plots in ./outputs
python -m fft_image_processing.main

# Real photo, edge enhancement (high-pass), save to a named folder
python -m fft_image_processing.main --image lena.png --filter high \
    --cutoff 0.08 --out outputs/lena_edges --prefix lena

# Already-noisy input + denoising
python -m fft_image_processing.main --image noisy.png --no-noise \
    --filter low --cutoff 0.20

# bf16 precision (Tenstorrent backend)
python -m fft_image_processing.main --precision bf16

# Band-pass with custom edges, more bench iters
python -m fft_image_processing.main --filter band --low 0.05 --high 0.30 \
    --bench-iters 10
```

Run from inside the folder also works:

```bash
cd tt_metal/programming_examples/fft_image_processing
python main.py --image my.png --filter low
```

## What you get

For each run you see:

* **Console**:
  - which backend is active (`tenstorrent` or `numpy_fallback`)
  - the per-stage timing
  - **MSE / PSNR** of the reconstructed image vs the clean reference
  - a table comparing **mean / min / max ms** for `custom` / `numpy` / `torch`
    plus the max numerical diff vs PyTorch
* **Outputs/ folder** (PNG):
  - `<prefix>_original.png` — clean reference
  - `<prefix>_noisy.png` — synthetic noisy input (if `--noise`)
  - `<prefix>_spectrum.png` — `|F|` (log, fftshift-ed)
  - `<prefix>_spectrum_filtered.png` — `|F * mask|`
  - `<prefix>_recon.png` — final reconstructed image
  - `<prefix>_panel.png` — all six tiles in a single figure for slides

## Module-level API (use it from your own scripts)

```python
import numpy as np
from fft_image_processing import (
    image_loader, fft_module, filters, metrics, visualization)

img = image_loader.load_image("photo.png", size=(256, 256))
noisy = image_loader.add_gaussian_noise(img, sigma=0.05)

F = fft_module.fft2(noisy, precision="fp32")           # Wormhole
F_shifted = fft_module.fftshift(F)
mask = filters.low_pass_filter(F_shifted.shape, cutoff=0.20)
F_filt = F_shifted * mask
recon = np.real(fft_module.ifft2(fft_module.ifftshift(F_filt))).astype(np.float32)

print("PSNR vs clean :", metrics.psnr(img, np.clip(recon, 0, 1)))
visualization.show_panel(
    [img, noisy, np.abs(F_shifted), mask, np.abs(F_filt), recon],
    ["original", "noisy", "|F|", "mask", "|F*mask|", "recon"],
    save_path="outputs/manual.png", cols=3)
```

## CLI flags

| Group | Flag | Default | Meaning |
|---|---|---|---|
| input | `--image PATH` | (synthetic) | input image; if omitted, a synthetic test image is generated |
| input | `--size N` | 256 | resize to N×N |
| input | `--synthetic-kind` | `mixed` | one of `mixed`, `checker`, `circles`, `gradient` |
| noise | `--noise` / `--no-noise` | off | toggle synthetic Gaussian noise |
| noise | `--sigma S` | 0.05 | noise std in [0, 1] |
| filter | `--filter K` | `low` | one of `low`, `high`, `band` |
| filter | `--cutoff C` | 0.20 | low/high cutoff (fraction of max radius) |
| filter | `--low / --high` | 0.05 / 0.30 | band-pass edges |
| filter | `--hard` | off | use hard ideal cutoff (default: smooth roll-off) |
| engine | `--engine {custom,numpy,torch}` | `custom` | which engine produces the displayed result |
| engine | `--precision {fp32,bf16}` | `fp32` | precision mode for the custom (Tenstorrent) backend |
| bench | `--bench-iters N` | 5 | timing iterations |
| bench | `--bench-warmup N` | 1 | warm-up iterations excluded from timing |
| bench | `--no-bench` | off | skip benchmarking |
| output | `--out DIR` | `outputs` | output folder |
| output | `--prefix P` | `run` | filename prefix |
| output | `--no-save` | off | don't write any files |

## Caveats / notes

* The 2-D FFT here is built as `row-FFTs` then `column-FFTs` over the
  Wormhole 1-D `tt_fft` engine. That's the canonical decomposition; we
  don't yet have a fused on-device 2-D kernel.
* The first call costs ~350 ms (kernel JIT + plan build); subsequent calls
  are warm. The benchmark's `--bench-warmup 1` hides the JIT cost.
* `--precision bf16` only affects the custom (Tenstorrent) backend.
  numpy and torch always run in fp32.
* Filter masks use a smooth (Butterworth-like) roll-off by default to
  avoid ringing artefacts. Use `--hard` to get the textbook ideal cutoff.
