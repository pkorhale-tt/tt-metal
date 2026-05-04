"""main.py — CLI entry point that wires every module together.

End-to-end flow:
    1. Load image (or generate synthetic).
    2. Optionally add Gaussian noise.
    3. Compute 2-D FFT (custom / numpy / torch).
    4. fftshift, build filter mask, multiply.
    5. ifftshift, inverse FFT, take real part.
    6. Save: original, noisy, |F| spectrum, |F| filtered, output.
    7. Benchmark all engines + print MSE / PSNR vs the clean reference.

Usage examples:
    # default:  synthetic 256x256, low-pass denoising, fp32, save to ./outputs
    python -m fft_image_processing.main

    # specific image, high-pass edge enhancement
    python -m fft_image_processing.main --image my.png --filter high --cutoff 0.08

    # already noisy input, low-pass denoise, save to custom dir
    python -m fft_image_processing.main --image noisy.png --no-noise \
                                         --filter low --cutoff 0.20 --out ./run1

    # bf16 precision mode (Tenstorrent backend)
    python -m fft_image_processing.main --precision bf16

    # band-pass + benchmark vs torch and numpy
    python -m fft_image_processing.main --filter band \
        --low 0.05 --high 0.30 --bench-iters 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# Allow `python main.py ...` from inside the folder as well as
# `python -m fft_image_processing.main ...` from the repo root.
if __package__ in (None, ""):
    HERE = os.path.dirname(os.path.abspath(__file__))
    PARENT = os.path.dirname(HERE)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)
    from fft_image_processing import (  # type: ignore
        image_loader, fft_module, torch_fft_module,
        filters, metrics, visualization, benchmarking)
else:
    from . import (image_loader, fft_module, torch_fft_module,
                   filters, metrics, visualization, benchmarking)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FFT-based image processing demo (Tenstorrent + PyTorch + NumPy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    g_in = p.add_argument_group("input")
    g_in.add_argument("--image", default=None,
                      help="path to input image. omit / 'synthetic' for built-in test image.")
    g_in.add_argument("--size", type=int, default=256,
                      help="resize to (size x size). default 256.")
    g_in.add_argument("--synthetic-kind", default="mixed",
                      choices=("mixed", "checker", "circles", "gradient"),
                      help="which synthetic image when --image is omitted.")

    g_n = p.add_argument_group("noise")
    g_n.add_argument("--noise", action="store_true",
                     help="add synthetic Gaussian noise on top of the input.")
    g_n.add_argument("--no-noise", dest="noise", action="store_false",
                     help="treat input as already noisy / clean (no extra noise).")
    g_n.set_defaults(noise=False)
    g_n.add_argument("--sigma", type=float, default=0.05,
                     help="std of synthetic noise in [0,1]. default 0.05.")

    g_f = p.add_argument_group("filter")
    g_f.add_argument("--filter", default="low", choices=("low", "high", "band"),
                     help="frequency filter type. default low (denoising).")
    g_f.add_argument("--cutoff", type=float, default=0.20,
                     help="low/high cutoff (fraction of max radius, 0..1).")
    g_f.add_argument("--low",  dest="low_cutoff",  type=float, default=0.05,
                     help="band-pass low edge.")
    g_f.add_argument("--high", dest="high_cutoff", type=float, default=0.30,
                     help="band-pass high edge.")
    g_f.add_argument("--hard", action="store_true",
                     help="use hard ideal cutoff (default = smooth roll-off).")

    g_e = p.add_argument_group("engine")
    g_e.add_argument("--engine", default="custom",
                     choices=("custom", "numpy", "torch"),
                     help="which engine to use for the *displayed* result.")
    g_e.add_argument("--precision", default="fp32", choices=("fp32", "bf16"),
                     help="precision mode for the custom (Tenstorrent) backend.")

    g_b = p.add_argument_group("benchmark")
    g_b.add_argument("--bench-iters", type=int, default=5,
                     help="iterations per engine for benchmarking. default 5.")
    g_b.add_argument("--bench-warmup", type=int, default=1,
                     help="warm-up iterations (excluded from timing). default 1.")
    g_b.add_argument("--no-bench", action="store_true",
                     help="skip the benchmark step.")
    g_b.add_argument("--bench-engines", default="auto",
                     help="comma-separated list of engines to benchmark "
                          "(custom,numpy,torch). 'auto' picks numpy+torch for "
                          "images larger than 64x64 to avoid 2N subprocess "
                          "calls through the per-call wormhole wrapper. "
                          "default: auto.")

    g_o = p.add_argument_group("output")
    g_o.add_argument("--out", default="outputs",
                     help="output directory for plots & images. default ./outputs.")
    g_o.add_argument("--prefix", default="run",
                     help="filename prefix for saved files.")
    g_o.add_argument("--no-save", action="store_true",
                     help="don't write any files to disk.")
    return p


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_dir = args.out if not args.no_save else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    backend = fft_module.backend_info()
    print()
    print("=" * 72)
    print("  FFT-based image processing")
    print("=" * 72)
    print(f"  custom backend  : {backend['backend']}")
    if backend["backend"] != "tenstorrent":
        print(f"     reason       : {backend.get('reason', '')}")
    print(f"  engine selected : {args.engine}")
    print(f"  precision       : {args.precision}")
    print()

    # ---------------- 1. Load image ----------------
    print("[1/8] Loading image...")
    if args.image is None:
        clean_ref = image_loader.make_synthetic_image(
            args.size, args.size, kind=args.synthetic_kind)
        print(f"      synthetic '{args.synthetic_kind}' {clean_ref.shape}")
    else:
        clean_ref = image_loader.load_image(
            args.image, size=(args.size, args.size), grayscale=True)
        print(f"      loaded {args.image} -> {clean_ref.shape}")

    # ---------------- 2. Optional noise ----------------
    if args.noise:
        noisy = image_loader.add_gaussian_noise(clean_ref, sigma=args.sigma)
        print(f"[2/8] Added Gaussian noise sigma={args.sigma}")
        input_img = noisy
    else:
        noisy = None
        input_img = clean_ref
        print("[2/8] No synthetic noise (input used as-is).")

    # ---------------- 3. FFT2 ----------------
    print(f"[3/8] FFT2 with engine={args.engine}...")
    t0 = time.time()
    if args.engine == "custom":
        spectrum = fft_module.fft2(input_img, precision=args.precision)
    elif args.engine == "numpy":
        spectrum = np.fft.fft2(input_img.astype(np.complex64))
    elif args.engine == "torch":
        spectrum = torch_fft_module.torch_fft2(input_img)
    else:
        raise ValueError(args.engine)
    fft_ms = (time.time() - t0) * 1000.0
    print(f"      done in {fft_ms:.2f} ms")

    # ---------------- 4. fftshift + filter ----------------
    print(f"[4/8] Apply {args.filter}-pass filter...")
    spectrum_shifted = fft_module.fftshift(spectrum)

    if args.filter == "band":
        mask = filters.make_filter("band", spectrum_shifted.shape,
                                   low_cutoff=args.low_cutoff,
                                   high_cutoff=args.high_cutoff,
                                   soft=not args.hard)
        filt_label = (f"band-pass [{args.low_cutoff:.2f}, "
                      f"{args.high_cutoff:.2f}] "
                      f"({'hard' if args.hard else 'soft'})")
    else:
        mask = filters.make_filter(args.filter, spectrum_shifted.shape,
                                   cutoff=args.cutoff, soft=not args.hard)
        filt_label = (f"{args.filter}-pass cutoff={args.cutoff:.2f} "
                      f"({'hard' if args.hard else 'soft'})")
    filtered_shifted = filters.apply_filter(spectrum_shifted, mask)
    print(f"      {filt_label}")

    # ---------------- 5. ifftshift + IFFT2 ----------------
    print(f"[5/8] IFFT2 with engine={args.engine}...")
    filtered_spectrum = fft_module.ifftshift(filtered_shifted)
    t0 = time.time()
    if args.engine == "custom":
        recon = fft_module.ifft2(filtered_spectrum, precision=args.precision)
    elif args.engine == "numpy":
        recon = np.fft.ifft2(filtered_spectrum)
    else:  # torch
        recon = torch_fft_module.torch_ifft2(filtered_spectrum)
    ifft_ms = (time.time() - t0) * 1000.0
    recon = np.real(recon).astype(np.float32)
    recon = np.clip(recon, 0.0, 1.0)
    print(f"      done in {ifft_ms:.2f} ms")

    # ---------------- 6. Visualisation ----------------
    print("[6/8] Building plots...")
    panels = [clean_ref]
    titles = ["original (clean reference)"]
    if noisy is not None:
        panels.append(noisy);                         titles.append(f"noisy (sigma={args.sigma})")
    panels.append(np.abs(spectrum_shifted));          titles.append("|F| (shifted)")
    panels.append(mask);                              titles.append(f"mask: {filt_label}")
    panels.append(np.abs(filtered_shifted));          titles.append("|F * mask|")
    panels.append(recon);                             titles.append(f"reconstructed ({args.engine})")

    if out_dir:
        panel_path = os.path.join(out_dir, f"{args.prefix}_panel.png")
        visualization.show_panel(
            panels, titles, save_path=panel_path,
            suptitle=f"FFT image pipeline | engine={args.engine} | filter={filt_label}",
            cols=3)
        visualization.save_image(clean_ref, os.path.join(out_dir, f"{args.prefix}_original.png"))
        if noisy is not None:
            visualization.save_image(noisy, os.path.join(out_dir, f"{args.prefix}_noisy.png"))
        visualization.save_image(recon, os.path.join(out_dir, f"{args.prefix}_recon.png"))
        visualization.show_spectrum(
            spectrum_shifted,
            save_path=os.path.join(out_dir, f"{args.prefix}_spectrum.png"))
        visualization.show_spectrum(
            filtered_shifted,
            save_path=os.path.join(out_dir, f"{args.prefix}_spectrum_filtered.png"),
            title=f"|F * mask| (log)  -- {filt_label}")
    else:
        print("      [--no-save] skipping disk writes")

    # ---------------- 7. Metrics ----------------
    print("[7/8] Metrics vs clean reference:")
    cands = {"reconstructed": recon}
    if noisy is not None: cands["noisy"] = noisy
    summary = metrics.summarise(clean_ref, cands, data_range=1.0)
    for k, v in summary.items():
        print(f"      {k:<14s}  MSE = {v['mse']:.6e}   PSNR = {v['psnr_db']:6.2f} dB")

    # ---------------- 8. Benchmark ----------------
    if not args.no_bench:
        if args.bench_engines == "auto":
            h, w = input_img.shape[:2]
            if max(h, w) <= 64:
                bench_engines = ("custom", "numpy", "torch")
            else:
                bench_engines = ("numpy", "torch")
                print(f"[8/8] image is {h}x{w} (> 64); excluding 'custom' "
                      f"from benchmark to avoid {2*h} subprocess launches.")
                print("      use --bench-engines custom,numpy,torch to force "
                      "it (slow), or --size 32 for a tiny demo.")
        else:
            bench_engines = tuple(s.strip() for s in args.bench_engines.split(",")
                                  if s.strip())
        print(f"[8/8] Benchmark engines: {bench_engines}")
        bench = benchmarking.benchmark_fft2(
            input_img,
            engines=bench_engines,
            precision=args.precision,
            iters=args.bench_iters, warmup=args.bench_warmup)
        ref = "torch" if "torch" in bench else "numpy"
        benchmarking.pretty_print(bench, reference=ref)
    else:
        print("[8/8] benchmark skipped (--no-bench)")

    print("Done.")
    if out_dir:
        print(f"All outputs in: {os.path.abspath(out_dir)}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
