"""benchmarking.py — head-to-head FFT timing across engines.

    benchmark_fft2(img, engines=('custom', 'numpy', 'torch'),
                   precision='fp32', iters=5, warmup=1)
        -> dict[engine] = { 'mean_ms': float, 'std_ms': float, 'min_ms': float,
                            'max_ms': float, 'output': np.ndarray }

    pretty_print(results, reference='torch')
        -> prints a one-screen table with mean / min / max + numerical diff
           vs `reference` engine.

The 'custom' engine routes through fft_module.fft2 (Tenstorrent if available,
numpy fallback otherwise). 'numpy' is np.fft.fft2 in fp32. 'torch' is
torch.fft.fft2 (CPU). All inputs are the same numpy array, so the timer
captures end-to-end host-visible latency.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

import numpy as np

from . import fft_module

try:
    from . import torch_fft_module
    _HAVE_TORCH = torch_fft_module.HAVE_TORCH
except Exception:
    _HAVE_TORCH = False
    torch_fft_module = None  # type: ignore


def _time_call(fn, *args, **kw):
    t0 = time.time()
    out = fn(*args, **kw)
    return out, (time.time() - t0) * 1000.0


def benchmark_fft2(
    img: np.ndarray,
    engines: Iterable[str] = ("custom", "numpy", "torch"),
    precision: str = "fp32",
    iters: int = 5,
    warmup: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Time each engine `iters` times after `warmup` warm-up calls.

    Returns a dict keyed by engine name. Engines that aren't available are
    silently skipped (with a printed note) so the demo never crashes mid-talk.
    """
    img = np.asarray(img)
    results: Dict[str, Dict[str, float]] = {}

    runners = {
        "custom": lambda x: fft_module.fft2(x, precision=precision),
        "numpy":  lambda x: np.fft.fft2(x.astype(np.complex64)),
    }
    if _HAVE_TORCH:
        runners["torch"] = lambda x: torch_fft_module.torch_fft2(x)

    for name in engines:
        if name not in runners:
            print(f"[bench] engine {name!r} not available, skipping")
            continue
        fn = runners[name]
        # warmup
        for _ in range(max(0, warmup)):
            _ = fn(img)
        times: list[float] = []
        last_out: Optional[np.ndarray] = None
        for _ in range(max(1, iters)):
            out, ms = _time_call(fn, img)
            times.append(ms)
            last_out = out
        arr = np.asarray(times)
        results[name] = {
            "mean_ms": float(arr.mean()),
            "std_ms":  float(arr.std()),
            "min_ms":  float(arr.min()),
            "max_ms":  float(arr.max()),
            "output":  last_out,  # type: ignore  # kept for diff reporting
        }
    return results


def pretty_print(results: Dict[str, Dict], reference: str = "torch") -> None:
    """Render a one-screen comparison table."""
    if not results:
        print("[bench] no results to print"); return

    ref = results.get(reference)
    ref_out = ref["output"] if ref is not None else None

    name_w = max(len("engine"), max(len(k) for k in results.keys()))
    print()
    print("-" * (name_w + 60))
    print(f"  {'engine':<{name_w}}   mean (ms)  min (ms)  max (ms)   "
          f"max|err| vs {reference if ref_out is not None else '-':<8}")
    print("-" * (name_w + 60))
    for name, r in results.items():
        diff_str = "-"
        if ref_out is not None:
            diff = float(np.max(np.abs(np.asarray(r["output"]) - np.asarray(ref_out))))
            diff_str = f"{diff:.3e}"
        print(f"  {name:<{name_w}}   {r['mean_ms']:8.2f}   "
              f"{r['min_ms']:7.2f}   {r['max_ms']:7.2f}   {diff_str}")
    print("-" * (name_w + 60))
    backend = fft_module.backend_info()
    print(f"  custom backend: {backend['backend']}")
    if backend["backend"] != "tenstorrent":
        print(f"    reason: {backend.get('reason', '')}")
    print()
