# `tt_fft` — PyTorch-style FFT on Tenstorrent Wormhole

A drop-in replacement for `torch.fft.fft` / `np.fft.fft` that runs on a
Wormhole card via the `fft_universal` (fp32) and `fft_universal_bf16`
(true bf16) pipelines. **Any length N ≥ 2** is supported — pow2, prime,
composite — and the device-side dispatcher picks the right algorithm
automatically.

---

## TL;DR — one screen for the presentation

Both lines below run on the Tenstorrent card. `tt_fft.randn` allocates the
input via `ttnn.rand` on the device; `tt_fft.fft` runs the universal FFT
pipeline. Nothing on the host CPU except the API call itself.

```python
import tt_fft

x = tt_fft.randn(1024)              # input on Tenstorrent (ttnn.rand)
X = tt_fft.fft(x)                   # FFT  on Tenstorrent (fft_universal)
y = tt_fft.ifft(X)                  # IFFT on Tenstorrent

X_bf16 = tt_fft.fft(x, precision='bf16')   # TRUE bf16 FPU compute
```

Other Tenstorrent-native input generators:

```python
tt_fft.rand(N)                  # uniform [-1, 1) complex via ttnn.rand
tt_fft.randn(N)                 # standard normal complex (Box-Muller on top of ttnn.rand)
tt_fft.tone(N, k=17)            # pure complex tone exp(2*pi*i*k*n/N)
tt_fft.chord(N, freqs=(50,120,240))   # sum of real sinusoids + noise
```

If `ttnn` isn't installed in the active env, the random generators fall
back to numpy automatically and print a one-line warning.

Mirrors `torch.fft`:

| this                                  | maps to                       |
|---------------------------------------|-------------------------------|
| `tt_fft.fft(x)`                       | `torch.fft.fft(x)`            |
| `tt_fft.ifft(X)`                      | `torch.fft.ifft(X)`           |
| `tt_fft.rfft(x)`                      | `torch.fft.rfft(x)`           |
| `tt_fft.fft2(img)`                    | `torch.fft.fft2(img)`         |
| `tt_fft.benchmark(N)`                 | timing helper                  |
| `tt_fft.device_path(N)`               | which algorithm will run      |

---

## Build (one time, on the Wormhole machine)

```bash
cmake -S . -B build -DBUILD_PROGRAMMING_EXAMPLES=ON
cmake --build build --target \
  metal_example_fft_universal_run \
  metal_example_fft_universal_bf16_run -j
```

The Python module looks for the binaries at:

* `./build/programming_examples/fft_universal/metal_example_fft_universal_run`
* `./build/programming_examples/fft_universal_bf16/metal_example_fft_universal_bf16_run`

If your build dir is elsewhere, set `TT_FFT_BUILD=/path/to/build` (or
override the two `TT_FFT_BIN_FP32` / `TT_FFT_BIN_BF16` env vars).

---

## Live demo (CLI) — copy/paste during the talk

From `tt_metal/programming_examples/fft_universal/python/`:

```bash
# 1. PyTorch vs Wormhole, side-by-side, with plot
python demo_presentation.py compare 1024 fp32 --plot compare.png

# 2. ANY N — show the dispatcher picks the right algorithm
python demo_presentation.py compare 97         # prime  -> Bluestein
python demo_presentation.py compare 60         # composite non-pow2
python demo_presentation.py compare 65536      # big pow2
python demo_presentation.py compare 1024 bf16  # TRUE bf16 path

# 3. Pure tone -> single spike (very visual)
python demo_presentation.py spike 1024 --k 17 --plot spike.png

# 4. Audio chord -> spectrum bars
python demo_presentation.py chord 4096 --plot chord.png

# 5. Round-trip:  x -> FFT -> IFFT -> x
python demo_presentation.py round_trip 1000

# 6. Speed sweep across many N
python demo_presentation.py bench --plot bench.png
```

---

## Live demo (Jupyter) — best on a projector

```bash
jupyter notebook demo.ipynb
```

The notebook walks through the same flow with rendered plots inline.

---

## How it works (one paragraph)

`tt_fft.fft(x)` writes the input numpy array to a tmp text file, invokes
the C++ binary `metal_example_fft_universal{,_bf16}_run`, and reads the
output back as numpy. The C++ binary calls into the existing
`fft_universal::fft` / `fft_universal_bf16::fft` host functions, which:

1. inspect `N` and pick an algorithm (Stockham, Bluestein, mixed-radix,
   or packed direct-DFT),
2. lay out the work across up to 64 Tensix cores,
3. run reader / compute / writer kernels via tt-metal CBs and matmul
   tiles (`fp32` SFPU on Stockham, FPU bf16+fp32 accumulator on the
   packed/Bluestein/mixed-radix paths), and
4. return the result.

No Python bindings, no extra C++ exports — just file I/O wrapping the
binary that already exists.

---

## API reference

```python
# FFT (runs on Tenstorrent)
tt_fft.fft(x, precision='fp32', verbose=False) -> np.ndarray (complex64)
tt_fft.ifft(X, precision='fp32', verbose=False) -> np.ndarray (complex64)
tt_fft.rfft(x, precision='fp32') -> np.ndarray  # first N//2 + 1 bins
tt_fft.fft2(img, precision='fp32') -> np.ndarray  # naive row+col 2-D

# Tenstorrent-native input generators (use ttnn.rand on the device)
tt_fft.rand(N, complex=True, seed=None) -> np.ndarray
tt_fft.randn(N, complex=True, seed=None) -> np.ndarray
tt_fft.tone(N, k=1) -> np.ndarray
tt_fft.chord(N, freqs=(50,120,240), amps=(1.0,0.6,0.3),
             noise=0.02, seed=7) -> np.ndarray

# ttnn interop (optional)
tt_fft.to_ttnn(x, dtype='bfloat16')  # numpy/torch -> ttnn.Tensor on device
tt_fft.from_ttnn(t)                  # ttnn.Tensor -> numpy

# Introspection / utilities
tt_fft.device_path(N) -> str        # which algorithm will run
tt_fft.benchmark(N, iters=20, precision='fp32') -> dict
tt_fft.set_binaries(fp32=..., bf16=...)
```

`x` may be a numpy array, a Python list, or a torch tensor; real inputs
are auto-promoted to complex.

---

## Troubleshooting

* `FileNotFoundError: ... metal_example_fft_universal_run` — build the
  binaries (see "Build" above) or set `TT_FFT_BUILD` / `TT_FFT_BIN_*`.
* `RuntimeError: tt_fft device call failed` — re-run the underlying binary
  by hand to see the device error:
  `./build/programming_examples/fft_universal/metal_example_fft_universal_run /tmp/in.txt /tmp/out.txt`.
* Cold (first) call is slow because the kernels JIT-compile; subsequent
  calls hit the cache and run in milliseconds. The `benchmark()` helper
  reports cold vs warm separately.
