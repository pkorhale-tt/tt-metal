# Host vs Device split — `ttnn.experimental.fft`

This document records, for every backend, exactly **which work runs on
the host CPU** and which runs on the Wormhole Tensix cores. It is
intended for the paper's "implementation" section so that the claims
about being "device-resident" can be made honestly.

The accounting is taken directly from the source on this branch
(`pkorhale/experimental-fft`).

---

## Universal call-site overhead (every backend)

The op is wrapped by `fft_program_factory.cpp::run_program()`. Before
any backend is invoked, for every call:

1. **Host fp32 materialisation of the input tensor.**
   `read_real_as_fp32()` copies the input from DRAM into a host
   `std::vector<float>` of size `B·N`. This is unconditional and
   happens once per real input and once per imag input (if provided).

   ```cpp
   // fft_program_factory.cpp
   const std::vector<float> in_re = read_real_as_fp32(in_re_tensor);
   ```

2. **Output materialisation back to a host fp32 buffer**, which is then
   written back into a freshly allocated device tensor.

3. **IFFT conjugate pre/post**:
   ```cpp
   const float scale = attrs.inverse ? (1.0f / N) : 1.0f;
   for (i = 0; i < N; ++i)
       work[i] = Complex{in_re[..], -in_im[..]};      // pre-conj
   // backend runs forward FFT into work
   for (i = 0; i < N; ++i)
       work[i] = scale * Complex{work[i].real(), -work[i].imag()};  // post-conj
   ```

So **every `ttnn.experimental.fft` call today pays one host round-trip
of the data** even before the backend starts. This is the most
important caveat for the paper: when you report "device time", be
explicit about whether the round-trip is included.

| Quantity                          | Host or Device?         |
|-----------------------------------|-------------------------|
| Input DRAM → host fp32 buffer     | host (memcpy + cast)    |
| Output host fp32 → DRAM           | host (memcpy + cast)    |
| IFFT conjugate + 1/N              | host O(N)               |
| Algorithm itself                  | mixed (see per-backend) |

> The `bench_host_device_split.py` script measures this delta so you
> can quote both "as the user sees it" and "with the data already on
> device" numbers in the paper.

---

## 1. `fft_stockham`

`stockham_host.hpp` (file header: *"Multi-pass Stockham (six-step /
Bailey 4-step)"*).

### On host
- **Plan construction.** For each (N, dtype) the orchestrator
  builds a `StockhamPlan` once and caches it (`plan_cache`). The plan
  captures the per-pass radices, the per-pass twiddle tables, CB
  configs, and the kernel binary handle. **First call pays plan +
  JIT cost; subsequent calls only pay dispatch.**
- **Twiddle table generation.** Per-pass twiddles are computed in
  `std::complex<float>`, tilized, and uploaded once when the plan
  is built. Not recomputed on subsequent calls.
- **B(N₁,N₂) → C(N₂,N₁) transpose** at the end of the Bailey 4-step
  decomposition is performed in **a pure host buffer shuffle**
  (see line 648 of `stockham_host.hpp`). For N ≤ 1024 (tile-fits) this
  step is skipped entirely.

### On device
- All butterflies, all twiddle multiplies, all bit-shuffling between
  passes. For tile-fits N ≤ 1024 the entire FFT is a single kernel
  dispatch.

---

## 2. `fft_universal` (mixed-radix + Bluestein)

### On host (factorization path)
- **Trial division of N** into small radices.
- **Per-stage twiddle tables** (chained Cooley-Tukey roots of unity).
  Cached per (N, dtype).
- For prime N: **chirp table `w[n] = exp(-i π n²/N)`** and the
  **`B_fft` reference spectrum** (length M = next_pow2(2N-1)) are
  precomputed in fp64 on host and uploaded once. They are reused on
  every subsequent call with the same N.

### On device
- Stockham sub-passes for the pow2 factors.
- For non-pow2 small factors (≤32) the packed batched kernel handles
  the radix in a single shot.
- For Bluestein: the **two length-M FFTs**, the **pointwise complex
  multiply `A·B_fft`**, and the **chirp pre/post multiplies** all
  dispatch through Stockham + small device ops on tile.

### Host-only summary table (Bluestein path)

| Step                                              | Where  | Per call?            |
|---------------------------------------------------|--------|----------------------|
| Trial division / factorization                    | host   | first call only      |
| Chirp `w[n]` table                                | host   | first call only      |
| `B_fft` reference spectrum                        | host   | first call only      |
| Padding x → length-M sequence                     | host   | every call           |
| Slicing M-length output → N                       | host   | every call           |
| Both length-M FFTs                                | device | every call           |
| Pointwise `A·B_fft`                               | device | every call           |
| Chirp pre/post multiplies                         | device | every call           |

The per-call host work for the Bluestein path is therefore O(M)
memcopies (padding + slicing) **plus** the universal
read/write described at the top of this file.

---

## 3. `fft_universal_bf16`

Same host/device split as `fft_universal` above. The differences are:

- All circular buffers are configured as `Float16_b`, not `Float32`.
- The compute kernel is the **FPU bf16-mantissa matmul** path; there
  is no SFPU fp32 alternative.
- The twiddle tables are still computed in `std::complex<float>` on
  host then **rounded to bf16 at upload time**.

There is no extra host computation — the bf16 specialisation is
entirely a dtype change.

---

## 4. `fft_universal_xl` ("Option B")

This is the only backend where a sizeable arithmetic step **runs on
host every call**.

### On host
- **XL plan factorization** N = F₁·F₂. Cached per N.
- **Outer twiddle table.** `OuterTwiddle` cache:
  `w[n₁·F₂ + k₂] = exp(-2πi · n₁ · k₂ / N)` is built **once per (N, F₁)
  in `std::complex<float>`** and reused on every subsequent call with
  the same shape. (No host arithmetic on the per-call hot path.)
- **Intermediate transpose** F₁×F₂ ↔ F₂×F₁ at the end of the four-step
  (same pure host buffer shuffle as `fft_stockham`'s Bailey-2 case).

### On device
- Both passes of length-F₂ FFTs (F₁ rows) and length-F₁ FFTs (F₂
  columns) dispatch through `fft_stockham`.
- The **outer twiddle multiply** is dispatched as device ops on the
  intermediate matrix — but the twiddle **values** themselves came
  from the host (Option B), as opposed to Option A which would build
  them on-tile.

### Paper-relevant fact

Option B trades **one extra host upload per N (cached)** for the
ability to support N > 2²⁰. There is **no per-call host arithmetic** in
the XL path beyond the universal read/write. So XL is still legitimate
to call "device-resident on the steady-state hot path", as long as the
description includes the one-shot twiddle upload at first call.

---

## Side-by-side summary

| Backend             | Per-call host arithmetic (excluding universal read/write/conj-trick) |
|---------------------|----------------------------------------------------------------------|
| `fft_stockham`      | N ≤ 1024: none. N > 1024: 1× transpose buffer shuffle.               |
| `fft_universal`     | Bluestein path: O(M) padding + O(N) slicing buffer shuffle.          |
| `fft_universal_bf16`| same as `fft_universal`                                              |
| `fft_universal_xl`  | 1× transpose buffer shuffle                                          |

> Everything labelled "first call only" or "cached per N" disappears
> as soon as you measure steady-state with program-cache enabled,
> which is exactly what `bench_latency.py` does (5 warmup calls,
> median of 50 measured calls).
