# Where the math actually runs — host vs device, with equations

This is the **math-explicit** companion to `HOST_VS_DEVICE.md`.
`HOST_VS_DEVICE.md` answers "which file / which line"; this document
answers **"which formula, with what symbols, evaluated where"**.

Every equation below is taken directly from the source on
`pkorhale/experimental-fft` — the file & line is cited in each
section so the paper can quote it without going stale.

Conventions used throughout:

- \(x \in \mathbb{C}^N\)  = forward-FFT input
- \(X \in \mathbb{C}^N\)  = forward-FFT output, \(X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-2\pi i n k / N}\)
- All "host" code runs in `std::complex<float>` (or `double` for trig
  arguments) inside the program-factory / `*_host.hpp` orchestrators.
- All "device" code runs on Tensix cores via Stockham radix-2
  butterflies (SFPU "precise" path) or FPU bf16-mantissa matmuls
  ("fast" path).

---

## 0. Universal pre/post that every call pays

Source: `device/fft_program_factory.cpp`, lines 110–155.

These steps run on the host once per call, regardless of which
backend the dispatcher picks. They are **the only host arithmetic**
on the forward path (everything else for "Stockham" and "XL" is
memcpy / tiling shuffles).

### Host: input materialisation
*Math:* none — it's a memcpy + dtype cast from DRAM into a host
`std::vector<float>` of length \(B \cdot N\):

```
in_re = read_real_as_fp32(input_real)        // host
in_im = read_real_as_fp32(input_imag) or 0   // host
```

### Host: IFFT conjugate trick (only on the inverse path)
*Math:* equivalent to the identity
\[
\text{ifft}(X) \;=\; \overline{\text{fft}(\overline{X})} \big/ N
\]

implemented as:

```
// host pre-multiply
work[n] = Re(in_re[n]) + i · (−in_im[n])         for n = 0..N−1

// device runs forward fft into work[]

// host post-multiply
out[k] = (1 / N) · ( Re(work[k]) + i · (−Im(work[k])) )
```

This is **2N real flops** on the host: one negate-imag before, one
negate-imag + scalar-multiply after.

### Host: output write-back
*Math:* none — memcpy + cast from host `std::vector<float>` back into
the output device tensor.

---

## 1. `fft_stockham`  (fp32 power-of-two, N ≤ 1M)

Source: `device/stockham_host.hpp`.

### Host math at plan time (cached per N)

#### 1a. Per-pass twiddle tables (Bailey 4-step inner)
Source: `stockham_host.hpp:443–465`.

For the four-step decomposition of an \(N = N_1 \cdot N_2\) FFT, the
"middle" twiddle multiplier is

\[
T[n_1, j] \;=\; e^{-2\pi i \, n_1 j / N},
\qquad n_1 \in [0, N_1),\; j \in [0, N_2).
\]

Built **on host in `double`** then rounded to `float` at upload:

```cpp
// stockham_host.hpp:451–463
tau_over_N = -2π / N;
for row in [0, N1):
    for j in [0, N2):
        angle      = tau_over_N · row · j;
        tile_r[j]  = cos(angle);
        tile_i[j]  = sin(angle);
```

Cost: \(\mathcal{O}(N_1 \cdot N_2) = \mathcal{O}(N)\) trig evaluations
**once per N**, then reused on every subsequent call with the same
shape (plan cache).

### Host math at call time

#### 1b. B(N₁, N₂) → C(N₂, N₁) transpose
Source: `stockham_host.hpp:648–657`.

After the column FFTs run on device, the orchestrator does a pure
index permutation on host:

\[
C[j, i] \;=\; B[i, j],
\qquad i \in [0, N_1),\; j \in [0, N_2).
\]

```cpp
for i in [0, N1):
    for j in [0, N2):
        C[j*N1 + i] = B[i*kTileElems + j];
```

**No arithmetic, just a memory shuffle.** For \(N \le 1024\) (tile-fits)
this step is skipped entirely.

### Device math

- Single-tile case (\(N \le 1024\)): one Stockham kernel performs
  \(\log_2 N\) radix-2 passes with the on-tile per-pass twiddles
  \(w_p[j] = e^{-2\pi i j / 2^p}\), \(p = 1..\log_2 N\).
- Multi-tile case: the same Stockham kernel runs once for each row of
  the N₁ × N₂ matrix, then the host transpose above, then once per
  column.

---

## 2. `fft_universal`  (fp32 non-pow2)

Source: `device/universal_host.hpp`.

This is the backend with the most interesting host math because of
**Bluestein's chirp-z** path for prime / large-prime-factor N.

### 2.1 Composite N (mixed-radix Cooley-Tukey)

#### Host math at plan time
Source: `universal_host.hpp:131–170` (`get_ct_plan`).

For \(N = N_1 \cdot N_2\) with \(\gcd(N_1, N_2) > 1\) trivially handled,
the standard Cooley-Tukey decomposition needs the **"twist" twiddle**

\[
W[n_1, k_2] \;=\; e^{-2\pi i \, n_1 k_2 / N}.
\]

Built on host in `double` once per (N₁, N₂):

```cpp
// universal_host.hpp:153–164
tau_over_N = -2π / N1 / N2;
for n1 in [0, N1):
    for k2 in [0, N2):
        a            = tau_over_N · n1 · k2;
        twiddle[n1*N2 + k2] = {cos(a), sin(a)};
```

#### Host math at call time
None beyond §0. The mixed-radix sub-passes (Stockham radix-2 + small
non-pow2 radices ≤ 32 in the packed kernel) all run on device.

### 2.2 Prime N (Bluestein chirp-z)

Source: `universal_host.hpp:79–129` (`BluesteinPlan`, `get_bluestein_plan`).

This is the most math-heavy host path in the entire op.

#### Setup
Let \(M = \text{next\_pow2}(2N - 1)\). The Bluestein identity is

\[
X[k] \;=\; \overline{w}[k] \;\cdot\; \bigl( a \;*\; b \bigr)[k],
\qquad
w[n] \;=\; e^{-i\pi n^2 / N},
\]

with

\[
a[n] \;=\; x[n] \cdot w[n],
\qquad
b[n] \;=\; \overline{w}[n].
\]

The convolution \(a * b\) of length \(N\) is computed as the
length-\(M\) **pointwise product** of two length-\(M\) FFTs.

#### Host math at plan time (cached per N)

**Chirp table** \(w[n] = e^{-i \pi n^2 / N}\), with the squared-index
reduction \(n^2 \bmod 2N\) for trig-argument stability:

```cpp
// universal_host.hpp:104–114
pi_over_N = π / N;
for n in [0, N):
    nn               = (uint64_t)n · (uint64_t)n;
    a                = pi_over_N · (nn mod 2N);
    chirp_fwd[n]     = {cos(a), -sin(a)};       // = exp(-i π n² / N)
```

**Padded reference sequence** \(b_{\text{ext}} \in \mathbb{C}^M\):

\[
b_{\text{ext}}[n] \;=\; \begin{cases}
1 & n = 0 \\
\overline{w}[n] = e^{+i\pi n^2/N} & 1 \le n < N \\
0 & N \le n \le M - N \\
\overline{w}[M-n] & M - N < n < M
\end{cases}
\]

```cpp
// universal_host.hpp:116–122
b_ext.assign(M, {0, 0});
b_ext[0] = {1, 0};
for n in [1, N):
    g            = conj(chirp_fwd[n]);
    b_ext[n]     = g;
    b_ext[M - n] = g;       // make b symmetric so its FFT is the reference spectrum
```

**Reference spectrum** \(B = \mathrm{FFT}_M(b_{\text{ext}})\):

```cpp
// universal_host.hpp:125
plan->B_fft = fft_stockham::fft(md, b_ext);   // <-- this one is on DEVICE
```

So plan-build:
- chirp table: \(\mathcal{O}(N)\) trig — **host**
- \(b_{\text{ext}}\) packing: \(\mathcal{O}(N)\) complex assignments — **host**
- \(B = \mathrm{FFT}_M(b_{\text{ext}})\): \(\mathcal{O}(M \log M)\) — **device**

Done **once per N**, cached.

#### Host math at call time

Per-call work:

1. **Chirp pre-multiply** \(a[n] = x[n] \cdot w[n]\), zero-padded to M.
   Host computes this on the input vector (N complex mults).
2. **Padding**: write \(a\) into a length-\(M\) buffer of zeros.
3. **(Device)** \(A = \mathrm{FFT}_M(a)\).
4. **(Device)** pointwise \(C[k] = A[k] \cdot B[k]\) for k ∈ [0, M).
5. **(Device)** \(c = \mathrm{IFFT}_M(C)\).
6. **Slice** length-\(M\) buffer to first \(N\) samples on host.
7. **Chirp post-multiply** \(X[k] = c[k] \cdot w[k]\) (N complex mults).

Per-call host arithmetic: **\(2N\) complex multiplies** (steps 1 and 7) **+
\(\mathcal{O}(M)\) memcpy** (steps 2 and 6).

---

## 3. `fft_universal_bf16`  (bf16 any N)

Source: `device/universal_bf16_host.hpp`.

Math is **identical to `fft_universal`** above. The only differences
are bit-width plumbing:

- All host `std::complex<float>` tables are downcast to bf16 at
  `WriteShard` time.
- The on-device compute kernel uses the **bf16 FPU matmul** path
  (`device/kernels/compute/packed_dft_bf16_compute.cpp` and friends)
  rather than the SFPU radix-2 path.

So the equations in §2 apply unchanged; the precision of every term
drops from `float32` to `bf16` (≈ 7 bits of mantissa).

A second 32×32 packed twiddle table for the small-N kernel is built
on host as

```
// universal_bf16_host.hpp:140
T[n, k] = exp(-2πi · k · n / N),      n, k ∈ [0, 32)
```

i.e. the standard DFT matrix.

---

## 4. `fft_universal_xl`  (fp32 power-of-two, 1M < N ≤ 16M/64M)

Source: `device/universal_xl_host.hpp`.

This is the only backend where a non-trivial twiddle table is
**physically built on host and uploaded as a device tensor** (Option B
of the file-header comment).

### Host math at plan time (cached per (N, F₁))

#### 4a. Outer twiddle table
Source: `universal_xl_host.hpp:55–83`.

Pick a factorisation \(N = F_1 \cdot M\) (\(F_1\) is the smallest
factor in the plan). The outer twiddle is

\[
W[n_1, k] \;=\; e^{-2\pi i \, n_1 k / N},
\qquad n_1 \in [0, F_1),\; k \in [0, M).
\]

Built **on host in `double`**, packed into a length-\(F_1 \cdot M\)
buffer, and uploaded once per `(N, F1)` pair:

```cpp
// universal_xl_host.hpp:67–78
two_pi_over_N = -2π / N;
for n1 in [0, F1):
    for k in [0, M):
        ang   = two_pi_over_N · n1 · k;
        w[n1*M + k] = {cos(ang), sin(ang)};
```

Cost: \(\mathcal{O}(F_1 \cdot M) = \mathcal{O}(N)\) trig calls — **once
per (N, F₁)**, cached across calls.

### Host math at call time

#### 4b. Bailey 4-step transpose
Same pure index permutation as §1b. **No arithmetic.**

### Device math

- Step 1: \(F_1\) length-\(M\) FFTs via `fft_stockham` (one per row).
- Step 2: pointwise multiply with the host-built outer twiddle
  \(W[n_1, k]\) — **device** evaluates the multiply, but the **values
  came from host arithmetic** above.
- Step 3: \(M\) length-\(F_1\) FFTs via `fft_stockham` (one per column).
- Step 4: final transpose (host index shuffle).

### N envelope, updated for both Wormhole and Blackhole

Source: `universal_xl_host.hpp:85–95`.

\[
\text{max-N}_{\text{XL}} \;=\;
\begin{cases}
2^{24} = 16\,777\,216 & \text{Wormhole} \\
2^{26} = 67\,108\,864 & \text{Blackhole}
\end{cases}
\]

(So the headline number from the previous answer was the Wormhole one;
Blackhole gets 4× thanks to ~2× DRAM BW and ~2× cores.)

---

## 5. IFFT (`ttnn.experimental.ifft`)

The conjugate-trick rewrite in §0 means **no backend needs a separate
inverse kernel**. The host arithmetic per IFFT call is:

- 1 pre-multiply by \((-1)\) on the imag part: \(N\) negations
- 1 post-multiply by \(1/N\) and 1 negation on the imag part:
  \(N\) multiplies + \(N\) negations

Total **\(3N\) extra host real flops** on top of the forward path.

---

## Compact "where does which formula live" table

| Formula                                              | Where evaluated     | Cost (per call)      | Cached? |
|------------------------------------------------------|---------------------|----------------------|---------|
| \(T[n_1, j] = e^{-2\pi i n_1 j / N}\)                | host                | \(\mathcal{O}(N)\) trig | yes, per N |
| Stockham radix-2 butterflies on tile                 | device              | \(\mathcal{O}(N \log N)\) flops | n/a |
| \(C[j, i] = B[i, j]\) transpose                      | host                | \(\mathcal{O}(N)\) memcpy | n/a |
| Cooley-Tukey twist \(W[n_1, k_2] = e^{-2\pi i n_1 k_2 / N}\) | host        | \(\mathcal{O}(N)\) trig | yes, per (N₁, N₂) |
| Bluestein chirp \(w[n] = e^{-i\pi n^2 / N}\)         | host                | \(\mathcal{O}(N)\) trig | yes, per N |
| Bluestein \(B = \mathrm{FFT}_M(b_{\text{ext}})\)     | device              | \(\mathcal{O}(M \log M)\) flops | yes, per N |
| Bluestein pre-mul \(a[n] = x[n] w[n]\)               | host                | \(N\) complex mults | n/a |
| Bluestein post-mul \(X[k] = c[k] w[k]\)              | host                | \(N\) complex mults | n/a |
| Pointwise \(C = A \cdot B\) inside Bluestein         | device              | \(M\) complex mults | n/a |
| XL outer twiddle \(W[n_1, k] = e^{-2\pi i n_1 k / N}\) | host              | \(\mathcal{O}(N)\) trig | yes, per (N, F₁) |
| Outer twiddle multiply against intermediate matrix   | device              | \(N\) complex mults | n/a |
| IFFT pre-conj                                        | host                | \(N\) negations | n/a |
| IFFT post-conj + 1/N scale                           | host                | \(N\) mults + \(N\) negations | n/a |

The take-away for the paper's "Implementation" section:

> **All inner-loop arithmetic is on device.** The host's responsibilities
> reduce to (a) building twiddle tables in `double` once per shape,
> (b) the universal data round-trip required by today's
> `fft_program_factory.cpp`, and (c) the \(2N\) chirp multiplies on the
> Bluestein path. Eliminating (b) and pushing (c) into a device-side
> prologue/epilogue kernel are the two known follow-ups documented in
> `RECOMMENDED_EXTRAS.md`.
