# Algorithms in `ttnn.experimental.fft`

This document describes every backend that the dispatcher in
`fft_device_operation.cpp::select_backend()` can route to, the math it
implements, and the **practical N range** it covers on Wormhole.

```
select_backend(dtype, N):
    BFLOAT16, any N            → UniversalBf16     (fft_universal_bf16)
    FLOAT32,  N is non-pow2    → Universal         (fft_universal)
    FLOAT32,  pow2, N ≤ 1M     → Stockham          (fft_stockham)
    FLOAT32,  pow2, N ≤ 16M    → UniversalXL       (fft_universal_xl)
    FLOAT32,  pow2, N > 16M    → throws (K=4 dispatcher not yet shipped)
```

The forward and inverse transforms share the same code path; IFFT is
realised by **conjugate-trick** at the program-factory level
(`fft_program_factory.cpp`):

```
ifft(X) = conj( fft( conj(X) ) ) / N
```

so every algorithm below applies to both directions; the host adds the
`1/N` scale and pre/post conjugation on the IFFT path.

---

## 1. `fft_stockham` (Stockham six-step / Bailey 4-step)

**File:** `device/stockham_host.hpp`
**Used for:** fp32, power-of-two N, **N ≤ 2²⁰ = 1 048 576**
**Precision:** "precise" (default) → SFPU radix-2; "fast" → FPU bf16-mantissa matmul

### Math

Stockham auto-sort FFT: at every pass the algorithm reads from a source
buffer and writes to a destination buffer, swapping the role each pass.
This eliminates the bit-reversal permutation of in-place Cooley-Tukey.

For N = 2ᵏ ≤ 1024 the whole FFT fits inside one Tensix tile (1024
complex elements per tile) and runs as a **single-shot** kernel.

For larger N (up to 2²⁰) the orchestrator decomposes N = N₁·N₂ (Bailey
four-step) and stages it as

1. column FFTs of length N₁,
2. apply outer twiddle wⁿ¹ᵏ²/ᴺ on tile,
3. row FFTs of length N₂,
4. transpose B(N₁,N₂) → C(N₂,N₁).

All four sub-passes run on device; the **transpose at the end of the
factorization** is the only host-touchable buffer copy (see
`HOST_VS_DEVICE.md` for the exact accounting).

### N envelope on Wormhole

| Tile-fits   | 2 ≤ N ≤ 1 024            | 1 kernel dispatch  |
| Bailey-2    | 2 048 ≤ N ≤ 1 048 576    | O(log₂N) dispatches |

Beyond 2²⁰ the **outer twiddle table** would not fit in L1; that is the
hand-off point to `universal_xl`.

---

## 2. `fft_universal` (mixed-radix + Bluestein)

**File:** `device/universal_host.hpp`
**Used for:** fp32, **non-pow2** N (any), including primes
**Precision:** same `precise` / `fast` knob as Stockham

### Math

For composite non-pow2 N the orchestrator factors

    N = p₁ · p₂ · … · pₖ

into small radices (typically ≤ 32) and runs a mixed-radix
Cooley-Tukey, reusing the Stockham kernel for the pow2 sub-passes.

For **prime** N (or any N whose factorization contains a large prime),
the orchestrator falls back to **Bluestein's algorithm** (chirp-z):

    X[k] = w^{-k²/2} · Σₙ ( x[n]·w^{-n²/2} ) · w^{(k-n)²/2}      (1)

which it rewrites as a convolution and evaluates with two length-M
FFTs, where **M = next_pow2(2N - 1)**.

Because Bluestein needs M to be a Stockham-acceptable pow2, the
constant `kStockhamMaxPow2 = 2²⁰` in `universal_host.hpp` caps **prime N
at N ≤ 524 288**. (For N above this cap on a non-pow2 path you would
add a host-glue chirp-z that dispatches the two FFTs through
`universal_xl` — see `RECOMMENDED_EXTRAS.md`.)

### N envelope on Wormhole

| Small N (≤32)             | packed batched kernel (1 dispatch)    |
| 33 ≤ composite N ≤ 2²⁰    | mixed-radix Cooley-Tukey               |
| 2 ≤ prime N ≤ 524 288     | Bluestein with M = next_pow2(2N-1)     |
| prime N > 524 288         | not currently supported                |

---

## 3. `fft_universal_bf16` (true-bf16 variant)

**File:** `device/universal_bf16_host.hpp`
**Used for:** **all bf16 inputs**, any N
**Precision:** there is no "precise" mode here — bf16 has no SFPU path,
so the FPU matmul kernel is always used

This is the bf16-only specialisation of `fft_universal`. It uses the
same orchestrator structure (small-N packed → mixed-radix → Bluestein)
but with all CB types set to `DataFormat::Float16_b` and the FPU matmul
compute kernel that uses bf16-mantissa multiplications natively.

### N envelope on Wormhole

Same shape as `fft_universal`. Throughput is typically **higher than
fp32-precise** because the FPU matmul kernel has higher arithmetic
intensity, at the cost of ~6 fewer bits of mantissa precision (~1e-2
to 1e-3 round-trip error depending on N).

---

## 4. `fft_universal_xl` (Option B: host outer twiddle)

**File:** `device/universal_xl_host.hpp`, `device/universal_xl_planner.hpp`
**Used for:** fp32, power-of-two N, **2²⁰ < N ≤ 2²⁴ = 16 777 216**
**Precision:** uses the same compute kernels as Stockham

### Math

For very large pow2 N where the outer twiddle table can no longer fit
in L1, the orchestrator picks a 2-factor split

    N = F₁ · F₂,   F₁ chosen by `pick_outer_factor()` as the smallest factor

and runs a 4-step Bailey:

1. F₁ row FFTs of length F₂ on device (Stockham backend),
2. outer twiddle applied **on host** to the F₁·F₂ matrix
   (`OuterTwiddle` cache: w[n₁·F₂ + k₂] = exp(−2πi n₁ k₂ / N)),
3. F₂ column FFTs of length F₁ on device,
4. final transpose F₂ × F₁ → N.

"Option B" is documented in the file header — it explicitly trades
device twiddle storage for one extra host pass. This trade is what
unlocks N > 2²⁰ on the current Wormhole hardware budget.

### N envelope on Wormhole

| 2²¹ ≤ N ≤ 2²⁴ (16 M)  | validated and shipping  |
| 2²⁴ < N ≤ 2²⁸          | not yet — needs the K=4 dispatcher and packed batch_fft_xl kernel; throws today |

---

## 5. IFFT (`ttnn.experimental.ifft`)

There is no separate IFFT backend. The same dispatcher is used; the
program factory applies `conj(input)` before dispatch and `conj(output)
/ N` after — pure host O(N) work that is invisible to all the
algorithm-level code.

This is why every backend's IFFT cost is the forward cost plus one
O(N) host pass.

---

## Summary table

| Backend             | dtype   | N constraint                  | Practical N                           | Host work per call                                                                  |
|---------------------|---------|-------------------------------|---------------------------------------|-------------------------------------------------------------------------------------|
| `fft_stockham`      | fp32    | pow2                          | 2 … 2²⁰ (1 M)                         | tensor read/write, plan setup (cached)                                              |
| `fft_universal`     | fp32    | non-pow2                      | composite ≤ 2²⁰, prime ≤ 524 288      | tensor read/write, factorization, Bluestein chirp tables (cached), padding/slicing |
| `fft_universal_bf16`| bf16    | any                           | composite ≤ 2²⁰, prime ≤ 524 288      | same as `fft_universal`                                                             |
| `fft_universal_xl`  | fp32    | pow2                          | 2²¹ … 2²⁴ (16 M)                      | tensor read/write, plan factorization (cached), **outer twiddle table on host**     |
| (gap, not shipped)  | fp32    | pow2                          | > 2²⁴                                 | needs K=4 dispatcher + packed batch_fft_xl                                          |
