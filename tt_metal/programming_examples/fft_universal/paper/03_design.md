# 3. Design: a planner–composer for any $N$

## 3.1 Why a single algorithm does not suffice

A one-size-fits-all FFT for an accelerator like Wormhole runs into three
mutually exclusive design pressures:

* **Tile efficiency.** The Tensix FPU is a $32 \times 32$ matrix engine.
  An FFT kernel that loads one length-$N$ signal into a $32 \times N/32$
  layout wastes $1 - N/(32 \cdot 32) = 1 - N/1024$ of the tile for any
  $N < 1024$. A length-$8$ FFT, run as a tiny pow-2 Stockham, occupies
  $8/1024 \approx 0.8\,\%$ of every tile it touches — and Bluestein on a
  prime $N = 7$ would internally need a length-$16$ Stockham, still at
  $\approx 1.6\,\%$.
* **PCIe efficiency.** Every device dispatch costs the same fixed
  PCIe-write + dispatch + PCIe-read tax (~10–30 µs on n300, dominated by
  `EnqueueMeshWorkload` and the blocking readback). An FFT algorithm that
  emits one dispatch per stage at $\log_2 N$ stages is bandwidth-bound by
  dispatch at small $N$ and bandwidth-free at large $N$.
* **Algorithmic coverage.** A radix-2 Cooley–Tukey only handles
  $N = 2^k$. A pure Bluestein handles any $N$ but spends $> 4N$ extra
  complex multiplies and two pow-2 FFTs of length $M = 2^{\lceil \log_2(2N-1) \rceil}$,
  which is wasteful when $N$ is already pow-2 or factors nicely.

We resolve these by **planning per $N$ and composing four device kernels**.

## 3.2 The four device kernels

| Kernel | Source | Job |
|--------|--------|-----|
| **Stockham single-tile** | `fft/` | One length-$N$ pow-2 FFT, $N \le 1024$, resident in one tile. Used as the leaf of every pow-2 path. |
| **Stockham multi-pass** | `fft_stockham/` | Pow-2 $N$ up to $2^{20}$ via four- and six-step Stockham orchestrated across all 64 cores. |
| **Packed direct DFT** (new) | `fft_universal/kernel/` | 32 sub-FFTs of length $\le 32$ per tile, one dispatch, computed as a complex $32 \times 32$ matmul. |
| **Batched complex multiply** | `fft_universal/` | Pointwise complex multiply used by Bluestein and by Cooley–Tukey's twiddle pass; per-row matmul, all 64 cores. |

The first two were contributed by Brown et al. and the broader Tenstorrent
community; the second two are new in this work.

## 3.3 The dispatch decision tree

For a given $N$, the host planner picks exactly **one** of four paths.
Figure 1 in the source repository (`flowchart.txt`) shows the decision
graphically; the rules are:

```
if N == 1:                    return x                        (identity)
elif 2 <= N <= 32:            packed direct DFT               (Path 0)
elif N is a power of two:     Stockham                        (Path A)
elif N is prime:              Bluestein (chirp-z)             (Path C)
else:                         Cooley–Tukey split  N = N1·N2   (Path D)
```

Path C and Path D both recurse: the inner length-$M$ FFTs in Bluestein
and the inner $N_1$ / $N_2$ sub-FFTs in Cooley–Tukey all re-enter the
top of the table. The recursion bottoms out either at a power of two
(handed off to Path A) or at $N \le 32$ (handed off to Path 0). This
gives the engine its key efficiency property: **every leaf is a single
device dispatch**.

## 3.4 Why this is a "planner–composer"

The arrangement is the same one used by FFTW [3]'s `planner` /
`executor` split and by cuFFT [2]'s `cufftPlan*` / `cufftExec*` split:
a per-$N$ plan object is built once and cached, holding all the data
that does not depend on the input values (twiddle tables, Bluestein
chirps, Cooley–Tukey split factors, JIT'd device kernels). Subsequent
calls for the same $N$ traverse the cached plan and do only:

1. tiny host glue (copy input into the plan's pinned scratch, mark
   indices in the per-pass transpose, advance recursion);
2. one or more `EnqueueMeshWorkload` calls into the cached programs;
3. a blocking read back into the plan's output scratch.

The host glue is $O(N)$ per call — strictly linear in problem size — and
amortised against the device compute, which is $O(N \log N)$. Asymptotically
the host fraction goes to zero with $N$; §5.3 measures the crossover.

## 3.5 Batched recursion

A subtle but important property of the design: when Path D splits
$N = N_1 \cdot N_2$, the resulting pass-1 must compute $N_1$
independent length-$N_2$ sub-FFTs and pass-2 must compute $N_2$
independent length-$N_1$ sub-FFTs. The engine carries a `count`
parameter through every recursion level: at the top of the tree
`count = 1`; after one split it becomes `count · N1` for pass-1 and
`count · N2` for pass-2; one more split squares it again.

This means **batching width grows multiplicatively with recursion
depth**. By the second or third level of CT split, `count` is large
enough that a single `batch_fft` dispatch saturates all 64 cores even
for very small sub-FFTs (which would otherwise be wasteful). The
packed DFT path (§4.2) is the asymptote of this idea: at length-32
leaves it always batches at least 32 sub-FFTs per tile.

## 3.6 Inverse FFT

We compute the IDFT through the conjugation trick:
$\mathrm{IFFT}(X) = \overline{\mathrm{FFT}(\overline{X})} / N$. This
costs one extra pass of pointwise conjugation (host, $O(N)$) and one
pass of pointwise division (host, $O(N)$) over a single forward call.
We measure the round-trip error in §5.4. All forward optimisations
(packed DFT, Bluestein, Cooley–Tukey) transparently apply to the
inverse without code duplication.
