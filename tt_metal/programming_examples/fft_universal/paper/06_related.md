# 6. Related work

**FFTs on Tenstorrent.** Brown, Davies, and Le Clair [1] published the
first 1-D FFT result on Wormhole in 2025: a single-Tensix radix-2
decimation-in-time at $N = 16\,384$ with a 2.8× advantage over a
single-thread fp32 CPU baseline. Their work is the foundation of
ours; the Stockham kernels we delegate to in §4.1 derive from the
same codebase. We extend their result in two orthogonal directions
(arbitrary $N$, all 64 cores) and add the host/device split that
their evaluation does not measure.

**FFTs on accelerators in general.** Modern GPU FFT libraries
(NVIDIA's cuFFT [2], AMD's rocFFT, Intel's oneMKL FFT) all adopt the
**planner–composer** pattern we use: per-$N$ plan objects are built
once and reused, internally selecting between a fixed set of
algorithms (small-$N$ direct DFT, pow-2 Stockham/Cooley–Tukey, mixed-
radix, Bluestein) based on factorisation of $N$. FFTW [3] is the
canonical academic instantiation of the same pattern on CPUs and
introduced the formal separation of *planning* from *executing*.
`fft_universal` is to Tenstorrent what cuFFT is to NVIDIA: a single
arbitrary-$N$ entry point backed by a small set of tightly optimised
kernels chosen at planning time.

**Bluestein and the chirp-z transform.** Bluestein's [5] original
algorithm and Rabiner & Schafer's chirp-z formulation [6] are the
standard tools for prime-length and arbitrary-length DFTs. Our
contribution is not to the algorithm but to its efficient mapping
onto Wormhole: fusing all sibling Bluestein calls into one
$M$-batched device dispatch and caching the constant spectrum.

**Small-$N$ direct DFT as a matmul.** Treating an $N \times N$
twiddle table as a constant matrix and computing the small-$N$ DFT
as a matmul is folklore; we are unaware of a prior publication that
applies the trick to a tile-based accelerator with 32-wide packing
of sibling DFTs per tile, as we do in §4.2.
