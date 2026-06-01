# Abstract

We present `fft_universal`, the first one-dimensional FFT implementation
on Tenstorrent's Wormhole accelerator that accepts arbitrary transform
lengths $N \ge 2$, not just powers of two. The implementation is built on
Tenstorrent's open-source `tt-metalium` runtime and pairs four
device-side kernels — small-$N$ packed direct DFT, in-tile pow-2
Stockham, multi-pass Stockham, and a batched complex multiplier — with
a thin host planner that splits any $N$ into one of four dispatch
paths: packed DFT for $N \le 32$, Stockham for power-of-two $N \le
2^{20}$, batched Bluestein chirp-$z$ for primes, and batched
Cooley–Tukey for composite non-powers-of-two. We show that adding the
packed direct-DFT kernel raises tile occupancy on the small-$N$ legs of
prime and composite transforms from under 3 % to between 6 % and 100 %,
which removes the dominant PCIe-bound cost in Bluestein and mixed-radix
recursion. We measure end-to-end host-to-device-to-host latency across
$N \in [2, 2^{20}]$ on a single Wormhole n300 card and report a host
overhead column for every measurement, giving the first published apples-
to-apples comparison of a non-power-of-two FFT on Tenstorrent against
the recent single-Tensix radix-2 result of Brown et al. [1].
