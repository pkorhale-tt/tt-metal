# Option C — K-step on-device Stockham kernel (design)

## Goal

Where Option A only fixes the twiddle-multiply tile cap, **Option C
collapses the entire K-step FFT into ONE device program**. No host
arithmetic, no host data shuffling, no PCIe round-trips between
passes.

## Why this matters at huge N

Today the K=3 dispatcher (Option B) does:

```
  host buffer ─┐
               ├─→ DRAM ─→ batch_fft (Pass 1) ─→ DRAM ─→ host
  host twiddle ─→ DRAM ─→ pass2_xl   (Pass 2) ─→ DRAM ─→ host  (Option A)
  host transpose ─→ DRAM ─→ batch_fft (Pass 3) ─→ DRAM ─→ host
```

Three PCIe round-trips per FFT. At N=1G that's 8 GB shuffled across
PCIe per call — the dominant cost.

Option C keeps everything DRAM-resident:

```
  host buffer ─→ DRAM ─→ kstep_kernel (Pass 1 + twid + Pass 2 + twid + Pass 3) ─→ DRAM ─→ host
```

ONE round-trip. Expected speed-up at N=1G: **5-10x** vs Option A,
because Option A still has 3 separate dispatches.

## Architecture

A single device program with **three orchestrated phases** running
across all 64 Tensix cores:

```
Phase 1: pass-1 sub-FFTs of length F1
  Each core handles a slab of (M*F3 / 64) sub-FFTs.
  Output written to DRAM intermediate buffer.

Phase 2: outer twiddle multiply over (F1*F2*F3) tiles
  Each core handles 1/64 of the tiles.
  Input AND output stay in DRAM (intermediate -> intermediate-2).

Phase 3: pass-2 sub-FFTs of length F2  (analogous)
Phase 4: middle twiddle
Phase 5: pass-3 sub-FFTs of length F3
Final reorder via NoC gathers when reading back to host.
```

## Sync points

Between phases, all 64 cores must agree the previous phase finished.
Two options:

* **A. Per-phase dispatch** — separate `EnqueueProgram` per phase, with
  `Finish` between. Cleanest; loses the "single dispatch" benefit but
  retains DRAM residency. **This is what I'd recommend as the first
  cut** — it's structurally simpler and gets 80% of the win.
* **B. In-program semaphores** — use `noc_semaphore_set/wait` to gate
  phases inside a single program. Faster (no dispatch overhead) but
  requires careful core-level orchestration code in EVERY kernel.

## DRAM layout

Two intermediate buffers (ping-pong), each sized `N * 8 bytes` (real +
imag fp32). At N=1G that's 16 GB per buffer x 2 = 32 GB. **This won't
fit in current Wormhole DRAM (typically 12 GB)**, which means Option C
in its naive form is gated at N <= ~750M for fp32.

For larger N you'd need:

* bf16 intermediates (cuts buffer size in half, fits up to 1.5G)
* OR streaming the intermediate through smaller chunks (much harder)

## Why this isn't a 3-day job

A K-step kernel is essentially a small DSL: phase descriptors, NoC
sync, transpose-in-place via NoC reads, twiddle table layout per
phase, multi-core load balancing for irregular factor sizes
(F3 != F1 != F2 in the general case). Each piece is a real chunk of
device kernel work.

Honest scope: **2-3 weeks for a working single-precision K=3 kernel**,
plus another 1-2 weeks for bf16 support if you want it. Test/regression
infrastructure on top.

## Recommended sequencing

1. Land Option B (this PR — done) so big-N works correctly today.
2. Land Option A (1 week) so big-N works without host arithmetic.
3. Profile B+A at N=8M, 64M, 256M. If PCIe is < 30% of total time,
   skip C. If it's > 50%, build C.
4. C is only worth it if your real workload runs many FFTs back-to-back
   at very large N. For one-shot calls, A is sufficient.

## Open questions before starting C

* Single fp32 or also bf16 intermediates? bf16 cuts memory but loses
  ~1 bit of precision per phase boundary.
* Single program with semaphores or per-phase dispatch? Recommend
  per-phase first; revisit only if dispatch overhead is measurable.
* Reorder on host or via NoC at readback? Host is trivial; NoC is
  faster but adds kernel complexity. Recommend host first.

When you decide to build it, ping me and I'll spec out the kernel
ABI per phase.
