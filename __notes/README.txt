================================================================================
  NOTES FOLDER — TT-METAL CONV_TRANSPOSE2D STUDY
================================================================================

This folder collects deep-dive notes about ttnn.conv_transpose2d (especially
the iSTFT use case) on Tenstorrent Wormhole hardware.

  01_conv_transpose2d_istft_envelope.txt
      Full end-to-end walkthrough of the test
        pytest.param(1, 1, 640, 160, 96, False, id="istft_envelope")
      Includes:
        - Parameter mapping
        - PyTorch reference math
        - Internal ttnn pipeline (auto-shard -> halo -> conv2d)
        - How H, W, full_input_W are derived
        - Auto-sharder picking 62 cores
        - 8x8 core-grid ASCII diagram
        - Per-core data BEFORE halo, AFTER halo
        - Per-core compute loop
        - End-to-end data-flow diagram
        - Why the case currently fails (NoC burst limit)

  09_polyphase_implementation_guide.txt
      Concrete patch instructions for adding the polyphase fast path
      to ttnn.conv_transpose2d. For an engineer who is going to
      actually write the code.
      Includes:
        - Chapter 1: Inventory (3 NEW files, 5 MODIFIED files,
                  zero touched in conv2d/sliding_window/halo)
        - Chapter 2: Add the POLYPHASE enum value (smallest patch)
        - Chapter 3: Implement detector + weight shuffle (with C++ code)
        - Chapter 4: The new conv_transpose2d_POLYPHASE function
                  (full skeleton: geometry, weight shuffle, pad,
                  S sub-convs, interleave, bias)
        - Chapter 5: Wire the dispatcher (early-return if-clause
                  in conv_transpose2d.cpp lines 1237-1294)
        - Chapter 6: CMakeLists.txt diff
        - Chapter 7: Tests (a new pytest file + a regression check)
        - Chapter 8: Day-by-day incremental milestones (8 weeks)
        - Chapter 9: Gotchas you will hit (pad/stack/bias placement,
                  tile alignment, mirror_kernel ordering)
        - Chapter 10: Rollout strategy (shadow -> opt-in -> default-on)
        - Chapter 11: Code-review checklist (15 items)
        - Chapter 12: Summary table of changes (~360 lines total)

  08_polyphase_decomposition_deep_dive.txt
      The single highest-leverage optimization for conv_transpose2d
      explained in depth: math derivation, hand-traced tiny example,
      what code it replaces, what code it keeps, and an 8-week
      implementation plan.
      Includes:
        - Chapter 1: The mathematical insight (with derivation)
        - Chapter 2: Why this is the right answer (numbers for iSTFT)
        - Chapter 3: A complete worked example by hand, proven against
                  the original stamp-and-overlap-add method
        - Chapter 3.5: A second worked example (x=[1,2], w=[1,2,3,4],
                  S=2) traced stick-by-stick through BOTH the current
                  ttnn pipeline (halo zero-interleaving, conv2d with
                  flipped weights, all 24 MACs counted) AND through
                  polyphase (2 sub-convs, 12 MACs counted), with a
                  side-by-side comparison table and discussion of how
                  the savings scale with stride S
        - Chapter 4: Picture of what gets replaced in the pipeline
                  ([C] weights, [D] sliding window, [E] halo, [F] conv2d)
        - Chapter 5: The polyphase algorithm step-by-step
        - Chapter 6: Side-by-side numbers for the iSTFT case
                  (10.1M MACs -> 61K MACs, etc.)
        - Chapter 7: Line-by-line code map of what to edit and where
        - Chapter 8: How to batch the S sub-convs efficiently
                  (grouped-conv trick)
        - Chapter 9: Edge cases (padding, output_padding, dilation,
                  groups, 2D transpose, mirror_kernel)
        - Chapter 10: Test plan for correctness
        - Chapter 11: Why this belongs IN ttnn (not in Python)
        - Chapter 12: Historical context (every other library does this)
        - Chapter 13: Concrete 8-week effort estimate
        - Chapter 14: One-page summary

  07_how_to_optimize_conv_transpose2d.txt
      Practical menu of optimization opportunities for conv_transpose2d.
      Every recommendation cites the exact file/function it would touch.
      Includes:
        - Section 1: Quick-win micro-optimizations (low effort)
        - Section 2: Reader & memory-layout fixes (unblocks iSTFT)
        - Section 3: Algorithmic restructuring (polyphase, FFT)
        - Section 4: Auto-shard / heuristic improvements
        - Section 5: Special-case fast paths
        - Section 6: Per-dtype / precision strategies
        - Section 7: System-level wins
        - Section 8: Recommended priority order (P0..P3)
        - Section 9: iSTFT-specific recipe (quickest path to a passing
                  test, then long-term polyphase plan)
        - Cheatsheet: which file to edit for each optimization

  06_files_index_for_conv_transpose2d.txt
      Complete code map: every file in the tt-metal repo that touches
      conv_transpose2d, organized by role, with a one-line purpose
      and the key functions/classes inside each.
      Includes:
        - Group A: conv_transpose2d core (the op itself)
        - Group B: conv2d engine (the underlying kernel)
        - Group C: conv2d on-device kernels (Tensix-side reader/writer/compute)
        - Group D: sliding_window (geometry, metadata)
        - Group E: halo (the data movement op + its kernels)
        - Group F: Python surface
        - Group G: build system
        - Group H: tests
        - Group I: sweep framework
        - Group J: documentation & model users
        - Group K: this notes folder
        - Recommended reading order for new developers
        - Top-10 files by importance

  05_how_halo_borrows_data_between_cores.txt
      Source-verified deep dive on the actual mechanism by which halo
      moves data between cores. Topic in one sentence: how does Core 1
      hand its boundary stick to Core 0?
      Includes:
        - The big surprise: halo PUSHES, not pulls (noc_async_write)
        - Push vs pull and why ttnn picks push (verified by the
                  static_assert(!remote_read) in halo_gather.cpp)
        - The three per-core metadata tables
                  (padding_config, local_config, remote_config)
        - Walked source code from halo_gather.cpp::kernel_main
        - A concrete example: Core 1 lending s2 to Core 0 and s3 to
                  Core 2 with the exact transfer triplets
        - How the host builds the tables (greedy stick coalescing)
        - Multiple borrows per core, BLOCK_SHARDED neighbors
        - Pad zeros via MEM_ZEROS_BASE
        - Timeline (when does each phase happen, when do barriers
                  apply)
        - FAQ (does C0 know about its borrow? what if reads start too
                  early? cost of one borrow? loopback writes?)

  04_why_coalesced_read_bytes_formula.txt
      Single-topic deep dive on the conv2d reader's NoC-burst formula:
            coalesced_read_bytes = K_w * in_channels_padded * dtype_size
      Includes:
        - The exact line in the kernel source where it is computed
        - Geometric meaning of each factor (with diagrams)
        - Why the reader coalesces in the first place (perf model)
        - Why in_channels_padded != in_channels (alignment rules)
        - The NoC hardware origin of the 8 KB burst cap
        - Working vs failing case comparison (small_k4 vs istft_envelope)
        - What ONE coalesced read actually does in L1
        - Why dilation_w > 1 takes a slower one-stick-per-read path
        - The full formula with reuse and alignment optimizations
        - Knobs you can turn to make a failing case fit (act_block_w_div,
                  bf16 vs fp32, DRAM slicing)

  03_sharding_and_halo_deep_dive.txt
      Focused, diagram-heavy deep dive on JUST two ideas:
        - How tensors are split across cores (sharding)
        - How halo helps neighboring cores cooperate
      Includes:
        - 8x8 chip grid drawing
        - Picture of the three sharding schemes
                  (HEIGHT vs WIDTH vs BLOCK)
        - The math the auto-sharder runs
                  (with worked examples for output_w = 34 and 15840)
        - The "boundary problem" that breaks naive sharding
        - Halo's three jobs (untilize, pad, gather) drawn step-by-step
        - The transpose-conv zero-interleave trick, with a hand-traced
                  9-output mini-example
        - A 4-core hand-traceable example that matches PyTorch
        - The iSTFT case scaled up
        - Anti-patterns and a fast debug checklist

  02_conv_transpose2d_beginners_journey.txt
      Plain-English deep dive aimed at a beginner. Start here if you have
      never thought about accelerators before. Covers:
        - 60-second tensor refresher (NCHW vs NHWC)
        - What a normal convolution does (cookie-cutter analogy)
        - What a transpose convolution does (stamping analogy)
                  + why the name "transpose"
        - Why iSTFT is a transpose convolution
        - The clever trick: zero-interleave + normal conv
        - Tenstorrent hardware basics (Tensix, L1, DRAM, NoC)
        - What sharding means and why it matters
        - The problem sharding creates and why halo exists
        - The full step-by-step JOURNEY of one conv_transpose2d call
        - A small, hand-traceable mini-example
        - Why the iSTFT envelope case breaks (in plain language)
        - Mental-model cheat sheet for any future shape
        - Glossary
        - Recommended study order

Naming convention:
  NN_topic.txt    where NN is a 2-digit ordinal so files sort chronologically.

Add new notes by creating a new file with the next ordinal number.
================================================================================
