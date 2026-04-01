FFT Float32 Single Core Example

This directory contains a single-core implementation of a Fast Fourier Transform (FFT) using Float32 precision on TT-Metalium. It demonstrates accurate complex multiplication and memory movement using full 32-bit floating-point math across L1 circular buffers.

---

File Breakdown

1. CMakeLists.txt
   Contains build targets to integrate fft_single_core into the programming_examples suite.

2. fft_single_core/fft_single_core.cpp
   CPU host code using the MeshDevice API.
   It creates a reference Float32 FFT calculation on the CPU and compares the results with the RISC-V hardware output.

3. fft_single_core/kernels/dataflow/

   * reader_fft_f32.cpp
     Reads input data from DRAM into L1 circular buffers.
   * writer_fft_f32.cpp
     Writes computed FFT output from L1 back to DRAM.

4. fft_single_core/kernels/compute/fft_compute_f32.cpp
   Performs the FFT mathematical transformation.
   Uses:
   MathFidelity::HiFi4
   .fp32_dest_acc_en = true
   This ensures high-precision Float32 accumulation.

---

Compilation Instructions

Inside your tt-metal repository root (example: /Users/pkorhale/tt-metal/), run:

1. Generate build files (skip if already configured):

cmake -B build -G "Unix Makefiles" -DBUILD_PROGRAMMING_EXAMPLES=ON

2. Compile the specific Float32 FFT target:

cmake --build build --target metal_example_fft_float32_single_core -j

---

Execution Instructions

Before running, set the architecture environment variable to trigger the correct just-in-time compilation for your hardware (e.g., Wormhole B0).

From the repository root, run:

ARCH_NAME=wormhole_b0 ./build/programming_examples/metal_example_fft_float32_single_core

---

Expected Results

During execution, the program compares hardware results with CPU reference results index-by-index.

Example output:

=== INDEX 0 ===
LHS R - Expected: ..., Actual: ...
LHS I - Expected: ..., Actual: ...
RHS R - Expected: ..., Actual: ...
RHS I - Expected: ..., Actual: ...

Verification: Max Diff = 0, Max RTol = 0.000000%
Test Passed

If everything matches exactly (within tolerance), the test will print "Test Passed".

If there is any mismatch, it will print the index and the expected vs actual values to help debug precision or algorithm issues.

---
