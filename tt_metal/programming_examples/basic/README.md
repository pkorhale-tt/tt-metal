FFT Basic Single Core Example

This directory contains the basic implementation of a Fast Fourier Transform (FFT) using the modern single-mesh TT-Metalium API. The kernel performs a radix-2 Cooley-Tukey FFT using Float32 precision on a single RISC-V compute core.

---

File Breakdown

1. CMakeLists.txt
   Integrates the metal_example_fft_basic target into the programming_examples build suite.

2. fft.cpp
   CPU host code responsible for:

   * Initializing the device using the MeshDevice API
   * Generating a mathematical test pattern
   * Transferring input data to device DRAM using MeshCommandQueue
   * Launching the RISC-V compute workload using MeshWorkload
   * Reading back results from the device
   * Validating results row-by-row on the CPU

3. kernels/dataflow/

   * reader.cpp
     Handles moving input data from DRAM to L1 memory.
   * writer.cpp
     Handles moving computed FFT results from L1 back to DRAM.

4. kernels/compute/compute.cpp
   Performs the FFT math iterations (radix-2 butterfly stages)
   Stores intermediate and final results inside L1 Circular Buffers on device hardware.

---

Compilation Instructions

From the tt-metal repository root (example: /Users/pkorhale/tt-metal/):

1. Generate Makefiles (run once if not already configured):

cmake -B build -G "Unix Makefiles" -DBUILD_PROGRAMMING_EXAMPLES=ON

2. Compile the basic FFT target:

cmake --build build --target metal_example_fft_basic -j

---

Execution Instructions

Because this kernel runs on Tenstorrent Wormhole hardware, you must set the ARCH_NAME environment variable.

Run from the repository root and pass the FFT size (must be a power of two, e.g., 1024):

ARCH_NAME=wormhole_b0 ./build/programming_examples/metal_example_fft_basic 1024

---

Expected Output

The executable performs:

1. A forward FFT pass
2. A backward (inverse) FFT pass
3. A verification step to confirm the inverse restores the original data

If successful, the output will end with something similar to:

Forwards FFT of size 1024: total time ... sec.
Backwards FFT of size 1024: total time ... sec.
Checked 1024 elements: 1024 match and 0 missmatched

This confirms:

* Forward FFT works correctly
* Inverse FFT restores original values
* No precision loss occurred across 1024 elements

---
