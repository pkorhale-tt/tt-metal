# Standard Operating Procedure (SOP): Testing FFT Kernels on TT-Metal

This document provides the standard commands and workflow for re-compiling and executing the single-core FFT kernels on a Tenstorrent Wormhole machine.

## Prerequisites

Ensure you are in the python environment and located in your TT-Metal directory:
```bash
cd /proj_sw/user_dev/pkorhale/fft_metal/tt-metal
```

## Step 1: Building the Executable

Whenever you make a change to a `.cpp` host file or a compute/dataflow kernel inside `programming_examples/basic_fft/fft_single_core/`, you must rebuild the target.

To efficiently recompile **only** the necessary files for `basic_fft`:
```bash
cmake --build build --target metal_example_basic_fft_single_core -j
```

*(Note: If you are modifying the original `fft` directory, swap the target name to `metal_example_fft_single_core` instead).*

## Step 2: Running the Executable

Because the shared Tenstorrent metalium library requires hardware-specific architecture bindings at runtime, you must prefix your standard execution command with `ARCH_NAME=wormhole_b0`.

To run your compiled test:
```bash
ARCH_NAME=wormhole_b0 ./build/programming_examples/metal_example_basic_fft_single_core
```

## Step 3: Verifying the Output

Upon running the executable, the Tenstorrent host application will:
1. Initialize the Wormhole machine (`tt_cluster`, `topology_discovery`)
2. Load random bfloat16 testing data into circular buffers.
3. Automatically execute the `.cpp` compute kernels on the RISC-V cores.
4. Compare the hardware output row-by-row with a local CPU FP32 reference calculation.

If the output from the hardware matches the mathematical reference within the strict 5% relative tolerance limit, the terminal will print:
```text
Verification summary:  Max Diff = 0.0, Max RTol = 0.00%
Test Passed
```

If it fails due to algorithmic mistakes or extreme precision loss, it will print `Test Failed` along with the exact expected and actual values for the first failed index.
