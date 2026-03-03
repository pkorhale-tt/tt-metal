Standard Operating Procedure (SOP): Testing FFT Kernels on TT-Metal (Wormhole)

This guide explains how to rebuild and run your single-core FFT kernel on the Tenstorrent Wormhole machine.

---

Prerequisites

1. Activate your Python environment.
2. Navigate to your TT-Metal directory:

cd /proj_sw/user_dev/pkorhale/fft_metal/tt-metal

---

Step 1: Rebuild After Any Code Change

Whenever you modify:

* Any .cpp host file
* Any compute kernel
* Any dataflow kernel

Inside:
programming_examples/basic_fft/fft_single_core/

Rebuild using:

cmake --build build --target metal_example_basic_fft_single_core -j

If you are modifying the original fft directory instead, use:

cmake --build build --target metal_example_fft_single_core -j

This command recompiles only the required FFT example.

---

Step 2: Run the Executable

You must always prefix the command with the architecture name:

ARCH_NAME=wormhole_b0 ./build/programming_examples/metal_example_basic_fft_single_core

Do not forget to include ARCH_NAME=wormhole_b0.

---

Step 3: What Happens When You Run It

When you execute the program, it will:

1. Initialize the Wormhole machine.
2. Load random bfloat16 input data.
3. Execute your FFT kernel on the RISC-V cores.
4. Compute a CPU FP32 reference result.
5. Compare the hardware output with the CPU result.

---

Step 4: Verifying the Output

If the FFT implementation is correct, you will see:

Verification summary:  Max Diff = 0.0, Max RTol = 0.00%
Test Passed

This means:

* The FFT algorithm is correct.
* The result is within the 5% relative tolerance limit.

If there is an issue, you will see:

Test Failed

The program will print:

* Expected value
* Actual hardware value
* Index where the mismatch occurred

This usually indicates:

* Algorithm mistake
* Incorrect twiddle factor
* Stage computation error
* Precision issue.

---
