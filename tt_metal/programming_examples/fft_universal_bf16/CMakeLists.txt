# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# fft_universal_bf16 — bfloat16 variant of fft_universal, with TRUE bf16
# compute on the Tensix FPU (matmul_tiles with bf16 srcA/srcB, fp32 DST
# accumulate, bf16 CB output). Phase 1 covers the packed direct-DFT path
# for N <= 32 only; N > 32 errors out pending the Phase 2 radix-32
# Stockham bf16 matmul kernel.

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

add_executable(metal_example_fft_universal_bf16_test
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_bf16_test.cpp)

add_executable(metal_example_fft_universal_bf16_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_bf16_benchmark.cpp)

add_executable(metal_example_fft_universal_bf16_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_bf16_demo.cpp)

add_executable(metal_example_fft_universal_bf16_precision
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_bf16_precision.cpp)

# File-IO runner — reads input from a file, runs FFT/IFFT, writes output.
# Backs the PyTorch-style Python wrapper (python/tt_fft.py).
add_executable(metal_example_fft_universal_bf16_run
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_bf16_run.cpp)

foreach(tgt
    metal_example_fft_universal_bf16_test
    metal_example_fft_universal_bf16_benchmark
    metal_example_fft_universal_bf16_demo
    metal_example_fft_universal_bf16_precision
    metal_example_fft_universal_bf16_run)
    target_link_libraries(${tgt} PRIVATE TT::Metalium)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..       # so "../fft_stockham/fft_stockham_host.cpp" resolves
        ${PROJECT_SOURCE_DIR}
    )
    set_target_properties(${tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_universal_bf16
    )
endforeach()
