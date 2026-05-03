# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Correctness harness for the universal FFT (any N >= 2).
add_executable(metal_example_fft_universal_test
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_test.cpp)

# End-to-end benchmark (any N).
add_executable(metal_example_fft_universal_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_benchmark.cpp)

# Minimal "how to call it" example.
add_executable(metal_example_fft_universal_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_demo.cpp)

# PyTorch comparison harness (pairs with compare_with_torch.py).
add_executable(metal_example_fft_universal_vs_torch
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_vs_torch.cpp)

# File-IO runner — reads input from a file, runs FFT/IFFT, writes output.
# Backs the PyTorch-style Python wrapper (python/tt_fft.py).
add_executable(metal_example_fft_universal_run
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_run.cpp)

foreach(tgt
    metal_example_fft_universal_test
    metal_example_fft_universal_benchmark
    metal_example_fft_universal_demo
    metal_example_fft_universal_vs_torch
    metal_example_fft_universal_run)
    target_link_libraries(${tgt} PRIVATE TT::Metalium)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..       # so "../fft/fft_host.cpp" and
                                             # "../fft_stockham/fft_stockham_host.cpp"
                                             # resolve
        ${PROJECT_SOURCE_DIR}
    )
    set_target_properties(${tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_universal
    )
endforeach()
