# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Correctness + timing harness for the Stockham orchestrator.
add_executable(metal_example_fft_stockham_test
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_stockham_test.cpp)

# Plan-cache speedup benchmark over Stockham (uses inner kernel's plan cache).
add_executable(metal_example_fft_stockham_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_stockham_benchmark.cpp)

# Minimal "how to call it" example.
add_executable(metal_example_fft_stockham_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_stockham_demo.cpp)

foreach(tgt
    metal_example_fft_stockham_test
    metal_example_fft_stockham_benchmark
    metal_example_fft_stockham_demo)
    target_link_libraries(${tgt} PRIVATE TT::Metalium)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..       # so "../fft/fft_host.cpp" resolves
        ${PROJECT_SOURCE_DIR}
    )
    set_target_properties(${tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_stockham
    )
endforeach()
