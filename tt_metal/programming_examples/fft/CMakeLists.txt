# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Correctness + timing harness
add_executable(metal_example_fft_test ${CMAKE_CURRENT_SOURCE_DIR}/fft_test.cpp)

# PyTorch-style one-shot FFT demo
add_executable(metal_example_fft_demo ${CMAKE_CURRENT_SOURCE_DIR}/fft_demo.cpp)

# tt-metal FFT dumper, used by compare_with_torch.py
add_executable(metal_example_fft_vs_torch ${CMAKE_CURRENT_SOURCE_DIR}/fft_vs_torch.cpp)

# User playground — edit signal in my_fft_app.cpp and rebuild
add_executable(my_fft_app ${CMAKE_CURRENT_SOURCE_DIR}/my_fft_app.cpp)

foreach(tgt metal_example_fft_test metal_example_fft_demo metal_example_fft_vs_torch my_fft_app)
    target_link_libraries(${tgt} PRIVATE TT::Metalium)
    target_include_directories(
        ${tgt}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}
    )
    set_target_properties(
        ${tgt}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft
    )
endforeach()
