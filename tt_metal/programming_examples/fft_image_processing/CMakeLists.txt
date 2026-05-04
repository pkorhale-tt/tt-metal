# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# fft_image_processing — single-process 2-D FFT runner (host-only orchestration
# around the existing fft_universal pipeline). Lives next to the Python image-
# processing demo so this folder is self-contained.

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

add_executable(metal_example_fft_image_processing_fft2_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/fft2_runner.cpp)

target_link_libraries(metal_example_fft_image_processing_fft2_runner
    PRIVATE TT::Metalium)

# Reuse the existing fft_universal host implementation (fft_universal_host.cpp
# is included directly by fft2_runner.cpp). The include paths below mirror what
# fft_universal/CMakeLists.txt exposes.
target_include_directories(metal_example_fft_image_processing_fft2_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../fft_universal
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}
)

set_target_properties(metal_example_fft_image_processing_fft2_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_image_processing
)
