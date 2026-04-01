# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

set(FFT_MULTI_CORE_OPT_PAPER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_multi_core.cpp
)

add_executable(metal_example_fft_float32_multi_core_optimised_paper ${FFT_MULTI_CORE_OPT_PAPER_SRC})

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_float32_multi_core_optimised_paper
    PRIVATE
        TT::Metalium
)

target_include_directories(
    metal_example_fft_float32_multi_core_optimised_paper
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    metal_example_fft_float32_multi_core_optimised_paper
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_float32_multicore_optimised_paper
)
