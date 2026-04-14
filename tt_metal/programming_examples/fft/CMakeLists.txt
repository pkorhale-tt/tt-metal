# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

set(FFT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_test.cpp
)

add_executable(metal_example_fft_test ${FFT_SRC})

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_test
    PRIVATE
        TT::Metalium
)

target_include_directories(
    metal_example_fft_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    metal_example_fft_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft
)
