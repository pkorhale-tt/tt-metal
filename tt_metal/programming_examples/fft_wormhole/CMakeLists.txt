# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

set(FFT_WORMHOLE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/host/fft_universal.cpp
)

add_executable(metal_example_fft_wormhole ${FFT_WORMHOLE_SRC})

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_wormhole
    PRIVATE
        TT::Metalium
)

target_include_directories(
    metal_example_fft_wormhole
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    metal_example_fft_wormhole
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_wormhole
)
