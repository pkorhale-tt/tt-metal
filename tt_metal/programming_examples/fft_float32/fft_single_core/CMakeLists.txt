# # SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# # SPDX-License-Identifier: Apache-2.0

# add_executable(metal_example_fft_float32_single_core fft_single_core.cpp)
# target_link_libraries(metal_example_fft_float32_single_core PRIVATE TT::Metalium)


# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

set(FFT_SINGLE_CORE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_single_core.cpp
)

add_executable(metal_example_fft_float32_single_core ${FFT_SINGLE_CORE_SRC})

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_float32_single_core
    PRIVATE
        TT::Metalium
)

target_include_directories(
    metal_example_fft_float32_single_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    metal_example_fft_float32_single_core
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fft_float32
)
