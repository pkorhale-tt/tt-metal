# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

add_executable(metal_example_fft_single_core)
target_sources(metal_example_fft_single_core PRIVATE fft_single_core.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_single_core
    PRIVATE
        TT::Metalium
)
