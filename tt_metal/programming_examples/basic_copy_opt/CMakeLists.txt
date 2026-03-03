# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

add_executable(metal_example_fft_basic_copy_opt)
target_sources(metal_example_fft_basic_copy_opt PRIVATE fft.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_basic_copy_opt
    PRIVATE
        TT::Metalium
)
