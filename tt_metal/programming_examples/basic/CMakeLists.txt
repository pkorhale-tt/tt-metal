# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(programming_examples_basic)

add_executable(metal_example_fft_basic fft.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_fft_basic
    PRIVATE
        TT::Metalium
)
