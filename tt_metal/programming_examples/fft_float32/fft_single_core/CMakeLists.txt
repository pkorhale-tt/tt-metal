# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

add_executable(metal_example_fft_float32_single_core fft_single_core.cpp)
target_link_libraries(metal_example_fft_float32_single_core PRIVATE TT::Metalium)