# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# Pure-host planner unit test: builds without device deps in CI.
# We still link against TT::Metalium for consistency, but the test only
# uses the planner header.
add_executable(metal_example_fft_universal_xl_planner_test
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_xl_planner_test.cpp)

# Device round-trip test (Option B path).
add_executable(metal_example_fft_universal_xl_test
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_xl_test.cpp)

# Minimal "how to call it" demo.
add_executable(metal_example_fft_universal_xl_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_universal_xl_demo.cpp)

foreach(tgt
    metal_example_fft_universal_xl_planner_test
    metal_example_fft_universal_xl_test
    metal_example_fft_universal_xl_demo)
    target_link_libraries(${tgt} PRIVATE TT::Metalium)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..   # so "../fft_stockham/..." resolves
        ${PROJECT_SOURCE_DIR}
    )
    set_target_properties(${tgt} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/programming_examples/fft_universal_xl
    )
endforeach()
