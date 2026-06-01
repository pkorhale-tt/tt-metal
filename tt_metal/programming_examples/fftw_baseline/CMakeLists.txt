# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Host-CPU FFTW baseline for the HPEC 2026 fft_universal paper.
# Builds only if libfftw3 is available on the host. Otherwise prints a
# warning and is skipped — does not block the rest of the build.

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3F QUIET fftw3f)
    pkg_check_modules(FFTW3F_THREADS QUIET fftw3f_threads)
endif()

if(NOT FFTW3F_FOUND)
    message(STATUS "fftw_baseline: libfftw3f not found (install libfftw3-dev). "
                   "Skipping fftw_baseline build.")
    return()
endif()

add_executable(fftw_baseline
    ${CMAKE_CURRENT_SOURCE_DIR}/fftw_baseline.cpp)

target_include_directories(fftw_baseline PRIVATE ${FFTW3F_INCLUDE_DIRS})
target_link_libraries(fftw_baseline PRIVATE ${FFTW3F_LIBRARIES})
if(FFTW3F_THREADS_FOUND)
    target_link_libraries(fftw_baseline PRIVATE ${FFTW3F_THREADS_LIBRARIES})
    target_compile_definitions(fftw_baseline PRIVATE FFTW_HAS_THREADS=1)
endif()

target_compile_features(fftw_baseline PRIVATE cxx_std_20)
set_target_properties(fftw_baseline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/fftw_baseline)

message(STATUS "fftw_baseline: enabled (fftw3f ${FFTW3F_VERSION})")
