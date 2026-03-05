import os

dirs = ['basic']
base_dir = '/Users/pkorhale/tt-metal/tt_metal/programming_examples'

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Reset any previous MeshDevice additions first just in case
    content = content.replace('auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);', 'IDevice* device = CreateDevice(0);')
    content = content.replace('device->close();', 'CloseDevice(device);')
    content = content.replace('device->mesh_command_queue()', 'device->command_queue()')
    content = content.replace('tt::tt_metal::distributed::MeshCommandQueue&', 'CommandQueue&')
    content = content.replace('#include <tt-metalium/distributed.hpp>\n', '')

    # Apply only header fixes
    content = content.replace('#include "host_api.hpp"', '#include <tt-metalium/host_api.hpp>')
    content = content.replace('#include "device.hpp"', '#include <tt-metalium/device.hpp>')
    
    # Kernel API includes
    content = content.replace('#include "compute_kernel_api/', '#include "api/compute/')
    content = content.replace('#include "dataflow_api.h"', '#include "api/dataflow/dataflow_api.h"')
    
    # Handle the un-used getElapsedTime Warning->Error
    content = content.replace('static double getElapsedTime(struct timeval start_time)', '[[maybe_unused]] static double getElapsedTime(struct timeval start_time)')
    content = content.replace('static double getElapsedTime(struct timeval);', '[[maybe_unused]] static double getElapsedTime(struct timeval);')

    with open(filepath, 'w') as f:
        f.write(content)

for d in dirs:
    dpath = os.path.join(base_dir, d)
    if not os.path.exists(dpath): 
        print(f"Skipping {dpath}, not found.")
        continue
    
    print(f"Processing {dpath}")
    for root, dirs_in, files in os.walk(dpath):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.hpp') or file.endswith('.h'):
                process_file(os.path.join(root, file))

    # Re-apply CMake template
    target_name = f'metal_example_fft_{d}'
    cmake_content = f"""# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(programming_examples_{d})

add_executable({target_name} fft.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    {target_name}
    PRIVATE
        TT::Metalium
)
"""
    with open(os.path.join(dpath, 'CMakeLists.txt'), 'w') as f:
        f.write(cmake_content)

print("Migration complete.")
