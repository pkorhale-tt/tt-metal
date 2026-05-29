// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::conv_transpose2d_polyphase {

using ttnn::prim::Conv2dConfig;

// Experimental polyphase-decomposed conv_transpose2d. Same signature as
// ttnn::conv_transpose2d but always takes the polyphase path (no detector).
//
// V1 restrictions (asserted at runtime):
//   - input_height == 1, kernel_h == 1, stride_h == 1
//   - padding == 0, dilation == (1,1), output_padding == 0
//   - groups == 1
//
// Exposed in Python as: ttnn.experimental.conv_transpose2d_polyphase(...)
ConvTranspose2dResultWithOptions conv_transpose2d_polyphase(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride = std::array<uint32_t, 2>{1, 1},
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding = std::array<uint32_t, 4>{0, 0, 0, 0},
    std::array<uint32_t, 2> output_padding = std::array<uint32_t, 2>{0, 0},
    std::array<uint32_t, 2> dilation = std::array<uint32_t, 2>{1, 1},
    uint32_t groups = 1,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const Conv2dConfig>& conv_config = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_config = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    bool mirror_kernel = true,
    bool return_output_dim = false,
    bool return_weights_and_bias = false);

}  // namespace ttnn::operations::experimental::conv_transpose2d_polyphase
