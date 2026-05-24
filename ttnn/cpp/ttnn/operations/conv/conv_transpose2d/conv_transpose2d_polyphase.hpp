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

namespace ttnn::operations::conv::conv_transpose2d {

using ttnn::prim::Conv2dConfig;

// ============================================================================
// Polyphase conv_transpose2d (V1)
// ============================================================================
//
// Computes ttnn::conv_transpose2d via polyphase decomposition:
//   - Shuffle the (mirrored) weight tensor into S_w sub-kernels of width K_p
//     = ceil(K_w / S_w).
//   - Pad the input with K_p - 1 zeros on the left and right along W.
//   - Run S_w standard 2D convolutions, each producing every S_w-th output.
//   - Interleave the S_w sub-outputs into the final output.
//
// V1 ONLY supports the 1D-as-2D regime:
//   input_h == 1, kernel_size[0] == 1, stride[0] == 1, padding_h == 0
//
// is_polyphase_friendly() (declared in prepare_conv_transpose2d_weights.hpp)
// should be checked first; this function asserts the V1 preconditions.

ConvTranspose2dResult conv_transpose2d_polyphase(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_config,
    const std::optional<const MemoryConfig>& memory_config,
    bool mirror_kernel);

}  // namespace ttnn::operations::conv::conv_transpose2d
