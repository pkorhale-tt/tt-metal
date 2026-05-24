// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/conv_transpose2d_polyphase/conv_transpose2d_polyphase.hpp"

#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d_polyphase.hpp"

namespace ttnn::operations::experimental::conv_transpose2d_polyphase {

namespace impl = ttnn::operations::conv::conv_transpose2d;

namespace {

ttnn::ConvTranspose2dResultWithOptions to_result_with_options(
    const ttnn::ConvTranspose2dResult& result, bool return_output_dim, bool return_weights_and_bias) {
    if (return_output_dim && return_weights_and_bias) {
        return std::make_tuple(
            std::get<0>(result),
            std::make_tuple(std::get<1>(result), std::get<2>(result)),
            std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    if (return_output_dim) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<1>(result), std::get<2>(result)));
    }
    if (return_weights_and_bias) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    return std::get<0>(result);
}

}  // namespace

ttnn::ConvTranspose2dResultWithOptions conv_transpose2d_polyphase(
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
    bool mirror_kernel,
    bool return_output_dim,
    bool return_weights_and_bias) {
    auto result = impl::conv_transpose2d_polyphase(
        input_tensor,
        weight_tensor,
        device,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        dtype,
        bias_tensor,
        conv_config,
        compute_config,
        memory_config,
        mirror_kernel);
    return to_result_with_options(result, return_output_dim, return_weights_and_bias);
}

}  // namespace ttnn::operations::experimental::conv_transpose2d_polyphase
