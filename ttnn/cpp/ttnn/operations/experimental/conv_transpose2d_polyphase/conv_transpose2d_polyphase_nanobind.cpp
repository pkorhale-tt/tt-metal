// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d_polyphase_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/conv_transpose2d_polyphase/conv_transpose2d_polyphase.hpp"

namespace ttnn::operations::experimental::conv_transpose2d_polyphase::detail {

void bind_experimental_conv_transpose2d_polyphase(nb::module_& mod) {
    const auto* doc = R"doc(
        EXPERIMENTAL: 2D transpose convolution implemented via polyphase decomposition.

        Replaces the default halo + zero-interleave + large-K conv2d pipeline with
        S_w parallel standard convolutions, each of kernel width K_p = ceil(K_w / S_w).

        Always takes the polyphase path (no auto-detector). Use ttnn.conv_transpose2d
        for the production path that decides based on shape heuristics.

        V1 restrictions (asserted at runtime):
          - input_height == 1, kernel_size[0] == 1, stride[0] == 1
          - padding == 0, dilation == (1, 1), output_padding == 0
          - groups == 1

        Args:
            input_tensor (ttnn.Tensor): NHWC input tensor.
            weight_tensor (ttnn.Tensor): IOHW weight tensor [C_in, C_out, 1, K_w].
            device (ttnn.MeshDevice): the device to use.
            in_channels (int)
            out_channels (int)
            batch_size (int)
            input_height (int): must be 1 in V1.
            input_width (int)
            kernel_size (tuple[int, int]): (1, K_w) in V1.
            stride (tuple[int, int]): (1, S_w).
            padding (tuple): must represent 0 padding in V1.
            output_padding (tuple): (0, 0) in V1.
            dilation (tuple): (1, 1) in V1.
            groups (int): 1 in V1.

        Keyword Args:
            bias_tensor (ttnn.Tensor, optional)
            dtype (DataType, optional)
            conv_config (ttnn.Conv2dConfig, optional)
            compute_config (ttnn.DeviceComputeKernelConfig, optional)
            memory_config (ttnn.MemoryConfig, optional)
            mirror_kernel (bool): defaults True; treats input weights as un-mirrored.
            return_output_dim (bool): if True, also return (H_out, W_out).
            return_weights_and_bias (bool): if True, also return preprocessed weights/bias.

        Returns:
            ttnn.Tensor or a tuple matching the return_* flags.
        )doc";

    ttnn::bind_function<"conv_transpose2d_polyphase">(
        mod,
        doc,
        &ttnn::operations::experimental::conv_transpose2d_polyphase::conv_transpose2d_polyphase,
        nb::kw_only(),
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("device"),
        nb::arg("in_channels"),
        nb::arg("out_channels"),
        nb::arg("batch_size"),
        nb::arg("input_height"),
        nb::arg("input_width"),
        nb::arg("kernel_size"),
        nb::arg("stride") = nb::cast(std::array<uint32_t, 2>{1, 1}),
        nb::arg("padding") = nb::cast(std::array<uint32_t, 2>{0, 0}),
        nb::arg("output_padding") = nb::cast(std::array<uint32_t, 2>{0, 0}),
        nb::arg("dilation") = nb::cast(std::array<uint32_t, 2>{1, 1}),
        nb::arg("groups") = 1,
        nb::arg("dtype") = nb::none(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("conv_config") = nb::none(),
        nb::arg("compute_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("mirror_kernel") = true,
        nb::arg("return_output_dim") = false,
        nb::arg("return_weights_and_bias") = false);
}

}  // namespace ttnn::operations::experimental::conv_transpose2d_polyphase::detail
