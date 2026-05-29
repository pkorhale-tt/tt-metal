// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d_polyphase.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::conv::conv_transpose2d {

namespace {

// V2.1 weight stacking.
//
// shuffle_weights_polyphase produces a host tensor of shape
// [S_w, C_out, C_in, K_h=1, K_p] in ROW_MAJOR. In linear memory:
//
//   addr(p, c, i, kh, kw) = p * (C_out * C_in * K_h * K_p)
//                         + c * (C_in * K_h * K_p)
//                         + i * (K_h * K_p)
//                         + kh * K_p
//                         + kw
//
// We want a 4D OIHW weight where output channel (p * C_out + c) holds
// phase p's kernel for original channel c. The address of element
// (p * C_out + c, i, kh, kw) in a [S_w * C_out, C_in, K_h, K_p] tensor is:
//
//   addr((p * C_out + c), i, kh, kw)
//       = (p * C_out + c) * (C_in * K_h * K_p) + i * (K_h * K_p) + kh * K_p + kw
//       = p * C_out * C_in * K_h * K_p + c * C_in * K_h * K_p + ...
//
// which equals the original 5D address. So the stacking is a pure logical
// reshape from rank-5 to rank-4 with no data motion at all.
ttnn::Tensor stack_weights_for_grouped_conv(const ttnn::Tensor& shuffled_weights) {
    const auto& shape = shuffled_weights.padded_shape();
    TT_FATAL(shape.rank() == 5, "shuffled_weights must be 5D, got rank {}", shape.rank());
    const uint32_t s_w = shape[0];
    const uint32_t c_out = shape[1];
    const uint32_t c_in = shape[2];
    const uint32_t k_h = shape[3];
    const uint32_t k_p = shape[4];
    return ttnn::reshape(shuffled_weights, ttnn::Shape{s_w * c_out, c_in, k_h, k_p});
}

// V2.1 bias stacking. Replicate every entry of the original C_out-wide bias
// S_w times so that big_bias[p * C_out + c] = bias[c]. The grouped conv2d
// adds big_bias to the corresponding output channel, which after the final
// reshape lands on every output position (each is "owned" by exactly one
// phase p), so every position gets bias[c] exactly once.
template <typename T>
ttnn::Tensor _stack_bias_impl(const ttnn::Tensor& bias, uint32_t s_w) {
    const auto& shape = bias.padded_shape();
    TT_FATAL(shape.rank() >= 1, "bias must have rank >= 1, got {}", shape.rank());
    const uint32_t c_out = shape[shape.rank() - 1];
    const ttnn::Shape out_shape{1u, 1u, 1u, s_w * c_out};

    auto compute = [s_w, c_out](const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto in_buf = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto out_buf = std::vector<T>(static_cast<uint64_t>(s_w) * c_out);
        for (uint32_t p = 0; p < s_w; ++p) {
            for (uint32_t c = 0; c < c_out; ++c) {
                out_buf[static_cast<uint64_t>(p) * c_out + c] = in_buf[c];
            }
        }
        return tt::tt_metal::HostBuffer(std::move(out_buf));
    };

    const TensorSpec out_spec(
        out_shape,
        tt::tt_metal::TensorLayout(bias.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    return ttnn::Tensor(bias.host_storage().transform(compute), out_spec, bias.tensor_topology());
}

ttnn::Tensor stack_bias_for_grouped_conv(const ttnn::Tensor& bias, uint32_t s_w) {
    ttnn::Tensor host_bias = tt::tt_metal::is_device_tensor(bias)
                                 ? ttnn::operations::core::from_device(bias)
                                 : bias;
    if (host_bias.layout() != Layout::ROW_MAJOR) {
        host_bias = ttnn::to_layout(host_bias, Layout::ROW_MAJOR);
    }
    switch (host_bias.dtype()) {
        case DataType::BFLOAT16: return _stack_bias_impl<::bfloat16>(host_bias, s_w);
        case DataType::FLOAT32: return _stack_bias_impl<float>(host_bias, s_w);
        case DataType::UINT32: return _stack_bias_impl<uint32_t>(host_bias, s_w);
        default: TT_THROW("Unsupported dtype for stack_bias_for_grouped_conv: {}", host_bias.dtype());
    }
}

}  // namespace

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
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding_,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_config,
    const std::optional<const MemoryConfig>& memory_config,
    bool mirror_kernel) {
   
    // get_pair_n4_padding returns std::array<uint32_t, 4> in
    // {pad_top, pad_bottom, pad_left, pad_right} order.
    const auto padding_n4 = sliding_window::get_pair_n4_padding(padding_);
    const uint32_t pad_top = padding_n4[0];
    const uint32_t pad_bottom = padding_n4[1];
    const uint32_t pad_left = padding_n4[2];
    const uint32_t pad_right = padding_n4[3];

    TT_FATAL(input_height == 1, "Polyphase V1 requires input_height == 1, got {}", input_height);
    TT_FATAL(kernel_size[0] == 1, "Polyphase V1 requires kernel_h == 1, got {}", kernel_size[0]);
    TT_FATAL(stride[0] == 1, "Polyphase V1 requires stride_h == 1, got {}", stride[0]);
    TT_FATAL(dilation[0] == 1 && dilation[1] == 1, "Polyphase V1 requires dilation == (1,1)");
    TT_FATAL(groups == 1, "Polyphase V1 does not support groups > 1");
    TT_FATAL(pad_top == 0 && pad_bottom == 0, "Polyphase V1 requires pad_h == 0");
    TT_FATAL(pad_left == 0 && pad_right == 0, "Polyphase V1 requires pad_w == 0 (TODO V2)");
    TT_FATAL(output_padding[0] == 0 && output_padding[1] == 0, "Polyphase V1 requires output_padding == (0,0)");

    const uint32_t k_w = kernel_size[1];
    const uint32_t s_w = stride[1];
    const uint32_t k_p = (k_w + s_w - 1) / s_w;  // ceil(K_w / S_w)
    const uint32_t t_in = input_width;
    const uint32_t t_out = (t_in - 1) * s_w + k_w;  // standard transpose-conv output length

    log_debug(tt::LogOp, "conv_transpose2d_polyphase: T={}, K={}, S={}, K_p={}, T_out={}", t_in, k_w, s_w, k_p, t_out);

    // Shuffle weights once on host: [C_in, C_out, 1, K_w] -> [S_w, C_out, C_in, 1, K_p]
    
    ttnn::Tensor weight_host = tt::tt_metal::is_device_tensor(weight_tensor)
                                   ? ttnn::operations::core::from_device(weight_tensor)
                                   : weight_tensor;
    if (!mirror_kernel) {//need to handle this case when mirror is true.
        log_warning(
            tt::LogOp,
            "polyphase: mirror_kernel=false is not supported in V1, treating as mirror_kernel=true. "
            "(Pre-mirrored weights would need to be un-mirrored first.)");
    }
    
    ttnn::Tensor shuffled_weights = shuffle_weights_polyphase(weight_host, s_w);

    const uint32_t conv_pad_left = (k_p > 0) ? (k_p - 1) : 0;
    const uint32_t conv_pad_right = (k_p > 0) ? (k_p - 1) : 0;
    const std::array<uint32_t, 4> conv_padding_n4 = {0u, 0u, conv_pad_left, conv_pad_right};
    log_debug(tt::LogOp, "polyphase using conv2d-internal padding L={} R={} (K_p={})", conv_pad_left, conv_pad_right, k_p);

    // V2.1: replace the S_w-iteration per-phase loop with a single grouped
    // conv2d whose output channels are all S_w phases stacked. After the
    // grouped conv:
    //
    //   big_out : (N, 1, T_phase, S_w * C_out)
    //   big_out[n, 0, t, p * C_out + c] = phase_p_output[n, 0, t, c]
    //
    // In ROW_MAJOR memory this is bit-identical to (N, 1, T_phase * S_w, C_out)
    // viewed as out[n, 0, t * S_w + p, c] = phase_p_output[n, 0, t, c], which
    // IS the polyphase interleave. So the entire merge step is a zero-copy
    // logical reshape -- no scatter, no concat, no permute.
    ttnn::Tensor stacked_weights = stack_weights_for_grouped_conv(shuffled_weights);

    std::optional<ttnn::Tensor> stacked_bias_opt;
    if (bias_tensor.has_value()) {
        stacked_bias_opt = stack_bias_for_grouped_conv(bias_tensor.value(), s_w);
    }
    // conv2d takes std::optional<const ttnn::Tensor>; cast a view of our owning
    // optional. We can't implicitly convert std::optional<T> -> std::optional<const T>.
    std::optional<const ttnn::Tensor> stacked_bias_for_conv =
        stacked_bias_opt.has_value() ? std::optional<const ttnn::Tensor>{stacked_bias_opt.value()}
                                     : std::optional<const ttnn::Tensor>{};

    const uint32_t grouped_out_channels = out_channels * s_w;
    log_debug(
        tt::LogOp,
        "polyphase V2.1: grouped conv2d with C_out_stacked={}, K_p={}, T_in={}, S_w={}",
        grouped_out_channels,
        k_p,
        t_in,
        s_w);

    auto conv_result = ttnn::conv2d(
        input_tensor,
        stacked_weights,
        device,
        in_channels,
        grouped_out_channels,
        batch_size,
        /*input_height=*/1,
        /*input_width=*/t_in,
        /*kernel_size=*/std::array<uint32_t, 2>{1, k_p},
        /*stride=*/std::array<uint32_t, 2>{1, 1},
        /*padding=*/conv_padding_n4,
        /*dilation=*/std::array<uint32_t, 2>{1, 1},
        /*groups=*/1,
        /*dtype=*/dtype,
        /*bias_tensor=*/stacked_bias_for_conv,
        /*conv_config=*/conv_config,
        /*compute_config=*/compute_config,
        /*memory_config=*/memory_config,
        /*dram_slice_config=*/std::nullopt,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);

    ttnn::Tensor big_out = std::get<ttnn::Tensor>(conv_result);

    // The final reshape must be ROW_MAJOR-preserving, otherwise the polyphase
    // interleave gets scrambled by tile-internal padding. Coerce once.
    if (big_out.memory_config().is_sharded()) {
        big_out = ttnn::sharded_to_interleaved(
            big_out,
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    }
    if (big_out.layout() != Layout::ROW_MAJOR) {
        big_out = ttnn::to_layout(
            big_out,
            Layout::ROW_MAJOR,
            /*dtype=*/std::nullopt,
            /*memory_config=*/
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    }

    const auto& big_shape = big_out.logical_shape();
    TT_FATAL(big_shape.rank() == 4, "Expected rank-4 conv output, got {}", big_shape.rank());
    const uint32_t n_out = big_shape[0];
    const uint32_t h_out = big_shape[1];
    const uint32_t t_phase = big_shape[2];
    const uint32_t c_combined = big_shape[3];
    TT_FATAL(h_out == 1, "Polyphase V2.1 expects H_out == 1, got {}", h_out);
    TT_FATAL(
        c_combined == grouped_out_channels,
        "Grouped conv output C dim ({}) != C_out * S_w ({}). conv2d may have padded C; "
        "V2.1 requires C_out * S_w to be tile-aligned. Got C_out={}, S_w={}",
        c_combined,
        grouped_out_channels,
        out_channels,
        s_w);
    TT_FATAL(t_phase == t_in + k_p - 1u,
             "Unexpected T_phase={} from conv2d (expected T_in + K_p - 1 = {})",
             t_phase, t_in + k_p - 1u);

    const uint32_t merged_t = t_phase * s_w;

    // Zero-copy interleave: ROW_MAJOR (N, 1, T_phase, S_w * C_out) shares the
    // exact same linear bytes as (N, 1, T_phase * S_w, C_out) under the
    // mapping (t, p, c) <-> (t * S_w + p, c). See header comment above.
    ttnn::Tensor flattened = ttnn::reshape(big_out, ttnn::Shape{n_out, 1u, merged_t, out_channels});

    // Trim trailing positions when K_w is not divisible by S_w
    // (merged_t = (T_in + K_p - 1) * S_w >= T_out = (T_in - 1) * S_w + K_w).
    ttnn::Tensor output;
    if (merged_t > t_out) {
        const std::array<uint32_t, 4> begins = {0u, 0u, 0u, 0u};
        const std::array<uint32_t, 4> ends = {n_out, 1u, t_out, out_channels};
        const std::array<uint32_t, 4> steps = {1u, 1u, 1u, 1u};
        output = ttnn::slice(flattened, begins, ends, steps);
    } else {
        TT_FATAL(merged_t == t_out, "merged_t {} != t_out {}", merged_t, t_out);
        output = flattened;
    }

    return ConvTranspose2dResult{output, /*OH=*/1, /*OW=*/t_out, stacked_weights, stacked_bias_opt};
}

}  // namespace ttnn::operations::conv::conv_transpose2d
