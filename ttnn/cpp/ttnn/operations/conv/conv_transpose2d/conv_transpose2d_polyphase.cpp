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
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::conv::conv_transpose2d {

namespace {

// Slice phase p out of a host-side 5D tensor shaped [S_w, C_out, C_in, 1, K_p].
// Returns a 4D host tensor [C_out, C_in, 1, K_p] (standard OIHW for conv2d).
template <typename T>
ttnn::Tensor _slice_phase_impl(const ttnn::Tensor& shuffled_weights, uint32_t phase) {
    const auto& shape = shuffled_weights.padded_shape();
    TT_FATAL(shape.rank() == 5, "shuffled_weights must be 5D, got rank {}", shape.rank());
    const uint32_t s_w = shape[0];
    const uint32_t c_out = shape[1];
    const uint32_t c_in = shape[2];
    const uint32_t k_h = shape[3];
    const uint32_t k_p = shape[4];
    TT_FATAL(phase < s_w, "phase {} out of range [0, {})", phase, s_w);

    const ttnn::Shape out_shape{c_out, c_in, k_h, k_p};
    const uint64_t per_phase_elems = static_cast<uint64_t>(c_out) * c_in * k_h * k_p;

    auto compute = [phase, per_phase_elems](const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto owned_buffer = std::vector<T>(per_phase_elems);
        const T* src = &input_buffer[static_cast<uint64_t>(phase) * per_phase_elems];
        std::copy(src, src + per_phase_elems, owned_buffer.begin());
        return tt::tt_metal::HostBuffer(std::move(owned_buffer));
    };

    const TensorSpec out_spec(
        out_shape,
        tt::tt_metal::TensorLayout(
            shuffled_weights.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    return ttnn::Tensor(
        shuffled_weights.host_storage().transform(compute), out_spec, shuffled_weights.tensor_topology());
}

ttnn::Tensor slice_phase_weights(const ttnn::Tensor& shuffled_weights, uint32_t phase) {
    ttnn::Tensor host_tensor = tt::tt_metal::is_device_tensor(shuffled_weights)
                                   ? ttnn::operations::core::from_device(shuffled_weights)
                                   : shuffled_weights;
    switch (host_tensor.dtype()) {
        case DataType::BFLOAT16: return _slice_phase_impl<::bfloat16>(host_tensor, phase);
        case DataType::FLOAT32: return _slice_phase_impl<float>(host_tensor, phase);
        case DataType::UINT32: return _slice_phase_impl<uint32_t>(host_tensor, phase);
        default: TT_THROW("Unsupported dtype for slice_phase_weights: {}", host_tensor.dtype());
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

    // Pre-allocate the merged output buffer once and scatter every phase's
    // result directly into the right strided positions via slice_write.
    // This replaces the old reshape + concat + reshape + slice chain with
    // a single allocation plus S_w cheap data-movement writes.
    //
    //   Per-phase output : (N, 1, T_phase, C_out) where T_phase = T_in + K_p - 1
    //   Merged output    : (N, 1, T_phase * S_w, C_out)
    //   Interleave law   : merged[..., p + j * S_w, :] = phase_p[..., j, :]
    //
    // slice_write with begins=(0,0,p,0), ends=(N,1,merged_t,C_out),
    // step=(1,1,S_w,1) implements exactly that scatter for phase p.
    const uint32_t merged_t = (t_in + k_p - 1u) * s_w;

    ttnn::Tensor output_buffer;
    bool output_allocated = false;

    ttnn::Tensor first_weight_on_device;
    std::optional<ttnn::Tensor> first_bias_on_device;

    for (uint32_t phase = 0; phase < s_w; ++phase) {
        ttnn::Tensor phase_w = slice_phase_weights(shuffled_weights, phase);

        // Bias only added to phase 0 (so total bias is added exactly once after interleave)
        std::optional<const ttnn::Tensor> bias_for_call =
            (phase == 0) ? bias_tensor : std::optional<const ttnn::Tensor>{};

        auto conv_result = ttnn::conv2d(
            input_tensor,
            phase_w,
            device,
            in_channels,
            out_channels,
            batch_size,
            /*input_height=*/1,
            /*input_width=*/t_in,
            /*kernel_size=*/std::array<uint32_t, 2>{1, k_p},
            /*stride=*/std::array<uint32_t, 2>{1, 1},
            /*padding=*/conv_padding_n4,
            /*dilation=*/std::array<uint32_t, 2>{1, 1},
            /*groups=*/1,
            /*dtype=*/dtype,
            /*bias_tensor=*/bias_for_call,
            /*conv_config=*/conv_config,
            /*compute_config=*/compute_config,
            /*memory_config=*/memory_config,
            /*dram_slice_config=*/std::nullopt,
            /*return_output_dim=*/false,
            /*return_weights_and_bias=*/false);

        ttnn::Tensor phase_out = std::get<ttnn::Tensor>(conv_result);

        // slice_write requires both tensors in ROW_MAJOR interleaved layout
        // (sharded outputs are unsupported for the strided-write path).
        if (phase_out.memory_config().is_sharded()) {
            phase_out = ttnn::sharded_to_interleaved(
                phase_out,
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
        }
        if (phase_out.layout() != Layout::ROW_MAJOR) {
            phase_out = ttnn::to_layout(
                phase_out,
                Layout::ROW_MAJOR,
                /*dtype=*/std::nullopt,
                /*memory_config=*/
                tt::tt_metal::MemoryConfig{
                    tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
        }

        // Allocate the merged DRAM buffer the first time around, using the
        // actual dtype produced by conv2d so slice_write sees matching dtypes.
        if (!output_allocated) {
            const uint32_t t_phase_actual = phase_out.logical_shape()[2];
            TT_FATAL(
                t_phase_actual * s_w == merged_t,
                "Unexpected T_phase={} from conv2d; expected {} (K_p={}, T_in={}, S_w={})",
                t_phase_actual,
                merged_t / s_w,
                k_p,
                t_in,
                s_w);
            output_buffer = tt::tt_metal::create_device_tensor(
                tt::tt_metal::TensorSpec(
                    ttnn::Shape{batch_size, 1u, merged_t, out_channels},
                    tt::tt_metal::TensorLayout(
                        phase_out.dtype(),
                        tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
                        tt::tt_metal::MemoryConfig{
                            tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                            tt::tt_metal::BufferType::DRAM})),
                device);
            output_allocated = true;
        }

        // Scatter phase `p`'s elements into output_buffer[..., p :: S_w, :].
        ttnn::experimental::slice_write(
            phase_out,
            output_buffer,
            ttnn::SmallVector<uint32_t>{0u, 0u, phase, 0u},
            ttnn::SmallVector<uint32_t>{batch_size, 1u, merged_t, out_channels},
            ttnn::SmallVector<uint32_t>{1u, 1u, s_w, 1u});

        if (phase == 0) {
            first_weight_on_device = phase_w;
            first_bias_on_device = bias_tensor;
        }
    }

    // Trim trailing positions when K_w is not divisible by S_w
    // (merged_t = (T_in + K_p - 1) * S_w >= T_out = (T_in - 1) * S_w + K_w).
    ttnn::Tensor output;
    if (merged_t > t_out) {
        const std::array<uint32_t, 4> begins = {0u, 0u, 0u, 0u};
        const std::array<uint32_t, 4> ends = {batch_size, 1u, t_out, out_channels};
        const std::array<uint32_t, 4> steps = {1u, 1u, 1u, 1u};
        output = ttnn::slice(output_buffer, begins, ends, steps);
    } else {
        TT_FATAL(merged_t == t_out, "merged_t {} != t_out {}", merged_t, t_out);
        output = output_buffer;
    }

    return ConvTranspose2dResult{output, /*OH=*/1, /*OW=*/t_out, first_weight_on_device, first_bias_on_device};
}

}  // namespace ttnn::operations::conv::conv_transpose2d
