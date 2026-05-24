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
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
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
    // ------------------------------------------------------------------
    // V1 preconditions
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // (1) Shuffle weights once on host: [C_in, C_out, 1, K_w] -> [S_w, C_out, C_in, 1, K_p]
    //     Each per-phase slice is in OIHW orientation and ready for ttnn::conv2d.
    // ------------------------------------------------------------------
    ttnn::Tensor weight_host = tt::tt_metal::is_device_tensor(weight_tensor)
                                   ? ttnn::operations::core::from_device(weight_tensor)
                                   : weight_tensor;
    if (!mirror_kernel) {
        log_warning(
            tt::LogOp,
            "polyphase: mirror_kernel=false is not supported in V1, treating as mirror_kernel=true. "
            "(Pre-mirrored weights would need to be un-mirrored first.)");
    }
    ttnn::Tensor shuffled_weights = shuffle_weights_polyphase(weight_host, s_w);

    // ------------------------------------------------------------------
    // (2) Pad the input with (K_p - 1) zeros on the left and right along W.
    //     This is the standard "same-padding" recipe for the polyphase
    //     sub-convs so that each phase produces exactly T_in outputs.
    // ------------------------------------------------------------------
    //
    // Input layout: NHWC = (N, 1, T_in, C_in).
    // Pad spec is per-dim {before, after}.
    ttnn::Tensor padded_input;
    if (k_p > 1) {
        ttnn::SmallVector<std::array<uint32_t, 2>> pad_spec = {
            {0, 0},          // N
            {0, 0},          // H
            {k_p - 1, k_p - 1},  // W
            {0, 0},          // C
        };
        padded_input = ttnn::pad(input_tensor, pad_spec, 0.0f, /*use_multicore=*/false);
    } else {
        padded_input = input_tensor;
    }

    const uint32_t padded_t = t_in + 2 * (k_p > 0 ? k_p - 1 : 0);
    log_debug(tt::LogOp, "polyphase padded input width: {}", padded_t);

    // ------------------------------------------------------------------
    // (3) Run S_w standard conv2ds. Each consumes the same padded input
    //     and a different per-phase weight slice. Output of each:
    //       (N, 1, T_phase, C_out)  where T_phase = padded_t - K_p + 1 = T_in + K_p - 1
    // ------------------------------------------------------------------
    std::vector<ttnn::Tensor> phase_outputs;
    phase_outputs.reserve(s_w);

    ttnn::Tensor first_weight_on_device;  // captured for return tuple
    std::optional<ttnn::Tensor> first_bias_on_device;

    for (uint32_t phase = 0; phase < s_w; ++phase) {
        ttnn::Tensor phase_w = slice_phase_weights(shuffled_weights, phase);

        // Bias only added to phase 0 (so total bias is added exactly once after interleave)
        std::optional<const ttnn::Tensor> bias_for_call =
            (phase == 0) ? bias_tensor : std::optional<const ttnn::Tensor>{};

        auto conv_result = ttnn::conv2d(
            padded_input,
            phase_w,
            device,
            in_channels,
            out_channels,
            batch_size,
            /*input_height=*/1,
            /*input_width=*/padded_t,
            /*kernel_size=*/std::array<uint32_t, 2>{1, k_p},
            /*stride=*/std::array<uint32_t, 2>{1, 1},
            /*padding=*/std::array<uint32_t, 2>{0, 0},
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

        // conv2d returns either a Tensor or a tuple depending on the last two flags.
        // We passed false/false so we know it's just the tensor variant.
        ttnn::Tensor phase_out = std::get<ttnn::Tensor>(conv_result);

        log_info(
            tt::LogOp,
            "[polyphase DBG] phase={} raw conv2d out: logical_shape={} layout={} is_sharded={}",
            phase,
            phase_out.logical_shape(),
            phase_out.layout(),
            phase_out.memory_config().is_sharded());

        // conv2d typically returns a TILE-layout, sharded-memory tensor.
        // Reshape/concat on tile-layout tensors does NOT preserve logical
        // element order (the tile structure aliases the dims).  To make the
        // downstream interleave operate on element-major data, convert to
        // ROW_MAJOR layout in interleaved DRAM memory before we touch it.
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
        log_info(
            tt::LogOp,
            "[polyphase DBG] phase={} after layout: logical_shape={} layout={} is_sharded={}",
            phase,
            phase_out.logical_shape(),
            phase_out.layout(),
            phase_out.memory_config().is_sharded());
        phase_outputs.push_back(std::move(phase_out));

        if (phase == 0) {
            first_weight_on_device = phase_w;
            first_bias_on_device = bias_tensor;
        }
    }

    // ------------------------------------------------------------------
    // (4) Interleave the S_w phase outputs into the final (N, 1, T_out, C_out) tensor.
    //
    //     We stack along a NEW phase axis, then transpose so the phase axis
    //     becomes the innermost-after-W axis, then reshape to merge phase*W
    //     into a single time dimension.
    //
    //     phase_outputs[p] : (N, 1, T_phase, C_out)
    //
    //     Step A: concat along a new leading dim ->  (S_w, N, 1, T_phase, C_out)
    //             (implemented as: unsqueeze each + concat dim=0)
    //     Step B: transpose so we end up with (N, 1, T_phase, S_w, C_out)
    //             We want time = phase + S_w * i  -> innermost time is phase changes fastest.
    //             So order along the W axis becomes:
    //                 [i=0,p=0], [i=0,p=1], ..., [i=0,p=S_w-1], [i=1,p=0], ...
    //             which requires phase to be the FAST axis -- yes, p innermost on W.
    //             We transpose the phase axis to be just before C.
    //     Step C: reshape (N, 1, T_phase * S_w, C_out)
    //     Step D: slice to (N, 1, T_out, C_out)
    //
    //     V1 uses straightforward host-friendly ops; can be fused later (V2).
    // ------------------------------------------------------------------
    //
    // To keep V1 simple, we use a simpler interleave: reshape each phase
    // output to (N*T_phase, C_out), stack to (N*T_phase, S_w, C_out),
    // reshape to (N*T_phase*S_w, C_out), then reshape to (N, 1, T_phase*S_w, C_out).
    // Finally slice to (N, 1, T_out, C_out).
    //
    // Actually the cleanest sequence (operating on (N, 1, T_phase, C_out) shapes):
    //   1) reshape each phase to (N, T_phase, 1, C_out)   [insert phase axis after T]
    //   2) concat all phases along dim=2 ->  (N, T_phase, S_w, C_out)
    //   3) reshape to (N, 1, T_phase * S_w, C_out)
    //   4) slice along W: [:, :, 0:T_out, :]
    std::vector<ttnn::Tensor> reshaped_phases;
    reshaped_phases.reserve(s_w);
    for (auto& po : phase_outputs) {
        // (N, 1, T_phase, C_out) -> (N, T_phase, 1, C_out)
        const auto& s = po.logical_shape();
        TT_FATAL(s.rank() == 4, "phase output expected rank 4, got {}", s.rank());
        const uint32_t n = s[0];
        const uint32_t h = s[1];
        const uint32_t w = s[2];
        const uint32_t c = s[3];
        TT_FATAL(h == 1, "phase output expected H == 1, got {}", h);
        ttnn::Shape new_shape{n, w, 1u, c};
        reshaped_phases.push_back(ttnn::reshape(po, new_shape));
    }
    log_info(
        tt::LogOp,
        "[polyphase DBG] before concat: each reshaped phase shape (rank {})",
        reshaped_phases[0].logical_shape().rank());
    log_info(tt::LogOp, "[polyphase DBG] reshaped_phases[0].logical_shape() = {}", reshaped_phases[0].logical_shape());

    // (N, T_phase, S_w, C_out)
    ttnn::Tensor stacked = ttnn::concat(reshaped_phases, /*dim=*/2);
    log_info(tt::LogOp, "[polyphase DBG] after concat: logical_shape = {}", stacked.logical_shape());

    // (N, 1, T_phase * S_w, C_out)
    const auto& ss = stacked.logical_shape();
    TT_FATAL(ss.rank() == 4, "stacked tensor expected rank 4, got {}", ss.rank());
    const uint32_t n = ss[0];
    const uint32_t merged_t = ss[1] * ss[2];
    const uint32_t c = ss[3];
    ttnn::Tensor flattened = ttnn::reshape(stacked, ttnn::Shape{n, 1u, merged_t, c});

    // Slice to exactly T_out elements along W
    ttnn::Tensor output;
    if (merged_t > t_out) {
        const std::array<uint32_t, 4> begins = {0u, 0u, 0u, 0u};
        const std::array<uint32_t, 4> ends = {n, 1u, t_out, c};
        const std::array<uint32_t, 4> steps = {1u, 1u, 1u, 1u};
        output = ttnn::slice(flattened, begins, ends, steps);
    } else {
        TT_FATAL(merged_t == t_out, "merged_t {} != t_out {}", merged_t, t_out);
        output = flattened;
    }

    return ConvTranspose2dResult{output, /*OH=*/1, /*OW=*/t_out, first_weight_on_device, first_bias_on_device};
}

}  // namespace ttnn::operations::conv::conv_transpose2d
