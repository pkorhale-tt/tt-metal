// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::fft_radix_pass — fused batched
// length-P FFT + optional post-twiddle complex multiply.  Single
// dispatch, single ProgramDescriptor, trace-safe.

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "fft_radix_pass_device_operation_types.hpp"
#include "fft_radix_pass_factory.hpp"

namespace ttnn::experimental::prim {

struct FftRadixPassDeviceOperation {
    using operation_attributes_t = FftRadixPassParams;
    using tensor_args_t          = FftRadixPassTensorArgs;

    // 2-tuple (real, imag) — same rationale as FFTDeviceOperation
    // (tt_stl reflection has no specialization for std::pair).
    using spec_return_value_t   = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    using program_factory_t = std::variant<FftRadixPassFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Public entry point used by the op-API layer (fft_radix_pass.cpp).
//   P            : FFT length per row (= last dim of input, pow-2 in [2, 1024])
//   twiddle_N2   : 0 → pure FFT.
//                  >0 → after FFT, multiply each row by twiddle row
//                       (row_idx % twiddle_N2) of T[n2, k] = exp(
//                       -2πi · n2 · k / (P · twiddle_N2)).  Used by the
//                       Pass-1 step of the two-pass / K-pass composite.
std::tuple<Tensor, Tensor> fft_radix_pass(
    const Tensor& input_real,
    const std::optional<Tensor>& input_imag,
    uint32_t P,
    uint32_t twiddle_N2);

}  // namespace ttnn::prim
