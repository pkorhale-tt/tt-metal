// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_nanobind.hpp"

#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"

namespace ttnn::operations::experimental::fft::detail {

void bind_experimental_fft_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            1-D Fast Fourier Transform (forward).

            Computes the discrete Fourier transform of the last dimension of
            ``input_real``. The input is treated as a real-valued signal of
            length ``N = input_real.shape[-1]``; leading dimensions are batched.

            Returns a pair ``(real, imag)`` of tensors with the same shape as
            the input, holding the natural-order complex spectrum
            ``X[0], X[1], ..., X[N-1]``.

            Phase 1 support matrix:

            +---------------+--------------------+----------------------+
            | dtype         | N                  | Backend              |
            +===============+====================+======================+
            | Float32       | pow2, N <= 1M      | fft_stockham (works) |
            +---------------+--------------------+----------------------+
            | Float32       | pow2, 1M < N <= 16M| fft_universal_xl     |
            |               |                    | (NOT YET WIRED)      |
            +---------------+--------------------+----------------------+
            | Float32       | non-pow2           | fft_universal        |
            |               |                    | (NOT YET WIRED)      |
            +---------------+--------------------+----------------------+
            | BFloat16      | any N              | fft_universal_bf16   |
            |               |                    | (NOT YET WIRED)      |
            +---------------+--------------------+----------------------+

            Equivalent PyTorch:

            .. code-block:: python

                X = torch.fft.fft(input_real)   # complex64
                real, imag = X.real, X.imag

            Args:
                * :attr:`input_real`: Float32 or BFloat16 ROW_MAJOR tensor.

            Returns:
                Tuple ``(real, imag)`` of Tensors.
        )doc";

    using OperationType = decltype(ttnn::fft);
    bind_registered_operation(
        mod,
        ttnn::fft,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_real) {
                return self(input_real);
            },
            nb::arg("input_real").noconvert()});
}

}  // namespace ttnn::operations::experimental::fft::detail
