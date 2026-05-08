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
                * :attr:`input_imag` (optional): Float32 or BFloat16
                  ROW_MAJOR tensor — same shape, dtype, layout as
                  ``input_real``. When supplied, the input is treated as
                  the complex signal ``input_real + i * input_imag``;
                  when omitted, the imaginary part is taken to be zero.

            Returns:
                Tuple ``(real, imag)`` of Tensors.

            Examples::

                # Real input (most common):
                spec_re, spec_im = ttnn.fft(x_real)

                # Complex input — e.g. when chaining FFT after another
                # operation that already produced a (real, imag) pair:
                spec_re, spec_im = ttnn.fft(x_real, x_imag)
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
            nb::arg("input_real").noconvert()},
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_real,
               const ttnn::Tensor& input_imag) {
                return self(input_real, input_imag);
            },
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert()});

    const auto* ifft_doc =
        R"doc(
            1-D Inverse Fast Fourier Transform.

            Reverses :func:`ttnn.fft`. Takes the (real, imag) halves of a
            spectrum, returns the (real, imag) of the reconstructed signal
            scaled by 1/N. For a real input ``x``::

                spec_re, spec_im = ttnn.fft(x)
                rec_re,  rec_im  = ttnn.ifft(spec_re, spec_im)
                # rec_re == x  (within fp32 noise);  rec_im ~ 0

            Phase 1 support matrix matches :func:`ttnn.fft`.

            Args:
                * :attr:`spectrum_real`: Float32 ROW_MAJOR tensor, real part.
                * :attr:`spectrum_imag`: Float32 ROW_MAJOR tensor, imag part.
                                          Same shape/dtype/layout as
                                          ``spectrum_real``.

            Returns:
                Tuple ``(real, imag)`` of Tensors, same shape as the inputs.
        )doc";

    using IFFTType = decltype(ttnn::ifft);
    bind_registered_operation(
        mod,
        ttnn::ifft,
        ifft_doc,
        ttnn::nanobind_overload_t{
            [](const IFFTType& self,
               const ttnn::Tensor& spectrum_real,
               const ttnn::Tensor& spectrum_imag) {
                return self(spectrum_real, spectrum_imag);
            },
            nb::arg("spectrum_real").noconvert(),
            nb::arg("spectrum_imag").noconvert()});
}

}  // namespace ttnn::operations::experimental::fft::detail
