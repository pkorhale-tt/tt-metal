// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_nanobind.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"

namespace ttnn::operations::experimental::fft::detail {

namespace {
// String → FFTPrecision (case-sensitive, lower-case for ergonomics).
// Anything else throws ValueError on the Python side.
inline FFTPrecision parse_precision(const std::string& s) {
    if (s == "precise") return FFTPrecision::Precise;
    if (s == "fast")    return FFTPrecision::Fast;
    throw std::invalid_argument(
        "ttnn.experimental.fft / ttnn.experimental.ifft: precision must be 'precise' or 'fast' "
        "(got '" + s + "').");
}
}  // namespace

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
                * :attr:`precision` (str, default ``"precise"``):
                  ``"precise"`` → SFPU true-fp32 path (matches
                  ``torch.fft`` precision; round-trip ~1e-7).
                  ``"fast"`` → FPU bf16-mantissa matmul (~10-30× faster
                  for small N but ~1e-3 round-trip). Only meaningful for
                  Float32 + non-pow2 N; ignored everywhere else.

            Returns:
                Tuple ``(real, imag)`` of Tensors.

            Examples::

                # Real input (most common; precise default matches torch):
                spec_re, spec_im = ttnn.experimental.fft(x_real)

                # Opt into the fast (bf16-mantissa) path for small N:
                spec_re, spec_im = ttnn.experimental.fft(x_real, precision="fast")

                # Complex input — e.g. when chaining FFT after another
                # operation that already produced a (real, imag) pair:
                spec_re, spec_im = ttnn.experimental.fft(x_real, x_imag)
        )doc";

    using OperationType = decltype(ttnn::experimental::fft);
    bind_registered_operation(
        mod,
        ttnn::experimental::fft,
        doc,
        // 1-arg (real input). precision defaults to "precise".
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_real,
               const std::string&  precision) {
                return self(input_real, parse_precision(precision));
            },
            nb::arg("input_real").noconvert(),
            nb::arg("precision") = std::string("precise")},
        // 2-arg (complex input). precision defaults to "precise".
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_real,
               const ttnn::Tensor& input_imag,
               const std::string&  precision) {
                return self(input_real, input_imag, parse_precision(precision));
            },
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert(),
            nb::arg("precision") = std::string("precise")});

    const auto* ifft_doc =
        R"doc(
            1-D Inverse Fast Fourier Transform.

            Reverses :func:`ttnn.experimental.fft`. Takes the (real, imag) halves of a
            spectrum, returns the (real, imag) of the reconstructed signal
            scaled by 1/N. For a real input ``x``::

                spec_re, spec_im = ttnn.experimental.fft(x)
                rec_re,  rec_im  = ttnn.experimental.ifft(spec_re, spec_im)
                # rec_re == x  (within fp32 noise);  rec_im ~ 0

            Phase 1 support matrix matches :func:`ttnn.experimental.fft`.

            Args:
                * :attr:`spectrum_real`: Float32 ROW_MAJOR tensor, real part.
                * :attr:`spectrum_imag`: Float32 ROW_MAJOR tensor, imag part.
                                          Same shape/dtype/layout as
                                          ``spectrum_real``.
                * :attr:`precision` (str, default ``"precise"``): same
                  selector as ``ttnn.experimental.fft``. See the forward op's docstring
                  for the trade-off.

            Returns:
                Tuple ``(real, imag)`` of Tensors, same shape as the inputs.
        )doc";

    using IFFTType = decltype(ttnn::experimental::ifft);
    bind_registered_operation(
        mod,
        ttnn::experimental::ifft,
        ifft_doc,
        ttnn::nanobind_overload_t{
            [](const IFFTType& self,
               const ttnn::Tensor& spectrum_real,
               const ttnn::Tensor& spectrum_imag,
               const std::string&  precision) {
                return self(spectrum_real, spectrum_imag,
                            parse_precision(precision));
            },
            nb::arg("spectrum_real").noconvert(),
            nb::arg("spectrum_imag").noconvert(),
            nb::arg("precision") = std::string("precise")});
}

}  // namespace ttnn::operations::experimental::fft::detail
