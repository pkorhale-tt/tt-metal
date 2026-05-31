// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fft_nanobind.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"
#include "ttnn/operations/experimental/fft/apply_twiddles.hpp"
#include "ttnn/operations/experimental/fft/apply_twiddles_xl.hpp"
#include "ttnn/operations/experimental/fft/transpose_rm.hpp"
#include "ttnn/operations/experimental/fft/fft_radix_pass.hpp"

namespace ttnn::operations::experimental::fft_binding::detail {

namespace {

using PrimPrecision = ttnn::experimental::prim::FFTPrecision;

// String → FFTPrecision (case-sensitive). Anything else → ValueError on the Python side.
PrimPrecision parse_precision(const std::string& s) {
    if (s == "precise") {
        return PrimPrecision::Precise;
    }
    if (s == "fast") {
        return PrimPrecision::Fast;
    }
    throw std::invalid_argument(
        "ttnn.experimental.fft / ttnn.experimental.ifft: precision must be 'precise' or 'fast' "
        "(got '" + s + "').");
}

// ---- Trampolines (stateless lambdas decay to plain function pointers,
//      which is what ttnn::bind_function / overload_t expects) ----

using TensorPair = std::tuple<ttnn::Tensor, ttnn::Tensor>;

TensorPair fft_real_trampoline(const ttnn::Tensor& input_real, std::string precision) {
    return ttnn::operations::experimental::fft(input_real, parse_precision(precision));
}

TensorPair fft_complex_trampoline(
    const ttnn::Tensor& input_real, const ttnn::Tensor& input_imag, std::string precision) {
    return ttnn::operations::experimental::fft(input_real, input_imag, parse_precision(precision));
}

TensorPair ifft_trampoline(
    const ttnn::Tensor& spectrum_real, const ttnn::Tensor& spectrum_imag, std::string precision) {
    return ttnn::operations::experimental::ifft(spectrum_real, spectrum_imag, parse_precision(precision));
}

TensorPair apply_twiddles_trampoline(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t N1,
    uint32_t N2) {
    return ttnn::operations::experimental::apply_twiddles(input_real, input_imag, N1, N2);
}

TensorPair apply_twiddles_xl_trampoline(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t P,
    uint32_t big_modulus,
    uint32_t full_N) {
    return ttnn::operations::experimental::apply_twiddles_xl(
        input_real, input_imag, P, big_modulus, full_N);
}

ttnn::Tensor transpose_rm_trampoline(const ttnn::Tensor& input) {
    return ttnn::operations::experimental::transpose_rm(input);
}

// fft_radix_pass has two overloads (with / without imag input) — bind
// each as its own trampoline so nanobind picks the right one based on
// the call signature.
TensorPair fft_radix_pass_real_trampoline(
    const ttnn::Tensor& input_real,
    uint32_t P,
    uint32_t twiddle_N2,
    uint32_t stride) {
    return ttnn::operations::experimental::fft_radix_pass(
        input_real, std::nullopt, P, twiddle_N2, stride);
}

TensorPair fft_radix_pass_complex_trampoline(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t P,
    uint32_t twiddle_N2,
    uint32_t stride) {
    return ttnn::operations::experimental::fft_radix_pass(
        input_real, input_imag, P, twiddle_N2, stride);
}

TensorPair fft_three_pass_trampoline(
    const ttnn::Tensor& input_real,
    uint32_t full_N,
    std::string precision) {
    return ttnn::operations::experimental::fft_three_pass(
        input_real, full_N, parse_precision(precision));
}

}  // namespace

void bind_experimental_fft_operation(nb::module_& mod) {
    const auto* fft_doc =
        R"doc(
            1-D Fast Fourier Transform (forward).

            Computes the discrete Fourier transform of the last dimension of
            ``input_real``. Leading dimensions are batched. Returns a pair
            ``(real, imag)`` of tensors with the same shape as the input,
            holding the natural-order complex spectrum
            ``X[0], X[1], ..., X[N-1]``.

            Equivalent PyTorch:

            .. code-block:: python

                X = torch.fft.fft(input_real)   # complex64
                real, imag = X.real, X.imag

            Args:
                * :attr:`input_real`: Float32 or BFloat16 ROW_MAJOR tensor.
                * :attr:`input_imag` (optional): same shape, dtype, layout as
                  ``input_real``. When supplied, the input is treated as the
                  complex signal ``input_real + i * input_imag``; when omitted,
                  the imaginary part is taken to be zero.
                * :attr:`precision` (str, default ``"precise"``):
                  ``"precise"`` → SFPU true-fp32 path (matches ``torch.fft``
                  precision; round-trip ~1e-7).
                  ``"fast"`` → FPU bf16-mantissa matmul (faster for small N
                  but ~1e-3 round-trip). Only meaningful for Float32 + non-pow2
                  N; ignored everywhere else.

            Returns:
                Tuple ``(real, imag)`` of Tensors.

            Examples::

                # Real input (most common; precise default matches torch):
                spec_re, spec_im = ttnn.experimental.fft(x_real)

                # Opt into the fast (bf16-mantissa) path for small N:
                spec_re, spec_im = ttnn.experimental.fft(x_real, precision="fast")

                # Complex input — chaining FFT after another op that already
                # produced a (real, imag) pair:
                spec_re, spec_im = ttnn.experimental.fft(x_real, x_imag)
        )doc";

    ttnn::bind_function<"fft", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        fft_doc,
        ttnn::overload_t(
            &fft_real_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("precision") = std::string("precise")),
        ttnn::overload_t(
            &fft_complex_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert(),
            nb::arg("precision") = std::string("precise")));

    const auto* ifft_doc =
        R"doc(
            1-D Inverse Fast Fourier Transform.

            Reverses :func:`ttnn.experimental.fft`. Takes the (real, imag)
            halves of a spectrum and returns the (real, imag) of the
            reconstructed signal scaled by 1/N. For a real input ``x``::

                spec_re, spec_im = ttnn.experimental.fft(x)
                rec_re,  rec_im  = ttnn.experimental.ifft(spec_re, spec_im)
                # rec_re == x  (within fp32 noise);  rec_im ~ 0

            Args:
                * :attr:`spectrum_real`: Float32 or BFloat16 ROW_MAJOR tensor, real part.
                * :attr:`spectrum_imag`: Same shape/dtype/layout as ``spectrum_real``.
                * :attr:`precision` (str, default ``"precise"``): same selector
                  as :func:`ttnn.experimental.fft`.

            Returns:
                Tuple ``(real, imag)`` of Tensors, same shape as the inputs.
        )doc";

    ttnn::bind_function<"ifft", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        ifft_doc,
        ttnn::overload_t(
            &ifft_trampoline,
            nb::arg("spectrum_real").noconvert(),
            nb::arg("spectrum_imag").noconvert(),
            nb::arg("precision") = std::string("precise")));

    const auto* apply_twiddles_doc =
        R"doc(
            Cooley-Tukey two-pass FFT between-pass elementwise complex multiply.

            Interprets the input ``(input_real, input_imag)`` as ``M`` rows of
            length ``N1`` (where ``M`` is the product of all leading dims) and
            multiplies each row by the corresponding row of the twiddle table

                T[n2, k1] = exp(-2*pi*i * n2 * k1 / (N1 * N2)),  n2 in [0, N2)

            broadcast over the leading-batch dimension as ``T[r % N2, :]``.

            This is a building block of the upcoming two-pass FFT composite;
            it is exposed for unit-testing the kernel in isolation and is not
            intended as a general-purpose user op.  Both ``N1`` and ``N2``
            must be powers of two with ``N1 in [2, 1024]`` and ``N2 in
            [1, 1024]``; ``M`` must be a multiple of ``N2``.

            Args:
                * :attr:`input_real`: Float32 or BFloat16 ROW_MAJOR tensor of
                  shape ``(..., N1)``.
                * :attr:`input_imag`: same shape / dtype / layout as
                  ``input_real``.
                * :attr:`N1`, :attr:`N2`: outer / inner factorisation of the
                  total FFT length ``N = N1 * N2``.

            Returns:
                Tuple ``(real, imag)`` of Tensors, same shape as the input.
        )doc";

    ttnn::bind_function<"apply_twiddles", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        apply_twiddles_doc,
        ttnn::overload_t(
            &apply_twiddles_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert(),
            nb::arg("N1"),
            nb::arg("N2")));

    const auto* apply_twiddles_xl_doc =
        R"doc(
            Large-modulus elementwise complex multiply:

                row_phase_idx = (row % big_modulus)
                out[r, k] = in[r, k] * exp(-2πi · row_phase_idx · k / full_N)

            Used by the fft_three_pass composite (N > 1M) as the between-
            pass-1-and-2 twiddle.  Unlike ``ttnn.experimental.apply_twiddles``
            (which caps at twiddle_N2 ≤ 1024), this op builds each twiddle
            row on-the-fly from a small per-(device, big_modulus, full_N)
            delta lookup, letting ``big_modulus`` scale to ``2^20``.

            Args:
                * :attr:`input_real`, :attr:`input_imag`: Float32 or BFloat16
                  ROW_MAJOR tensors of shape ``(..., P)``.  ``M`` = product
                  of leading dims, must be a multiple of ``big_modulus``.
                * :attr:`P`: row length, pow-2 in ``[2, 1024]``.
                * :attr:`big_modulus`: twiddle row modulus, pow-2 in
                  ``[1, 2^20]``.
                * :attr:`full_N`: angle denominator, pow-2, ``>= big_modulus``.

            Returns:
                Tuple ``(real, imag)`` of Tensors, same shape as the input.
        )doc";

    ttnn::bind_function<"apply_twiddles_xl", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        apply_twiddles_xl_doc,
        ttnn::overload_t(
            &apply_twiddles_xl_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert(),
            nb::arg("P"),
            nb::arg("big_modulus"),
            nb::arg("full_N")));

    const auto* transpose_rm_doc =
        R"doc(
            Precision-preserving inner-axis transpose for ROW_MAJOR
            fp32 / bf16 tensors.  Swaps the last two dims of ``input``
            (shape ``(..., A, C)`` → ``(..., C, A)``).

            Unlike the standard :func:`ttnn.transpose`, which silently
            downcasts to bf16 for ROW_MAJOR fp32 paths, this op is pure
            data movement (32×32 tile permute via the NoC) so fp32 input
            yields bit-exact fp32 output.

            Used as a building block in the two-pass FFT composite
            (commit 3c) where the inter-pass data layout demands
            full-precision transposition.

            Constraints:
                * Both ``A`` and ``C`` must be multiples of 32 and ≥ 32.
                * Layout must be ROW_MAJOR.
                * Dtype must be Float32 or BFloat16.
        )doc";

    ttnn::bind_function<"transpose_rm", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        transpose_rm_doc,
        ttnn::overload_t(
            &transpose_rm_trampoline,
            nb::arg("input").noconvert()));

    const auto* fft_radix_pass_doc =
        R"doc(
            Fused [batched length-P FFT  +  optional post-twiddle complex
            multiply].  Building block for the K-pass Cooley-Tukey
            composite that handles N > 1M.

            Interprets the input as ``M`` rows of length ``P`` (where ``M``
            is the product of all leading dims) and replaces each row
            with its length-``P`` DFT:

                y[r, k] = sum_{n=0}^{P-1} x[r, n] * exp(-2πi n k / P)

            If ``twiddle_N2 != 0``, the output is then multiplied by

                row_idx = (r / stride) % twiddle_N2       # stride defaults to 1
                y[r, k] *= exp(-2πi · row_idx · k / (P·twiddle_N2))

            which is exactly the between-pass twiddle of a two-pass FFT
            decomposition (``stride=1``) or the Pass-2 twiddle of a three-
            pass FFT (``stride=N3`` to pick the n1 twiddle without an
            extra transpose).  Passing ``twiddle_N2=0`` makes this a pure
            batched FFT (equivalent to ``ttnn.experimental.fft`` on the
            same input).

            The two-arg form passes only the real input (imag implicitly
            zero); the three-arg form takes a complex (real+imag) input.

            Constraints:
                * P pow-2 in [2, 1024]
                * M pow-2 and >= 1
                * twiddle_N2 == 0 or pow-2 in [1, 1024]
                * stride pow-2 in [1, M] dividing M, and (M/stride) % twiddle_N2 == 0
                * fp32 or bf16; ROW_MAJOR layout
        )doc";

    ttnn::bind_function<"fft_radix_pass", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        fft_radix_pass_doc,
        ttnn::overload_t(
            &fft_radix_pass_real_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("P"),
            nb::arg("twiddle_N2") = 0u,
            nb::arg("stride")     = 1u),
        ttnn::overload_t(
            &fft_radix_pass_complex_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("input_imag").noconvert(),
            nb::arg("P"),
            nb::arg("twiddle_N2") = 0u,
            nb::arg("stride")     = 1u));

    const auto* fft_three_pass_doc =
        R"doc(
            Three-pass Cooley–Tukey composite FFT for very large N
            (2^20 < ``full_N`` ≤ 2^30).

            ⚠ **Pre-shaped input required**: the input must already be
            shaped as ``(B·N1·N2, N3)`` (or with extra leading batch dims
            collapsed; last two dims = ``(N1·N2, N3)``).  The
            factorization is auto-picked from ``full_N`` via "max-N3 then
            balance N1/N2"; both ``N1, N2, N3`` are pow-2 in ``[32, 1024]``.

            Why: the implicit ``(B, full_N) → (B·N1·N2, N3)`` reshape
            would require streaming an ``full_N``-element row through one
            CB tile per core, which blows L1 for ``full_N > ~256K``.
            Callers do the equivalent ``torch.view(B·N1·N2, N3)`` on the
            host (a metadata-only torch view) before ``ttnn.from_torch``,
            so the device buffer is allocated with the small per-row
            page_size from the start.

            Output: returned in the factored shape ``(B·N1, N2, N3)``.
            Recover natural-order ``(B, full_N)`` via
            ``to_torch().reshape(B, full_N)`` on the host (cheap — the
            FFT chain already arranges ``k = k1·N2·N3 + k2·N3 + k3``
            naturally, so the host reshape is a torch view).

            Args:
                * :attr:`input_real`: Float32 or BFloat16 ROW_MAJOR tensor.
                  Shape must end in ``(N1·N2, N3)`` for the
                  ``pick_three_factorization(full_N)`` factorization.
                * :attr:`full_N`: logical FFT length.  Pow-2 in
                  ``[2^21, 2^30]``.  Must factor as ``N1·N2·N3`` with each
                  factor pow-2 in ``[32, 1024]``.
                * :attr:`precision` (default ``"precise"``): same as
                  :func:`ttnn.experimental.fft`.

            Returns:
                Tuple ``(real, imag)`` of Tensors, shape ``(B·N1, N2, N3)``.

            Example::

                N = 1 << 21
                N1, N2, N3 = pick_three_factorization(N)   # (64, 32, 1024)
                x = torch.randn(1, N).view(N1 * N2, N3)    # host view, no copy
                tt_x = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                re, im = ttnn.experimental.fft_three_pass(tt_x, full_N=N)
                out = (torch.complex(ttnn.to_torch(re), ttnn.to_torch(im))
                       .reshape(1, N))
        )doc";

    ttnn::bind_function<"fft_three_pass", ttnn::unique_string{"ttnn.experimental."}>(
        mod,
        fft_three_pass_doc,
        ttnn::overload_t(
            &fft_three_pass_trampoline,
            nb::arg("input_real").noconvert(),
            nb::arg("full_N"),
            nb::arg("precision") = std::string("precise")));
}

}  // namespace ttnn::operations::experimental::fft_binding::detail
