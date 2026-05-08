// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FFT program factory — full-backend dispatcher.
//
// Routes each call of `ttnn::experimental::fft / ifft` to one of four
// device-resident orchestrators originally developed under
// tt_metal/programming_examples/. Selection is by (dtype, N, is_pow2):
//
//     fp32 + pow2  + N <= 1M    →  fft_stockham         (4-pass Stockham)
//     fp32 + pow2  + N <= 16M   →  fft_universal_xl     (2-level Cooley–Tukey)
//     fp32 + non-pow2           →  fft_universal        (mixed-radix / Bluestein)
//     bf16 + any N              →  fft_universal_bf16   (true-bf16 FPU matmul)
//
// The orchestrators are header-only inline modules; including their
// `*_host.cpp` files pulls in the kernel-launch code at translation-unit
// scope and keeps every kernel reference (`CreateKernel("...")`) pointing
// at the existing programming_examples source tree. Migration of those
// kernel paths into ttnn/cpp/.../fft/device/kernels/ is staged in a
// follow-up PR; the kernels are already copied there (Phase 2-A).
//
// This file is "host-orchestrated, device-executed" — every FFT pass
// runs on Tensix cores via the orchestrator's MeshWorkload enqueues.
// We funnel through the orchestrator's host API rather than building
// one big fused Program because:
//   * the orchestrators already implement every algorithm correctness-
//     tested end-to-end in the programming_examples;
//   * a fused single-Program rewrite (Phase 3) is a large, separate
//     undertaking that should not block landing the public ttnn API.
//
// IFFT path: y = conj(fft(conj(X))) / N — applied per-row on host
// around the forward backend call. No new device code required.

#include "fft_program_factory.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>

// Backend orchestrators — header-only inline modules. Each has #pragma once;
// transitive includes (notably fft_stockham) collapse cleanly.
#include "tt_metal/programming_examples/fft_stockham/fft_stockham_host.cpp"
#include "tt_metal/programming_examples/fft_universal/fft_universal_host.cpp"
#include "tt_metal/programming_examples/fft_universal_xl/fft_universal_xl_host.cpp"
#include "tt_metal/programming_examples/fft_universal_bf16/fft_universal_bf16_host.cpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ttnn::experimental::prim {

namespace {

using Complex  = std::complex<float>;
using DataType = tt::tt_metal::DataType;
using tt::tt_metal::distributed::MeshDevice;

// Renamed `is_pow2_local` (rather than `is_pow2`) so the Unity build can
// merge this TU with fft_device_operation.cpp — which has its own
// anonymous-namespace `is_pow2` — without ODR collision.
constexpr bool is_pow2_local(uint32_t n) {
    return n != 0u && (n & (n - 1u)) == 0u;
}

// Pick + invoke the right backend for a single length-N row.
// Mirrors `select_backend()` in fft_device_operation.cpp; that one
// rejects unsupported (dtype, N) at validate time, so by the time we
// reach here every branch is known-good.
std::vector<Complex> fft_one_row(
    std::shared_ptr<MeshDevice>& md,
    DataType                     dtype,
    const std::vector<Complex>&  signal) {

    if (dtype == DataType::BFLOAT16) {
        return fft_universal_bf16::fft(md, signal);
    }
    // Float32
    const uint32_t N = static_cast<uint32_t>(signal.size());
    if (!is_pow2_local(N))           return fft_universal::fft(md, signal);
    if (N <= 1u * 1024u * 1024u)     return fft_stockham::fft(md, signal);
    return fft_universal_xl::fft(md, signal);
}

// ── Tensor I/O helpers (handle fp32 ↔ bf16 at the host boundary) ────────────

// Read a Tensor's full payload as a flat fp32 vector. For BFLOAT16 inputs
// we explicitly call `to_vector<bfloat16>` and widen on host so we avoid
// any silent cast that the templated `to_vector<float>` overload would
// otherwise reject.
std::vector<float> read_real_as_fp32(const Tensor& t) {
    if (t.dtype() == DataType::BFLOAT16) {
        const auto buf = t.to_vector<bfloat16>();
        std::vector<float> out(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) {
            out[i] = static_cast<float>(buf[i]);
        }
        return out;
    }
    return t.to_vector<float>();
}

// Build an output Tensor matching `spec` (dtype/shape/layout/memory) from
// a host fp32 buffer, narrowing to bf16 if the spec requires it. The
// returned tensor lives on `device`.
Tensor write_real_with_spec(
    std::vector<float>&&     buf,
    const ttnn::TensorSpec&  spec,
    MeshDevice*              device) {

    if (spec.data_type() == DataType::BFLOAT16) {
        std::vector<bfloat16> bf(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) {
            bf[i] = bfloat16(buf[i]);
        }
        return Tensor::from_vector(std::move(bf), spec, device);
    }
    return Tensor::from_vector(std::move(buf), spec, device);
}

// Drives the per-row FFT loop and replaces the output tensors with new
// device tensors holding the (real, imag) spectrum halves.
//
// We replace the outputs (rather than writing into the buffers allocated
// by `create_output_tensors`) because the orchestrators each manage
// their own DRAM buffers and return a host vector; the framework-supplied
// blank tensors become unreferenced and are freed by RAII.
//
// Phase 2 limitation: drops the per-shard TensorTopology that the
// device_operation layer imputes onto the blank outputs. We currently
// only test on single-device dispatch, so the loss is invisible to
// callers; a Phase-3 fused-Program rewrite will preserve the originals.
void run_backend_fft(
    const FFTParams&            attrs,
    const FFTTensorArgs&        tensor_args,
    std::pair<Tensor, Tensor>&  tensor_return_value) {

    const auto& in_re_tensor = tensor_args.input_real;
    const auto& shape        = in_re_tensor.logical_shape();
    TT_FATAL(shape.size() >= 1u, "fft: tensor has rank 0");

    const uint32_t N     = shape[-1];
    const uint64_t total = in_re_tensor.logical_volume();
    TT_FATAL(total % N == 0u,
             "fft: total volume {} not divisible by N {}.", total, N);
    const uint64_t batches = total / N;

    const auto dtype = in_re_tensor.dtype();

    // 1. Materialise host-side fp32 inputs.
    const std::vector<float> in_re = read_real_as_fp32(in_re_tensor);
    std::vector<float>       in_im;
    if (attrs.inverse) {
        TT_FATAL(tensor_args.input_imag.has_value(),
                 "fft (inverse): input_imag is required.");
        in_im = read_real_as_fp32(*tensor_args.input_imag);
        TT_FATAL(in_im.size() == in_re.size(),
                 "fft (inverse): real / imag size mismatch ({} vs {}).",
                 in_re.size(), in_im.size());
    } else {
        in_im.assign(in_re.size(), 0.0f);
    }
    TT_FATAL(in_re.size() == total,
             "fft: read returned {} elements, expected {}.",
             in_re.size(), total);

    // 2. Wrap the device pointer in a no-op-deleter shared_ptr — the
    //    orchestrators all take `shared_ptr<MeshDevice>`, but we don't
    //    own the lifetime here (the tensor does).
    auto* device_raw = in_re_tensor.device();
    auto md = std::shared_ptr<MeshDevice>(
        device_raw, [](MeshDevice*){});

    std::vector<float>   out_re(total);
    std::vector<float>   out_im(total);
    std::vector<Complex> work(N);

    // IFFT via conjugate trick: y = conj(fft(conj(X))) / N.
    const float scale = attrs.inverse ? (1.0f / static_cast<float>(N)) : 1.0f;

    // 3. Per-row dispatch through the selected backend.
    for (uint64_t b = 0u; b < batches; ++b) {
        const uint64_t off = b * static_cast<uint64_t>(N);

        if (attrs.inverse) {
            for (uint32_t i = 0u; i < N; ++i) {
                work[i] = Complex{in_re[off + i], -in_im[off + i]};
            }
        } else {
            for (uint32_t i = 0u; i < N; ++i) {
                work[i] = Complex{in_re[off + i], in_im[off + i]};
            }
        }

        const auto X = fft_one_row(md, dtype, work);

        if (attrs.inverse) {
            for (uint32_t k = 0u; k < N; ++k) {
                out_re[off + k] =  X[k].real() * scale;
                out_im[off + k] = -X[k].imag() * scale;
            }
        } else {
            for (uint32_t k = 0u; k < N; ++k) {
                out_re[off + k] = X[k].real();
                out_im[off + k] = X[k].imag();
            }
        }
    }

    // 4. Replace outputs with fresh device tensors. compute_output_specs
    //    guarantees output spec mirrors input spec (dtype/shape/layout).
    const auto& spec = in_re_tensor.tensor_spec();
    tensor_return_value.first  = write_real_with_spec(std::move(out_re), spec, device_raw);
    tensor_return_value.second = write_real_with_spec(std::move(out_im), spec, device_raw);
}

}  // namespace

FFTProgramFactory::cached_program_t FFTProgramFactory::create(
    const FFTParams&            operation_attributes,
    const FFTTensorArgs&        tensor_args,
    std::pair<Tensor, Tensor>&  tensor_return_value) {

    // Run the dispatched backend. The orchestrator handles its own
    // EnqueueMeshWorkload + read-back, so by the time this returns the
    // spectrum is already populated on `tensor_return_value`.
    run_backend_fft(operation_attributes, tensor_args, tensor_return_value);

    // Empty outer Program — the FFT work happens inside the orchestrator
    // calls above, not in a Program owned by this factory.
    tt::tt_metal::Program program{};
    const uint32_t N = tensor_args.input_real.logical_shape()[-1];

    return cached_program_t{
        std::move(program),
        FFTSharedVariables{
            .kernel_ids = {},
            .cores      = {},
            .N          = N,
        }};
}

void FFTProgramFactory::override_runtime_arguments(
    cached_program_t&           /*cached_program*/,
    const FFTParams&            operation_attributes,
    const FFTTensorArgs&        tensor_args,
    std::pair<Tensor, Tensor>&  tensor_return_value) {
    // Cache-hit path — re-dispatch the backend. The cached "program" is
    // empty; there are no kernel runtime args to update.
    run_backend_fft(operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim

// =====================================================================
// PHASE 3 TODO — single fused on-device Program
// =====================================================================
// The current factory is "host-orchestrated, device-executed": each
// length-N row runs through one of four orchestrators that own their
// own MeshWorkloads and command-queue enqueues. This is correct and
// fast enough for the public API to land, but it has two costs:
//
//   1. B device dispatches per call for a [B, ..., N] tensor; a fused
//      Program could batch the outer dimensions into one workload.
//   2. The orchestrators' kernel paths still resolve to
//      tt_metal/programming_examples/fft*/kernel/...; a self-contained
//      ttnn op should resolve to the in-tree copies under
//      ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/
//      (already staged in Phase 2-A).
//
// Phase 3 work:
//   * Refactor each orchestrator's `fft()` into
//     `build_<backend>_program(md, N, in_buf, out_re_buf, out_im_buf)
//        -> {Program, std::vector<KernelHandle>, std::vector<CoreCoord>}`
//     emitting one Program per call, no internal enqueues.
//   * Retarget all CreateKernel(...) paths to the in-tree kernels dir.
//   * Wire override_runtime_arguments() to update buffer addresses on
//     program-cache hits (currently a re-dispatch, which is fine but
//     wastes the cached Program object).
//   * Inverse path can stay as the conjugate trick or be inlined as a
//     short SFPU pass appended to the writer kernel.
