# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
A/B comparison tests for ttnn.experimental.conv_transpose2d_polyphase.

Compares against:
  1. torch.nn.functional.conv_transpose1d  (the correctness oracle)
  2. ttnn.conv_transpose2d                  (the existing path; may xfail for
                                              very large kernels due to the
                                              NOC_MAX_BURST_SIZE static_assert)

Run:
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    pytest tests/ttnn/unit_tests/operations/conv/test_conv_transpose2d_polyphase.py -v -s
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc


# (batch, C_in, C_out, K, S, T_in, label, expect_current_works)
SHAPES = [
    # Tiny shape -- both paths should work, sanity check correctness.
    (1, 1, 1, 8, 4, 8, "tiny_k8_s4", True),
    # Vocoder-style shape -- both work, polyphase should be faster.
    (1, 16, 16, 16, 8, 32, "vocoder_k16_s8", True),
    # iSTFT envelope -- current path FAILS to compile (NOC burst limit).
    (1, 1, 1, 640, 160, 96, "istft_envelope", False),
    # iSTFT cos/sin -- current path FAILS to compile.
    (1, 321, 1, 640, 160, 96, "istft_cos_sin", False),
]


def _build_inputs(batch, c_in, c_out, k, t_in, dtype=torch.float32):
    """Random reproducible torch tensors. Returns (x_torch, w_torch)."""
    torch.manual_seed(0)
    x = torch.randn(batch, c_in, t_in, dtype=dtype)
    # IOHW for ttnn: [C_in, C_out, K_h=1, K_w]
    w = torch.randn(c_in, c_out, 1, k, dtype=dtype)
    return x, w


def _torch_reference(x, w, stride):
    """PyTorch reference using conv_transpose1d. w is IOHW [C_in, C_out, 1, K]."""
    w1d = w.squeeze(2)  # [C_in, C_out, K]
    return F.conv_transpose1d(x, w1d, stride=stride)


def _to_ttnn(x_torch, w_torch, device):
    """Convert (N, C, T) torch -> (N, 1, T, C) ttnn NHWC; w stays IOHW."""
    # Input NHWC: unsqueeze H=1 then move channels last
    x_nhwc = x_torch.unsqueeze(2).permute(0, 2, 3, 1).contiguous()  # (N, 1, T, C)
    x_tt = ttnn.from_torch(
        x_nhwc,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_tt = ttnn.from_torch(w_torch, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return x_tt, w_tt


def _from_ttnn(out_tt):
    """Convert (N, 1, T, C) ttnn -> (N, C, T) torch."""
    out = ttnn.to_torch(out_tt)  # (N, 1, T, C)
    if out.dim() == 4:
        return out.squeeze(1).permute(0, 2, 1).contiguous()  # (N, C, T)
    return out


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize("batch,c_in,c_out,k,s,t_in,label,current_works", SHAPES)
def test_polyphase_correctness(device, batch, c_in, c_out, k, s, t_in, label, current_works):
    """Verify polyphase output matches PyTorch reference (PCC > 0.99)."""
    x_torch, w_torch = _build_inputs(batch, c_in, c_out, k, t_in)
    ref = _torch_reference(x_torch, w_torch, stride=s)  # (N, C_out, T_out)

    x_tt, w_tt = _to_ttnn(x_torch, w_torch, device)

    out_tt = ttnn.experimental.conv_transpose2d_polyphase(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=c_in,
        out_channels=c_out,
        batch_size=batch,
        input_height=1,
        input_width=t_in,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, 0),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        mirror_kernel=True,
    )

    out_torch = _from_ttnn(out_tt)
    assert out_torch.shape == ref.shape, f"shape mismatch: got {out_torch.shape}, want {ref.shape}"

    if label == "tiny_k8_s4":
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
        print(f"\n[DBG {label}] x_torch          = {x_torch.flatten().tolist()}")
        print(f"[DBG {label}] w_torch          = {w_torch.flatten().tolist()}")
        print(f"[DBG {label}] reference y[0..]={ref.flatten().tolist()}")
        print(f"[DBG {label}] polyphase y[0..]={out_torch.flatten().tolist()}")
        diff = (ref - out_torch).abs().flatten()
        print(f"[DBG {label}] abs diff         = {diff.tolist()}")
        print(f"[DBG {label}] ratio (poly/ref) = {(out_torch.flatten() / ref.flatten()).tolist()}")

    pcc_ok, pcc_msg = comp_pcc(ref, out_torch, pcc=0.99)
    assert pcc_ok, f"[{label}] polyphase PCC failed: {pcc_msg}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize("batch,c_in,c_out,k,s,t_in,label,current_works", SHAPES)
def test_polyphase_vs_current(device, batch, c_in, c_out, k, s, t_in, label, current_works):
    """Side-by-side: polyphase vs the existing ttnn.conv_transpose2d path.

    For shapes where the current path is known to fail at compile time
    (NOC burst limit), it is wrapped in try/except and reported as xfail.
    """
    x_torch, w_torch = _build_inputs(batch, c_in, c_out, k, t_in)
    ref = _torch_reference(x_torch, w_torch, stride=s)

    # --- current path ---
    cur_status = "ok"
    cur_t_ms = float("inf")
    cur_pcc = 0.0
    try:
        x_tt, w_tt = _to_ttnn(x_torch, w_torch, device)
        t0 = time.perf_counter()
        out_cur = ttnn.conv_transpose2d(
            input_tensor=x_tt,
            weight_tensor=w_tt,
            device=device,
            in_channels=c_in,
            out_channels=c_out,
            batch_size=batch,
            input_height=1,
            input_width=t_in,
            kernel_size=(1, k),
            stride=(1, s),
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            mirror_kernel=True,
        )
        ttnn.synchronize_device(device)
        cur_t_ms = (time.perf_counter() - t0) * 1e3
        out_torch_cur = _from_ttnn(out_cur)
        cur_pcc_ok, _ = comp_pcc(ref, out_torch_cur, pcc=0.99)
        cur_pcc = float(cur_pcc_ok)
    except Exception as e:
        cur_status = f"FAIL: {type(e).__name__}: {str(e)[:120]}"

    # --- polyphase path ---
    x_tt, w_tt = _to_ttnn(x_torch, w_torch, device)
    t0 = time.perf_counter()
    out_poly = ttnn.experimental.conv_transpose2d_polyphase(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=c_in,
        out_channels=c_out,
        batch_size=batch,
        input_height=1,
        input_width=t_in,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, 0),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        mirror_kernel=True,
    )
    ttnn.synchronize_device(device)
    poly_t_ms = (time.perf_counter() - t0) * 1e3
    out_torch_poly = _from_ttnn(out_poly)
    poly_pcc_ok, poly_pcc_msg = comp_pcc(ref, out_torch_poly, pcc=0.99)

    # --- report ---
    speedup = cur_t_ms / poly_t_ms if poly_t_ms > 0 and cur_t_ms != float("inf") else float("inf")
    print(
        f"\n[{label}] K={k} S={s} C_in={c_in} C_out={c_out} T_in={t_in}\n"
        f"  current  : {cur_t_ms:9.3f} ms  status={cur_status}\n"
        f"  polyphase: {poly_t_ms:9.3f} ms  pcc={poly_pcc_ok}\n"
        f"  speedup  : {speedup:.2f}x"
    )

    assert poly_pcc_ok, f"[{label}] polyphase PCC failed: {poly_pcc_msg}"
    if current_works:
        # Polyphase should at minimum match correctness when the current path
        # also works. Speedup is not asserted -- this is just a regression
        # guard for shapes where both paths must be correct.
        assert cur_status == "ok", f"[{label}] current path unexpectedly failed: {cur_status}"
