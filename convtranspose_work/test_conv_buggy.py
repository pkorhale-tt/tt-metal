"""Unit tests for ``ttnn.conv_transpose2d`` used as a 1D op (``H=1``).

Calls ``ttnn.conv_transpose2d`` directly (no wrapper) and compares against
``torch.nn.functional.conv_transpose1d`` via PCC.

Two parameter groups:
- ``small``: tiny shapes that should always succeed and exercise correctness.
- ``istft``: the actual shapes used by ``TtIstft`` (n_fft=640, hop=160) — these
  reproduce the NoC burst-size failure in the conv2d reader kernel
  (``coalesced_read_bytes <= NOC_MAX_BURST_SIZE``) and are marked ``xfail``
  until ``ttnn.conv_transpose2d`` supports very large kernel widths.

Run:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    pytest models/demos/inworld_tts/tt/test_conv_transpose1d.py -v -s
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    yield dev
    ttnn.close_device(dev)


# (in_channels, out_channels, kernel_size, stride, T, with_bias)
_SMALL_CASES = [
    pytest.param(8, 8, 4, 2, 16, True, id="small_k4_s2_T16_bias"),
    pytest.param(8, 8, 4, 2, 16, False, id="small_k4_s2_T16_nobias"),
    pytest.param(16, 8, 8, 4, 32, True, id="small_k8_s4_T32_bias"),
    pytest.param(32, 1, 16, 8, 32, False, id="reduce_to_1ch_k16_s8"),
    pytest.param(1, 1, 16, 8, 32, False, id="envelope_like_k16_s8"),
]

# Real ISTFT shapes (n_fft=640, hop=160, T=96). Currently fails inside
# ttnn.conv_transpose2d's halo+conv2d reader (NOC_MAX_BURST_SIZE static_assert).
_ISTFT_CASES = [
    pytest.param(321, 1, 640, 160, 96, False, id="istft_cos_or_sin"),
    pytest.param(1, 1, 640, 160, 96, False, id="istft_envelope"),
]


def _run_case(device, in_channels, out_channels, kernel_size, stride, T, with_bias):
    """Drive ``ttnn.conv_transpose2d`` with ``H=1`` for a 1D transposed conv.

    PyTorch reference ``F.conv_transpose1d`` weight is ``[in, out, K]``; we
    unsqueeze axis 2 so the conv2d sees ``[in, out, 1, K]``. ``mirror_kernel=True``
    makes ttnn's kernel orientation match PyTorch (no manual flip needed).
    """
    torch.manual_seed(0)

    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32)
    bias = torch.randn(out_channels, dtype=torch.float32) if with_bias else None
    x_pt = torch.randn(1, in_channels, T, dtype=torch.float32)

    ref = F.conv_transpose1d(x_pt, weight, bias=bias, stride=stride).contiguous()  # [1, out, T_out]

    # NHWC for ttnn: [B, H=1, W=T, C_in]
    x_nhwc = x_pt.transpose(1, 2).unsqueeze(1).contiguous()
    x_tt = ttnn.from_torch(
        x_nhwc,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Conv weight: [in, out, K] -> [in, out, 1, K] for conv2d (H=1).
    weight_tt = ttnn.from_torch(weight.unsqueeze(2), dtype=ttnn.float32)
    bias_tt = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32) if with_bias else None

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.float32,
        deallocate_activation=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_layout=None,
        act_block_h_override=32,
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
    )

    y_tt, [_out_h, _out_w], _ = ttnn.conv_transpose2d(
        input_tensor=x_tt,
        weight_tensor=weight_tt,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=bias_tt,
        kernel_size=(1, kernel_size),
        stride=(1, stride),
        padding=(0, 0),
        output_padding=(0, 0),
        batch_size=1,
        input_height=1,
        input_width=T,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=1,
        mirror_kernel=True,  # match PyTorch ConvTranspose1d kernel orientation
        dram_slice_config=ttnn.Conv2dL1FullSliceConfig,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.float32,
    )

    y = ttnn.to_torch(y_tt)  # [B, 1, T_out, out]
    while y.dim() > 3 and y.shape[1] == 1:
        y = y.squeeze(1)
    y = y.transpose(1, 2).contiguous()  # [1, out, T_out]

    assert y.shape == ref.shape, f"Shape mismatch: tt {y.shape} vs ref {ref.shape}"
    pcc = compute_pcc(ref, y)
    print(f"in={in_channels} out={out_channels} k={kernel_size} s={stride} T={T} " f"bias={with_bias}: PCC={pcc:.6f}")
    return pcc


@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,T,with_bias", _SMALL_CASES)
def test_conv_transpose2d_as_1d_small(device, in_channels, out_channels, kernel_size, stride, T, with_bias):
    pcc = _run_case(device, in_channels, out_channels, kernel_size, stride, T, with_bias)
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"


@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,T,with_bias", _ISTFT_CASES)
def test_conv_transpose2d_as_1d_istft_shapes(device, in_channels, out_channels, kernel_size, stride, T, with_bias):
    pcc = _run_case(device, in_channels, out_channels, kernel_size, stride, T, with_bias)
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
