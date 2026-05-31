# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test for ttnn.transpose on fp32/bf16 ROW_MAJOR tensors.
This is what the FFT two-pass composite (commit 3c) depends on:
each between-pass transpose must work for shapes (B, A, C) -> (B, C, A)
where A,C are pow-2 in [2, 1024] and B is the user's batch.

Run:
    pytest -x tests/ttnn/unit_tests/operations/experimental/fft/test_transpose_smoke.py -v
"""

import pytest
import torch
import ttnn


_DTYPES = [
    (ttnn.float32, torch.float32, "fp32"),
    (ttnn.bfloat16, torch.bfloat16, "bf16"),
]

# Shapes that match the composite's needs:
#   (B, N1, N2)  ->  (B, N2, N1)
# Covering small / mid / large factorisations, plus the always-square
# corner case used at N=4096, N=16K, N=1M.
_SHAPES = [
    (1,   32,   64),     # N=2048  — smallest two-pass
    (1,   64,   64),     # N=4096
    (1,   64,  128),     # N=8192
    (1,  128,  128),     # N=16384
    (1,  256,  128),     # N=32768
    (1,  256,  256),     # N=65536
    (1,  512,  512),     # N=262144
    (1, 1024, 1024),     # N=1M    — largest two-pass
    (2,  256,  128),     # B>1, B*N1 needs to fit dispatch
    (4,  128,  128),
]


@pytest.mark.parametrize("tt_dtype,torch_dtype,label", _DTYPES, ids=[d[2] for d in _DTYPES])
@pytest.mark.parametrize("B,A,C", _SHAPES, ids=[f"B{b}x{a}x{c}" for (b, a, c) in _SHAPES])
def test_transpose_rowmajor(device, B, A, C, tt_dtype, torch_dtype, label):
    torch.manual_seed(0)
    x_fp32 = torch.randn(B, A, C, dtype=torch.float32)
    x = x_fp32.to(torch_dtype)

    tt_x = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_y = ttnn.transpose(tt_x, -2, -1)
    got = ttnn.to_torch(tt_y).to(torch.float32)
    ref = x.to(torch.float32).transpose(-2, -1).contiguous()

    # bit-exact for fp32 (pure data movement, no math);
    # allow tiny tolerance for bf16 only if device does any internal conv.
    tol = 0.0 if label == "fp32" else 1e-6
    diff = (got - ref).abs().max().item()
    assert diff <= tol, f"[{label}] {B}x{A}x{C} transpose max abs diff {diff:.2e}"
