from __future__ import annotations

import numpy as np
import pytest
import torch

from qwenburst.ops import cuda_ops
from qwenburst.lowbit_reference import dequant_symmetric_lowbit_cpu
from qwenburst.quantize import quantize_symmetric_lowbit

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize("bits", [3, 4])
def test_lowbit_gemv_matches_dequant_matvec(bits: int):
    ops = cuda_ops()
    torch.manual_seed(7 + bits)
    rows, cols, group = 37, 513, 128
    w = torch.randn(rows, cols, dtype=torch.float32) * 0.2
    x = torch.randn(cols, dtype=torch.float16)
    packed, scales, meta = quantize_symmetric_lowbit(w, group_size=group, bits=bits)
    ref_w = dequant_symmetric_lowbit_cpu(packed, scales, meta["cols"], meta["group_size"], meta["bits"])
    ref = (ref_w @ x.float()).to(torch.float16)

    q = torch.from_numpy(np.asarray(packed)).cuda().contiguous()
    s = torch.from_numpy(np.asarray(scales)).cuda().contiguous()
    xc = x.cuda().contiguous()
    y = ops.lowbit_gemv(q, s, xc, meta["cols"], meta["group_size"], meta["bits"])
    torch.cuda.synchronize()
    assert torch.allclose(y.cpu().float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bits", [3, 4])
def test_lowbit_gemv_pair_matches_two_single_gemvs(bits: int):
    ops = cuda_ops()
    torch.manual_seed(70 + bits)
    rows, cols, group = 41, 257, 128
    wa = torch.randn(rows, cols, dtype=torch.float32) * 0.2
    wb = torch.randn(rows, cols, dtype=torch.float32) * 0.2
    x = torch.randn(cols, dtype=torch.float16)
    pa, sa, meta = quantize_symmetric_lowbit(wa, group_size=group, bits=bits)
    pb, sb, _ = quantize_symmetric_lowbit(wb, group_size=group, bits=bits)
    qa = torch.from_numpy(np.asarray(pa)).cuda().contiguous()
    qb = torch.from_numpy(np.asarray(pb)).cuda().contiguous()
    sca = torch.from_numpy(np.asarray(sa)).cuda().contiguous()
    scb = torch.from_numpy(np.asarray(sb)).cuda().contiguous()
    xc = x.cuda().contiguous()
    ya = ops.lowbit_gemv(qa, sca, xc, meta["cols"], meta["group_size"], meta["bits"])
    yb = ops.lowbit_gemv(qb, scb, xc, meta["cols"], meta["group_size"], meta["bits"])
    pa_out, pb_out = ops.lowbit_gemv_pair(qa, sca, qb, scb, xc, meta["cols"], meta["group_size"], meta["bits"])
    torch.cuda.synchronize()
    assert torch.allclose(pa_out.float(), ya.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(pb_out.float(), yb.float(), atol=2e-2, rtol=2e-2)
