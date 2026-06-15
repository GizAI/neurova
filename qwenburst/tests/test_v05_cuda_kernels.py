from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_qwen_rmsnorm_cuda_plus_one():
    import qwenburst_cuda
    x = torch.randn(2, 512, device="cuda", dtype=torch.float16)
    w = torch.randn(512, device="cuda", dtype=torch.float16) * 0.01
    y = qwenburst_cuda.rmsnorm_qwen(x, w, 1e-6)
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * (1.0 + w.float())
    assert torch.allclose(y.float(), ref, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize(
    ("bits", "packed", "expected"),
    [
        (4, [0x80, 0x97, 0x0F], [-8.0, 0.0, -1.0, 1.0, 7.0]),
        (3, [0xE0, 0x01], [-4.0, 0.0, 3.0, -4.0]),
    ],
)
def test_lowbit_row_dequant_cuda_matches_cpu(bits, packed, expected):
    import qwenburst_cuda
    qweight = torch.tensor([packed], device="cuda", dtype=torch.uint8)
    scales = torch.ones((1, 1), device="cuda", dtype=torch.float16)
    row = torch.tensor(0, device="cuda", dtype=torch.long)
    out = qwenburst_cuda.lowbit_row_dequant(qweight, scales, row, len(expected), 128, bits)
    assert out.cpu().tolist() == expected
