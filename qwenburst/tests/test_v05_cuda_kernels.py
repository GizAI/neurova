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


def test_depthwise_conv_update_scan_matches_single_token_loop():
    import qwenburst_cuda

    torch.manual_seed(0)
    state_loop = torch.randn(32, 3, device="cuda", dtype=torch.float16)
    state_scan = state_loop.clone()
    x = torch.randn(7, 32, device="cuda", dtype=torch.float16)
    weight = torch.randn(32, 4, device="cuda", dtype=torch.float16)
    bias = torch.randn(32, device="cuda", dtype=torch.float16)

    loop_rows = []
    for row in x:
        loop_rows.append(qwenburst_cuda.depthwise_conv_update(state_loop, row.contiguous(), weight, bias))
    y_loop = torch.stack(loop_rows, dim=0)
    y_scan = qwenburst_cuda.depthwise_conv_update_scan(state_scan, x.contiguous(), weight, bias)
    torch.cuda.synchronize()

    assert torch.equal(y_scan, y_loop)
    assert torch.equal(state_scan, state_loop)


def test_gdn_recurrent_ab_scan_matches_single_token_loop():
    import qwenburst_cuda

    torch.manual_seed(0)
    tokens = 5
    kv_heads = 4
    v_heads = 8
    head_dim = 128
    q = (torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(tokens, v_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a = (torch.randn(tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    b = (torch.randn(tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a_log = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 1.0).contiguous()
    dt_bias = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    state_loop = (torch.randn(v_heads, head_dim, head_dim, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_scan = state_loop.clone()

    loop_rows = []
    for t in range(tokens):
        loop_rows.append(
            qwenburst_cuda.gdn_recurrent_ab(
                q[t],
                k[t],
                v[t],
                a[t],
                b[t],
                a_log,
                dt_bias,
                state_loop,
            )
        )
    y_loop = torch.stack(loop_rows, dim=0)
    y_scan = qwenburst_cuda.gdn_recurrent_ab_scan(q, k, v, a, b, a_log, dt_bias, state_scan)
    torch.cuda.synchronize()

    assert torch.equal(y_scan, y_loop)
    assert torch.equal(state_scan, state_loop)


def test_silu_mul_cuda_matches_torch():
    import qwenburst_cuda

    torch.manual_seed(0)
    gate = torch.randn(4096, device="cuda", dtype=torch.float16)
    up = torch.randn(4096, device="cuda", dtype=torch.float16)
    y = qwenburst_cuda.silu_mul(gate.contiguous(), up.contiguous())
    ref = (torch.nn.functional.silu(gate.float()) * up.float()).half()
    torch.cuda.synchronize()
    assert torch.allclose(y, ref, atol=2e-3, rtol=2e-3)
