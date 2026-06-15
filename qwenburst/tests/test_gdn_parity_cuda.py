from __future__ import annotations

import pytest
import torch

from qwenburst.gdn_reference import gdn_recurrent_reference
from qwenburst.ops import cuda_ops


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_gdn_recurrent_cuda_matches_reference():
    ops = cuda_ops()
    torch.manual_seed(123)
    kv_heads = 2
    v_heads = 6
    q = (torch.randn(kv_heads, 128, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(kv_heads, 128, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(v_heads, 128, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    g = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 0.05).contiguous()
    beta = torch.sigmoid(torch.randn(v_heads, device="cuda", dtype=torch.float32)).to(torch.float16).contiguous()
    state_cuda = (torch.randn(v_heads, 128, 128, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_ref = state_cuda.detach().cpu().clone()

    out = ops.gdn_recurrent(q, k, v, g, beta, state_cuda)
    ref, ref_state = gdn_recurrent_reference(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), state_ref, inplace=True)

    torch.cuda.synchronize()
    assert torch.allclose(out.cpu().float(), ref.float(), atol=5e-3, rtol=5e-3)
    assert torch.allclose(state_cuda.cpu().float(), ref_state.float(), atol=5e-3, rtol=5e-3)
