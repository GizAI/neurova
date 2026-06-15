from __future__ import annotations

import torch

from qwenburst.gdn_reference import gdn_recurrent_reference


def test_gdn_reference_shapes_and_inplace_update():
    torch.manual_seed(0)
    q = torch.randn(2, 128, dtype=torch.float16) * 0.1
    k = torch.randn(2, 128, dtype=torch.float16) * 0.1
    v = torch.randn(6, 128, dtype=torch.float16) * 0.1
    g = torch.full((6,), -0.05, dtype=torch.float32)
    beta = torch.full((6,), 0.5, dtype=torch.float16)
    state = torch.zeros(6, 128, 128, dtype=torch.float16)
    out, new_state = gdn_recurrent_reference(q, k, v, g, beta, state, inplace=True)
    assert out.shape == (6, 128)
    assert new_state.data_ptr() == state.data_ptr()
    assert torch.isfinite(out).all()
    assert state.abs().sum() > 0


def test_gdn_reference_is_deterministic():
    torch.manual_seed(1)
    args = [
        torch.randn(1, 128, dtype=torch.float16) * 0.1,
        torch.randn(1, 128, dtype=torch.float16) * 0.1,
        torch.randn(3, 128, dtype=torch.float16) * 0.1,
        torch.randn(3, dtype=torch.float32) * 0.01 - 0.1,
        torch.sigmoid(torch.randn(3, dtype=torch.float32)).to(torch.float16),
    ]
    s0 = torch.randn(3, 128, 128, dtype=torch.float16) * 0.01
    o1, s1 = gdn_recurrent_reference(*args, s0.clone(), inplace=True)
    o2, s2 = gdn_recurrent_reference(*args, s0.clone(), inplace=True)
    assert torch.equal(o1, o2)
    assert torch.equal(s1, s2)
