from __future__ import annotations

import pytest
import torch

from langburst.ops import cuda_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_argmax_many_out_matches_torch():
    ops = cuda_ops()
    torch.manual_seed(11)
    logits = torch.randn(5, 1009, device="cuda", dtype=torch.float16)
    out = torch.empty(5, device="cuda", dtype=torch.long)
    ops.argmax_many_out(logits.contiguous(), out)
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), torch.argmax(logits, dim=-1).cpu())


def test_count_prefix_matches():
    ops = cuda_ops()
    proposed = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
    verified = torch.tensor([1, 2, 9, 4], device="cuda", dtype=torch.long)
    out = ops.count_prefix_matches(proposed, verified)
    torch.cuda.synchronize()
    assert int(out.item()) == 2
