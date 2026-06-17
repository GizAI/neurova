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


def test_resolve_greedy_speculative_matches_reference_shape():
    ops = cuda_ops()
    drafts = torch.tensor([10, 11, 12, 20, 21], device="cuda", dtype=torch.long)
    targets = torch.tensor([10, 99, 12, 20, 21], device="cuda", dtype=torch.long)
    bonuses = torch.tensor([13, 30, 22], device="cuda", dtype=torch.long)
    cu_drafts = torch.tensor([3, 3, 5], device="cuda", dtype=torch.int32)
    scheduled = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)

    token_matrix, sampled, rejected, accepted = ops.resolve_greedy_speculative(
        drafts,
        targets,
        bonuses,
        cu_drafts,
        scheduled,
    )
    torch.cuda.synchronize()

    assert sampled.cpu().tolist() == [2, 1, 3]
    assert rejected.cpu().tolist() == [2, 0, 0]
    assert accepted.cpu().tolist() == [1, 0, 2]
    assert token_matrix.cpu().tolist()[0][:2] == [10, 99]
    assert token_matrix.cpu().tolist()[1][:1] == [30]
    assert token_matrix.cpu().tolist()[2][:3] == [20, 21, 22]
