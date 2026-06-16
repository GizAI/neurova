from __future__ import annotations

import pytest

from langburst.engines.native.cuda_graph import CudaGraphBucketPlanner


def test_cuda_graph_bucket_planner_selects_next_static_shape():
    planner = CudaGraphBucketPlanner(batch_sizes=(1, 2, 4), query_lens=(1, 2, 8))

    bucket = planner.select(batch_size=3, query_len=2, speculative_tokens=1)

    assert bucket.batch_size == 4
    assert bucket.query_len == 2
    assert bucket.speculative_tokens == 1
    assert bucket.target_tokens_per_request == 3


def test_cuda_graph_bucket_planner_rejects_overflow_shape():
    planner = CudaGraphBucketPlanner(batch_sizes=(1, 2), query_lens=(1,))

    with pytest.raises(ValueError, match="batch_size"):
        planner.select(batch_size=3, query_len=1)
