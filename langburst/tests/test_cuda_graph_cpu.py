from __future__ import annotations

import pytest

from langburst.engines.native.cuda_graph import CudaGraphBucketPlanner, CudaGraphKey, CudaGraphReplayCache


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


def test_cuda_graph_replay_cache_disabled_rejects_capture():
    cache = CudaGraphReplayCache(enabled=False)
    key = CudaGraphKey(batch_size=1, query_len=1, speculative_tokens=0, context_bucket=512)

    with pytest.raises(RuntimeError, match="disabled"):
        cache.capture(key, lambda: None)


def test_cuda_graph_replay_cache_requires_cuda_when_enabled():
    cache = CudaGraphReplayCache(enabled=True)
    key = CudaGraphKey(batch_size=1, query_len=1, speculative_tokens=0, context_bucket=512)

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        pytest.skip("CUDA capture behavior is covered by GPU integration tests")

    with pytest.raises(RuntimeError, match="requires CUDA"):
        cache.capture(key, lambda: None)


def test_cuda_graph_replay_cache_replays_static_cuda_callable():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA graph replay requires CUDA")

    cache = CudaGraphReplayCache(enabled=True, warmup_steps=0)
    key = CudaGraphKey(batch_size=1, query_len=1, speculative_tokens=0, context_bucket=1)
    x = torch.zeros((1,), device="cuda")

    def step() -> None:
        x.add_(1.0)

    cache.replay_or_capture(key, step)
    torch.cuda.synchronize()
    assert float(x.item()) == 1.0

    cache.replay_or_capture(key, step)
    torch.cuda.synchronize()
    assert float(x.item()) == 2.0
