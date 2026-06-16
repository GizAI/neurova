from __future__ import annotations

from pathlib import Path

from langburst.engines.native_impl.model_runner import BatchedModelRunner
from langburst.correctness import run_batch_path_parity
from langburst.core.features import RuntimeFeatures
from langburst.engines.native_impl.runtime import RuntimeEngine
from langburst.engines.native_impl.scheduler import ContinuousBatchScheduler

from test_adapter_runtime_cpu import ToyAdapter


def test_batched_model_runner_prefill_then_decode_step(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    scheduler = ContinuousBatchScheduler(max_num_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2)
    runner = BatchedModelRunner(engine=engine, scheduler=scheduler)
    a = runner.add_request("a", [1, 2])
    b = runner.add_request("b", [3])

    prefill = runner.execute_step(device="cpu")

    assert prefill is not None
    assert prefill.batch.request_ids == ["a", "b"]
    assert prefill.sampled_counts == [1, 1]
    assert prefill.tokens_by_request() == {"a": [2], "b": [1]}
    assert a.computed_tokens == 2
    assert b.computed_tokens == 1
    assert a.last_sampled_token == 2
    assert b.last_sampled_token == 1

    decode = runner.execute_step(device="cpu")

    assert decode is not None
    assert decode.batch.request_ids == ["a", "b"]
    assert decode.sampled_counts == [1, 1]
    assert decode.rejected_counts == [0, 0]
    assert decode.tokens_by_request() == {"a": [3], "b": [2]}
    assert a.token_ids[-1] == 3
    assert b.token_ids[-1] == 2


def test_batch_path_parity_splits_target_speculative_and_prefix_axes(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("stateful").with_overrides(prefill_chunk_size=2),
    )

    result = run_batch_path_parity(
        engine,
        engine.encode_prompt("abcdefghijklmnopq"),
        features=engine.features,
        max_new_tokens=2,
    )

    assert result.target_only_match
    assert result.speculative_match
    assert result.prefix_cache_match
    assert result.prefix_cache_hit_tokens >= 2


def test_batched_model_runner_releases_state_on_finish(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    runner = BatchedModelRunner(
        engine=engine,
        scheduler=ContinuousBatchScheduler(max_num_requests=1, max_num_batched_tokens=2, prefill_chunk_size=2),
    )
    row = runner.add_request("a", [1])

    assert row.state_index in runner.state_store.states
    assert runner.state_store.stats().allocated_states == 1
    assert runner.finish_request("a") is row
    assert row.state_index not in runner.state_store.states
    assert runner.state_store.stats().allocated_states == 0


def test_batched_model_runner_rolls_back_scheduler_when_state_allocate_fails(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    scheduler = ContinuousBatchScheduler(max_num_requests=1, max_num_batched_tokens=2, prefill_chunk_size=2)
    runner = BatchedModelRunner(engine=engine, scheduler=scheduler)

    def fail_allocate(_state_index):
        raise RuntimeError("synthetic state allocation failure")

    runner.state_store.allocate = fail_allocate  # type: ignore[method-assign]
    try:
        runner.add_request("a", [1])
    except RuntimeError as exc:
        assert "synthetic state allocation failure" in str(exc)
    else:
        raise AssertionError("state allocation failure should propagate")

    assert scheduler.get_request("a") is None
    assert scheduler.stats().active_requests == 0
    assert scheduler.stats().waiting_requests == 0


def test_batched_model_runner_accepts_draft_and_emits_bonus(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    runner = BatchedModelRunner(
        engine=engine,
        scheduler=ContinuousBatchScheduler(max_num_requests=1, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    row = runner.add_request("a", [1, 2])
    runner.execute_step(device="cpu")
    row.draft_token_ids = [3]

    step = runner.execute_step(device="cpu")

    assert step is not None
    assert step.tokens_by_request() == {"a": [3, 4]}
    assert step.sampled_counts == [2]
    assert step.rejected_counts == [0]
    assert row.token_ids[-2:] == [3, 4]


def test_batched_model_runner_rejects_bad_draft(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    runner = BatchedModelRunner(
        engine=engine,
        scheduler=ContinuousBatchScheduler(max_num_requests=1, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    row = runner.add_request("a", [1, 2])
    runner.execute_step(device="cpu")
    row.draft_token_ids = [99]

    step = runner.execute_step(device="cpu")

    assert step is not None
    assert step.tokens_by_request() == {"a": [3]}
    assert step.sampled_counts == [1]
    assert step.rejected_counts == [1]
    assert row.token_ids[-1] == 3
    assert runner.state_store.get(row.state_index).pos == 3


def test_batched_model_runner_uses_state_store_indices(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    runner = BatchedModelRunner(
        engine=engine,
        scheduler=ContinuousBatchScheduler(max_num_requests=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    first = runner.add_request("a", [1])
    second = runner.add_request("b", [2])

    assert runner.state_store.get(first.state_index).pos == 0
    assert runner.state_store.get(second.state_index).pos == 0

    step = runner.execute_step(device="cpu")

    assert step is not None
    assert runner.state_store.stats().summary()["active_state_indices"] == [0, 1]
    assert runner.state_store.get(first.state_index).pos == 1
    assert runner.state_store.get(second.state_index).pos == 1
