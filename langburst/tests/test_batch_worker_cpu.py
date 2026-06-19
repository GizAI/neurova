from __future__ import annotations

from pathlib import Path
import time

from langburst.engines.native.batch_worker import BatchGenerationHandle, BatchGenerationWorker
from langburst.engines.native.model_runner import BatchedModelRunner
from langburst.engines.native.runtime import GenerationConfig, RuntimeEngine
from langburst.engines.native.scheduler import ContinuousBatchScheduler

from test_adapter_runtime_cpu import ToyAdapter


def test_batch_generation_handle_records_finish_detail_for_length():
    handle = BatchGenerationHandle(
        request_id="r1",
        max_new_tokens=1,
        generation_config=GenerationConfig(ignore_eos=True),
    )

    assert handle.push_tokens([11]) is True
    metrics = handle.metrics()

    assert handle.finish_reason == "length"
    assert handle.finish_detail == "max_new_tokens"
    assert metrics["finish_reason"] == "length"
    assert metrics["finish_detail"] == "max_new_tokens"


def test_batch_generation_handle_records_finish_detail_for_eos():
    handle = BatchGenerationHandle(
        request_id="r1",
        max_new_tokens=8,
        eos_token_ids=(2,),
        generation_config=GenerationConfig(ignore_eos=False),
    )

    assert handle.push_tokens([2]) is True

    assert handle.finish_reason == "stop"
    assert handle.finish_detail == "eos_token:2"


class FailingRunner:
    def __init__(self):
        self.engine = type("Engine", (), {"lock": __import__("threading").Lock()})()
        self.finished: list[str] = []

    def add_request(self, request_id, token_ids, **kwargs):
        return None

    def execute_step(self, *, device=None):
        raise RuntimeError("boom")

    def finish_request(self, request_id):
        self.finished.append(request_id)


class OneStepRunner:
    def __init__(self):
        self.engine = type("Engine", (), {"lock": __import__("threading").Lock()})()
        self.finished: list[str] = []
        self._request_id: str | None = None
        self._sent = False

    def add_request(self, request_id, token_ids, **kwargs):
        self._request_id = request_id

    def execute_step(self, *, device=None):
        if self._request_id is None or self._sent:
            return None
        self._sent = True
        request_id = self._request_id

        class Step:
            def tokens_by_request(self):
                return {request_id: [42]}

        return Step()

    def finish_request(self, request_id):
        self.finished.append(request_id)


class IdleRunner:
    def __init__(self):
        import threading

        self.engine = type("Engine", (), {"lock": threading.Lock()})()
        self.added = threading.Event()
        self.finished: list[str] = []

    def add_request(self, request_id, token_ids, **kwargs):
        self.added.set()

    def execute_step(self, *, device=None):
        return None

    def finish_request(self, request_id):
        self.finished.append(request_id)


class CapacityRunner:
    def __init__(self):
        import threading

        self.engine = type("Engine", (), {"lock": threading.Lock()})()
        self.scheduler = type("Scheduler", (), {"max_num_requests": 1})()
        self.added: list[str] = []
        self.finished: list[str] = []

    def add_request(self, request_id, token_ids, **kwargs):
        if len(self.added) - len(self.finished) >= 1:
            raise AssertionError("worker admitted a pending request while capacity was full")
        self.added.append(request_id)

    def execute_step(self, *, device=None):
        return None

    def finish_request(self, request_id):
        self.finished.append(request_id)


class MultiCapacityIdleRunner:
    def __init__(self, capacity: int = 2):
        import threading

        self.engine = type("Engine", (), {"lock": threading.Lock()})()
        self.scheduler = type("Scheduler", (), {"max_num_requests": int(capacity)})()
        self.added: list[str] = []
        self.finished: list[str] = []

    def add_request(self, request_id, token_ids, **kwargs):
        self.added.append(request_id)

    def execute_step(self, *, device=None):
        return None

    def finish_request(self, request_id):
        self.finished.append(request_id)


def test_batch_generation_worker_batches_two_requests(tmp_path: Path):
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
    worker = BatchGenerationWorker(runner=runner, device="cpu", max_wait_s=0.02)
    try:
        first = worker.submit([1, 2], max_new_tokens=2, request_id="a")
        second = worker.submit([3], max_new_tokens=2, request_id="b")

        assert first.wait_ids(timeout=2.0) == [2, 3]
        assert second.wait_ids(timeout=2.0) == [1, 2]
        assert scheduler.stats().total_scheduled_batches >= 2
        assert worker.stats()["active_requests"] == 0
        assert first.metrics()["output_tokens"] == 2
        assert first.metrics()["ttft_s"] is not None
        assert worker.stats()["completed_requests"] == 2
        assert worker.stats()["completed_output_tokens"] == 4
    finally:
        worker.shutdown()


def test_batch_generation_worker_passes_sampling_config_through_runner(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    scheduler = ContinuousBatchScheduler(max_num_requests=1, max_num_batched_tokens=4, prefill_chunk_size=2)
    runner = BatchedModelRunner(engine=engine, scheduler=scheduler)
    worker = BatchGenerationWorker(runner=runner, device="cpu", max_wait_s=0.001)
    try:
        handle = worker.submit(
            [1, 2],
            max_new_tokens=1,
            generation_config=GenerationConfig(temperature=0.8, top_k=2),
            request_id="sampled",
        )
        assert handle.wait_ids(timeout=2.0)
        row = scheduler.finish_request("sampled")
        assert row is None
        assert handle.generation_config.temperature == 0.8
        assert handle.generation_config.top_k == 2
    finally:
        worker.shutdown()


def test_batch_generation_worker_releases_runner_state_on_failure():
    runner = FailingRunner()
    worker = BatchGenerationWorker(runner=runner, device="cpu", max_wait_s=0.001)
    try:
        handle = worker.submit([1], max_new_tokens=1, request_id="bad")
        try:
            handle.wait_ids(timeout=2.0)
            raise AssertionError("failure should propagate")
        except RuntimeError as exc:
            assert "boom" in str(exc)
        assert runner.finished == ["bad"]
    finally:
        worker.shutdown()


def test_batch_generation_worker_releases_runner_state_before_done():
    runner = OneStepRunner()
    worker = BatchGenerationWorker(runner=runner, device="cpu", max_wait_s=0.001)
    try:
        handle = worker.submit([1], max_new_tokens=1, request_id="one")
        assert handle.wait_ids(timeout=2.0) == [42]
        assert runner.finished == ["one"]
        assert worker.stats()["active_requests"] == 0
    finally:
        worker.shutdown()


def test_batch_generation_worker_cancel_releases_active_request():
    runner = IdleRunner()
    worker = BatchGenerationWorker(runner=runner, device="cpu", max_wait_s=0.001)
    try:
        handle = worker.submit([1], max_new_tokens=8, request_id="cancel-me")
        assert runner.added.wait(timeout=2.0)
        handle.cancel()
        assert handle.wait_ids(timeout=2.0) == []
        assert runner.finished == ["cancel-me"]
        assert worker.stats()["active_requests"] == 0
    finally:
        worker.shutdown()


def test_batch_generation_handle_stops_repeated_token_loop():
    handle = BatchGenerationHandle(
        request_id="repeat-1",
        max_new_tokens=32,
        generation_config=GenerationConfig(
            repetition_stop_ngram_size=4,
            repetition_stop_repeats=6,
        ),
    )

    assert handle.push_tokens([7, 7, 7, 7, 7, 7]) is True
    assert handle.finish_reason == "repetition"


def test_batch_generation_handle_stops_repeated_ngram_loop():
    handle = BatchGenerationHandle(
        request_id="repeat-2",
        max_new_tokens=32,
        generation_config=GenerationConfig(
            repetition_stop_ngram_size=4,
            repetition_stop_repeats=4,
        ),
    )

    assert handle.push_tokens([10, 11, 10, 11, 10, 11, 10, 11]) is True
    assert handle.finish_reason == "repetition"


def test_batch_generation_handle_stops_repeated_phrase_loop():
    phrase = list(range(100, 114))
    handle = BatchGenerationHandle(
        request_id="repeat-phrase",
        max_new_tokens=128,
        generation_config=GenerationConfig(
            repetition_stop_ngram_size=32,
            repetition_stop_repeats=4,
        ),
    )

    assert handle.push_tokens(phrase * 4) is True
    assert handle.finish_reason == "repetition"
    assert handle.finish_detail == "repetition_ngram"


def test_batch_generation_handle_does_not_stop_non_repeated_long_tail():
    handle = BatchGenerationHandle(
        request_id="no-repeat-phrase",
        max_new_tokens=128,
        generation_config=GenerationConfig(
            repetition_stop_ngram_size=32,
            repetition_stop_repeats=4,
        ),
    )

    assert handle.push_tokens(list(range(100, 156))) is False
    assert handle.finish_reason == "stop"


def test_batch_generation_handle_respects_min_repetition_ngram_size():
    handle = BatchGenerationHandle(
        request_id="repeat-min-ngram",
        max_new_tokens=128,
        generation_config=GenerationConfig(
            repetition_stop_min_ngram_size=8,
            repetition_stop_ngram_size=16,
            repetition_stop_repeats=2,
        ),
    )

    assert handle.push_tokens([1, 2, 1, 2, 1, 2, 1, 2]) is False
    phrase = list(range(100, 108))
    assert handle.push_tokens(phrase * 2) is True
    assert handle.finish_reason == "repetition"


def test_batch_generation_worker_does_not_admit_pending_when_active_capacity_is_full():
    runner = CapacityRunner()
    worker = BatchGenerationWorker(runner=runner, device="cpu", max_wait_s=0.001)
    try:
        first = worker.submit([1], max_new_tokens=8, request_id="first")
        second = worker.submit([2], max_new_tokens=8, request_id="second")
        deadline = time.monotonic() + 2.0
        while runner.added != ["first"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.added == ["first"]
        first.cancel()
        assert first.wait_ids(timeout=2.0) == []
        deadline = time.monotonic() + 2.0
        while runner.added != ["first", "second"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.added == ["first", "second"]
        second.cancel()
        assert second.wait_ids(timeout=2.0) == []
        assert runner.added == ["first", "second"]
        assert runner.finished == ["first", "second"]
    finally:
        worker.shutdown()


def test_batch_generation_worker_defers_exclusive_request_until_active_rows_finish():
    runner = MultiCapacityIdleRunner(capacity=2)
    worker = BatchGenerationWorker(
        runner=runner,
        device="cpu",
        max_wait_s=0.001,
        exclusive_prefill_tokens=4,
    )
    try:
        short = worker.submit([1], max_new_tokens=8, request_id="short")
        long = worker.submit([2, 3, 4, 5], max_new_tokens=8, request_id="long")

        deadline = time.monotonic() + 0.2
        while runner.added != ["short"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.added == ["short"]

        short.cancel()
        assert short.wait_ids(timeout=2.0) == []
        deadline = time.monotonic() + 2.0
        while runner.added != ["short", "long"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.added == ["short", "long"]

        long.cancel()
        assert long.wait_ids(timeout=2.0) == []
        assert runner.finished == ["short", "long"]
    finally:
        worker.shutdown()


def test_batch_generation_worker_keeps_later_rows_out_while_exclusive_request_is_active():
    runner = MultiCapacityIdleRunner(capacity=2)
    worker = BatchGenerationWorker(
        runner=runner,
        device="cpu",
        max_wait_s=0.001,
        exclusive_prefill_tokens=4,
    )
    try:
        long = worker.submit([1, 2, 3, 4], max_new_tokens=8, request_id="long")
        short = worker.submit([5], max_new_tokens=8, request_id="short")

        deadline = time.monotonic() + 0.2
        while runner.added != ["long"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.added == ["long"]

        long.prefill_done_monotonic = time.monotonic()
        time.sleep(0.05)
        assert runner.added == ["long"]

        long.cancel()
        assert long.wait_ids(timeout=2.0) == []
        deadline = time.monotonic() + 2.0
        while runner.added != ["long", "short"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert runner.added == ["long", "short"]

        short.cancel()
        assert short.wait_ids(timeout=2.0) == []
        assert runner.finished == ["long", "short"]
    finally:
        worker.shutdown()
