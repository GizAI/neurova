from __future__ import annotations

from pathlib import Path

from qwenburst.core.batch_worker import BatchGenerationWorker
from qwenburst.core.model_runner import BatchedModelRunner
from qwenburst.core.runtime import RuntimeEngine
from qwenburst.core.scheduler import ContinuousBatchScheduler

from test_adapter_runtime_cpu import ToyAdapter


class FailingRunner:
    def __init__(self):
        self.engine = type("Engine", (), {"lock": __import__("threading").Lock()})()
        self.finished: list[str] = []

    def add_request(self, request_id, token_ids):
        return None

    def execute_step(self, *, device=None):
        raise RuntimeError("boom")

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
