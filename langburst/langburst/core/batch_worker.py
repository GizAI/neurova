from __future__ import annotations

from dataclasses import dataclass, field
import statistics
import queue
import threading
import time
import uuid
from typing import Sequence

from .model_runner import BatchedModelRunner


_STOP = object()


@dataclass
class BatchGenerationHandle:
    request_id: str
    max_new_tokens: int
    eos_token_ids: tuple[int, ...] = ()
    output_queue: queue.Queue[object] = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    generated: list[int] = field(default_factory=list)
    error: BaseException | None = None
    created_monotonic: float = field(default_factory=time.monotonic)
    admitted_monotonic: float | None = None
    first_token_monotonic: float | None = None
    finished_monotonic: float | None = None
    token_monotonic: list[float] = field(default_factory=list)

    def push_tokens(self, token_ids: Sequence[int]) -> bool:
        should_finish = False
        if self.cancelled.is_set():
            return True
        eos = set(int(t) for t in self.eos_token_ids)
        for token in token_ids:
            if self.cancelled.is_set():
                should_finish = True
                break
            if len(self.generated) >= self.max_new_tokens:
                should_finish = True
                break
            token_id = int(token)
            if token_id in eos:
                should_finish = True
                break
            now = time.monotonic()
            if self.first_token_monotonic is None:
                self.first_token_monotonic = now
            self.generated.append(token_id)
            self.token_monotonic.append(now)
            self.output_queue.put(token_id)
            if len(self.generated) >= self.max_new_tokens:
                should_finish = True
                break
        return should_finish

    def cancel(self) -> None:
        self.cancelled.set()

    def finish(self) -> None:
        if not self.done.is_set():
            self.finished_monotonic = time.monotonic()
            self.done.set()
            self.output_queue.put(_STOP)

    def fail(self, exc: BaseException) -> None:
        self.error = exc
        self.finished_monotonic = time.monotonic()
        self.done.set()
        self.output_queue.put(_STOP)

    def wait_ids(self, timeout: float | None = None) -> list[int]:
        if not self.done.wait(timeout):
            raise TimeoutError(f"batch generation timed out for {self.request_id}")
        if self.error is not None:
            raise self.error
        return list(self.generated)

    def iter_token_ids(self):
        while True:
            item = self.output_queue.get()
            if item is _STOP:
                if self.error is not None:
                    raise self.error
                return
            yield int(item)

    def metrics(self) -> dict[str, float | int | str | None]:
        end = self.finished_monotonic or time.monotonic()
        admitted = self.admitted_monotonic
        first = self.first_token_monotonic
        output_tokens = len(self.generated)
        queue_wait_s = (admitted - self.created_monotonic) if admitted is not None else None
        ttft_s = (first - self.created_monotonic) if first is not None else None
        e2e_s = end - self.created_monotonic
        decode_s = (end - first) if first is not None else None
        inter_token_latencies = [
            b - a for a, b in zip(self.token_monotonic, self.token_monotonic[1:])
        ]
        return {
            "request_id": self.request_id,
            "output_tokens": output_tokens,
            "queue_wait_s": queue_wait_s,
            "ttft_s": ttft_s,
            "e2e_s": e2e_s,
            "decode_s": decode_s,
            "e2e_tok_s": output_tokens / max(e2e_s, 1e-9),
            "decode_tok_s": (
                output_tokens / max(decode_s, 1e-9)
                if decode_s is not None and decode_s > 0
                else None
            ),
            "mean_itl_s": (
                statistics.fmean(inter_token_latencies)
                if inter_token_latencies
                else None
            ),
        }


class BatchGenerationWorker:
    """Continuous-batching generation worker.

    This is intentionally small, but it mirrors vLLM's serving split: request
    handlers enqueue work, the worker owns the model runner, and postprocessing
    is driven by sampled/rejected counts from the runner.
    """

    def __init__(
        self,
        *,
        runner: BatchedModelRunner,
        device: str,
        max_wait_s: float = 0.002,
    ) -> None:
        if max_wait_s < 0:
            raise ValueError("max_wait_s must be >= 0")
        self.runner = runner
        self.device = device
        self.max_wait_s = float(max_wait_s)
        self._pending: queue.Queue[tuple[BatchGenerationHandle, list[int]]] = queue.Queue()
        self._active: dict[str, BatchGenerationHandle] = {}
        self._completed: list[dict[str, float | int | str | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="langburst-batch-worker", daemon=True)
        self._thread.start()

    def submit(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
        request_id: str | None = None,
    ) -> BatchGenerationHandle:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        handle = BatchGenerationHandle(
            request_id=request_id or f"qb-{uuid.uuid4().hex}",
            max_new_tokens=int(max_new_tokens),
            eos_token_ids=tuple(int(t) for t in eos_token_ids),
        )
        self._pending.put((handle, [int(t) for t in prompt_ids]))
        return handle

    def shutdown(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def stats(self) -> dict[str, object]:
        completed = list(self._completed)
        return {
            "active_requests": len(self._active),
            "pending_requests": self._pending.qsize(),
            "completed_requests": len(completed),
            "completed_output_tokens": sum(int(row["output_tokens"] or 0) for row in completed),
            "mean_ttft_s": _mean_metric(completed, "ttft_s"),
            "mean_e2e_s": _mean_metric(completed, "e2e_s"),
            "mean_decode_tok_s": _mean_metric(completed, "decode_tok_s"),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain_pending(wait_for_first=not self._active)
                if not self._active:
                    continue
                # The runner thread is the vLLM-style owner for batched greedy
                # requests. The engine lock prevents legacy streaming/sampling
                # paths from mutating the same model/state objects concurrently.
                with self.runner.engine.lock:
                    self._release_cancelled_active()
                    if not self._active:
                        continue
                    step = self.runner.execute_step(device=self.device)
                if step is None:
                    time.sleep(0.001)
                    self._release_cancelled_active()
                    continue
                for req_id, token_ids in step.tokens_by_request().items():
                    handle = self._active.get(req_id)
                    if handle is None:
                        continue
                    should_finish = handle.push_tokens(token_ids)
                    if should_finish:
                        self._finish_active(req_id, handle)
            except BaseException as exc:
                for req_id, handle in list(self._active.items()):
                    handle.fail(exc)
                    self.runner.finish_request(req_id)
                self._active.clear()

    def _drain_pending(self, *, wait_for_first: bool) -> None:
        deadline = time.monotonic() + self.max_wait_s
        first_timeout = self.max_wait_s if wait_for_first else 0
        try:
            item = self._pending.get(timeout=first_timeout)
        except queue.Empty:
            return
        self._admit(item)
        scheduler = getattr(self.runner, "scheduler", None)
        capacity = int(getattr(scheduler, "max_num_requests", 1))
        # vLLM admits already-queued requests up to the active batch capacity
        # before running a model step. State allocation can be nontrivial on the
        # first request, so a pure wall-clock drain deadline can accidentally
        # split an otherwise ready batch and destroy TTFT/throughput.
        while len(self._active) < capacity:
            try:
                self._admit(self._pending.get_nowait())
            except queue.Empty:
                break
        while time.monotonic() < deadline:
            if len(self._active) >= capacity:
                break
            try:
                self._admit(self._pending.get_nowait())
            except queue.Empty:
                break

    def _admit(self, item: tuple[BatchGenerationHandle, list[int]]) -> None:
        handle, prompt_ids = item
        if handle.cancelled.is_set():
            handle.finish()
            return
        try:
            self.runner.add_request(handle.request_id, prompt_ids)
            handle.admitted_monotonic = time.monotonic()
            self._active[handle.request_id] = handle
        except BaseException as exc:
            handle.fail(exc)

    def _finish_active(self, req_id: str, handle: BatchGenerationHandle) -> None:
        self.runner.finish_request(req_id)
        self._active.pop(req_id, None)
        handle.finish()
        self._completed.append(handle.metrics())

    def _release_cancelled_active(self) -> None:
        for req_id, handle in list(self._active.items()):
            if handle.cancelled.is_set():
                self._finish_active(req_id, handle)


def _mean_metric(rows: Sequence[dict[str, object]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return float(statistics.fmean(vals))
