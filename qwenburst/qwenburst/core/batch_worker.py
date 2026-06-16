from __future__ import annotations

from dataclasses import dataclass, field
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
    generated: list[int] = field(default_factory=list)
    error: BaseException | None = None

    def push_tokens(self, token_ids: Sequence[int]) -> None:
        eos = set(int(t) for t in self.eos_token_ids)
        for token in token_ids:
            if len(self.generated) >= self.max_new_tokens:
                break
            token_id = int(token)
            if token_id in eos:
                self.finish()
                break
            self.generated.append(token_id)
            self.output_queue.put(token_id)
            if len(self.generated) >= self.max_new_tokens:
                self.finish()
                break

    def finish(self) -> None:
        if not self.done.is_set():
            self.done.set()
            self.output_queue.put(_STOP)

    def fail(self, exc: BaseException) -> None:
        self.error = exc
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
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qwenburst-batch-worker", daemon=True)
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

    def stats(self) -> dict[str, int]:
        return {
            "active_requests": len(self._active),
            "pending_requests": self._pending.qsize(),
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
                    step = self.runner.execute_step(device=self.device)
                if step is None:
                    time.sleep(0.001)
                    continue
                for req_id, token_ids in step.tokens_by_request().items():
                    handle = self._active.get(req_id)
                    if handle is None:
                        continue
                    handle.push_tokens(token_ids)
                    if handle.done.is_set():
                        self.runner.finish_request(req_id)
                        self._active.pop(req_id, None)
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
        while time.monotonic() < deadline:
            try:
                self._admit(self._pending.get_nowait())
            except queue.Empty:
                break

    def _admit(self, item: tuple[BatchGenerationHandle, list[int]]) -> None:
        handle, prompt_ids = item
        try:
            self.runner.add_request(handle.request_id, prompt_ids)
            self._active[handle.request_id] = handle
        except BaseException as exc:
            handle.fail(exc)
