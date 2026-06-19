from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import statistics
import queue
import threading
import time
import uuid
from typing import Sequence

from .cuda_memory import CudaMemoryPolicy
from .model_runner import BatchedModelRunner
from .runtime import GenerationConfig


_STOP = object()


def _detached_exception(exc: BaseException) -> BaseException:
    """Return a lightweight exception that cannot retain CUDA tensor frames.

    CUDA OOM tracebacks often include model-forward frames whose locals still
    reference large tensors. Keeping the original exception on a request handle
    can therefore keep the whole runtime resident after recovery. Preserve the
    exception type for upstream handlers, but intentionally drop traceback,
    cause, and context.
    """

    message = str(exc)
    if exc.__class__.__module__ == "torch.cuda" and exc.__class__.__name__ == "OutOfMemoryError":
        try:
            import torch

            out: BaseException = torch.cuda.OutOfMemoryError(message)
        except Exception:
            out = RuntimeError(message)
    else:
        out = RuntimeError(message)
    out.__traceback__ = None
    out.__cause__ = None
    out.__context__ = None
    return out


@dataclass
class BatchGenerationHandle:
    request_id: str
    max_new_tokens: int
    eos_token_ids: tuple[int, ...] = ()
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    prompt_tokens: int = 0
    prompt_cache_key: str | None = None
    prefix_cache_enabled: bool = True
    stop_sequences: tuple[tuple[int, ...], ...] = ()
    include_stop_str_in_output: bool = False
    output_queue: queue.Queue[object] = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    generated: list[int] = field(default_factory=list)
    error: BaseException | None = None
    created_monotonic: float = field(default_factory=time.monotonic)
    admitted_monotonic: float | None = None
    prefill_done_monotonic: float | None = None
    first_token_monotonic: float | None = None
    finished_monotonic: float | None = None
    token_monotonic: list[float] = field(default_factory=list)
    cached_input_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0
    finish_reason: str = "stop"
    finish_detail: str | None = None

    def push_tokens(self, token_ids: Sequence[int]) -> bool:
        should_finish = False
        if self.cancelled.is_set():
            self.finish_reason = "cancelled"
            self.finish_detail = "cancelled"
            return True
        eos = set(int(t) for t in self.eos_token_ids)
        for token in token_ids:
            if self.cancelled.is_set():
                self.finish_reason = "cancelled"
                self.finish_detail = "cancelled"
                should_finish = True
                break
            if len(self.generated) >= self.max_new_tokens:
                self.finish_reason = "length"
                self.finish_detail = "max_new_tokens"
                should_finish = True
                break
            token_id = int(token)
            cfg = self.generation_config
            can_stop = len(self.generated) >= int(cfg.min_new_tokens)
            if can_stop and not cfg.ignore_eos and token_id in eos:
                self.finish_reason = "stop"
                self.finish_detail = f"eos_token:{token_id}"
                should_finish = True
                break
            if can_stop and token_id in set(int(t) for t in cfg.stop_token_ids):
                self.finish_reason = "stop"
                self.finish_detail = f"stop_token_id:{token_id}"
                should_finish = True
                break
            now = time.monotonic()
            if self.first_token_monotonic is None:
                self.first_token_monotonic = now
            self.generated.append(token_id)
            self.token_monotonic.append(now)
            self.output_queue.put(token_id)
            if can_stop and self._matched_repetition_stop():
                self.finish_reason = "repetition"
                self.finish_detail = "repetition_ngram"
                should_finish = True
                break
            matched_stop = self._matched_stop_sequence()
            if can_stop and matched_stop:
                self.finish_reason = "stop"
                self.finish_detail = f"stop_sequence:{len(matched_stop)}"
                if not self.include_stop_str_in_output:
                    remove_n = len(matched_stop)
                    if remove_n:
                        del self.generated[-remove_n:]
                    # Streaming cannot retract already emitted token chunks. The
                    # token list is corrected for non-streaming and accounting;
                    # callers that need strict string-stop streaming should use
                    # token stops or include_stop_str_in_output.
                should_finish = True
                break
            if len(self.generated) >= self.max_new_tokens:
                self.finish_reason = "length"
                self.finish_detail = "max_new_tokens"
                should_finish = True
                break
        return should_finish

    def _matched_stop_sequence(self) -> tuple[int, ...]:
        if not self.stop_sequences:
            return ()
        for seq in self.stop_sequences:
            if seq and len(self.generated) >= len(seq) and tuple(self.generated[-len(seq) :]) == tuple(seq):
                return tuple(seq)
        return ()

    def _matched_repetition_stop(self) -> bool:
        cfg = self.generation_config
        max_n = int(getattr(cfg, "repetition_stop_ngram_size", 0) or 0)
        min_n = max(1, int(getattr(cfg, "repetition_stop_min_ngram_size", 1) or 1))
        repeats = int(getattr(cfg, "repetition_stop_repeats", 0) or 0)
        if max_n <= 0 or repeats <= 1 or min_n > max_n:
            return False
        for n in range(min_n, max_n + 1):
            total = n * repeats
            if len(self.generated) < total:
                continue
            tail = self.generated[-total:]
            unit = tail[-n:]
            if all(tail[i : i + n] == unit for i in range(0, total, n)):
                return True
        return False

    def cancel(self) -> None:
        self.cancelled.set()
        # Wake any streaming waiter immediately. The worker still owns model
        # state cleanup, but the HTTP side must not block on a cancelled queue.
        self.output_queue.put(_STOP)

    def finish(self) -> None:
        if not self.done.is_set():
            self.finished_monotonic = time.monotonic()
            self.done.set()
            self.output_queue.put(_STOP)

    def fail(self, exc: BaseException) -> None:
        self.error = _detached_exception(exc)
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

    def poll_output(self, timeout: float = 0.25) -> tuple[int | None, bool]:
        """Return ``(token_id, done)`` without blocking indefinitely.

        Streaming HTTP handlers need a polling boundary so client disconnects
        can cancel long prefill requests before they pin arena slots or KV
        blocks. `iter_token_ids()` remains the simple blocking API for tests and
        non-streaming waiters.
        """

        try:
            item = self.output_queue.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            return None, False
        if item is _STOP:
            if self.error is not None:
                raise self.error
            return None, True
        return int(item), False

    def metrics(self) -> dict[str, float | int | str | None]:
        end = self.finished_monotonic or time.monotonic()
        admitted = self.admitted_monotonic
        first = self.first_token_monotonic
        prefill_done = self.prefill_done_monotonic
        output_tokens = len(self.generated)
        queue_wait_s = (admitted - self.created_monotonic) if admitted is not None else None
        prefill_s = (prefill_done - admitted) if admitted is not None and prefill_done is not None else None
        uncached_prompt_tokens = max(0, int(self.prompt_tokens) - int(self.cached_input_tokens))
        ttft_s = (first - self.created_monotonic) if first is not None else None
        e2e_s = end - self.created_monotonic
        decode_s = (end - first) if first is not None else None
        inter_token_latencies = [
            b - a for a, b in zip(self.token_monotonic, self.token_monotonic[1:])
        ]
        return {
            "request_id": self.request_id,
            "prompt_tokens": int(self.prompt_tokens),
            "output_tokens": output_tokens,
            "cached_input_tokens": int(self.cached_input_tokens),
            "accepted_prediction_tokens": int(self.accepted_prediction_tokens),
            "rejected_prediction_tokens": int(self.rejected_prediction_tokens),
            "finish_reason": self.finish_reason,
            "finish_detail": self.finish_detail,
            "queue_wait_s": queue_wait_s,
            "prefill_s": prefill_s,
            "prefill_tok_s": (
                uncached_prompt_tokens / max(prefill_s, 1e-9)
                if prefill_s is not None and prefill_s > 0
                else None
            ),
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

    This is intentionally small, but it mirrors the reference runtime's serving split: request
    handlers enqueue work, the worker owns the model runner, and postprocessing
    is driven by sampled/rejected counts from the runner.
    """

    def __init__(
        self,
        *,
        runner: BatchedModelRunner,
        device: str,
        max_wait_s: float = 0.002,
        exclusive_prefill_tokens: int | None = None,
    ) -> None:
        if max_wait_s < 0:
            raise ValueError("max_wait_s must be >= 0")
        if exclusive_prefill_tokens is not None and exclusive_prefill_tokens < 1:
            raise ValueError("exclusive_prefill_tokens must be >= 1")
        self.runner = runner
        self.device = device
        self.max_wait_s = float(max_wait_s)
        self.exclusive_prefill_tokens = int(exclusive_prefill_tokens) if exclusive_prefill_tokens is not None else None
        self._pending: queue.Queue[tuple[BatchGenerationHandle, list[int]]] = queue.Queue()
        self._deferred: deque[tuple[BatchGenerationHandle, list[int]]] = deque()
        self._active: dict[str, BatchGenerationHandle] = {}
        self._completed: list[dict[str, float | int | str | None]] = []
        self._cuda_memory_policy = CudaMemoryPolicy.from_env()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="langburst-batch-worker", daemon=True)
        self._thread.start()

    def submit(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
        generation_config: GenerationConfig | None = None,
        prompt_cache_key: str | None = None,
        prefix_cache_enabled: bool = True,
        stop_sequences: tuple[tuple[int, ...], ...] = (),
        include_stop_str_in_output: bool = False,
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
            generation_config=generation_config or GenerationConfig(),
            prompt_tokens=len(prompt_ids),
            prompt_cache_key=prompt_cache_key,
            prefix_cache_enabled=bool(prefix_cache_enabled),
            stop_sequences=tuple(tuple(int(t) for t in seq) for seq in stop_sequences),
            include_stop_str_in_output=bool(include_stop_str_in_output),
        )
        self._pending.put((handle, [int(t) for t in prompt_ids]))
        self._update_runner_pressure_count()
        return handle

    def shutdown(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        for req_id, handle in list(self._active.items()):
            handle.cancel()
            try:
                self.runner.finish_request(req_id)
            except Exception:
                pass
        self._active.clear()
        while True:
            try:
                handle, _prompt_ids = self._pending.get_nowait()
            except queue.Empty:
                break
            handle.cancel()
        while self._deferred:
            handle, _prompt_ids = self._deferred.popleft()
            handle.cancel()
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
                # The runner thread is the continuous-serving owner for batched greedy
                # requests. The engine lock prevents legacy streaming/sampling
                # paths from mutating the same model/state objects concurrently.
                with self.runner.engine.lock:
                    self._release_cancelled_active()
                    if not self._active:
                        continue
                    self._update_runner_pressure_count()
                    step = self.runner.execute_step(device=self.device)
                    if step is not None:
                        self._mark_prefill_done()
                        for req_id, token_ids in step.tokens_by_request().items():
                            handle = self._active.get(req_id)
                            if handle is None:
                                continue
                            batch = getattr(step, "batch", None)
                            try:
                                if batch is None:
                                    raise ValueError
                                row_idx = batch.request_ids.index(req_id)
                                handle.accepted_prediction_tokens += max(0, int(step.sampled_counts[row_idx]) - 1)
                                handle.rejected_prediction_tokens += int(step.rejected_counts[row_idx])
                            except (ValueError, IndexError):
                                pass
                            should_finish = handle.push_tokens(token_ids)
                            if should_finish:
                                self._finish_active(req_id, handle)
                if step is None:
                    time.sleep(0.001)
                    with self.runner.engine.lock:
                        self._release_cancelled_active()
                    continue
            except BaseException as exc:
                detached = _detached_exception(exc)
                for req_id, handle in list(self._active.items()):
                    handle.fail(detached)
                    self.runner.finish_request(req_id)
                self._active.clear()
                clear = getattr(self.runner, "clear", None)
                if callable(clear):
                    clear()
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

    def _drain_pending(self, *, wait_for_first: bool) -> None:
        scheduler = getattr(self.runner, "scheduler", None)
        capacity = int(getattr(scheduler, "max_num_requests", 1))
        if len(self._active) >= capacity:
            return
        if self._active_exclusive_request():
            return
        deadline = time.monotonic() + self.max_wait_s
        first_timeout = self.max_wait_s if wait_for_first else 0
        item = self._get_pending(timeout=first_timeout)
        if item is None:
            return
        if not self._try_admit(item):
            return
        # external serving engine admits already-queued requests up to the active batch capacity
        # before running a model step. State allocation can be nontrivial on the
        # first request, so a pure wall-clock drain deadline can accidentally
        # split an otherwise ready batch and destroy TTFT/throughput.
        while len(self._active) < capacity:
            item = self._get_pending_nowait()
            if item is None:
                break
            if not self._try_admit(item):
                break
        self._update_runner_pressure_count()
        while time.monotonic() < deadline:
            if len(self._active) >= capacity:
                break
            item = self._get_pending_nowait()
            if item is None:
                break
            if not self._try_admit(item):
                break

    def _get_pending(self, *, timeout: float) -> tuple[BatchGenerationHandle, list[int]] | None:
        if self._deferred:
            return self._deferred.popleft()
        try:
            return self._pending.get(timeout=timeout)
        except queue.Empty:
            return None

    def _get_pending_nowait(self) -> tuple[BatchGenerationHandle, list[int]] | None:
        if self._deferred:
            return self._deferred.popleft()
        try:
            return self._pending.get_nowait()
        except queue.Empty:
            return None

    def _try_admit(self, item: tuple[BatchGenerationHandle, list[int]]) -> bool:
        handle, _prompt_ids = item
        if self._active_exclusive_request():
            self._deferred.appendleft(item)
            return False
        if self._is_exclusive_request(handle) and self._active:
            self._deferred.appendleft(item)
            return False
        self._admit(item)
        return True

    def _is_exclusive_request(self, handle: BatchGenerationHandle) -> bool:
        threshold = self.exclusive_prefill_tokens
        return threshold is not None and int(handle.prompt_tokens) >= int(threshold)

    def _active_exclusive_request(self) -> bool:
        return any(self._is_exclusive_request(handle) for handle in self._active.values())

    def _admit(self, item: tuple[BatchGenerationHandle, list[int]]) -> None:
        handle, prompt_ids = item
        if handle.cancelled.is_set():
            handle.finish()
            return
        try:
            self.runner.add_request(
                handle.request_id,
                prompt_ids,
                generation_config=handle.generation_config,
                prompt_cache_key=handle.prompt_cache_key,
                prefix_cache_enabled=handle.prefix_cache_enabled,
            )
            scheduler = getattr(self.runner, "scheduler", None)
            get_request = getattr(scheduler, "get_request", None)
            if callable(get_request):
                row = get_request(handle.request_id)
                if row is not None:
                    handle.cached_input_tokens = int(getattr(row, "prefix_cache_hit_tokens", 0) or 0)
            handle.admitted_monotonic = time.monotonic()
            self._active[handle.request_id] = handle
        except BaseException as exc:
            try:
                self.runner.finish_request(handle.request_id)
            except Exception:
                pass
            if exc.__class__.__module__ == "torch.cuda" and exc.__class__.__name__ == "OutOfMemoryError":
                clear = getattr(self.runner, "clear", None)
                if callable(clear):
                    clear()
                self._release_idle_cuda_cache()
            handle.fail(exc)

    def _mark_prefill_done(self) -> None:
        scheduler = getattr(self.runner, "scheduler", None)
        get_request = getattr(scheduler, "get_request", None)
        if not callable(get_request):
            return
        now = time.monotonic()
        for req_id, handle in list(self._active.items()):
            if handle.prefill_done_monotonic is not None:
                continue
            row = get_request(req_id)
            if row is not None and not row.is_prefilling:
                handle.prefill_done_monotonic = now

    def _finish_active(self, req_id: str, handle: BatchGenerationHandle) -> None:
        self.runner.finish_request(req_id)
        self._active.pop(req_id, None)
        handle.finish()
        self._completed.append(handle.metrics())
        if not self._active:
            reset_spec = getattr(self.runner, "reset_speculative_tracker", None)
            if callable(reset_spec):
                reset_spec()
            release_caches = getattr(self.runner, "release_idle_runtime_caches", None)
            if callable(release_caches):
                release_caches()
        self._release_idle_cuda_cache()

    def _cancel_active(self, req_id: str, handle: BatchGenerationHandle) -> None:
        self.runner.finish_request(req_id)
        self._active.pop(req_id, None)
        handle.finish()
        self._completed.append(handle.metrics())
        if not self._active:
            reset_spec = getattr(self.runner, "reset_speculative_tracker", None)
            if callable(reset_spec):
                reset_spec()
            release_caches = getattr(self.runner, "release_idle_runtime_caches", None)
            if callable(release_caches):
                release_caches()
        self._release_idle_cuda_cache()

    def _release_cancelled_active(self) -> None:
        for req_id, handle in list(self._active.items()):
            if handle.cancelled.is_set():
                self._cancel_active(req_id, handle)

    def _release_idle_cuda_cache(self) -> None:
        try:
            self._cuda_memory_policy.release_idle_cache(active_requests=len(self._active))
        except Exception:
            pass

    def _update_runner_pressure_count(self) -> None:
        set_count = getattr(self.runner, "set_serving_pressure_request_count", None)
        if not callable(set_count):
            return
        set_count(len(self._active) + self._pending.qsize() + len(self._deferred))


def _mean_metric(rows: Sequence[dict[str, object]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return float(statistics.fmean(vals))
