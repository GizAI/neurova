from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import threading
from typing import Sequence

from .block_table import KVBlockTable
from .cuda_graph import CudaGraphBucketPlanner
from ..speculative_batch import DecodeBatchPlan, DecodeInputBuffers, DecodeRequestState, build_decode_batch_plan


@dataclass(frozen=True)
class SchedulerStats:
    max_active_requests: int
    max_queued_requests: int
    active_requests: int
    queued_requests: int
    total_admitted: int
    total_completed: int
    total_rejected: int
    total_timed_out: int

    def summary(self) -> dict[str, int]:
        return {
            "max_active_requests": self.max_active_requests,
            "max_queued_requests": self.max_queued_requests,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_admitted": self.total_admitted,
            "total_completed": self.total_completed,
            "total_rejected": self.total_rejected,
            "total_timed_out": self.total_timed_out,
        }


class AdmissionController:
    """Minimal request admission layer.

    This is not continuous batching yet.  It is the serving boundary that lets
    LangBurst evolve toward continuous-serving scheduling without scattering semaphores
    through server handlers.
    """

    def __init__(self, *, max_active_requests: int = 1, max_queued_requests: int = 0, admission_timeout_s: float | None = None) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be >= 1")
        if max_queued_requests < 0:
            raise ValueError("max_queued_requests must be >= 0")
        if admission_timeout_s is not None and admission_timeout_s < 0:
            raise ValueError("admission_timeout_s must be >= 0")
        self.max_active_requests = int(max_active_requests)
        self.max_queued_requests = int(max_queued_requests)
        self.admission_timeout_s = admission_timeout_s
        self._slots = threading.Semaphore(self.max_active_requests)
        self._lock = threading.Lock()
        self._active = 0
        self._queued = 0
        self._total_admitted = 0
        self._total_completed = 0
        self._total_rejected = 0
        self._total_timed_out = 0

    def acquire(self, *, timeout_s: float | None = None):
        timeout_s = self.admission_timeout_s if timeout_s is None else timeout_s
        return _RequestLease(self, timeout_s=timeout_s)

    def stats(self) -> SchedulerStats:
        with self._lock:
            return SchedulerStats(
                max_active_requests=self.max_active_requests,
                max_queued_requests=self.max_queued_requests,
                active_requests=self._active,
                queued_requests=self._queued,
                total_admitted=self._total_admitted,
                total_completed=self._total_completed,
                total_rejected=self._total_rejected,
                total_timed_out=self._total_timed_out,
            )


class _RequestLease:
    def __init__(self, scheduler: AdmissionController, *, timeout_s: float | None) -> None:
        self.scheduler = scheduler
        self.timeout_s = timeout_s
        self.acquired = False

    def __enter__(self):
        scheduler = self.scheduler
        with scheduler._lock:
            if scheduler._active >= scheduler.max_active_requests and scheduler._queued >= scheduler.max_queued_requests:
                scheduler._total_rejected += 1
                raise TimeoutError("request scheduler queue is full")
            scheduler._queued += 1
        try:
            if self.timeout_s is None:
                scheduler._slots.acquire()
                ok = True
            else:
                ok = scheduler._slots.acquire(timeout=max(0.0, self.timeout_s))
            if not ok:
                with scheduler._lock:
                    scheduler._total_timed_out += 1
                raise TimeoutError("request scheduler admission timed out")
            self.acquired = True
            with scheduler._lock:
                scheduler._queued -= 1
                scheduler._active += 1
                scheduler._total_admitted += 1
            return self
        except Exception:
            with scheduler._lock:
                scheduler._queued = max(0, scheduler._queued - 1)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.acquired:
            return
        scheduler = self.scheduler
        with scheduler._lock:
            scheduler._active = max(0, scheduler._active - 1)
            scheduler._total_completed += 1
        scheduler._slots.release()


@dataclass(frozen=True)
class ContinuousBatchSchedulerStats:
    max_num_requests: int
    max_num_batched_tokens: int
    prefill_chunk_size: int
    waiting_requests: int
    active_requests: int
    total_added: int
    total_finished: int
    total_scheduled_batches: int
    total_scheduled_tokens: int

    def summary(self) -> dict[str, int]:
        return {
            "max_num_requests": self.max_num_requests,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "prefill_chunk_size": self.prefill_chunk_size,
            "waiting_requests": self.waiting_requests,
            "active_requests": self.active_requests,
            "total_added": self.total_added,
            "total_finished": self.total_finished,
            "total_scheduled_batches": self.total_scheduled_batches,
            "total_scheduled_tokens": self.total_scheduled_tokens,
        }


class ContinuousBatchScheduler:
    """continuous-serving request scheduler boundary.

    It owns request rows and produces `DecodeBatchPlan` objects.  Model adapters
    own the actual DecodeState tensors; this scheduler only decides which token
    rows enter the next target-model step.
    """

    def __init__(
        self,
        *,
        max_num_requests: int = 8,
        max_num_batched_tokens: int = 256,
        prefill_chunk_size: int = 64,
        block_table: KVBlockTable | None = None,
        cuda_graph_planner: CudaGraphBucketPlanner | None = None,
    ) -> None:
        if max_num_requests < 1:
            raise ValueError("max_num_requests must be >= 1")
        if max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        self.max_num_requests = int(max_num_requests)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.block_table = block_table
        self.cuda_graph_planner = cuda_graph_planner
        self._waiting: OrderedDict[str, DecodeRequestState] = OrderedDict()
        self._active: OrderedDict[str, DecodeRequestState] = OrderedDict()
        self._next_state_index = 0
        self._total_added = 0
        self._total_finished = 0
        self._total_scheduled_batches = 0
        self._total_scheduled_tokens = 0
        self._buffers: DecodeInputBuffers | None = None

    def add_request(self, request_id: str, token_ids: Sequence[int]) -> DecodeRequestState:
        if request_id in self._waiting or request_id in self._active:
            raise ValueError(f"duplicate request_id: {request_id}")
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        row = DecodeRequestState(
            request_id=request_id,
            state_index=self._next_state_index,
            token_ids=[int(t) for t in token_ids],
        )
        self._next_state_index += 1
        self._waiting[request_id] = row
        if self.block_table is not None:
            self.block_table.ensure_tokens(request_id, len(row.token_ids))
        self._total_added += 1
        self._admit_waiting()
        return row

    def finish_request(self, request_id: str) -> DecodeRequestState | None:
        row = self._active.pop(request_id, None)
        if row is None:
            row = self._waiting.pop(request_id, None)
        if row is not None:
            if self.block_table is not None:
                self.block_table.release(request_id)
            self._total_finished += 1
        return row

    def get_request(self, request_id: str) -> DecodeRequestState | None:
        return self._active.get(request_id) or self._waiting.get(request_id)

    def schedule(self, *, device: str = "cpu") -> DecodeBatchPlan | None:
        self._admit_waiting()
        if not self._active:
            return None
        selected: list[DecodeRequestState] = []
        scheduled_tokens: list[int] = []
        token_budget = self.max_num_batched_tokens

        # Decode first, then prefill. This mirrors the reference runtime's scheduling shape and
        # prevents long prefill chunks from starving active decoders.
        active_rows = list(self._active.values())
        for row in [r for r in active_rows if not r.is_prefilling]:
            n = 1 + len(row.draft_token_ids or [])
            if selected and n > token_budget:
                continue
            if n <= token_budget:
                selected.append(row)
                scheduled_tokens.append(n)
                token_budget -= n
        # Stateful hybrid adapters cannot safely mix decode rows and multi-token
        # prefill rows in one target-model call until every layer consumes a
        # unified paged/block plan. Keep the external serving engine decode-priority rule strict:
        # decode ticks batch decode rows only, then prefill resumes on the next
        # tick. This avoids falling through to per-row legacy KV writes for
        # paged-KV arena states.
        if selected:
            token_budget = 0
        for row in [r for r in active_rows if r.is_prefilling]:
            n = min(row.prefill_remaining, self.prefill_chunk_size, token_budget)
            if n <= 0:
                continue
            selected.append(row)
            scheduled_tokens.append(n)
            token_budget -= n
            if token_budget <= 0:
                break
        if not selected:
            return None
        if self.block_table is not None:
            for row, n in zip(selected, scheduled_tokens):
                self.block_table.ensure_tokens(row.request_id, row.computed_tokens + n)
        graph_bucket: tuple[int, int, int] | None = None
        if self.cuda_graph_planner is not None:
            try:
                bucket = self.cuda_graph_planner.select(
                    batch_size=len(selected),
                    query_len=max(scheduled_tokens),
                    speculative_tokens=max((len(row.draft_token_ids or []) for row in selected), default=0),
                )
                graph_bucket = (bucket.batch_size, bucket.query_len, bucket.speculative_tokens)
            except ValueError:
                # external serving engine falls back to eager for shapes outside captured buckets.
                # LangBurst graph capture is still optional; scheduling must not
                # reject valid prefill/decode work because no graph bucket fits.
                graph_bucket = None
        buffers = self._input_buffers(device)
        batch = build_decode_batch_plan(
            selected,
            device=device,
            buffers=buffers,
            max_prefill_tokens_per_request=self.prefill_chunk_size,
            scheduled_tokens_per_request=scheduled_tokens,
            cuda_graph_bucket=graph_bucket,
        )
        if self.block_table is not None:
            block_tables = self.block_table.block_table_tensor(batch.request_ids, device=device)
            slot_mapping = self.block_table.slot_mapping_tensor(
                batch.request_ids,
                batch.query_start_loc,
                batch.positions,
                device=device,
            )
            batch = build_decode_batch_plan(
                selected,
                device=device,
                buffers=buffers,
                max_prefill_tokens_per_request=self.prefill_chunk_size,
                scheduled_tokens_per_request=scheduled_tokens,
                cuda_graph_bucket=graph_bucket,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
            )
        self._total_scheduled_batches += 1
        self._total_scheduled_tokens += batch.num_tokens
        return batch

    def stats(self) -> ContinuousBatchSchedulerStats:
        return ContinuousBatchSchedulerStats(
            max_num_requests=self.max_num_requests,
            max_num_batched_tokens=self.max_num_batched_tokens,
            prefill_chunk_size=self.prefill_chunk_size,
            waiting_requests=len(self._waiting),
            active_requests=len(self._active),
            total_added=self._total_added,
            total_finished=self._total_finished,
            total_scheduled_batches=self._total_scheduled_batches,
            total_scheduled_tokens=self._total_scheduled_tokens,
        )

    def clear(self) -> None:
        self._waiting.clear()
        self._active.clear()
        self._buffers = None
        if self.block_table is not None:
            self.block_table.clear()

    def _admit_waiting(self) -> None:
        while self._waiting and len(self._active) < self.max_num_requests:
            request_id, row = self._waiting.popitem(last=False)
            self._active[request_id] = row

    def _input_buffers(self, device: str) -> DecodeInputBuffers:
        if self._buffers is None or str(self._buffers.device) != str(device):
            self._buffers = DecodeInputBuffers(
                max_num_requests=self.max_num_requests,
                max_num_tokens=self.max_num_batched_tokens,
                device=device,
            )
        return self._buffers
