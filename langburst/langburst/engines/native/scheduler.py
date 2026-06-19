from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import threading
import time
from typing import Sequence

from .block_table import KVBlockTable
from .cuda_graph import CudaGraphBucketPlanner
from ...speculative_batch import DecodeBatchPlan, DecodeInputBuffers, DecodeRequestState, build_decode_batch_plan


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
    context_tiers: tuple[int, ...] = ()
    context_tier_slots: tuple[int, ...] = ()
    active_by_tier: tuple[int, ...] = ()

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
            "context_tiers": self.context_tiers,
            "context_tier_slots": self.context_tier_slots,
            "active_by_tier": self.active_by_tier,
        }


class AdmissionController:
    """Minimal request admission layer.

    This is not continuous batching yet.  It is the serving boundary that lets
    LangBurst evolve toward continuous-serving scheduling without scattering semaphores
    through server handlers.
    """

    def __init__(
        self,
        *,
        max_active_requests: int = 1,
        max_queued_requests: int = 0,
        admission_timeout_s: float | None = None,
        context_tiers: Sequence[int] = (),
        context_tier_slots: Sequence[int] = (),
        allow_context_overflow: bool = False,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError("max_active_requests must be >= 1")
        if max_queued_requests < 0:
            raise ValueError("max_queued_requests must be >= 0")
        if admission_timeout_s is not None and admission_timeout_s < 0:
            raise ValueError("admission_timeout_s must be >= 0")
        if bool(context_tiers) != bool(context_tier_slots):
            raise ValueError("context_tiers and context_tier_slots must be set together")
        if len(context_tiers) != len(context_tier_slots):
            raise ValueError("context_tiers and context_tier_slots must have the same length")
        pairs = tuple(sorted((int(tier), int(slots)) for tier, slots in zip(context_tiers, context_tier_slots)))
        if any(tier < 1 or slots < 1 for tier, slots in pairs):
            raise ValueError("context tiers and slots must be >= 1")
        self.max_active_requests = int(max_active_requests)
        self.max_queued_requests = int(max_queued_requests)
        self.admission_timeout_s = admission_timeout_s
        self.context_tiers = tuple(tier for tier, _slots in pairs)
        self.context_tier_slots = tuple(slots for _tier, slots in pairs)
        self.allow_context_overflow = bool(allow_context_overflow)
        self._condition = threading.Condition()
        self._active = 0
        self._queued = 0
        self._active_by_tier = [0 for _tier in self.context_tiers]
        self._total_admitted = 0
        self._total_completed = 0
        self._total_rejected = 0
        self._total_timed_out = 0

    def acquire(self, *, timeout_s: float | None = None, prompt_tokens: int | None = None):
        timeout_s = self.admission_timeout_s if timeout_s is None else timeout_s
        return _RequestLease(self, timeout_s=timeout_s, prompt_tokens=prompt_tokens)

    def stats(self) -> SchedulerStats:
        with self._condition:
            return SchedulerStats(
                max_active_requests=self.max_active_requests,
                max_queued_requests=self.max_queued_requests,
                active_requests=self._active,
                queued_requests=self._queued,
                total_admitted=self._total_admitted,
                total_completed=self._total_completed,
                total_rejected=self._total_rejected,
                total_timed_out=self._total_timed_out,
                context_tiers=self.context_tiers,
                context_tier_slots=self.context_tier_slots,
                active_by_tier=tuple(self._active_by_tier),
            )

    def _first_fitting_tier_index(self, prompt_tokens: int | None) -> int | None:
        if not self.context_tiers:
            return None
        tokens = int(prompt_tokens or 0)
        for idx, tier in enumerate(self.context_tiers):
            if tokens <= tier:
                return idx
        if self.allow_context_overflow:
            return len(self.context_tiers) - 1
        raise ValueError(f"prompt too long for configured context tiers: tokens={tokens} max={self.context_tiers[-1]}")

    def _select_available_tier_locked(self, prompt_tokens: int | None) -> int | None:
        tier_index = self._first_fitting_tier_index(prompt_tokens)
        if tier_index is None:
            return None if self._active < self.max_active_requests else -1
        for idx in range(tier_index, len(self.context_tiers)):
            if self._active_by_tier[idx] < self.context_tier_slots[idx]:
                return idx
        return -1

    def _can_admit_locked(self, tier_index: int | None) -> bool:
        if self._active >= self.max_active_requests:
            return False
        if tier_index == -1:
            return False
        if tier_index is not None and self._active_by_tier[tier_index] >= self.context_tier_slots[tier_index]:
            return False
        return True


class _RequestLease:
    def __init__(self, scheduler: AdmissionController, *, timeout_s: float | None, prompt_tokens: int | None) -> None:
        self.scheduler = scheduler
        self.timeout_s = timeout_s
        self.prompt_tokens = prompt_tokens
        self.tier_index: int | None = None
        self.acquired = False

    def __enter__(self):
        scheduler = self.scheduler
        deadline = None if self.timeout_s is None else time.monotonic() + max(0.0, self.timeout_s)
        with scheduler._condition:
            queued = False
            try:
                scheduler._first_fitting_tier_index(self.prompt_tokens)
                if scheduler._active >= scheduler.max_active_requests and scheduler._queued >= scheduler.max_queued_requests:
                    scheduler._total_rejected += 1
                    raise TimeoutError("request scheduler queue is full")
                scheduler._queued += 1
                queued = True
                while True:
                    self.tier_index = scheduler._select_available_tier_locked(self.prompt_tokens)
                    if scheduler._can_admit_locked(self.tier_index):
                        break
                    if deadline is None:
                        scheduler._condition.wait()
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        scheduler._total_timed_out += 1
                        raise TimeoutError("request scheduler admission timed out")
                    scheduler._condition.wait(timeout=remaining)
                self.acquired = True
                scheduler._queued -= 1
                queued = False
                scheduler._active += 1
                if self.tier_index is not None:
                    scheduler._active_by_tier[self.tier_index] += 1
                scheduler._total_admitted += 1
                return self
            except Exception:
                if queued:
                    scheduler._queued = max(0, scheduler._queued - 1)
                scheduler._condition.notify_all()
                raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.acquired:
            return
        scheduler = self.scheduler
        with scheduler._condition:
            scheduler._active = max(0, scheduler._active - 1)
            if self.tier_index is not None:
                scheduler._active_by_tier[self.tier_index] = max(0, scheduler._active_by_tier[self.tier_index] - 1)
            scheduler._total_completed += 1
            scheduler._condition.notify_all()


@dataclass(frozen=True)
class ContinuousBatchSchedulerStats:
    max_num_requests: int
    max_num_batched_tokens: int
    prefill_chunk_size: int
    max_prefill_rows_per_batch: int
    decode_prefill_interleave_steps: int
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
            "max_prefill_rows_per_batch": self.max_prefill_rows_per_batch,
            "decode_prefill_interleave_steps": self.decode_prefill_interleave_steps,
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
        max_prefill_rows_per_batch: int = 0,
        decode_prefill_interleave_steps: int = 16,
        block_table: KVBlockTable | None = None,
        cuda_graph_planner: CudaGraphBucketPlanner | None = None,
        kv_window_tokens: int | None = None,
    ) -> None:
        if max_num_requests < 1:
            raise ValueError("max_num_requests must be >= 1")
        if max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1")
        if max_prefill_rows_per_batch < 0:
            raise ValueError("max_prefill_rows_per_batch must be >= 0")
        if decode_prefill_interleave_steps < 1:
            raise ValueError("decode_prefill_interleave_steps must be >= 1")
        self.max_num_requests = int(max_num_requests)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.max_prefill_rows_per_batch = int(max_prefill_rows_per_batch)
        self.decode_prefill_interleave_steps = int(decode_prefill_interleave_steps)
        self.block_table = block_table
        self.cuda_graph_planner = cuda_graph_planner
        self.kv_window_tokens = int(kv_window_tokens) if kv_window_tokens is not None else None
        if self.kv_window_tokens is not None and self.kv_window_tokens < 1:
            raise ValueError("kv_window_tokens must be >= 1")
        self._waiting: OrderedDict[str, DecodeRequestState] = OrderedDict()
        self._active: OrderedDict[str, DecodeRequestState] = OrderedDict()
        self._next_state_index = 0
        self._total_added = 0
        self._total_finished = 0
        self._total_scheduled_batches = 0
        self._total_scheduled_tokens = 0
        self._decode_only_ticks = 0
        self._spec_decode_only_ticks = 0
        self._buffers: DecodeInputBuffers | None = None

    def add_request(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        kv_window_tokens: int | None = None,
    ) -> DecodeRequestState:
        if request_id in self._waiting or request_id in self._active:
            raise ValueError(f"duplicate request_id: {request_id}")
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        row = DecodeRequestState(
            request_id=request_id,
            state_index=self._next_state_index,
            token_ids=[int(t) for t in token_ids],
            kv_window_tokens=self._resolve_kv_window_tokens(kv_window_tokens),
        )
        self._next_state_index += 1
        self._waiting[request_id] = row
        if self.block_table is not None:
            self.block_table.ensure_tokens(request_id, row.kv_token_capacity)
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

    def cap_active_draft_tokens(self, max_draft_tokens: int) -> None:
        max_draft = max(0, int(max_draft_tokens))
        for row in self._active.values():
            if row.num_draft_tokens <= max_draft:
                continue
            if max_draft == 0:
                row.clear_draft_tokens()
            elif row.draft_token_ids_tensor is not None:
                row.draft_token_ids_tensor = row.draft_token_ids_tensor[:max_draft].contiguous()
            elif row.draft_token_ids is not None:
                row.draft_token_ids = list(row.draft_token_ids[:max_draft])

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
        decode_rows = [r for r in active_rows if not r.is_prefilling]
        speculative_decode_rows = [r for r in decode_rows if r.has_draft_tokens]
        plain_decode_rows = [r for r in decode_rows if not r.has_draft_tokens]
        if speculative_decode_rows and plain_decode_rows:
            # Speculative verification and ordinary decode cannot share one
            # fixed-shape target pass yet. Alternate the two classes so a
            # steady stream of speculative drafts cannot starve plain rows.
            if self._spec_decode_only_ticks > 0:
                decode_rows = plain_decode_rows
            else:
                decode_rows = speculative_decode_rows
        elif speculative_decode_rows:
            # Speculative verification has a different fixed-shape target
            # contract from plain decode.  Keep one scheduled batch descriptor
            # uniform instead of mixing rows that require forward_verify_batch
            # with rows that should run ordinary forward_batch_logits.
            decode_rows = speculative_decode_rows
        prefill_rows = [r for r in active_rows if r.is_prefilling]
        prefill_due = bool(decode_rows and prefill_rows and self._decode_only_ticks >= self.decode_prefill_interleave_steps)
        for row in ([] if prefill_due else decode_rows):
            n = 1 + row.num_draft_tokens
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
        prefill_rows_selected = 0
        for row in prefill_rows:
            if self.max_prefill_rows_per_batch and prefill_rows_selected >= self.max_prefill_rows_per_batch:
                break
            n = min(row.prefill_remaining, self.prefill_chunk_size, token_budget)
            if n <= 0:
                continue
            selected.append(row)
            scheduled_tokens.append(n)
            token_budget -= n
            prefill_rows_selected += 1
            if token_budget <= 0:
                break
        if not selected:
            return None
        selected_has_decode = any(not row.is_prefilling for row in selected)
        selected_has_prefill = any(row.is_prefilling for row in selected)
        if selected_has_decode and prefill_rows and not selected_has_prefill:
            self._decode_only_ticks += 1
        elif selected_has_prefill:
            self._decode_only_ticks = 0
        selected_has_spec_decode = any((not row.is_prefilling) and row.has_draft_tokens for row in selected)
        selected_has_plain_decode = any((not row.is_prefilling) and not row.has_draft_tokens for row in selected)
        if selected_has_spec_decode and plain_decode_rows:
            self._spec_decode_only_ticks += 1
        elif selected_has_plain_decode or not speculative_decode_rows:
            self._spec_decode_only_ticks = 0
        if self.block_table is not None:
            for row, n in zip(selected, scheduled_tokens):
                self.block_table.ensure_tokens(row.request_id, min(row.computed_tokens + n, row.kv_token_capacity))
        graph_bucket: tuple[int, int, int] | None = None
        if self.cuda_graph_planner is not None:
            try:
                bucket = self.cuda_graph_planner.select(
                    batch_size=len(selected),
                    query_len=max(scheduled_tokens),
                    speculative_tokens=max((row.num_draft_tokens for row in selected), default=0),
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
            max_prefill_rows_per_batch=self.max_prefill_rows_per_batch,
            decode_prefill_interleave_steps=self.decode_prefill_interleave_steps,
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
        self._decode_only_ticks = 0
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

    def _resolve_kv_window_tokens(self, explicit: int | None) -> int | None:
        value = explicit if explicit is not None else self.kv_window_tokens
        if value is None:
            return None
        value = int(value)
        if value < 1:
            raise ValueError("kv_window_tokens must be >= 1")
        return value
