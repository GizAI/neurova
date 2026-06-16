from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .runtime import RuntimeEngine, sample_next
from .scheduler import ContinuousBatchScheduler
from .features import RuntimeFeatures
from .state_store import BatchStateStore
from ..speculative_batch import DecodeBatchPlan, DecodeRequestState, apply_decode_post_update


@dataclass(frozen=True)
class BatchedStepOutput:
    batch: DecodeBatchPlan
    sampled_token_ids: list[list[int]]
    sampled_counts: list[int]
    rejected_counts: list[int]

    def tokens_by_request(self) -> dict[str, list[int]]:
        return {
            req_id: list(tokens[: self.sampled_counts[i]])
            for i, (req_id, tokens) in enumerate(zip(self.batch.request_ids, self.sampled_token_ids))
        }


class BatchedModelRunner:
    """Small vLLM-style model runner for QwenBurst.

    It is the first serving loop that owns the full contract:
    scheduler -> DecodeBatchPlan -> RuntimeEngine.forward_batch -> sample ->
    sampled/rejected post-update.
    """

    def __init__(
        self,
        *,
        engine: RuntimeEngine,
        scheduler: ContinuousBatchScheduler,
        features: RuntimeFeatures | None = None,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.features = engine.resolve_plan(features).effective
        self.state_store = BatchStateStore(engine=engine, features=self.features, max_slots=scheduler.max_num_requests)

    @property
    def _states(self) -> dict[int, Any]:
        """Compatibility view for older tests; new code uses state_store."""
        return self.state_store.states

    def add_request(self, request_id: str, token_ids: Sequence[int]) -> DecodeRequestState:
        row = self.scheduler.add_request(request_id, token_ids)
        self.state_store.allocate(row.state_index)
        return row

    def finish_request(self, request_id: str) -> DecodeRequestState | None:
        row = self.scheduler.finish_request(request_id)
        if row is not None:
            self.state_store.release(row.state_index)
        return row

    @torch.no_grad()
    def execute_step(self, *, device: str | None = None) -> BatchedStepOutput | None:
        batch = self.scheduler.schedule(device=device or self.engine.device)
        if batch is None:
            return None
        rows = [self._row(req_id) for req_id in batch.request_ids]
        states = self.state_store.get_many(row.state_index for row in rows)
        logits_by_row = self.engine.forward_batch_logits(batch, states)
        sampled_token_ids: list[list[int]] = []
        sampled_counts: list[int] = []
        rejected_counts: list[int] = []
        for row_idx, (row, row_logits) in enumerate(zip(rows, logits_by_row)):
            was_prefilling = row.is_prefilling
            finishes_prefill = was_prefilling and (row.computed_tokens + batch.num_scheduled_tokens[row_idx] >= row.total_len)
            if was_prefilling and not finishes_prefill:
                sampled_token_ids.append([])
                sampled_counts.append(0)
                rejected_counts.append(0)
                continue
            if not row_logits:
                raise RuntimeError(f"row {row.request_id!r} did not return logits")
            if row.draft_token_ids:
                tokens, sampled_n, rejected_n = self._sample_speculative_row(row, row_logits)
                sampled_token_ids.append(tokens)
                sampled_counts.append(sampled_n)
                rejected_counts.append(rejected_n)
                continue
            token = sample_next(row_logits[-1], _GreedyConfig())
            sampled_token_ids.append([int(token)])
            sampled_counts.append(1)
            rejected_counts.append(0)
        apply_decode_post_update(
            rows,
            batch=batch,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )
        return BatchedStepOutput(
            batch=batch,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )

    def _row(self, request_id: str) -> DecodeRequestState:
        row = self.scheduler.get_request(request_id)
        if row is None:
            raise KeyError(f"unknown scheduled request: {request_id}")
        return row

    def _sample_speculative_row(self, row: DecodeRequestState, logits_rows: list[torch.Tensor]) -> tuple[list[int], int, int]:
        drafts = [int(t) for t in (row.draft_token_ids or [])]
        if not drafts:
            token = sample_next(logits_rows[-1], _GreedyConfig())
            return [int(token)], 1, 0
        if len(logits_rows) < len(drafts) + 1:
            raise RuntimeError("speculative row did not return enough logits")
        accepted = 0
        for idx, draft in enumerate(drafts):
            target = int(sample_next(logits_rows[idx], _GreedyConfig()))
            if target != draft:
                tokens = drafts[:accepted] + [target]
                sampled = accepted + 1
                return tokens, sampled, len(logits_rows) - sampled
            accepted += 1
        bonus = int(sample_next(logits_rows[len(drafts)], _GreedyConfig()))
        tokens = drafts + [bonus]
        sampled = len(tokens)
        return tokens, sampled, len(logits_rows) - sampled


@dataclass(frozen=True)
class _GreedyConfig:
    temperature: float = 0.0
    top_k: int = 0
