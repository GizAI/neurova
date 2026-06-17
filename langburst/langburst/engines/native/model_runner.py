from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import time
from typing import Sequence

import torch

from .runtime import GenerationConfig, RuntimeEngine, sample_next
from .scheduler import ContinuousBatchScheduler
from ...core.features import RuntimeFeatures
from .cuda_memory import CudaMemoryPolicy
from .kv_policy import KVCachePolicy
from .prefix_cache import RadixPrefixCache
from .state_store import BatchStateStore
from ...speculation import DraftRequest, SpeculativeAcceptanceTracker
from ...ops import cuda_ops
from ...speculative_batch import (
    DecodeBatchPlan,
    DecodeRequestState,
    NativeSpecDecodeMetadata,
    apply_decode_post_update,
    select_decode_batch_rows,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchedStepOutput:
    batch: DecodeBatchPlan
    sampled_token_ids: list[list[int]]
    sampled_counts: list[int]
    rejected_counts: list[int]
    accepted_draft_counts: list[int]

    def tokens_by_request(self) -> dict[str, list[int]]:
        return {
            req_id: list(tokens[: self.sampled_counts[i]])
            for i, (req_id, tokens) in enumerate(zip(self.batch.request_ids, self.sampled_token_ids))
        }


class BatchedModelRunner:
    """Small continuous-serving model runner for LangBurst.

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
        max_state_pool_size: int | None = None,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.features = engine.resolve_plan(features).effective
        kv_num_blocks = scheduler.block_table.num_blocks if scheduler.block_table is not None else None
        kv_block_size = scheduler.block_table.block_size if scheduler.block_table is not None else None
        if max_state_pool_size is None:
            max_slots = scheduler.max_num_requests
        else:
            max_slots = min(int(max_state_pool_size), scheduler.max_num_requests)
        self.state_store = BatchStateStore(
            engine=engine,
            features=self.features,
            max_slots=max_slots,
            kv_num_blocks=kv_num_blocks,
            kv_block_size=kv_block_size,
        )
        block_table = self.scheduler.block_table
        self.kv_cache_policy = KVCachePolicy.from_env(
            enabled=bool(self.features.prefix_cache),
            block_table=block_table,
        )
        self.prefix_cache = RadixPrefixCache(**self.kv_cache_policy.prefix_cache_kwargs(block_table=block_table))
        self.cuda_memory_policy = CudaMemoryPolicy.from_env()
        self.speculative_policy = engine.resolve_policy(self.features).speculative
        self.speculative_tracker = SpeculativeAcceptanceTracker(self.speculative_policy)
        self._speculative_prepare_skips: dict[str, int] = {}

    def add_request(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        generation_config: GenerationConfig | None = None,
        prompt_cache_key: str | None = None,
        prefix_cache_enabled: bool = True,
    ) -> DecodeRequestState:
        row = self.scheduler.add_request(request_id, token_ids)
        row.generation_config = generation_config or GenerationConfig()
        row.prompt_cache_key = prompt_cache_key
        row.prefix_cache_enabled = bool(prefix_cache_enabled)
        try:
            state = self.state_store.allocate(row.state_index)
            self._apply_prefix_cache(row, state)
        except BaseException:
            self.scheduler.finish_request(request_id)
            raise
        return row

    def finish_request(self, request_id: str) -> DecodeRequestState | None:
        row = self.scheduler.finish_request(request_id)
        if row is not None:
            self.state_store.release(row.state_index)
            self._trim_cuda_cache_after_request()
        return row

    @torch.no_grad()
    def execute_step(self, *, device: str | None = None) -> BatchedStepOutput | None:
        batch = self.scheduler.schedule(device=device or self.engine.device)
        if batch is None:
            return None
        rows = [self._row(req_id) for req_id in batch.request_ids]
        was_prefilling = [row.is_prefilling for row in rows]
        for row in rows:
            if int(row.state_index) not in self.state_store.states:
                self.state_store.allocate(row.state_index)
        states = self.state_store.get_many(row.state_index for row in rows)
        batch = self._with_physical_state_indices(batch, rows, device=device or self.engine.device)
        step_start = time.perf_counter()
        fast_spec = self._execute_verify_batch_if_possible(batch, rows, states)
        if fast_spec is not None:
            sampled_token_ids, sampled_counts, rejected_counts, accepted_draft_counts = fast_spec
            apply_decode_post_update(
                rows,
                batch=batch,
                sampled_token_ids=sampled_token_ids,
                sampled_counts=sampled_counts,
                rejected_counts=rejected_counts,
            )
            self._truncate_speculative_kv_blocks(rows)
            self._record_speculative_accounting(
                batch=batch,
                accepted_draft_counts=accepted_draft_counts,
                elapsed_ms=(time.perf_counter() - step_start) * 1000.0,
                output_tokens=sum(sampled_counts),
            )
            self._prepare_native_nextn_drafts(rows, states, sampled_counts, rejected_counts)
            return BatchedStepOutput(
                batch=batch,
                sampled_token_ids=sampled_token_ids,
                sampled_counts=sampled_counts,
                rejected_counts=rejected_counts,
                accepted_draft_counts=accepted_draft_counts,
            )
        if batch.spec_decode_metadata is not None or any(row.has_draft_tokens for row in rows):
            raise RuntimeError("speculative rows must use the production forward_verify_batch hot path")
        logits_by_row = self.engine.forward_batch_logits(batch, states)
        sampled_token_ids: list[list[int]] = [[] for _ in rows]
        sampled_counts: list[int] = [0 for _ in rows]
        rejected_counts: list[int] = [0 for _ in rows]
        accepted_draft_counts: list[int] = [0 for _ in rows]
        plain_rows: list[int] = []
        plain_logits: list[torch.Tensor] = []
        for row_idx, (row, row_logits) in enumerate(zip(rows, logits_by_row)):
            row_was_prefilling = row.is_prefilling
            finishes_prefill = row_was_prefilling and (row.computed_tokens + batch.num_scheduled_tokens[row_idx] >= row.total_len)
            if row_was_prefilling and not finishes_prefill:
                continue
            if not row_logits:
                raise RuntimeError(f"row {row.request_id!r} did not return logits")
            plain_rows.append(row_idx)
            plain_logits.append(row_logits[-1])
        if plain_rows:
            plain_tokens = self._sample_plain_rows([rows[i] for i in plain_rows], plain_logits)
            for row_idx, token in zip(plain_rows, plain_tokens):
                sampled_token_ids[row_idx] = [int(token)]
                sampled_counts[row_idx] = 1
        apply_decode_post_update(
            rows,
            batch=batch,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )
        self._truncate_speculative_kv_blocks(rows)
        self._record_speculative_accounting(
            batch=batch,
            accepted_draft_counts=accepted_draft_counts,
            elapsed_ms=(time.perf_counter() - step_start) * 1000.0 if batch.spec_decode_metadata is not None else None,
            output_tokens=sum(sampled_counts) if batch.spec_decode_metadata is not None else None,
        )
        if batch.spec_decode_metadata is None:
            self.speculative_tracker.record_baseline(
                elapsed_ms=(time.perf_counter() - step_start) * 1000.0,
                output_tokens=sum(sampled_counts),
            )
        self._prepare_native_nextn_drafts(rows, states, sampled_counts, rejected_counts)
        self._store_prefix_cache_rows(rows, states, was_prefilling)
        return BatchedStepOutput(
            batch=batch,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
            accepted_draft_counts=accepted_draft_counts,
        )

    def _execute_verify_batch_if_possible(
        self,
        batch: DecodeBatchPlan,
        rows: list[DecodeRequestState],
        states: list[object],
    ) -> tuple[list[list[int]], list[int], list[int], list[int]] | None:
        verify_batch = getattr(self.engine.model, "forward_verify_batch", None)
        if not callable(verify_batch):
            return None
        if not rows or any(row.is_prefilling or not row.has_draft_tokens for row in rows):
            return None
        if any(_config_needs_configured_sampling(_generation_config_from_row(row)) for row in rows):
            return None
        metadata = batch.spec_decode_metadata
        if metadata is None:
            return None
        spec_plan = select_decode_batch_rows(batch, list(range(len(rows))))
        verify_results = verify_batch(spec_plan, states)
        if len(verify_results) != len(rows):
            raise RuntimeError("verify batch result count mismatch")
        if not all(bool(getattr(result, "state_already_committed", False)) for result in verify_results):
            raise RuntimeError(
                "forward_verify_batch must commit exactly the sampled prefix; "
                "rollback-style verifier results are not accepted in production"
            )
        sampled_token_ids: list[list[int]] = [[] for _ in rows]
        sampled_counts: list[int] = [0 for _ in rows]
        rejected_counts: list[int] = [0 for _ in rows]
        accepted_draft_counts = self._resolve_verify_batch_decision(
            metadata=metadata,
            verify_results=verify_results,
            rows=rows,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )
        return sampled_token_ids, sampled_counts, rejected_counts, accepted_draft_counts

    def _resolve_verify_batch_decision(
        self,
        *,
        metadata: NativeSpecDecodeMetadata,
        verify_results: Sequence[object],
        rows: Sequence[DecodeRequestState],
        sampled_token_ids: list[list[int]],
        sampled_counts: list[int],
        rejected_counts: list[int],
    ) -> list[int]:
        if metadata.batch_size != len(rows) or len(verify_results) != len(rows):
            raise ValueError("verify result batch size mismatch")
        decision = getattr(verify_results[0], "speculative_decision", None) if verify_results else None
        if decision is None:
            raise RuntimeError("production verify batch must return a SpeculativeGPUDecision")
        if not all(getattr(result, "speculative_decision", None) is decision for result in verify_results):
            raise RuntimeError("verify batch returned inconsistent speculative decisions")
        return self._apply_speculative_gpu_decision(
            decision,
            row_indices=list(range(len(rows))),
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )

    def _trim_cuda_cache_after_request(self) -> None:
        active_requests = self.scheduler.stats().active_requests
        self.cuda_memory_policy.release_idle_cache(active_requests=active_requests)

    def _truncate_speculative_kv_blocks(self, rows: Sequence[DecodeRequestState]) -> None:
        block_table = getattr(self.scheduler, "block_table", None)
        if block_table is None:
            return
        truncate = getattr(block_table, "truncate_tokens", None)
        if not callable(truncate):
            return
        for row in rows:
            truncate(row.request_id, int(row.computed_tokens))

    def _row(self, request_id: str) -> DecodeRequestState:
        row = self.scheduler.get_request(request_id)
        if row is None:
            raise KeyError(f"unknown scheduled request: {request_id}")
        return row

    def _with_physical_state_indices(
        self,
        batch: DecodeBatchPlan,
        rows: Sequence[DecodeRequestState],
        *,
        device: str | torch.device,
    ) -> DecodeBatchPlan:
        physical = [
            self.state_store.physical_index(int(row.state_index))
            for row in rows
        ]
        physical_tensor = torch.tensor(
            physical,
            device=torch.device(device),
            dtype=torch.int32,
        )
        return replace(batch, state_indices=physical_tensor)

    def clear(self) -> None:
        self.scheduler.clear()
        self.state_store.clear()
        self.prefix_cache.clear()

    def prefix_cache_summary(self) -> dict[str, object]:
        summary: dict[str, object] = self.prefix_cache.stats().summary()
        summary["policy"] = self.kv_cache_policy.summary()
        return summary

    def memory_policy_summary(self) -> dict[str, object]:
        return {
            "kv_cache": self.kv_cache_policy.summary(),
            "cuda_cache": self.cuda_memory_policy.summary(),
        }

    def speculative_summary(self) -> dict[str, object]:
        summary = self.speculative_tracker.summary()
        summary["features_enabled"] = bool(self.features.speculative_decoding)
        summary["proposer_available"] = self.engine.speculative_proposer is not None
        summary["prepare_skips"] = dict(sorted(self._speculative_prepare_skips.items()))
        return summary

    def reset_speculative_tracker(self) -> None:
        self.speculative_tracker = SpeculativeAcceptanceTracker(self.speculative_policy)
        self._speculative_prepare_skips.clear()

    def release_idle_runtime_caches(self) -> None:
        clear_model_caches = getattr(self.engine.model, "clear_runtime_caches", None)
        if callable(clear_model_caches):
            clear_model_caches()

    def _record_speculative_prepare_skip(self, reason: str) -> None:
        self._speculative_prepare_skips[reason] = self._speculative_prepare_skips.get(reason, 0) + 1

    def _max_reusable_prefix_len(self, token_count: int) -> int:
        if token_count <= 1:
            return 0
        max_len = token_count - 1
        if self.scheduler.block_table is not None:
            block_size = self.scheduler.block_table.block_size
            max_len = (max_len // block_size) * block_size
        return max(0, int(max_len))

    def _apply_prefix_cache(self, row: DecodeRequestState, state: object) -> None:
        if not bool(getattr(row, "prefix_cache_enabled", True)):
            return
        max_prefix_len = self._max_reusable_prefix_len(len(row.token_ids))
        if max_prefix_len <= 0:
            return
        hit = self.prefix_cache.lookup(row.token_ids, max_prefix_len=max_prefix_len, namespace=row.prompt_cache_key)
        if hit is None or hit.prefix_len <= 0:
            return
        copy_from = getattr(state, "copy_from_", None)
        if not callable(copy_from):
            return
        block_table = self.scheduler.block_table
        if block_table is not None and hit.block_ids:
            block_table.reset_to_prefix_blocks(row.request_id, hit.block_ids)
            copy_from(hit.state, copy_attention=False)
        else:
            copy_from(hit.state, copy_attention=True)
        row.computed_tokens = int(hit.prefix_len)
        row.prefix_cache_hit_tokens = int(hit.prefix_len)

    def _store_prefix_cache_rows(
        self,
        rows: list[DecodeRequestState],
        states: list[object],
        was_prefilling: list[bool],
    ) -> None:
        if not self.features.prefix_cache:
            return
        block_table = self.scheduler.block_table
        block_size = block_table.block_size if block_table is not None else 0
        for row, state, did_prefill in zip(rows, states, was_prefilling):
            if not did_prefill or row.computed_tokens <= 0:
                continue
            if not bool(getattr(row, "prefix_cache_enabled", True)):
                continue
            prefix_len = int(row.computed_tokens)
            if int(row.prefix_cache_hit_tokens) >= prefix_len:
                continue
            if block_size > 0 and prefix_len % block_size != 0:
                continue
            fork = getattr(state, "fork", None)
            if not callable(fork):
                continue
            decision = self.kv_cache_policy.admit_prefix_store(
                prefix_len=prefix_len,
                block_table=block_table,
            )
            if not decision.allowed:
                self.prefix_cache.clear()
                continue
            block_ids: tuple[int, ...] = ()
            if block_table is not None:
                block_ids = block_table.pin_prefix_blocks(row.request_id, prefix_len)
            cached_state = fork(clone_attention=block_table is None)
            self.prefix_cache.insert(
                row.token_ids[:prefix_len],
                cached_state,
                block_ids=block_ids,
                namespace=row.prompt_cache_key,
            )

    def _apply_speculative_gpu_decision(
        self,
        resolved,
        *,
        row_indices: Sequence[int],
        sampled_token_ids: list[list[int] | None] | list[list[int]],
        sampled_counts: list[int],
        rejected_counts: list[int],
    ) -> list[int]:
        token_rows = resolved.output_rows()
        sampled_list = resolved.output_length_list()
        rejected_list = resolved.rejected_count_list()
        accepted_list = resolved.accepted_draft_count_list()
        for local_row, global_row in enumerate(row_indices):
            tokens = token_rows[local_row]
            sampled_token_ids[global_row] = tokens
            sampled_counts[global_row] = sampled_list[local_row]
            rejected_counts[global_row] = rejected_list[local_row]
        return accepted_list

    def _record_speculative_accounting(
        self,
        *,
        batch: DecodeBatchPlan,
        accepted_draft_counts: Sequence[int],
        elapsed_ms: float | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if batch.spec_decode_metadata is None:
            return
        self.speculative_tracker.record(
            accepted_counts=[int(v) for v in accepted_draft_counts],
            verified_counts=[int(v) for v in batch.num_draft_tokens_per_request],
            elapsed_ms=elapsed_ms,
            output_tokens=output_tokens,
        )

    def _sample_plain_rows(self, rows: list[DecodeRequestState], logits_rows: list[torch.Tensor]) -> list[int]:
        if not logits_rows:
            return []
        if len(rows) != len(logits_rows):
            raise ValueError("row/logit count mismatch")
        if any(_config_needs_configured_sampling(_generation_config_from_row(row)) for row in rows):
            out: list[int] = []
            for row, logits in zip(rows, logits_rows):
                cfg = _generation_config_from_row(row)
                token = int(
                    sample_next(
                        logits,
                        cfg,
                        history=row.token_ids,
                        generated=row.token_ids[int(row.prefill_len or 0) :],
                        sample_index=row.sample_index,
                    )
                )
                row.sample_index += 1
                out.append(token)
            return out
        if len(logits_rows) == 1:
            logits = logits_rows[0].contiguous()
            if logits.device.type == "cuda":
                sampled = cuda_ops().argmax(logits).reshape(1)
            else:
                sampled = torch.argmax(logits, dim=-1).reshape(1).to(dtype=torch.long)
            return [int(t) for t in sampled.detach().cpu().tolist()]
        logits = torch.stack([row.contiguous() for row in logits_rows], dim=0).contiguous()
        if logits.device.type == "cuda":
            sampled = cuda_ops().argmax_many(logits)
        else:
            sampled = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        return [int(t) for t in sampled.detach().cpu().tolist()]

    def _prepare_native_nextn_drafts(
        self,
        rows: list[DecodeRequestState],
        states: list[object],
        sampled_counts: list[int],
        rejected_counts: list[int],
    ) -> None:
        proposer = self.engine.speculative_proposer
        if proposer is None:
            self._record_speculative_prepare_skip("proposer_unavailable")
            return
        if not self.features.speculative_decoding:
            self._record_speculative_prepare_skip("feature_disabled")
            return
        if not self.speculative_tracker.should_propose():
            for row in rows:
                row.clear_draft_tokens()
            self._record_speculative_prepare_skip("policy_disabled")
            return
        if not self._has_speculative_vram_headroom():
            for row in rows:
                row.clear_draft_tokens()
            self._record_speculative_prepare_skip("low_free_vram")
            return
        max_draft = int(self.speculative_tracker.current_max_draft())
        pending: list[tuple[int, DraftRequest]] = []
        for row_idx, (row, state, sampled_n, rejected_n) in enumerate(zip(rows, states, sampled_counts, rejected_counts, strict=True)):
            if int(sampled_n) <= 0:
                self._record_speculative_prepare_skip("no_sampled_token")
                continue
            if int(rejected_n) != 0:
                self._record_speculative_prepare_skip("rejected_previous_step")
                continue
            if row.last_sampled_token is None:
                self._record_speculative_prepare_skip("missing_last_sampled_token")
                continue
            cfg = _generation_config_from_row(row)
            sampling_reason = _configured_sampling_reason(cfg)
            if sampling_reason is not None:
                self._record_speculative_prepare_skip(f"configured_sampling:{sampling_reason}")
                continue
            raw_hidden = getattr(state, "last_raw_hidden", None)
            if raw_hidden is None:
                self._record_speculative_prepare_skip("missing_raw_hidden")
                continue
            pending.append(
                (
                    row_idx,
                    DraftRequest(
                        history=row.token_ids,
                        max_draft=max_draft,
                        signals={
                            "raw_hidden": raw_hidden,
                            "first_token": torch.tensor(
                                int(row.last_sampled_token),
                                device=raw_hidden.device,
                                dtype=torch.long,
                            ),
                            "pos": int(row.computed_tokens),
                        },
                    ),
                )
            )
        if not pending:
            self._record_speculative_prepare_skip("no_pending_rows")
            return
        row_indices = [idx for idx, _request in pending]
        requests = [request for _idx, request in pending]
        try:
            propose_batch = getattr(proposer, "propose_tensors_batch", None)
            if callable(propose_batch):
                drafts = propose_batch(requests)
            else:
                propose_tensors = getattr(proposer, "propose_tensors", None)
                if not callable(propose_tensors):
                    return
                drafts = torch.stack([propose_tensors(request).reshape(-1) for request in requests], dim=0).contiguous()
        except Exception:
            log.exception("native speculative proposer failed; clearing draft tokens")
            for idx in row_indices:
                rows[idx].clear_draft_tokens()
            return
        draft_rows = drafts[:, :max_draft].to(dtype=torch.long).contiguous()
        for local_idx, idx in enumerate(row_indices):
            row_tensor = draft_rows[local_idx].reshape(-1).contiguous()
            rows[idx].draft_token_ids_tensor = row_tensor
            rows[idx].draft_token_ids = None
        if not row_indices:
            self._record_speculative_prepare_skip("no_draft_rows")

    def _has_speculative_vram_headroom(self) -> bool:
        min_free_mib = int(self.speculative_policy.min_free_vram_mib)
        if min_free_mib <= 0:
            return True
        if not torch.cuda.is_available() or not str(self.engine.device).startswith("cuda"):
            return True
        free_bytes, _total_bytes = torch.cuda.mem_get_info(torch.device(self.engine.device))
        return free_bytes >= min_free_mib * 1024 * 1024

def _config_needs_configured_sampling(cfg: GenerationConfig) -> bool:
    return _configured_sampling_reason(cfg) is not None


def _configured_sampling_reason(cfg: GenerationConfig) -> str | None:
    if float(cfg.temperature) > 0:
        return "temperature"
    if int(cfg.top_k) > 0:
        return "top_k"
    if float(cfg.top_p) < 1.0:
        return "top_p"
    if float(cfg.min_p) > 0.0:
        return "min_p"
    if float(cfg.repetition_penalty) != 1.0:
        return "repetition_penalty"
    if float(cfg.presence_penalty) != 0.0:
        return "presence_penalty"
    if float(cfg.frequency_penalty) != 0.0:
        return "frequency_penalty"
    if int(cfg.no_repeat_ngram_size) > 0:
        return "no_repeat_ngram"
    if bool(cfg.logit_bias):
        return "logit_bias"
    if bool(cfg.bad_token_ids):
        return "bad_token_ids"
    return None


def _generation_config_from_row(row: object) -> GenerationConfig:
    cfg = getattr(row, "generation_config", None)
    if isinstance(cfg, GenerationConfig):
        return cfg
    return GenerationConfig()
