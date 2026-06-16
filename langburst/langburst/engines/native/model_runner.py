from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from typing import Any, Sequence

import torch

from .runtime import GenerationConfig, RuntimeEngine, sample_next
from .scheduler import ContinuousBatchScheduler
from ...core.features import RuntimeFeatures
from .prefix_cache import RadixPrefixCache
from .state_store import BatchStateStore
from ...speculation import DraftRequest
from ...ops import cuda_ops
from ...speculative_batch import DecodeBatchPlan, DecodeRequestState, NativeSpecDecodeMetadata, apply_decode_post_update, resolve_greedy_speculative_metadata, select_decode_batch_rows


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
        self.prefix_cache = RadixPrefixCache(
            enabled=bool(self.features.prefix_cache),
            min_prefix_tokens=max(1, int(block_table.block_size if block_table is not None else 16)),
            release_blocks=block_table.release_pinned_blocks if block_table is not None else None,
        )
        self.speculative_policy = engine.resolve_policy(self.features).speculative

    def add_request(
        self,
        request_id: str,
        token_ids: Sequence[int],
        *,
        generation_config: GenerationConfig | None = None,
        prompt_cache_key: str | None = None,
        external_state: object | None = None,
        release_callback: object | None = None,
    ) -> DecodeRequestState:
        row = self.scheduler.add_request(request_id, token_ids)
        row.generation_config = generation_config or GenerationConfig()
        row.prompt_cache_key = prompt_cache_key
        try:
            if external_state is None:
                state = self.state_store.allocate(row.state_index)
                self._apply_prefix_cache(row, state)
            else:
                state = self.state_store.attach_external(
                    row.state_index,
                    external_state,
                    release_callback=release_callback,
                )
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
        spec_snapshots = self._snapshot_speculative_rows(batch, rows, states)
        fast_spec = self._execute_verify_batch_if_possible(batch, rows, states)
        if fast_spec is not None:
            sampled_token_ids, sampled_counts, rejected_counts, commit_states = fast_spec
            self._rollback_rejected_speculative_rows(
                batch,
                rows,
                states,
                spec_snapshots,
                sampled_token_ids,
                sampled_counts,
                rejected_counts,
                commit_states,
            )
            apply_decode_post_update(
                rows,
                batch=batch,
                sampled_token_ids=sampled_token_ids,
                sampled_counts=sampled_counts,
                rejected_counts=rejected_counts,
            )
            self._prepare_native_nextn_drafts(rows, states, sampled_counts, rejected_counts)
            return BatchedStepOutput(
                batch=batch,
                sampled_token_ids=sampled_token_ids,
                sampled_counts=sampled_counts,
                rejected_counts=rejected_counts,
            )
        logits_by_row = self.engine.forward_batch_logits(batch, states)
        sampled_token_ids: list[list[int]] = [[] for _ in rows]
        sampled_counts: list[int] = [0 for _ in rows]
        rejected_counts: list[int] = [0 for _ in rows]
        plain_rows: list[int] = []
        plain_logits: list[torch.Tensor] = []
        speculative_rows: list[int] = []
        for row_idx, (row, row_logits) in enumerate(zip(rows, logits_by_row)):
            row_was_prefilling = row.is_prefilling
            finishes_prefill = row_was_prefilling and (row.computed_tokens + batch.num_scheduled_tokens[row_idx] >= row.total_len)
            if row_was_prefilling and not finishes_prefill:
                continue
            if not row_logits:
                raise RuntimeError(f"row {row.request_id!r} did not return logits")
            if row.has_draft_tokens:
                speculative_rows.append(row_idx)
                continue
            plain_rows.append(row_idx)
            plain_logits.append(row_logits[-1])
        if speculative_rows:
            spec_tokens, spec_sampled, spec_rejected = self._sample_speculative_rows_batch(
                batch,
                speculative_rows,
                [rows[i] for i in speculative_rows],
                [logits_by_row[i] for i in speculative_rows],
            )
            for row_idx, tokens, sampled_n, rejected_n in zip(speculative_rows, spec_tokens, spec_sampled, spec_rejected, strict=True):
                sampled_token_ids[row_idx] = tokens
                sampled_counts[row_idx] = sampled_n
                rejected_counts[row_idx] = rejected_n
        if plain_rows:
            plain_tokens = self._sample_plain_rows([rows[i] for i in plain_rows], plain_logits)
            for row_idx, token in zip(plain_rows, plain_tokens):
                sampled_token_ids[row_idx] = [int(token)]
                sampled_counts[row_idx] = 1
        self._rollback_rejected_speculative_rows(
            batch,
            rows,
            states,
            spec_snapshots,
            sampled_token_ids,
            sampled_counts,
            rejected_counts,
            None,
        )
        apply_decode_post_update(
            rows,
            batch=batch,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )
        self._prepare_native_nextn_drafts(rows, states, sampled_counts, rejected_counts)
        for row, state, did_prefill in zip(rows, states, was_prefilling, strict=True):
            if did_prefill and not row.is_prefilling:
                cache = getattr(state, "_prefill_fp16_kv", None)
                if cache is not None:
                    cache.clear()
        self._store_prefix_cache_rows(rows, states, was_prefilling)
        return BatchedStepOutput(
            batch=batch,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )

    def _execute_verify_batch_if_possible(
        self,
        batch: DecodeBatchPlan,
        rows: list[DecodeRequestState],
        states: list[object],
    ) -> tuple[list[list[int]], list[int], list[int], list[list[object] | None]] | None:
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
        sampled_token_ids: list[list[int]] = [[] for _ in rows]
        sampled_counts: list[int] = [0 for _ in rows]
        rejected_counts: list[int] = [0 for _ in rows]
        self._resolve_greedy_verify_results(
            metadata=metadata,
            verify_results=verify_results,
            rows=rows,
            sampled_token_ids=sampled_token_ids,
            sampled_counts=sampled_counts,
            rejected_counts=rejected_counts,
        )
        commit_states = [getattr(result, "commit_states", None) for result in verify_results]
        return sampled_token_ids, sampled_counts, rejected_counts, commit_states

    def _resolve_greedy_verify_results(
        self,
        *,
        metadata: NativeSpecDecodeMetadata,
        verify_results: Sequence[object],
        rows: Sequence[DecodeRequestState],
        sampled_token_ids: list[list[int]],
        sampled_counts: list[int],
        rejected_counts: list[int],
    ) -> None:
        if metadata.batch_size != len(rows) or len(verify_results) != len(rows):
            raise ValueError("verify result batch size mismatch")
        bonus_logits = [getattr(result, "logits") for result in verify_results]
        bonus_ids = self._argmax_many_tensor([logit.contiguous() for logit in bonus_logits])
        target_rows: list[torch.Tensor] = []
        for row_idx, result in enumerate(verify_results):
            draft_n = int(metadata.num_draft_tokens[row_idx])
            if draft_n <= 0:
                continue
            targets = getattr(result, "target_ids").reshape(-1)
            if int(targets.numel()) < draft_n:
                raise RuntimeError("verify result did not return enough target ids")
            target_rows.append(targets[:draft_n].to(device=bonus_ids.device, dtype=torch.long))
        target_ids = torch.cat(target_rows, dim=0) if target_rows else torch.empty((0,), device=bonus_ids.device, dtype=torch.long)
        decisions = resolve_greedy_speculative_metadata(
            metadata,
            target_token_ids=target_ids,
            bonus_token_ids=bonus_ids,
            scheduled_token_counts=[row.num_draft_tokens + 1 for row in rows],
        )
        for row_idx, decision in enumerate(decisions):
            sampled_token_ids[row_idx] = decision.token_ids
            sampled_counts[row_idx] = decision.sampled_count
            rejected_counts[row_idx] = decision.rejected_count

    def _trim_cuda_cache_after_request(self) -> None:
        raw = os.environ.get(
            "LANGBURST_TRIM_CACHE_AFTER_REQUEST",
            os.environ.get("LANGBURST_TRIM_CACHE_DURING_PREFILL", "1"),
        ).strip().lower()
        if raw in {"0", "false", "off", "no"}:
            return
        if not torch.cuda.is_available():
            return
        threshold_raw = os.environ.get("LANGBURST_TRIM_CACHE_FREE_BELOW_MIB", "768").strip()
        try:
            free_below_mib = int(threshold_raw)
        except ValueError as exc:
            raise ValueError("LANGBURST_TRIM_CACHE_FREE_BELOW_MIB must be an integer MiB value") from exc
        if free_below_mib > 0:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= free_below_mib * 1024 * 1024:
                return
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def _row(self, request_id: str) -> DecodeRequestState:
        row = self.scheduler.get_request(request_id)
        if row is None:
            raise KeyError(f"unknown scheduled request: {request_id}")
        return row

    def clear(self) -> None:
        self.scheduler.clear()
        self.state_store.clear()
        self.prefix_cache.clear()

    def prefix_cache_summary(self) -> dict[str, int]:
        return self.prefix_cache.stats().summary()

    def _max_reusable_prefix_len(self, token_count: int) -> int:
        if token_count <= 1:
            return 0
        max_len = token_count - 1
        if self.scheduler.block_table is not None:
            block_size = self.scheduler.block_table.block_size
            max_len = (max_len // block_size) * block_size
        return max(0, int(max_len))

    def _apply_prefix_cache(self, row: DecodeRequestState, state: object) -> None:
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
            prefix_len = int(row.computed_tokens)
            if int(row.prefix_cache_hit_tokens) >= prefix_len:
                continue
            if block_size > 0 and prefix_len % block_size != 0:
                continue
            fork = getattr(state, "fork", None)
            if not callable(fork):
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

    def _sample_speculative_row(self, row: DecodeRequestState, logits_rows: list[torch.Tensor]) -> tuple[list[int], int, int]:
        drafts = [int(t) for t in row.draft_tensor(device=self.engine.device).detach().cpu().tolist()]
        if not drafts:
            token = sample_next(logits_rows[-1], _generation_config_from_row(row))
            return [int(token)], 1, 0
        if len(logits_rows) < len(drafts) + 1:
            raise RuntimeError("speculative row did not return enough logits")
        accepted = 0
        cfg = _generation_config_from_row(row)
        for idx, draft in enumerate(drafts):
            target = int(sample_next(logits_rows[idx], cfg))
            if target != draft:
                tokens = drafts[:accepted] + [target]
                sampled = accepted + 1
                return tokens, sampled, len(logits_rows) - sampled
            accepted += 1
        bonus = int(sample_next(logits_rows[len(drafts)], cfg))
        tokens = drafts + [bonus]
        sampled = len(tokens)
        return tokens, sampled, len(logits_rows) - sampled

    def _sample_speculative_rows_batch(
        self,
        batch: DecodeBatchPlan,
        row_indices: list[int],
        rows: list[DecodeRequestState],
        logits_by_row: list[list[torch.Tensor]],
    ) -> tuple[list[list[int]], list[int], list[int]]:
        """Resolve NEXTN rows using one batched greedy argmax contract.

        This is the compressed vLLM-style sampler boundary for native serving:
        target logits for draft positions and bonus positions are flattened,
        argmaxed in batches, then reduced to per-request accepted prefixes.
        Rows that require configured sampling fall back to the exact scalar path.
        """

        if len(rows) != len(logits_by_row):
            raise ValueError("row/logits count mismatch")
        sampled_token_ids: list[list[int] | None] = [None] * len(rows)
        sampled_counts: list[int] = [0] * len(rows)
        rejected_counts: list[int] = [0] * len(rows)
        if len(row_indices) != len(rows):
            raise ValueError("row_indices and rows length mismatch")
        greedy_rows: list[int] = []
        greedy_logits_by_row: list[list[torch.Tensor]] = []

        for row_idx, (row, logits_rows) in enumerate(zip(rows, logits_by_row, strict=True)):
            cfg = _generation_config_from_row(row)
            if _config_needs_configured_sampling(cfg):
                tokens, sampled_n, rejected_n = self._sample_speculative_row(row, logits_rows)
                sampled_token_ids[row_idx] = tokens
                sampled_counts[row_idx] = sampled_n
                rejected_counts[row_idx] = rejected_n
                continue
            if not row.has_draft_tokens:
                tokens, sampled_n, rejected_n = self._sample_speculative_row(row, logits_rows)
                sampled_token_ids[row_idx] = tokens
                sampled_counts[row_idx] = sampled_n
                rejected_counts[row_idx] = rejected_n
                continue
            drafts = [int(t) for t in (row.draft_token_ids or [])] if row.draft_token_ids_tensor is None else [0] * row.num_draft_tokens
            if len(logits_rows) < len(drafts) + 1:
                raise RuntimeError("speculative row did not return enough logits")
            greedy_rows.append(row_idx)
            greedy_logits_by_row.append(logits_rows[: len(drafts) + 1])

        if greedy_rows:
            metadata = batch.spec_decode_metadata
            if metadata is None:
                raise RuntimeError("speculative batch is missing metadata")
            self._resolve_greedy_speculative_metadata(
                metadata=metadata.select_rows([row_indices[i] for i in greedy_rows]),
                row_indices=greedy_rows,
                logits_by_row=greedy_logits_by_row,
                sampled_token_ids=sampled_token_ids,
                sampled_counts=sampled_counts,
                rejected_counts=rejected_counts,
            )

        for row_idx in range(len(rows)):
            if sampled_token_ids[row_idx] is not None:
                continue
            raise RuntimeError("speculative sampler left a row unresolved")

        return (
            [list(tokens or []) for tokens in sampled_token_ids],
            sampled_counts,
            rejected_counts,
        )

    def _resolve_greedy_speculative_metadata(
        self,
        *,
        metadata: NativeSpecDecodeMetadata,
        row_indices: list[int],
        logits_by_row: list[list[torch.Tensor]],
        sampled_token_ids: list[list[int] | None],
        sampled_counts: list[int],
        rejected_counts: list[int],
    ) -> None:
        if metadata.batch_size != len(row_indices) or len(row_indices) != len(logits_by_row):
            raise ValueError("speculative metadata batch size mismatch")
        flat_logits = [logit for row_logits in logits_by_row for logit in row_logits]
        if not flat_logits:
            return
        target_logits = [flat_logits[int(i)] for i in metadata.target_logits_indices.detach().cpu().tolist()]
        bonus_logits = [flat_logits[int(i)] for i in metadata.bonus_logits_indices.detach().cpu().tolist()]
        target_ids = self._argmax_many_tensor(target_logits)
        bonus_ids = self._argmax_many_tensor(bonus_logits)
        decisions = resolve_greedy_speculative_metadata(
            metadata,
            target_token_ids=target_ids,
            bonus_token_ids=bonus_ids,
            scheduled_token_counts=[len(row_logits) for row_logits in logits_by_row],
        )
        for global_row, decision in zip(row_indices, decisions, strict=True):
            tokens = decision.token_ids
            sampled_token_ids[global_row] = tokens
            sampled_counts[global_row] = decision.sampled_count
            rejected_counts[global_row] = decision.rejected_count

    def _argmax_many_tensor(self, logits_rows: list[torch.Tensor]) -> torch.Tensor:
        if not logits_rows:
            return torch.empty((0,), dtype=torch.long, device=self.engine.device)
        if len(logits_rows) == 1:
            logits = logits_rows[0].contiguous()
            if logits.device.type == "cuda":
                return cuda_ops().argmax(logits).reshape(1).to(device=logits.device, dtype=torch.long)
            return torch.argmax(logits, dim=-1).reshape(1).to(dtype=torch.long)
        logits = torch.stack([row.contiguous() for row in logits_rows], dim=0).contiguous()
        if logits.device.type == "cuda":
            return cuda_ops().argmax_many(logits).to(device=logits.device, dtype=torch.long)
        return torch.argmax(logits, dim=-1).to(dtype=torch.long)

    def _argmax_many_tensors(self, logits_rows: list[torch.Tensor]) -> list[int]:
        if not logits_rows:
            return []
        token_ids = self._argmax_many_tensor(logits_rows)
        return [int(t) for t in token_ids.detach().cpu().tolist()]

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
                return [int(cuda_ops().argmax(logits).detach().cpu().item())]
            return [int(torch.argmax(logits, dim=-1).item())]
        logits = torch.stack([row.contiguous() for row in logits_rows], dim=0).contiguous()
        if logits.device.type == "cuda":
            sampled = cuda_ops().argmax_many(logits)
        else:
            sampled = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        return [int(t) for t in sampled.detach().cpu().tolist()]

    def _snapshot_speculative_rows(
        self,
        batch: DecodeBatchPlan,
        rows: list[DecodeRequestState],
        states: list[object],
    ) -> dict[int, object]:
        snapshots: dict[int, object] = {}
        for row_idx, (row, state) in enumerate(zip(rows, states)):
            if not row.has_draft_tokens:
                continue
            snapshot_fn = getattr(state, "speculative_write_snapshot", None)
            if not callable(snapshot_fn):
                continue
            snapshots[row_idx] = snapshot_fn(int(batch.num_scheduled_tokens[row_idx]))
        return snapshots

    def _rollback_rejected_speculative_rows(
        self,
        batch: DecodeBatchPlan,
        rows: list[DecodeRequestState],
        states: list[object],
        snapshots: dict[int, object],
        sampled_token_ids: list[list[int]],
        sampled_counts: list[int],
        rejected_counts: list[int],
        commit_states: Sequence[Sequence[object] | None] | None,
    ) -> None:
        for row_idx, rejected_n in enumerate(rejected_counts):
            if int(rejected_n) <= 0:
                continue
            state = states[row_idx]
            scheduled_n = int(batch.num_scheduled_tokens[row_idx])
            commit_n = scheduled_n - int(rejected_n)
            if commit_n > 0 and commit_states is not None and row_idx < len(commit_states):
                row_commits = commit_states[row_idx]
                if row_commits is not None and len(row_commits) >= commit_n:
                    copy_from = getattr(state, "copy_from_", None)
                    if callable(copy_from):
                        copy_from(row_commits[commit_n - 1], copy_attention=True)
                        continue
            snapshot = snapshots.get(row_idx)
            if snapshot is None:
                continue
            restore = getattr(snapshot, "restore_", None)
            if not callable(restore):
                continue
            restore(state)
            if commit_n <= 0:
                continue
            start, _end = batch.row_spans[row_idx]
            commit_ids = [int(t) for t in batch.input_ids[start : start + commit_n].detach().cpu().tolist()]
            if len(commit_ids) != commit_n:
                raise RuntimeError(
                    f"speculative rollback for {rows[row_idx].request_id!r} "
                    f"needs {commit_n} commit tokens, got {len(commit_ids)}"
                )
            for token_id in commit_ids:
                self.engine.forward_one(int(token_id), state, return_logits=False)

    def _prepare_native_nextn_drafts(
        self,
        rows: list[DecodeRequestState],
        states: list[object],
        sampled_counts: list[int],
        rejected_counts: list[int],
    ) -> None:
        proposer = self.engine.speculative_proposer
        if proposer is None or not self.features.speculative_decoding:
            return
        pending: list[tuple[int, DraftRequest]] = []
        for row_idx, (row, state, sampled_n, rejected_n) in enumerate(zip(rows, states, sampled_counts, rejected_counts, strict=True)):
            if int(sampled_n) <= 0:
                continue
            if int(rejected_n) != 0:
                continue
            if row.last_sampled_token is None:
                continue
            cfg = _generation_config_from_row(row)
            if _config_needs_configured_sampling(cfg):
                continue
            raw_hidden = getattr(state, "last_raw_hidden", None)
            if raw_hidden is None:
                continue
            snapshot_fn = getattr(state, "speculative_write_snapshot", None)
            if not callable(snapshot_fn):
                continue
            try:
                snapshot_fn(1 + self.speculative_policy.max_draft)
            except Exception:
                # Paged-KV arena states need a page-table/lookahead snapshot,
                # not the canonical ring snapshot. Until that contract exists,
                # do not attach drafts that would be impossible to roll back.
                row.clear_draft_tokens()
                continue
            pending.append(
                (
                    row_idx,
                    DraftRequest(
                        history=row.token_ids,
                        max_draft=int(self.speculative_policy.max_draft),
                        signals={
                            "raw_hidden": raw_hidden,
                            "first_token": torch.tensor(
                                int(row.last_sampled_token),
                                device=raw_hidden.device,
                                dtype=torch.long,
                            ),
                            "pos": int(getattr(state, "pos", 0)),
                        },
                    ),
                )
            )
        if not pending:
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
            for idx in row_indices:
                rows[idx].clear_draft_tokens()
            return
        draft_rows = drafts[:, : int(self.speculative_policy.max_draft)].to(dtype=torch.long).contiguous()
        for local_idx, idx in enumerate(row_indices):
            row_tensor = draft_rows[local_idx].reshape(-1).contiguous()
            rows[idx].draft_token_ids_tensor = row_tensor
            rows[idx].draft_token_ids = None

def _config_needs_configured_sampling(cfg: GenerationConfig) -> bool:
    return any(
        (
            float(cfg.temperature) > 0,
            int(cfg.top_k) > 0,
            float(cfg.top_p) < 1.0,
            float(cfg.min_p) > 0.0,
            float(cfg.repetition_penalty) != 1.0,
            float(cfg.presence_penalty) != 0.0,
            float(cfg.frequency_penalty) != 0.0,
            int(cfg.no_repeat_ngram_size) > 0,
            bool(cfg.logit_bias),
            bool(cfg.bad_token_ids),
            bool(cfg.suppress_tokens),
        )
    )


def _generation_config_from_row(row: object) -> GenerationConfig:
    cfg = getattr(row, "generation_config", None)
    if isinstance(cfg, GenerationConfig):
        return cfg
    return GenerationConfig()
