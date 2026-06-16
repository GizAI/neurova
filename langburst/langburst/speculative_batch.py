from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .ops import cuda_ops


@dataclass(frozen=True)
class SpeculativeBatchPlan:
    """Single-request spec-decode execution plan.

    This is the langburst equivalent of the external serving engine spec-decode input contract:
    target verification receives the newly sampled token followed by draft
    tokens, then sampler output is reduced to sampled/rejected counts for state
    post-processing.
    """

    first_token: torch.Tensor
    draft_tokens: torch.Tensor
    emit_first: bool = True

    @property
    def num_draft_tokens(self) -> int:
        return int(self.draft_tokens.numel())

    @property
    def num_target_tokens(self) -> int:
        return 1 + self.num_draft_tokens

    def target_token_ids(self) -> list[int]:
        tokens: list[torch.Tensor] = [self.first_token.reshape(())]
        tokens.extend(token.reshape(()) for token in self.draft_tokens.reshape(-1))
        return [int(token.detach().cpu().item()) for token in tokens]


@dataclass(frozen=True)
class SpeculativeBatchResult:
    accepted_draft_tokens: int
    verified_draft_tokens: int
    rejected_tokens: int
    commit_tokens: list[torch.Tensor]
    emitted_tokens: list[torch.Tensor]
    all_draft_accepted: bool

    @property
    def num_sampled(self) -> int:
        return len(self.commit_tokens)

    @property
    def state_delta_tokens(self) -> int:
        return self.num_sampled


@dataclass(frozen=True)
class NativeSpecDecodeMetadata:
    """vLLM-shaped speculative decode metadata for native serving.

    The target model returns flattened logits for each scheduled speculative
    row as `[draft-check logits..., bonus logits]`.  This object is the single
    source of truth for which logits validate draft tokens and which logits
    produce the bonus token.
    """

    draft_token_ids: torch.Tensor
    num_draft_tokens: list[int]
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor

    @property
    def max_spec_len(self) -> int:
        return max(self.num_draft_tokens, default=0)

    @property
    def batch_size(self) -> int:
        return len(self.num_draft_tokens)

    def draft_rows(self) -> list[list[int]]:
        out: list[list[int]] = []
        start = 0
        draft_ids = [int(t) for t in self.draft_token_ids.detach().cpu().tolist()]
        for draft_n in self.num_draft_tokens:
            end = start + int(draft_n)
            out.append(draft_ids[start:end])
            start = end
        return out

    def select_rows(self, row_indices: Sequence[int]) -> "NativeSpecDecodeMetadata":
        selected_counts: list[int] = []
        selected_tensors: list[torch.Tensor] = []
        offsets = [0]
        for draft_n in self.num_draft_tokens:
            offsets.append(offsets[-1] + int(draft_n))
        for row in row_indices:
            row_i = int(row)
            start, end = offsets[row_i], offsets[row_i + 1]
            selected_counts.append(end - start)
            if end > start:
                selected_tensors.append(self.draft_token_ids[start:end])
        if selected_tensors:
            flat = torch.cat(selected_tensors, dim=0).to(dtype=torch.long).contiguous()
        else:
            flat = torch.empty((0,), dtype=torch.long, device=self.draft_token_ids.device)
        return NativeSpecDecodeMetadata.from_flat_drafts(
            flat,
            selected_counts,
            device=self.draft_token_ids.device,
        )

    @classmethod
    def from_flat_drafts(
        cls,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: Sequence[int],
        *,
        device: torch.device | str | None = None,
    ) -> "NativeSpecDecodeMetadata":
        device = torch.device(device) if device is not None else draft_token_ids.device
        counts = [int(n) for n in num_draft_tokens]
        cu_draft: list[int] = []
        cu_sampled: list[int] = []
        target_indices: list[int] = []
        bonus_indices: list[int] = []
        logits_indices: list[int] = []
        draft_total = 0
        sampled_total = 0
        for draft_n in counts:
            row_start = sampled_total
            target_indices.extend(range(row_start, row_start + draft_n))
            bonus_indices.append(row_start + draft_n)
            logits_indices.extend(range(row_start, row_start + draft_n + 1))
            draft_total += draft_n
            sampled_total += draft_n + 1
            cu_draft.append(draft_total)
            cu_sampled.append(sampled_total)
        return cls(
            draft_token_ids=draft_token_ids.to(device=device, dtype=torch.long).reshape(-1).contiguous(),
            num_draft_tokens=counts,
            cu_num_draft_tokens=torch.tensor(cu_draft, dtype=torch.int32, device=device),
            cu_num_sampled_tokens=torch.tensor(cu_sampled, dtype=torch.int32, device=device),
            target_logits_indices=torch.tensor(target_indices, dtype=torch.long, device=device),
            bonus_logits_indices=torch.tensor(bonus_indices, dtype=torch.long, device=device),
            logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=device),
        )

    @classmethod
    def from_draft_rows(
        cls,
        draft_token_rows: Sequence[Sequence[int]],
        *,
        device: torch.device | str = "cpu",
    ) -> "NativeSpecDecodeMetadata":
        num_draft_tokens = [len(row) for row in draft_token_rows]
        flat_drafts = [int(token) for row in draft_token_rows for token in row]
        device = torch.device(device)
        cu_draft: list[int] = []
        cu_sampled: list[int] = []
        target_indices: list[int] = []
        bonus_indices: list[int] = []
        logits_indices: list[int] = []
        draft_total = 0
        sampled_total = 0
        for draft_n in num_draft_tokens:
            row_start = sampled_total
            target_indices.extend(range(row_start, row_start + draft_n))
            bonus_indices.append(row_start + draft_n)
            logits_indices.extend(range(row_start, row_start + draft_n + 1))
            draft_total += draft_n
            sampled_total += draft_n + 1
            cu_draft.append(draft_total)
            cu_sampled.append(sampled_total)
        return cls(
            draft_token_ids=torch.tensor(flat_drafts, dtype=torch.long, device=device),
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=torch.tensor(cu_draft, dtype=torch.int32, device=device),
            cu_num_sampled_tokens=torch.tensor(cu_sampled, dtype=torch.int32, device=device),
            target_logits_indices=torch.tensor(target_indices, dtype=torch.long, device=device),
            bonus_logits_indices=torch.tensor(bonus_indices, dtype=torch.long, device=device),
            logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=device),
        )


@dataclass(frozen=True)
class SpeculativeBatchDecision:
    """Per-request result of resolving target verifier logits.

    The token list follows the continuous serving contract: it contains the
    accepted draft prefix, plus either the target correction token or the bonus
    token when every draft was accepted. The already-scheduled first sampled
    token is part of the target input/state update, not this emitted list.
    """

    token_ids: list[int]
    sampled_count: int
    rejected_count: int
    accepted_draft_tokens: int
    verified_draft_tokens: int

    @property
    def all_drafts_accepted(self) -> bool:
        return self.accepted_draft_tokens == self.verified_draft_tokens


def make_speculative_batch_plan(first_token: torch.Tensor, draft_tokens: torch.Tensor, *, emit_first: bool = True) -> SpeculativeBatchPlan:
    draft = draft_tokens.reshape(-1).to(device=first_token.device, dtype=torch.long)
    return SpeculativeBatchPlan(first_token=first_token.reshape(()).to(dtype=torch.long), draft_tokens=draft, emit_first=emit_first)


def count_accepted_prefix(draft_tokens: torch.Tensor, target_token_ids: torch.Tensor) -> int:
    draft = draft_tokens.reshape(-1).to(dtype=torch.long)
    target = target_token_ids.reshape(-1).to(device=draft.device, dtype=torch.long)
    if draft.numel() != target.numel():
        raise ValueError("draft and target token tensors must have the same length")
    if draft.numel() == 0:
        return 0
    if draft.device.type == "cuda":
        return int(cuda_ops().count_prefix_matches(draft.contiguous(), target.contiguous())[0].detach().cpu().item())
    matches = target.eq(draft)
    mismatch = torch.nonzero(~matches, as_tuple=False)
    return int(draft.numel()) if mismatch.numel() == 0 else int(mismatch[0].item())


def resolve_greedy_speculative_metadata(
    metadata: NativeSpecDecodeMetadata,
    *,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    scheduled_token_counts: Sequence[int] | None = None,
) -> list[SpeculativeBatchDecision]:
    """Resolve vLLM-shaped speculative metadata into per-row decisions.

    `target_token_ids` is flattened in `metadata.target_logits_indices` order
    and validates draft tokens. `bonus_token_ids` is one token per row in
    `metadata.bonus_logits_indices` order. This is the single reducer used by
    both the generic batched verifier path and the transaction verifier path.
    """

    target_ids = target_token_ids.reshape(-1).to(dtype=torch.long)
    bonus_ids = bonus_token_ids.reshape(-1).to(dtype=torch.long)
    if int(bonus_ids.numel()) != metadata.batch_size:
        raise ValueError("bonus_token_ids must contain one token per speculative row")
    if scheduled_token_counts is not None and len(scheduled_token_counts) != metadata.batch_size:
        raise ValueError("scheduled_token_counts must match speculative metadata batch size")
    decisions: list[SpeculativeBatchDecision] = []
    draft_offset = 0
    target_offset = 0
    for row_idx, draft_n_raw in enumerate(metadata.num_draft_tokens):
        draft_n = int(draft_n_raw)
        drafts = metadata.draft_token_ids[draft_offset : draft_offset + draft_n].to(dtype=torch.long)
        targets = target_ids[target_offset : target_offset + draft_n].to(device=drafts.device, dtype=torch.long)
        if int(targets.numel()) < draft_n:
            raise RuntimeError("target_token_ids did not include enough verifier tokens")
        draft_offset += draft_n
        target_offset += draft_n
        accepted = count_accepted_prefix(drafts, targets)
        if accepted < draft_n:
            tokens_tensor = torch.cat([drafts[:accepted], targets[accepted : accepted + 1]], dim=0)
        else:
            tokens_tensor = torch.cat(
                [drafts, bonus_ids[row_idx : row_idx + 1].to(device=drafts.device, dtype=torch.long)],
                dim=0,
            )
        tokens = [int(t) for t in tokens_tensor.detach().cpu().tolist()]
        sampled_count = len(tokens)
        scheduled_count = int(scheduled_token_counts[row_idx]) if scheduled_token_counts is not None else draft_n + 1
        rejected_count = scheduled_count - sampled_count
        if rejected_count < 0:
            raise RuntimeError(f"sampled more tokens than scheduled: {sampled_count} > {scheduled_count}")
        decisions.append(
            SpeculativeBatchDecision(
                token_ids=tokens,
                sampled_count=sampled_count,
                rejected_count=rejected_count,
                accepted_draft_tokens=accepted,
                verified_draft_tokens=draft_n,
            )
        )
    return decisions


def resolve_speculative_batch(
    plan: SpeculativeBatchPlan,
    *,
    target_token_ids: torch.Tensor,
) -> SpeculativeBatchResult:
    """Resolve accepted/rejected counts and exact commit/emission tokens.

    `target_token_ids[i]` is the target model token that validates
    `plan.draft_tokens[i]`.  On reject, external serving engine samples the correcting target token
    and treats the remaining candidate logits as rejected; langburst mirrors that
    contract for the single-request path.
    """

    verified = plan.num_draft_tokens
    accepted = count_accepted_prefix(plan.draft_tokens, target_token_ids)
    first = plan.first_token.reshape(())
    accepted_drafts = [token.reshape(()) for token in plan.draft_tokens[:accepted]]

    if accepted == verified:
        commit_tokens = [first, *accepted_drafts]
        emitted = ([first] if plan.emit_first else []) + accepted_drafts
        return SpeculativeBatchResult(
            accepted_draft_tokens=accepted,
            verified_draft_tokens=verified,
            rejected_tokens=0,
            commit_tokens=commit_tokens,
            emitted_tokens=emitted,
            all_draft_accepted=True,
        )

    correct = target_token_ids[accepted].reshape(())
    commit_tokens = [first, *accepted_drafts, correct]
    emitted = ([first] if plan.emit_first else []) + accepted_drafts + [correct]
    return SpeculativeBatchResult(
        accepted_draft_tokens=accepted,
        verified_draft_tokens=verified,
        rejected_tokens=verified - accepted,
        commit_tokens=commit_tokens,
        emitted_tokens=emitted,
        all_draft_accepted=False,
    )


def tensor_token_ids(tokens: Sequence[torch.Tensor]) -> list[int]:
    return [int(token.detach().cpu().item()) for token in tokens]


@dataclass
class DecodeRequestState:
    """Scheduler-visible request row.

    It is intentionally model-light: adapters own the actual DecodeState/KV/GDN
    buffers, while the scheduler owns token IDs, positions, and draft metadata.
    """

    request_id: str
    state_index: int
    token_ids: list[int]
    prefill_len: int | None = None
    computed_tokens: int = 0
    last_sampled_token: int | None = None
    draft_token_ids: list[int] | None = None
    draft_token_ids_tensor: torch.Tensor | None = None
    prefix_cache_hit_tokens: int = 0
    prompt_cache_key: str | None = None
    generation_config: Any | None = None
    sample_index: int = 0

    def __post_init__(self) -> None:
        if self.prefill_len is None:
            self.prefill_len = len(self.token_ids)

    @property
    def total_len(self) -> int:
        return len(self.token_ids)

    @property
    def prefill_remaining(self) -> int:
        return max(0, int(self.prefill_len or 0) - self.computed_tokens)

    @property
    def is_prefilling(self) -> bool:
        return self.prefill_remaining > 0

    @property
    def num_draft_tokens(self) -> int:
        if self.draft_token_ids_tensor is not None:
            return int(self.draft_token_ids_tensor.numel())
        return len(self.draft_token_ids or [])

    @property
    def has_draft_tokens(self) -> bool:
        return self.num_draft_tokens > 0

    def draft_tensor(self, *, device: torch.device | str) -> torch.Tensor:
        if self.draft_token_ids_tensor is not None:
            return self.draft_token_ids_tensor.to(device=device, dtype=torch.long).reshape(-1).contiguous()
        if self.draft_token_ids:
            return torch.tensor(self.draft_token_ids, dtype=torch.long, device=device)
        return torch.empty((0,), dtype=torch.long, device=device)

    def clear_draft_tokens(self) -> None:
        self.draft_token_ids = None
        self.draft_token_ids_tensor = None


@dataclass(frozen=True)
class DecodeBatchPlan:
    """continuous-serving compact input batch for langburst.

    `input_ids`, `positions`, `query_start_loc`, `logits_indices`, and
    `cu_num_logits` mirror the concepts used by the reference runtime's `InputBatch`, but this
    object is pure Python/Torch and small enough to use in CPU tests and as a
    future CUDA Graph/static-buffer contract.
    """

    request_ids: list[str]
    state_indices: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    logits_indices: torch.Tensor
    cu_num_logits: torch.Tensor
    row_spans: tuple[tuple[int, int], ...]
    num_scheduled_tokens: list[int]
    num_draft_tokens_per_request: list[int]
    is_prefill: list[bool]
    cuda_graph_bucket: tuple[int, int, int] | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    spec_decode_metadata: NativeSpecDecodeMetadata | None = None

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def num_reqs(self) -> int:
        return self.num_requests

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def idx_mapping(self) -> torch.Tensor:
        return self.state_indices

    @property
    def num_draft_tokens(self) -> int:
        return int(sum(self.num_draft_tokens_per_request))

    @property
    def num_logits(self) -> int:
        return int(self.logits_indices.numel())


class DecodeInputBuffers:
    """Reusable continuous-serving tensors for batch input preparation."""

    def __init__(self, *, max_num_requests: int, max_num_tokens: int, device: torch.device | str = "cpu") -> None:
        if max_num_requests < 1:
            raise ValueError("max_num_requests must be >= 1")
        if max_num_tokens < 1:
            raise ValueError("max_num_tokens must be >= 1")
        self.max_num_requests = int(max_num_requests)
        self.max_num_tokens = int(max_num_tokens)
        self.device = torch.device(device)
        self.input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.long, device=self.device)
        self.positions = torch.zeros((self.max_num_tokens,), dtype=torch.long, device=self.device)
        self.query_start_loc = torch.zeros((self.max_num_requests + 1,), dtype=torch.int32, device=self.device)
        self.seq_lens = torch.zeros((self.max_num_requests,), dtype=torch.int32, device=self.device)
        self.logits_indices = torch.zeros((self.max_num_tokens,), dtype=torch.long, device=self.device)
        self.cu_num_logits = torch.zeros((self.max_num_requests + 1,), dtype=torch.int32, device=self.device)
        self.state_indices = torch.zeros((self.max_num_requests,), dtype=torch.int32, device=self.device)

    def check_capacity(self, *, num_requests: int, num_tokens: int, num_logits: int) -> None:
        if num_requests > self.max_num_requests:
            raise ValueError(f"num_requests={num_requests} exceeds buffer capacity {self.max_num_requests}")
        if num_tokens > self.max_num_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds buffer capacity {self.max_num_tokens}")
        if num_logits > self.max_num_tokens:
            raise ValueError(f"num_logits={num_logits} exceeds buffer capacity {self.max_num_tokens}")


def select_decode_batch_rows(plan: DecodeBatchPlan, row_indices: Sequence[int]) -> DecodeBatchPlan:
    """Return a compact plan containing a subset of request rows.

    This keeps the vLLM-shaped speculative metadata and paged-KV tensors aligned
    when the native runner routes speculative rows through a verifier-specific
    model entry point instead of the generic logits path.
    """

    if not row_indices:
        raise ValueError("row_indices must not be empty")
    rows = [int(row) for row in row_indices]
    device = plan.input_ids.device
    request_ids: list[str] = []
    state_indices: list[torch.Tensor] = []
    input_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    query_start: list[int] = [0]
    seq_lens: list[torch.Tensor] = []
    logits_indices: list[int] = []
    cu_num_logits: list[int] = [0]
    row_spans: list[tuple[int, int]] = []
    scheduled: list[int] = []
    draft_counts: list[int] = []
    prefill_flags: list[bool] = []
    slot_parts: list[torch.Tensor] = []

    for row in rows:
        start, end = plan.row_spans[row]
        n = int(end) - int(start)
        if n <= 0:
            raise ValueError(f"selected row {row} has no scheduled tokens")
        request_ids.append(plan.request_ids[row])
        state_indices.append(plan.state_indices[row].reshape(1))
        input_parts.append(plan.input_ids[start:end])
        position_parts.append(plan.positions[start:end])
        seq_lens.append(plan.seq_lens[row].reshape(1))
        new_start = query_start[-1]
        new_end = new_start + n
        row_spans.append((new_start, new_end))
        query_start.append(new_end)
        scheduled.append(int(plan.num_scheduled_tokens[row]))
        draft_counts.append(int(plan.num_draft_tokens_per_request[row]))
        prefill_flags.append(bool(plan.is_prefill[row]))
        if n:
            logits_indices.extend(range(new_start, new_end))
        cu_num_logits.append(cu_num_logits[-1] + n)
        if plan.slot_mapping is not None:
            slot_parts.append(plan.slot_mapping[start:end])

    block_tables = plan.block_tables[rows].contiguous() if plan.block_tables is not None else None
    slot_mapping = torch.cat(slot_parts, dim=0).contiguous() if slot_parts else None
    metadata = plan.spec_decode_metadata.select_rows(rows) if plan.spec_decode_metadata is not None else None
    return DecodeBatchPlan(
        request_ids=request_ids,
        state_indices=torch.cat(state_indices, dim=0).to(device=device, dtype=torch.int32).contiguous(),
        input_ids=torch.cat(input_parts, dim=0).to(device=device, dtype=torch.long).contiguous(),
        positions=torch.cat(position_parts, dim=0).to(device=device, dtype=torch.long).contiguous(),
        query_start_loc=torch.tensor(query_start, dtype=torch.int32, device=device),
        seq_lens=torch.cat(seq_lens, dim=0).to(device=device, dtype=torch.int32).contiguous(),
        logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=device),
        cu_num_logits=torch.tensor(cu_num_logits, dtype=torch.int32, device=device),
        row_spans=tuple(row_spans),
        num_scheduled_tokens=scheduled,
        num_draft_tokens_per_request=draft_counts,
        is_prefill=prefill_flags,
        cuda_graph_bucket=plan.cuda_graph_bucket,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        spec_decode_metadata=metadata,
    )


def build_decode_batch_plan(
    requests: Sequence[DecodeRequestState],
    *,
    device: torch.device | str = "cpu",
    buffers: DecodeInputBuffers | None = None,
    max_prefill_tokens_per_request: int | None = None,
    scheduled_tokens_per_request: Sequence[int] | None = None,
    cuda_graph_bucket: tuple[int, int, int] | None = None,
    block_tables: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    include_sampled_token: bool = True,
) -> DecodeBatchPlan:
    """Build one scheduled target batch from request rows.

    Prefill rows schedule the next prompt chunk. Decode rows schedule
    `last_sampled_token + draft_token_ids`, matching the reference runtime's speculative target
    input convention.
    """

    if not requests:
        raise ValueError("at least one request is required")
    request_ids: list[str] = []
    state_indices: list[int] = []
    input_ids: list[int] = []
    positions: list[int] = []
    query_start: list[int] = [0]
    seq_lens: list[int] = []
    logits_indices: list[int] = []
    cu_num_logits: list[int] = [0]
    scheduled: list[int] = []
    draft_counts: list[int] = []
    draft_token_rows: list[list[int]] = []
    draft_token_tensors: list[torch.Tensor] = []
    draft_copy_ranges: list[tuple[int, torch.Tensor]] = []
    prefill_flags: list[bool] = []

    if scheduled_tokens_per_request is not None and len(scheduled_tokens_per_request) != len(requests):
        raise ValueError("scheduled_tokens_per_request must match request count")

    for row_idx, req in enumerate(requests):
        request_ids.append(req.request_id)
        state_indices.append(int(req.state_index))
        start = len(input_ids)
        if req.is_prefilling:
            prefill_flags.append(True)
            n = req.prefill_remaining
            if max_prefill_tokens_per_request is not None:
                n = min(n, max(1, int(max_prefill_tokens_per_request)))
            if scheduled_tokens_per_request is not None:
                n = min(n, max(1, int(scheduled_tokens_per_request[row_idx])))
            chunk = req.token_ids[req.computed_tokens : req.computed_tokens + n]
            pos = list(range(req.computed_tokens, req.computed_tokens + len(chunk)))
            num_logits = 1 if chunk else 0
            num_draft = 0
            drafts_for_metadata: list[int] = []
        else:
            prefill_flags.append(False)
            if include_sampled_token and req.last_sampled_token is not None:
                chunk = [int(req.last_sampled_token)]
                first_pos = req.computed_tokens
            else:
                chunk = []
                first_pos = req.computed_tokens
            draft_tensor = req.draft_tensor(device=(buffers.device if buffers is not None else device))
            drafts = [int(t) for t in (req.draft_token_ids or [])] if req.draft_token_ids_tensor is None else []
            num_draft = int(draft_tensor.numel())
            if req.draft_token_ids_tensor is None:
                chunk.extend(drafts)
            else:
                draft_start = len(input_ids) + len(chunk)
                chunk.extend([0] * num_draft)
                if num_draft:
                    draft_copy_ranges.append((draft_start, draft_tensor))
                    draft_token_tensors.append(draft_tensor)
            pos = list(range(first_pos, first_pos + len(chunk)))
            num_logits = len(chunk)
            drafts_for_metadata = drafts if req.draft_token_ids_tensor is None else []
        input_ids.extend(chunk)
        positions.extend(pos)
        scheduled.append(len(chunk))
        draft_counts.append(num_draft)
        draft_token_rows.append(drafts_for_metadata)
        query_start.append(len(input_ids))
        seq_lens.append(req.computed_tokens + len(chunk))
        if num_logits:
            logits_start = len(input_ids) - num_logits
            logits_indices.extend(range(logits_start, logits_start + num_logits))
        cu_num_logits.append(cu_num_logits[-1] + num_logits)
        if len(input_ids) == start:
            raise ValueError(f"request {req.request_id!r} scheduled no tokens")

    num_requests = len(request_ids)
    num_tokens = len(input_ids)
    num_logits = len(logits_indices)
    row_spans = tuple((int(query_start[i]), int(query_start[i + 1])) for i in range(num_requests))
    metadata_device = buffers.device if buffers is not None else torch.device(device)
    if any(draft_counts):
        if draft_token_tensors:
            flat_tensors: list[torch.Tensor] = []
            tensor_iter = iter(draft_token_tensors)
            for count, row in zip(draft_counts, draft_token_rows, strict=True):
                if int(count) <= 0:
                    continue
                if row:
                    flat_tensors.append(torch.tensor(row, dtype=torch.long, device=metadata_device))
                else:
                    flat_tensors.append(next(tensor_iter).to(device=metadata_device, dtype=torch.long).reshape(-1))
            flat_drafts = torch.cat(flat_tensors, dim=0).contiguous() if flat_tensors else torch.empty((0,), dtype=torch.long, device=metadata_device)
            spec_decode_metadata = NativeSpecDecodeMetadata.from_flat_drafts(flat_drafts, draft_counts, device=metadata_device)
        else:
            spec_decode_metadata = NativeSpecDecodeMetadata.from_draft_rows(draft_token_rows, device=metadata_device)
    else:
        spec_decode_metadata = None
    if buffers is not None:
        buffers.check_capacity(num_requests=num_requests, num_tokens=num_tokens, num_logits=num_logits)
        buffers.state_indices[:num_requests] = torch.tensor(state_indices, dtype=torch.int32, device=buffers.device)
        buffers.input_ids[:num_tokens] = torch.tensor(input_ids, dtype=torch.long, device=buffers.device)
        for draft_start, draft_tensor in draft_copy_ranges:
            draft_n = int(draft_tensor.numel())
            buffers.input_ids[draft_start : draft_start + draft_n].copy_(draft_tensor.to(device=buffers.device, dtype=torch.long))
        buffers.positions[:num_tokens] = torch.tensor(positions, dtype=torch.long, device=buffers.device)
        buffers.query_start_loc[: num_requests + 1] = torch.tensor(query_start, dtype=torch.int32, device=buffers.device)
        buffers.seq_lens[:num_requests] = torch.tensor(seq_lens, dtype=torch.int32, device=buffers.device)
        buffers.logits_indices[:num_logits] = torch.tensor(logits_indices, dtype=torch.long, device=buffers.device)
        buffers.cu_num_logits[: num_requests + 1] = torch.tensor(cu_num_logits, dtype=torch.int32, device=buffers.device)
        return DecodeBatchPlan(
            request_ids=request_ids,
            state_indices=buffers.state_indices[:num_requests],
            input_ids=buffers.input_ids[:num_tokens],
            positions=buffers.positions[:num_tokens],
            query_start_loc=buffers.query_start_loc[: num_requests + 1],
            seq_lens=buffers.seq_lens[:num_requests],
            logits_indices=buffers.logits_indices[:num_logits],
            cu_num_logits=buffers.cu_num_logits[: num_requests + 1],
            row_spans=row_spans,
            num_scheduled_tokens=scheduled,
            num_draft_tokens_per_request=draft_counts,
            is_prefill=prefill_flags,
            cuda_graph_bucket=cuda_graph_bucket,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            spec_decode_metadata=spec_decode_metadata,
        )

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    for draft_start, draft_tensor in draft_copy_ranges:
        draft_n = int(draft_tensor.numel())
        input_tensor[draft_start : draft_start + draft_n].copy_(draft_tensor.to(device=input_tensor.device, dtype=torch.long))
    return DecodeBatchPlan(
        request_ids=request_ids,
        state_indices=torch.tensor(state_indices, dtype=torch.int32, device=device),
        input_ids=input_tensor,
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        query_start_loc=torch.tensor(query_start, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=device),
        cu_num_logits=torch.tensor(cu_num_logits, dtype=torch.int32, device=device),
        row_spans=row_spans,
        num_scheduled_tokens=scheduled,
        num_draft_tokens_per_request=draft_counts,
        is_prefill=prefill_flags,
        cuda_graph_bucket=cuda_graph_bucket,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        spec_decode_metadata=spec_decode_metadata,
    )


def sampled_and_rejected_counts(
    *,
    sampled_counts: torch.Tensor,
    cu_num_logits: torch.Tensor,
    is_prefill: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch-runtime-compatible sampled/rejected counts for each request."""

    sampled = sampled_counts.to(dtype=torch.int32).clone()
    logits_per_req = (cu_num_logits[1:] - cu_num_logits[:-1]).to(dtype=torch.int32)
    if is_prefill is not None:
        sampled = torch.where(is_prefill.to(device=sampled.device, dtype=torch.bool), torch.zeros_like(sampled), sampled)
    rejected = logits_per_req - sampled
    if is_prefill is not None:
        rejected = torch.where(is_prefill.to(device=rejected.device, dtype=torch.bool), torch.zeros_like(rejected), rejected)
    return sampled, rejected


def apply_decode_post_update(
    requests: Sequence[DecodeRequestState],
    *,
    batch: DecodeBatchPlan,
    sampled_token_ids: Sequence[Sequence[int]],
    sampled_counts: Sequence[int],
    rejected_counts: Sequence[int],
) -> None:
    """Apply continuous-serving sampled/rejected post-update to request rows."""

    if len(requests) != batch.num_requests:
        raise ValueError("request count and batch row count differ")
    if len(sampled_counts) != batch.num_requests or len(rejected_counts) != batch.num_requests:
        raise ValueError("sampled/rejected counts must match batch rows")
    if len(sampled_token_ids) != batch.num_requests:
        raise ValueError("sampled token rows must match batch rows")

    for row, req in enumerate(requests):
        sampled_n = int(sampled_counts[row])
        rejected_n = int(rejected_counts[row])
        query_len = int(batch.num_scheduled_tokens[row])
        if sampled_n:
            row_tokens = [int(t) for t in sampled_token_ids[row][:sampled_n]]
            req.token_ids.extend(row_tokens)
            req.last_sampled_token = row_tokens[-1]
        computed_delta = query_len - rejected_n
        if computed_delta:
            req.computed_tokens += computed_delta
        req.clear_draft_tokens()
