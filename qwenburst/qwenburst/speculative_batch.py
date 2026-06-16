from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .ops import cuda_ops


@dataclass(frozen=True)
class SpeculativeBatchPlan:
    """Single-request spec-decode execution plan.

    This is the qwenburst equivalent of the vLLM spec-decode input contract:
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


def resolve_speculative_batch(
    plan: SpeculativeBatchPlan,
    *,
    target_token_ids: torch.Tensor,
) -> SpeculativeBatchResult:
    """Resolve accepted/rejected counts and exact commit/emission tokens.

    `target_token_ids[i]` is the target model token that validates
    `plan.draft_tokens[i]`.  On reject, vLLM samples the correcting target token
    and treats the remaining candidate logits as rejected; qwenburst mirrors that
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


@dataclass(frozen=True)
class DecodeBatchPlan:
    """vLLM-style compact input batch for qwenburst.

    `input_ids`, `positions`, `query_start_loc`, `logits_indices`, and
    `cu_num_logits` mirror the concepts used by vLLM's `InputBatch`, but this
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
    cuda_graph_bucket: tuple[int, int, int] | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None

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
    """Reusable vLLM-style tensors for batch input preparation."""

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
    `last_sampled_token + draft_token_ids`, matching vLLM's speculative target
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

    if scheduled_tokens_per_request is not None and len(scheduled_tokens_per_request) != len(requests):
        raise ValueError("scheduled_tokens_per_request must match request count")

    for row_idx, req in enumerate(requests):
        request_ids.append(req.request_id)
        state_indices.append(int(req.state_index))
        start = len(input_ids)
        if req.is_prefilling:
            n = req.prefill_remaining
            if max_prefill_tokens_per_request is not None:
                n = min(n, max(1, int(max_prefill_tokens_per_request)))
            if scheduled_tokens_per_request is not None:
                n = min(n, max(1, int(scheduled_tokens_per_request[row_idx])))
            chunk = req.token_ids[req.computed_tokens : req.computed_tokens + n]
            pos = list(range(req.computed_tokens, req.computed_tokens + len(chunk)))
            num_logits = 1 if chunk else 0
            num_draft = 0
        else:
            if include_sampled_token and req.last_sampled_token is not None:
                chunk = [int(req.last_sampled_token)]
                first_pos = req.computed_tokens
            else:
                chunk = []
                first_pos = req.computed_tokens
            drafts = [int(t) for t in (req.draft_token_ids or [])]
            chunk.extend(drafts)
            pos = list(range(first_pos, first_pos + len(chunk)))
            num_logits = len(chunk)
            num_draft = len(drafts)
        input_ids.extend(chunk)
        positions.extend(pos)
        scheduled.append(len(chunk))
        draft_counts.append(num_draft)
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
    if buffers is not None:
        buffers.check_capacity(num_requests=num_requests, num_tokens=num_tokens, num_logits=num_logits)
        buffers.state_indices[:num_requests] = torch.tensor(state_indices, dtype=torch.int32, device=buffers.device)
        buffers.input_ids[:num_tokens] = torch.tensor(input_ids, dtype=torch.long, device=buffers.device)
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
            cuda_graph_bucket=cuda_graph_bucket,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
        )

    return DecodeBatchPlan(
        request_ids=request_ids,
        state_indices=torch.tensor(state_indices, dtype=torch.int32, device=device),
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        query_start_loc=torch.tensor(query_start, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        logits_indices=torch.tensor(logits_indices, dtype=torch.long, device=device),
        cu_num_logits=torch.tensor(cu_num_logits, dtype=torch.int32, device=device),
        row_spans=row_spans,
        num_scheduled_tokens=scheduled,
        num_draft_tokens_per_request=draft_counts,
        cuda_graph_bucket=cuda_graph_bucket,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
    )


def sampled_and_rejected_counts(
    *,
    sampled_counts: torch.Tensor,
    cu_num_logits: torch.Tensor,
    is_prefill: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return vLLM-compatible sampled/rejected counts for each request."""

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
    """Apply vLLM-style sampled/rejected post-update to request rows."""

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
        req.draft_token_ids = None
