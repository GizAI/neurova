from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..ops import cuda_ops
from ..speculative_batch import NativeSpecDecodeMetadata, resolve_speculative_gpu


@dataclass(frozen=True)
class SpeculativeBatchPlan:
    """Research/test single-request speculative verifier oracle."""

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
class SpeculativeBatchDecision:
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
    resolved = resolve_speculative_gpu(
        metadata,
        target_token_ids=target_token_ids,
        bonus_token_ids=bonus_token_ids,
        scheduled_token_counts=scheduled_token_counts
        if scheduled_token_counts is not None
        else [draft_n + 1 for draft_n in metadata.num_draft_tokens],
    )
    token_rows = resolved.output_rows()
    sampled_counts = resolved.output_length_list()
    rejected_counts = resolved.rejected_count_list()
    accepted_counts = resolved.accepted_draft_count_list()
    return [
        SpeculativeBatchDecision(
            token_ids=token_rows[row_idx],
            sampled_count=sampled_counts[row_idx],
            rejected_count=rejected_counts[row_idx],
            accepted_draft_tokens=accepted_counts[row_idx],
            verified_draft_tokens=int(draft_n),
        )
        for row_idx, draft_n in enumerate(metadata.num_draft_tokens)
    ]


def resolve_speculative_batch(
    plan: SpeculativeBatchPlan,
    *,
    target_token_ids: torch.Tensor,
) -> SpeculativeBatchResult:
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
