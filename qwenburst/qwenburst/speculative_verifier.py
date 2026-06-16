from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch

from .speculative_batch import make_speculative_batch_plan, resolve_speculative_batch
from .speculation import DraftRequest, SpeculativeProposer

VerifierMode = Literal["sequential", "transaction_block"]


@dataclass
class VerifierStep:
    tokens: list[torch.Tensor]
    logits: torch.Tensor
    raw_hidden: torch.Tensor
    accepted: int = 0
    verified: int = 0
    rejected: int = 0


class NativeNextNVerifier:
    """Exact target verifier for Qwen native MTP/NEXTN proposals.

    This is the single speculative verification contract used by runtime and
    benchmarks.  Proposers only suggest tokens; this verifier is the only object
    allowed to mutate the target DecodeState.
    """

    def __init__(
        self,
        *,
        model: object,
        proposer: SpeculativeProposer,
        sample_next: Callable[[torch.Tensor], torch.Tensor],
        max_draft: int = 1,
        mode: VerifierMode = "sequential",
    ) -> None:
        if max_draft < 1:
            raise ValueError("max_draft must be >= 1")
        if mode not in {"sequential", "transaction_block"}:
            raise ValueError(f"unknown verifier mode: {mode}")
        self.model = model
        self.proposer = proposer
        self.sample_next = sample_next
        self.max_draft = int(max_draft)
        self.mode: VerifierMode = mode
        self.accepted = 0
        self.verified = 0
        self.rejected = 0
        self._pending_first: torch.Tensor | None = None

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.verified if self.verified else 0.0

    def step(
        self,
        *,
        logits: torch.Tensor,
        raw_hidden: torch.Tensor,
        state: object,
        remaining_tokens: int,
    ) -> VerifierStep:
        if remaining_tokens <= 0:
            raise ValueError("remaining_tokens must be positive")
        if self._pending_first is None:
            first = self.sample_next(logits)
            emit_first = True
        else:
            first = self._pending_first
            self._pending_first = None
            emit_first = False
        remaining_after_first = remaining_tokens - (1 if emit_first else 0)
        candidates = self._propose(
            raw_hidden=raw_hidden,
            first_token=first,
            state=state,
            remaining_after_first=remaining_after_first,
            logits_device=logits.device,
        )
        if candidates is None or candidates.numel() == 0:
            next_logits, next_raw = self.model.forward_one(
                first,
                state,
                return_hidden=True,
                return_raw_hidden=False,
            )
            return VerifierStep(tokens=([first] if emit_first else []), logits=next_logits, raw_hidden=next_raw)
        if self.mode == "transaction_block":
            return self._step_transaction_block(first, candidates, state, emit_first=emit_first, remaining_tokens=remaining_tokens)
        return self._step_sequential(first, candidates, state, emit_first=emit_first)

    def _propose(
        self,
        *,
        raw_hidden: torch.Tensor,
        first_token: torch.Tensor,
        state: object,
        remaining_after_first: int,
        logits_device: torch.device,
    ) -> torch.Tensor | None:
        if remaining_after_first <= 0:
            return None
        request = DraftRequest(
            history=(),
            max_draft=min(self.max_draft, remaining_after_first),
            signals={
                "raw_hidden": raw_hidden,
                "first_token": first_token,
                "pos": int(getattr(state, "pos", 0)),
            },
        )
        propose_tensors = getattr(self.proposer, "propose_tensors", None)
        propose_tensor = getattr(self.proposer, "propose_tensor", None)
        if callable(propose_tensors):
            return propose_tensors(request).reshape(-1).to(device=logits_device, dtype=torch.long)
        if callable(propose_tensor):
            return propose_tensor(request).reshape(1).to(device=logits_device, dtype=torch.long)
        proposal = self.proposer.propose(request)
        if not proposal.tokens:
            return None
        return torch.tensor([int(token) for token in proposal.tokens], device=logits_device, dtype=torch.long)

    def _step_sequential(self, first: torch.Tensor, candidates: torch.Tensor, state: object, *, emit_first: bool = True) -> VerifierStep:
        logits, raw_hidden = self.model.forward_one(
            first,
            state,
            return_hidden=True,
            return_raw_hidden=False,
        )
        emitted = [first] if emit_first else []
        accepted = 0
        verified = 0
        rejected = 0
        for candidate in candidates:
            target_next = self.sample_next(logits)
            verified += 1
            if bool(torch.equal(candidate.reshape(()), target_next.reshape(()))):
                logits, raw_hidden = self.model.forward_one(
                    candidate,
                    state,
                    return_hidden=True,
                    return_raw_hidden=False,
                )
                emitted.append(candidate)
                accepted += 1
                continue
            logits, raw_hidden = self.model.forward_one(
                target_next,
                state,
                return_hidden=True,
                return_raw_hidden=False,
            )
            emitted.append(target_next)
            rejected = 1
            break
        self.accepted += accepted
        self.verified += verified
        self.rejected += rejected
        return VerifierStep(
            tokens=emitted,
            logits=logits,
            raw_hidden=raw_hidden,
            accepted=accepted,
            verified=verified,
            rejected=rejected,
        )

    def _step_transaction_block(
        self,
        first: torch.Tensor,
        candidates: torch.Tensor,
        state: object,
        *,
        emit_first: bool = True,
        remaining_tokens: int,
    ) -> VerifierStep:
        snapshot_fn = getattr(state, "speculative_write_snapshot", None)
        if not callable(snapshot_fn):
            return self._step_sequential(first, candidates, state)
        plan = make_speculative_batch_plan(first, candidates, emit_first=emit_first)
        token_ids = plan.target_token_ids()
        snapshot = snapshot_fn(len(token_ids))
        verify_block = getattr(self.model, "forward_verify_block", None)
        if callable(verify_block):
            result = verify_block(token_ids, state, num_candidates=int(candidates.numel()))
            target_ids = result.target_ids
            final_logits = result.logits
            final_raw_hidden = result.hidden
        else:
            result = self.model.forward_block(
                token_ids,
                state,
                return_logits=True,
                logits_mode="all",
                commit=True,
            )
            target_ids = self._target_ids_for_candidates(result.logits, candidates.numel())
            final_logits = result.logits[-1]
            final_raw_hidden = result.raw_hiddens[-1]
        batch_result = resolve_speculative_batch(plan, target_token_ids=target_ids)
        accepted = batch_result.accepted_draft_tokens
        verified = batch_result.verified_draft_tokens
        if not batch_result.all_draft_accepted:
            snapshot.restore_(state)
            logits: torch.Tensor | None = None
            raw_hidden: torch.Tensor | None = None
            for token in batch_result.commit_tokens:
                logits, raw_hidden = self.model.forward_one(
                    token,
                    state,
                    return_hidden=True,
                    return_raw_hidden=False,
                )
            assert logits is not None and raw_hidden is not None
            self.accepted += accepted
            self.verified += verified
            self.rejected += 1
            return VerifierStep(
                tokens=batch_result.emitted_tokens,
                logits=logits,
                raw_hidden=raw_hidden,
                accepted=accepted,
                verified=verified,
                rejected=1,
            )
        accepted = verified
        emitted = batch_result.emitted_tokens
        if len(emitted) < remaining_tokens:
            bonus = self.sample_next(final_logits)
            self._pending_first = bonus.reshape(())
            emitted.append(self._pending_first)
        self.accepted += accepted
        self.verified += verified
        return VerifierStep(
            tokens=emitted,
            logits=final_logits,
            raw_hidden=final_raw_hidden,
            accepted=accepted,
            verified=verified,
        )

    def _target_ids_for_candidates(self, logits: Sequence[torch.Tensor], num_candidates: int) -> torch.Tensor:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if len(logits) < num_candidates:
            raise RuntimeError("target verifier did not return enough logits")
        check_logits = torch.stack([logits[i].contiguous() for i in range(num_candidates)], dim=0)
        if check_logits.device.type == "cuda":
            return torch.argmax(check_logits, dim=-1).to(device=check_logits.device, dtype=torch.long)
        return torch.argmax(check_logits, dim=-1).to(dtype=torch.long)


def tensor_ids(tokens: Sequence[torch.Tensor]) -> list[int]:
    return [int(token.detach().cpu().item()) for token in tokens]
