from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class DraftProposal:
    method: str
    tokens: list[int]


@dataclass(frozen=True)
class DraftRequest:
    """Inputs visible to a proposer before target verification.

    Token-only proposers can use only `history`. Learned proposers such as
    native MTP, EAGLE, and Medusa can consume optional `signals` populated by a
    model adapter without changing the target verifier/commit path.
    """

    history: Sequence[int]
    max_draft: int
    signals: Mapping[str, Any] | None = None


class SpeculativeProposer(Protocol):
    method: str

    def propose(self, request: DraftRequest) -> DraftProposal: ...


@dataclass(frozen=True)
class SpeculativeProbeResult:
    method: str
    total: int
    accepted: int
    min_accept_rate: float

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def viable(self) -> bool:
        return self.total > 0 and self.accept_rate >= self.min_accept_rate


@dataclass(frozen=True)
class SpeculativeBenchmarkResult:
    method: str
    tokens: int
    target_seconds: float
    speculative_seconds: float
    accepted_second_tokens: int
    verified_steps: int
    identical: bool

    @property
    def target_tok_s(self) -> float:
        return self.tokens / self.target_seconds if self.target_seconds > 0 else 0.0

    @property
    def speculative_tok_s(self) -> float:
        return self.tokens / self.speculative_seconds if self.speculative_seconds > 0 else 0.0

    @property
    def speedup(self) -> float:
        return self.speculative_tok_s / self.target_tok_s if self.target_tok_s > 0 else 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted_second_tokens / self.verified_steps if self.verified_steps else 0.0

    @property
    def keep(self) -> bool:
        return self.identical and self.accept_rate > 0.0 and self.speedup > 1.03

