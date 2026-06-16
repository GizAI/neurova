from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

MAX_NATIVE_NEXTN_DRAFT = 10


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
    rollback_tokens: int = 0
    fallback_reason: str | None = None
    min_speedup: float = 1.03

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
        return (
            self.identical
            and self.fallback_reason is None
            and self.accept_rate > 0.0
            and self.speedup > self.min_speedup
        )


@dataclass(frozen=True)
class SpeculativeDecodePolicy:
    """Runtime policy for exact native MTP/NEXTN verification.

    Speculation is only allowed to stay enabled when the target verifier keeps
    greedy identity and the measured path is speed-positive.  This object keeps
    that contract shared by runtime, CLI probes, and benchmarks.
    """

    max_draft: int = 1
    verifier_mode: str = "transaction_block"
    adaptive: bool = True
    min_verified: int = 1
    accept_threshold: float = 1.0
    max_rejections: int | None = None
    min_speedup: float = 1.03

    def __post_init__(self) -> None:
        if self.max_draft < 1 or self.max_draft > MAX_NATIVE_NEXTN_DRAFT:
            raise ValueError(f"max_draft must be in [1, {MAX_NATIVE_NEXTN_DRAFT}]")
        if self.verifier_mode not in {"sequential", "transaction_block"}:
            raise ValueError("verifier_mode must be sequential or transaction_block")
        if self.min_verified < 1:
            raise ValueError("min_verified must be >= 1")
        if self.accept_threshold < 0.0 or self.accept_threshold > 1.0:
            raise ValueError("accept_threshold must be in [0, 1]")
        if self.max_rejections is not None and self.max_rejections < 0:
            raise ValueError("max_rejections must be >= 0")
        if self.min_speedup <= 0:
            raise ValueError("min_speedup must be positive")


@dataclass
class SpeculativeDecodeStats:
    method: str = "native_nextn"
    max_draft: int = 1
    verifier_mode: str = "transaction_block"
    accepted_draft_tokens: int = 0
    verified_draft_tokens: int = 0
    verifier_steps: int = 0
    rejected_steps: int = 0
    rollback_tokens: int = 0
    fallback_reason: str | None = None

    @property
    def accept_rate(self) -> float:
        if self.verified_draft_tokens <= 0:
            return 0.0
        return self.accepted_draft_tokens / self.verified_draft_tokens


@dataclass(frozen=True)
class SpeculativeDecodeResult:
    ids: list[int]
    stats: SpeculativeDecodeStats
