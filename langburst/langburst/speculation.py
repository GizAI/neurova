from __future__ import annotations

from dataclasses import dataclass, field
import math
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
class TargetVerification:
    target_ids: Any
    logits: Any
    raw_hidden: Any


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
    min_verified: int = 32
    accept_threshold: float = 0.30
    max_rejections: int | None = None
    min_speedup: float = 1.03
    latency_ema_alpha: float = 0.20
    latency_min_verified: int = 8
    draft_candidates: tuple[int, ...] = (1, 2, 4, 8)
    min_free_vram_mib: int = 256

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
        if not 0.0 < self.latency_ema_alpha <= 1.0:
            raise ValueError("latency_ema_alpha must be in (0, 1]")
        if self.latency_min_verified < 1:
            raise ValueError("latency_min_verified must be >= 1")
        if not self.draft_candidates:
            raise ValueError("draft_candidates must not be empty")
        for candidate in self.draft_candidates:
            if candidate < 1 or candidate > MAX_NATIVE_NEXTN_DRAFT:
                raise ValueError(f"draft candidate must be in [1, {MAX_NATIVE_NEXTN_DRAFT}]")
        if self.min_free_vram_mib < 0:
            raise ValueError("min_free_vram_mib must be >= 0")


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


@dataclass
class SpeculativeAcceptanceTracker:
    """Request-batch runtime gate for native NEXTN/MTP.

    The serving path should keep speculation only when it increases accepted
    tokens per target verifier pass.  This tracker is deliberately independent
    of proposer/model details so native MTP, EAGLE, or another proposer can
    share the same accounting contract.
    """

    policy: SpeculativeDecodePolicy
    accepted_draft_tokens: int = 0
    verified_draft_tokens: int = 0
    verifier_steps: int = 0
    rejected_steps: int = 0
    baseline_ms_per_token_ema: float | None = None
    speculative_ms_per_output_token_ema: float | None = None
    target_pass_ms_ema: float | None = None
    draft_pass_ms_ema: float | None = None
    # candidate K -> (accepted_draft_tokens, verified_draft_tokens, steps, spec_ms_per_output_token_ema)
    draft_totals: dict[int, tuple[int, int, int, float | None]] = field(default_factory=dict)
    champion_draft: int | None = None

    def _ema(self, current: float | None, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            return current if current is not None else 0.0
        if current is None:
            return float(value)
        alpha = float(self.policy.latency_ema_alpha)
        return (1.0 - alpha) * current + alpha * float(value)

    def record_baseline(self, *, elapsed_ms: float, output_tokens: int) -> None:
        tokens = int(output_tokens)
        if tokens <= 0:
            return
        self.baseline_ms_per_token_ema = self._ema(
            self.baseline_ms_per_token_ema,
            float(elapsed_ms) / float(tokens),
        )

    def record(
        self,
        *,
        accepted_counts: Sequence[int],
        verified_counts: Sequence[int],
        elapsed_ms: float | None = None,
        output_tokens: int | None = None,
        target_pass_ms: float | None = None,
        draft_pass_ms: float | None = None,
    ) -> None:
        if len(accepted_counts) != len(verified_counts):
            raise ValueError("accepted_counts and verified_counts length mismatch")
        for accepted, verified in zip(accepted_counts, verified_counts, strict=True):
            verified_n = int(verified)
            if verified_n <= 0:
                continue
            accepted_n = int(accepted)
            if accepted_n < 0 or accepted_n > verified_n:
                raise ValueError("accepted draft count must be in [0, verified]")
            self.verified_draft_tokens += verified_n
            self.accepted_draft_tokens += accepted_n
            self.verifier_steps += 1
            if accepted_n < verified_n:
                self.rejected_steps += 1
            old_accepted, old_verified, old_steps, old_ms = self.draft_totals.get(verified_n, (0, 0, 0, None))
            next_ms = old_ms
            if elapsed_ms is not None and output_tokens is not None and int(output_tokens) > 0:
                next_ms = self._ema(old_ms, float(elapsed_ms) / float(output_tokens))
            self.draft_totals[verified_n] = (
                old_accepted + accepted_n,
                old_verified + verified_n,
                old_steps + 1,
                next_ms,
            )
        if elapsed_ms is not None and output_tokens is not None and int(output_tokens) > 0:
            self.speculative_ms_per_output_token_ema = self._ema(
                self.speculative_ms_per_output_token_ema,
                float(elapsed_ms) / float(output_tokens),
            )
        if target_pass_ms is not None:
            self.target_pass_ms_ema = self._ema(self.target_pass_ms_ema, float(target_pass_ms))
        if draft_pass_ms is not None:
            self.draft_pass_ms_ema = self._ema(self.draft_pass_ms_ema, float(draft_pass_ms))

    @property
    def accept_rate(self) -> float:
        if self.verified_draft_tokens <= 0:
            return 0.0
        return self.accepted_draft_tokens / self.verified_draft_tokens

    @property
    def mean_accepted_per_pass(self) -> float:
        if self.verifier_steps <= 0:
            return 0.0
        return self.accepted_draft_tokens / self.verifier_steps

    def should_propose(self) -> bool:
        if not self.policy.adaptive:
            return True
        if self.verified_draft_tokens < self.policy.min_verified:
            return True
        if self.policy.max_rejections is not None and self.rejected_steps > self.policy.max_rejections:
            return False
        if self.accept_rate < float(self.policy.accept_threshold):
            return False
        if self.verified_draft_tokens < self.policy.latency_min_verified:
            return True
        if self.baseline_ms_per_token_ema is None or self.speculative_ms_per_output_token_ema is None:
            return True
        return self.speculative_ms_per_output_token_ema < (
            self.baseline_ms_per_token_ema / float(self.policy.min_speedup)
        )

    def current_max_draft(self) -> int:
        candidates = tuple(
            sorted(
                {
                    int(v)
                    for v in self.policy.draft_candidates
                    if 1 <= int(v) <= int(self.policy.max_draft)
                }
            )
        )
        if not candidates:
            return int(self.policy.max_draft)
        if not self.policy.adaptive:
            return candidates[-1]
        if self.verified_draft_tokens < self.policy.min_verified:
            for candidate in candidates:
                _accepted, _verified, steps, _ms = self.draft_totals.get(candidate, (0, 0, 0, None))
                if steps < self.policy.latency_min_verified:
                    self.champion_draft = candidate
                    return int(candidate)
            self.champion_draft = candidates[0]
            return int(candidates[0])
        if not self.should_propose():
            return candidates[0]
        best = candidates[0]
        best_score = -1.0
        best_ms = float("inf")
        for candidate in candidates:
            accepted, _verified, steps, spec_ms = self.draft_totals.get(candidate, (0, 0, 0, None))
            if steps <= 0:
                self.champion_draft = candidate
                return int(candidate)
            if (
                self.baseline_ms_per_token_ema is not None
                and spec_ms is not None
                and spec_ms >= self.baseline_ms_per_token_ema / float(self.policy.min_speedup)
            ):
                continue
            score = float(accepted) / float(steps)
            if score > best_score or (score == best_score and spec_ms is not None and spec_ms < best_ms):
                best = candidate
                best_score = score
                best_ms = spec_ms if spec_ms is not None else best_ms
        self.champion_draft = best
        return int(best)

    def summary(self) -> dict[str, object]:
        return {
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "verified_draft_tokens": self.verified_draft_tokens,
            "verifier_steps": self.verifier_steps,
            "rejected_steps": self.rejected_steps,
            "accept_rate": self.accept_rate,
            "mean_accepted_per_pass": self.mean_accepted_per_pass,
            "baseline_ms_per_token_ema": self.baseline_ms_per_token_ema,
            "speculative_ms_per_output_token_ema": self.speculative_ms_per_output_token_ema,
            "target_pass_ms_ema": self.target_pass_ms_ema,
            "draft_pass_ms_ema": self.draft_pass_ms_ema,
            "latency_min_verified": self.policy.latency_min_verified,
            "min_free_vram_mib": self.policy.min_free_vram_mib,
            "current_max_draft": self.current_max_draft(),
            "draft_totals": {
                int(k): {
                    "accepted_draft_tokens": v[0],
                    "verified_draft_tokens": v[1],
                    "steps": v[2],
                    "speculative_ms_per_output_token_ema": v[3],
                }
                for k, v in self.draft_totals.items()
            },
            "champion_draft": self.champion_draft,
            "should_propose": self.should_propose(),
        }


@dataclass(frozen=True)
class SpeculativeDecodeResult:
    ids: list[int]
    stats: SpeculativeDecodeStats
