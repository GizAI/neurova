from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class RequestUsage:
    """OpenAI-style token and latency accounting for one generation request."""

    prompt_tokens: int
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0
    requested_completion_tokens: int | None = None
    finish_reason: str | None = None
    finish_detail: str | None = None
    queue_wait_s: float | None = None
    prefill_s: float | None = None
    prefill_tok_s: float | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    decode_s: float | None = None
    e2e_tok_s: float | None = None
    decode_tok_s: float | None = None
    mean_itl_s: float | None = None
    created_monotonic: float = field(default_factory=time.monotonic)

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, int(self.prompt_tokens) - int(self.cached_input_tokens))

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def apply_metrics(self, metrics: dict[str, Any]) -> None:
        self.completion_tokens = int(metrics.get("output_tokens") or self.completion_tokens)
        self.cached_input_tokens = int(metrics.get("cached_input_tokens") or self.cached_input_tokens)
        self.accepted_prediction_tokens = int(metrics.get("accepted_prediction_tokens") or 0)
        self.rejected_prediction_tokens = int(metrics.get("rejected_prediction_tokens") or 0)
        if metrics.get("finish_reason") is not None:
            self.finish_reason = str(metrics["finish_reason"])
        if metrics.get("finish_detail") is not None:
            self.finish_detail = str(metrics["finish_detail"])
        for key in ("queue_wait_s", "prefill_s", "prefill_tok_s", "ttft_s", "e2e_s", "decode_s", "e2e_tok_s", "decode_tok_s", "mean_itl_s"):
            value = metrics.get(key)
            if value is not None:
                setattr(self, key, float(value))

    def finish_now(self, *, completion_tokens: int, cached_input_tokens: int = 0) -> None:
        end = time.monotonic()
        self.completion_tokens = int(completion_tokens)
        self.cached_input_tokens = int(cached_input_tokens)
        self.e2e_s = end - self.created_monotonic
        self.ttft_s = self.e2e_s if completion_tokens else None
        self.decode_s = self.e2e_s if completion_tokens else None
        self.e2e_tok_s = completion_tokens / max(self.e2e_s, 1e-9)
        self.decode_tok_s = self.e2e_tok_s if completion_tokens else None

    def openai_usage(self) -> dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "prompt_tokens_details": {
                "cached_tokens": int(self.cached_input_tokens),
                "uncached_tokens": int(self.uncached_input_tokens),
            },
            "completion_tokens_details": {
                "reasoning_tokens": int(self.reasoning_tokens),
                "accepted_prediction_tokens": int(self.accepted_prediction_tokens),
                "rejected_prediction_tokens": int(self.rejected_prediction_tokens),
            },
        }

    def performance(self) -> dict[str, Any]:
        return {
            "queue_wait_s": self.queue_wait_s,
            "prefill_s": self.prefill_s,
            "prefill_tok_s": self.prefill_tok_s,
            "ttft_s": self.ttft_s,
            "e2e_s": self.e2e_s,
            "decode_s": self.decode_s,
            "e2e_tok_s": self.e2e_tok_s,
            "decode_tok_s": self.decode_tok_s,
            "mean_itl_s": self.mean_itl_s,
            "requested_completion_tokens": self.requested_completion_tokens,
            "finish_reason": self.finish_reason,
            "finish_detail": self.finish_detail,
        }
