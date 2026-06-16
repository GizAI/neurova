from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...core.features import RuntimeFeatures, RuntimePlan
from ...speculation import SpeculativeDecodePolicy

DEFAULT_SPECULATIVE_VERIFIER = "transaction_block"
SPECULATIVE_VERIFIER_CHOICES = ("sequential", "transaction_block")
_UNSET = object()


@dataclass(frozen=True)
class RuntimePolicyResolver:
    """Single source of truth for env/autotune/default execution policy."""

    env: Mapping[str, str] | None = None

    def _env(self) -> Mapping[str, str]:
        return self.env if self.env is not None else os.environ

    def verifier_mode(self, value: str | None = None) -> str:
        env = self._env()
        raw = value if value is not None else env.get("LANGBURST_SPECULATIVE_VERIFIER")
        if raw is None or raw == "":
            return DEFAULT_SPECULATIVE_VERIFIER
        mode = str(raw).strip().lower().replace("-", "_")
        if mode not in SPECULATIVE_VERIFIER_CHOICES:
            choices = ", ".join(SPECULATIVE_VERIFIER_CHOICES)
            raise ValueError(f"LANGBURST_SPECULATIVE_VERIFIER must be one of: {choices}")
        return mode

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = self._env().get(name)
        if raw is None or raw == "":
            return default
        text = raw.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be one of: 1/0, true/false, on/off")

    def _autotune_policy_values(self) -> dict[str, object]:
        raw = self._env().get("LANGBURST_MTP_AUTOTUNE_JSON")
        if raw is None or raw == "":
            return {}
        path = Path(raw).expanduser()
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(payload, dict) or not payload.get("keep", False):
            return {}
        policy = payload.get("policy")
        if not isinstance(policy, dict):
            return {}
        values: dict[str, object] = {}
        for key in (
            "max_draft",
            "verifier_mode",
            "adaptive",
            "min_verified",
            "accept_threshold",
            "max_rejections",
            "min_speedup",
        ):
            if key in policy:
                values[key] = policy[key]
        return values

    def _choose_value(
        self,
        explicit: object,
        env_name: str,
        tuned: dict[str, object],
        tuned_name: str,
        default: object,
    ) -> object:
        if explicit is not None and explicit is not _UNSET:
            return explicit
        raw = self._env().get(env_name)
        if raw is not None and raw != "":
            return raw
        if tuned_name in tuned:
            return tuned[tuned_name]
        return default

    def speculative_policy(
        self,
        *,
        max_draft: int | None = None,
        verifier_mode: str | None = None,
        adaptive: bool | None = None,
        min_verified: int | None = None,
        accept_threshold: float | None = None,
        max_rejections: int | None | object = _UNSET,
        min_speedup: float | None = None,
    ) -> SpeculativeDecodePolicy:
        tuned = self._autotune_policy_values()
        chosen_max_draft = self._choose_value(max_draft, "LANGBURST_MTP_MAX_DRAFT", tuned, "max_draft", 1)
        chosen_adaptive = self._choose_value(adaptive, "LANGBURST_MTP_ADAPTIVE", tuned, "adaptive", True)
        chosen_min_verified = self._choose_value(min_verified, "LANGBURST_MTP_MIN_VERIFIED", tuned, "min_verified", 1)
        chosen_accept_threshold = self._choose_value(
            accept_threshold,
            "LANGBURST_MTP_ACCEPT_THRESHOLD",
            tuned,
            "accept_threshold",
            1.0,
        )
        chosen_max_rejections = self._choose_value(
            max_rejections,
            "LANGBURST_MTP_MAX_REJECTIONS",
            tuned,
            "max_rejections",
            None,
        )
        chosen_min_speedup = self._choose_value(min_speedup, "LANGBURST_MTP_MIN_SPEEDUP", tuned, "min_speedup", 1.03)
        chosen_verifier_mode = (
            str(verifier_mode)
            if verifier_mode is not None
            else str(
                self._choose_value(
                    None,
                    "LANGBURST_SPECULATIVE_VERIFIER",
                    tuned,
                    "verifier_mode",
                    DEFAULT_SPECULATIVE_VERIFIER,
                )
            )
        )
        return SpeculativeDecodePolicy(
            max_draft=int(chosen_max_draft),
            verifier_mode=self.verifier_mode(chosen_verifier_mode),
            adaptive=(
                bool(chosen_adaptive)
                if isinstance(chosen_adaptive, bool)
                else self._env_bool(
                    "LANGBURST_MTP_ADAPTIVE",
                    bool(str(chosen_adaptive).strip().lower() in {"1", "true", "yes", "on"}),
                )
            ),
            min_verified=int(chosen_min_verified),
            accept_threshold=float(chosen_accept_threshold),
            max_rejections=(
                None
                if chosen_max_rejections is None or str(chosen_max_rejections).strip().lower() in {"none", "null", "-1"}
                else int(chosen_max_rejections)
            ),
            min_speedup=float(chosen_min_speedup),
        )

    def execution_policy(
        self,
        plan: RuntimePlan,
        *,
        speculative: SpeculativeDecodePolicy | None = None,
    ) -> "ExecutionPolicy":
        return ExecutionPolicy(features=plan.effective, speculative=speculative or self.speculative_policy())


@dataclass(frozen=True)
class ExecutionPolicy:
    """Resolved per-request execution policy.

    RuntimeFeatures owns feature gates. SpeculativeDecodePolicy owns native
    NEXTN tuning values. Downstream decode code should consume this resolved
    object instead of re-reading env or reconstructing speculative defaults.
    """

    features: RuntimeFeatures
    speculative: SpeculativeDecodePolicy

    @classmethod
    def from_plan(
        cls,
        plan: RuntimePlan,
        *,
        speculative: SpeculativeDecodePolicy | None = None,
    ) -> "ExecutionPolicy":
        return RuntimePolicyResolver().execution_policy(plan, speculative=speculative)

    def summary(self) -> dict[str, object]:
        return {
            "features": self.features.summary(),
            "speculative": {
                "max_draft": self.speculative.max_draft,
                "verifier_mode": self.speculative.verifier_mode,
                "adaptive": self.speculative.adaptive,
                "min_verified": self.speculative.min_verified,
                "accept_threshold": self.speculative.accept_threshold,
                "max_rejections": self.speculative.max_rejections,
                "min_speedup": self.speculative.min_speedup,
            },
        }
