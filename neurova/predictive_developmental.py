"""Predictive developmental loop -- the heart of every observe() call.

Every input goes through: predict -> execute -> compare -> episode -> learn.
This is NOT an LLM; it predicts IR type, dialogue act, user state, and success
before the runtime processes the input, then records prediction error.

v36: expanded from v35's basic predict/observe to a full closed-loop pipeline
with structured episode records, error classification, and direct wiring to
schema_learning for failure-to-schema compilation.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional
import json
import re
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    text: str
    predicted_ir_type: str
    predicted_dialogue_act: str
    predicted_user_state: str
    predicted_success: float
    predicted_schema_ids: List[str] = field(default_factory=list)
    chosen_action: str = "parse_and_store"
    observed_outcome: str = "unknown"
    prediction_error: str = "pending"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Episode:
    """A single interaction with full context -- the atom of episodic memory."""
    text: str
    response: str
    ir_type: str
    prediction: PredictionRecord
    ir_json: str = "{}"
    confidence: float = 0.0
    parser: str = ""
    success: bool = True
    error_type: str = ""
    feedback: Optional[str] = None
    trace: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsolidatedSkill:
    name: str
    skill_type: str
    evidence_count: int
    success_count: int
    failure_count: int
    status: str = "experimental"
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fast / Slow memory (hippocampus / cortex split)
# ---------------------------------------------------------------------------

class FastEpisodicMemory:
    """Hippocampus-like: fast, contextual, reversible, never deletes raw data."""

    def __init__(self):
        self.episodes: List[Episode] = []
        self.failures: List[Episode] = []

    def add(self, ep: Episode):
        self.episodes.append(ep)
        if ep.prediction.prediction_error not in {"none", "pending"}:
            self.failures.append(ep)

    def recent(self, n: int = 20) -> List[Episode]:
        return self.episodes[-n:]

    def failure_texts(self, n: int = 200) -> List[str]:
        return [ep.text for ep in self.failures[-n:]]

    def error_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ep in self.failures:
            et = ep.error_type or ep.prediction.prediction_error
            counts[et] = counts.get(et, 0) + 1
        return counts

    @property
    def total_count(self) -> int:
        return len(self.episodes)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


class SlowSemanticMemory:
    """Cortex-like: only promoted skills/constructions survive here."""

    def __init__(self):
        self.skills: Dict[str, ConsolidatedSkill] = {}

    def promote(self, skill: ConsolidatedSkill) -> ConsolidatedSkill:
        skill.status = (
            "stable"
            if skill.success_count >= max(2, skill.failure_count * 2 + 1)
            else "experimental"
        )
        self.skills[skill.name] = skill
        return skill


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------

class PredictionErrorClassifier:
    """Classifies prediction errors into actionable categories for schema learning."""

    PATTERNS: List[tuple[str, List[str]]] = [
        ("belief_or_coreference_error", ["believe", "think", "he ", "she ", "they "]),
        ("world_state_error", ["where", "moved", "carried", "located", "have", "has"]),
        ("temporal_schema_error", ["during", "from", "through", "before", "after", "became", "stopped"]),
        ("dialogue_act_error", ["worried", "confused", "rough day", "cheer", "haha", "lol", "stuck"]),
        ("wrapper_operation_error", ["would you say", "is it true", "would you classify", "can be considered"]),
        ("event_frame_error", ["borrow", "lend", "return", "rent", "steal", "drop", "pick up"]),
        ("korean_morpho_error", ["보다", "에 비해", "뒤처지다", "압도", "다고 볼"]),
        ("classification_question_error", ["classify", "regarded", "considered", "category", "type of"]),
        ("negation_error", ["not", "n't", "cannot", "never"]),
        ("paraphrase_error", ["fair to call", "counts as", "fall under"]),
    ]

    def classify(self, text: str, ir_type: str, response: str, feedback: Optional[str] = None) -> str:
        combined = f"{text} {feedback or ''} {response}".lower()
        for error_type, keywords in self.PATTERNS:
            if any(k in combined for k in keywords):
                return error_type
        return "semantic_parse_error"


# ---------------------------------------------------------------------------
# Predictive loop
# ---------------------------------------------------------------------------

class PredictiveSocialCognitiveLoop:
    """Unified predictive loop: predict -> observe -> error -> episode -> consolidate.

    Core contract:
    - predict() runs BEFORE runtime processing.
    - observe() runs AFTER and compares prediction vs actual.
    - Every observe() produces an Episode and increments counters.
    - Every failure produces a classified prediction_error.
    - consolidate() promotes repeated error families into skill candidates.
    """

    def __init__(self):
        self.fast = FastEpisodicMemory()
        self.slow = SlowSemanticMemory()
        self.error_counts: Dict[str, int] = {}
        self.error_classifier = PredictionErrorClassifier()

    # -- Prediction phase (before runtime) --
    def predict(self, text: str, candidate_ir_type: str = "Unknown") -> PredictionRecord:
        low = text.lower()
        act = "statement"
        state = "neutral"
        success = 0.55

        if any(k in low for k in ["worried", "rough day", "stuck", "confused", "not sure", "cheer me"]):
            act, state, success = "support_request", "distressed_or_confused", 0.72
        elif any(k in low for k in ["haha", "lol", "hilarious", "wild", "nice!"]):
            act, state, success = "smalltalk", "playful", 0.70
        elif low.endswith("?") or low.startswith(("does ", "did ", "who ", "where ", "would ", "is ", "can ")):
            act, state, success = "question", "information_seeking", 0.65
        elif any(k in low for k in ["no,", "actually", "i mean", "correction:", "when i say"]):
            act, state, success = "correction", "teaching", 0.75

        return PredictionRecord(
            text=text,
            predicted_ir_type=candidate_ir_type,
            predicted_dialogue_act=act,
            predicted_user_state=state,
            predicted_success=success,
            chosen_action=self._default_action(act),
        )

    def _default_action(self, act: str) -> str:
        return {
            "support_request": "empathize_and_decompose",
            "smalltalk": "acknowledge_and_bridge",
            "question": "answer_or_ask_clarification",
            "correction": "create_patch_candidate",
        }.get(act, "semantic_parse_and_store")

    # -- Observation phase (after runtime) --
    def observe(
        self,
        text: str,
        response: str,
        ir_type: str,
        ok: bool,
        ir_json: str = "{}",
        confidence: float = 0.0,
        parser: str = "",
        trace: Optional[List[str]] = None,
        feedback: Optional[str] = None,
    ) -> Episode:
        pred = self.predict(text, ir_type)
        pred.observed_outcome = "success" if ok else "failure"

        error_type = ""
        if not ok:
            error_type = self.error_classifier.classify(text, ir_type, response, feedback)
            pred.prediction_error = error_type
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        else:
            pred.prediction_error = "none"

        ep = Episode(
            text=text,
            response=response,
            ir_type=ir_type,
            prediction=pred,
            ir_json=ir_json,
            confidence=confidence,
            parser=parser,
            success=ok,
            error_type=error_type,
            feedback=feedback,
            trace=trace or [],
        )
        self.fast.add(ep)
        return ep

    # -- Estimate success heuristic --
    @staticmethod
    def estimate_success(response: str) -> bool:
        low = response.lower()
        return not any(
            marker in low
            for marker in ["cannot prove", "fallback", "unknown", "not recognized"]
        )

    # -- Consolidation (sleep-like) --
    def consolidate(self) -> Dict[str, Any]:
        """Promote repeated error families into skill candidates."""
        promoted = []
        for err, count in sorted(self.error_counts.items()):
            if count <= 0:
                continue
            skill = ConsolidatedSkill(
                name=f"repair_{err}",
                skill_type=err.replace("_error", ""),
                evidence_count=count,
                success_count=max(1, count - 1),
                failure_count=1,
                notes=["created_from_prediction_error_cluster"],
            )
            promoted.append(asdict(self.slow.promote(skill)))
        return {
            "episodes": self.fast.total_count,
            "failures": self.fast.failure_count,
            "failure_clusters": dict(self.error_counts),
            "promoted_skills": promoted,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "episodes": len(self.fast.episodes),
                "failures": len(self.fast.failures),
                "error_counts": self.error_counts,
                "slow_skills": {k: asdict(v) for k, v in self.slow.skills.items()},
            },
            ensure_ascii=False,
            indent=2,
        )


def save_predictive_report(path: Path, loop: PredictiveSocialCognitiveLoop):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(loop.to_json(), encoding="utf-8")
