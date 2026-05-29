from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List
from .memory import EvidenceGraphMemory


@dataclass
class PromotionReport:
    candidate_id: str
    promoted: bool
    score: float
    detail: Dict


class RegressionGatedSelfImprover:
    """Failure-driven promotion gate.

    It does not mutate core code. It turns failures into candidate lessons/tests and
    promotes only if supplied regression checks pass.
    """
    def __init__(self, memory: EvidenceGraphMemory):
        self.memory = memory

    def propose_from_failure(self, task: str, failure_type: str, lesson: str) -> str:
        self.memory.log_trajectory(task, "failure", failure_type, {"lesson": lesson}, lesson)
        return self.memory.add_promotion_candidate("lesson", {"task": task, "failure_type": failure_type, "lesson": lesson}, status="candidate")

    def promote_if_passes(self, candidate_id: str, checks: List[Callable[[], bool]]) -> PromotionReport:
        results = []
        for check in checks:
            try:
                results.append(bool(check()))
            except Exception:
                results.append(False)
        score = sum(results) / max(1, len(results))
        promoted = bool(results) and all(results)
        status = "promoted" if promoted else "rejected"
        self.memory.conn.execute("UPDATE promotion_candidates SET status=?, score=?, detail_json=? WHERE id=?", (status, score, str({"checks": results}), candidate_id))
        self.memory.log_action("PROMOTION_GATE", candidate_id, {"checks": results, "status": status}, score)
        self.memory.conn.commit()
        return PromotionReport(candidate_id, promoted, score, {"checks": results, "status": status})


# ===========================================================================
# V36 Evolution: Self Model
# ===========================================================================

@dataclass
class SelfModel:
    """BrainOS's model of its own capabilities, weaknesses, and priorities."""
    known_domains: list = field(default_factory=list)
    unknown_domains: list = field(default_factory=list)
    weak_areas: dict = field(default_factory=dict)  # area -> failure_rate
    strong_areas: dict = field(default_factory=dict)  # area -> success_rate
    confidence_overall: float = 0.5
    total_episodes: int = 0
    total_schemas: int = 0
    total_errors: int = 0
    active_goals: list = field(default_factory=list)
    learning_priorities: list = field(default_factory=list)

    def update_from_stats(self, episodes: int, errors: int, schemas: int, error_counts: dict) -> None:
        self.total_episodes = episodes
        self.total_errors = errors
        self.total_schemas = schemas
        total = max(1, episodes)
        self.weak_areas = {k: v / total for k, v in error_counts.items()}
        self.learning_priorities = sorted(self.weak_areas, key=self.weak_areas.get, reverse=True)[:5]
        self.confidence_overall = max(0.1, 1.0 - (errors / total))

    def i_know(self, domain: str) -> bool:
        return domain in self.known_domains

    def i_dont_know(self, domain: str) -> bool:
        return domain in self.unknown_domains or domain not in self.known_domains

    def report(self) -> dict:
        return {
            "confidence": round(self.confidence_overall, 3),
            "total_episodes": self.total_episodes,
            "total_errors": self.total_errors,
            "total_schemas": self.total_schemas,
            "weak_areas": {k: round(v, 3) for k, v in list(self.weak_areas.items())[:10]},
            "learning_priorities": self.learning_priorities[:5],
        }


class SelfModelManager:
    """Manages the SelfModel by querying memory and predictive loop stats."""

    def __init__(self, memory, predictive_loop=None, schema_substrate=None):
        self.memory = memory
        self.predictive_loop = predictive_loop
        self.schema_substrate = schema_substrate
        self.model = SelfModel()

    def refresh(self) -> SelfModel:
        episodes = 0
        errors = 0
        error_counts = {}

        if self.predictive_loop:
            episodes = self.predictive_loop.fast.total_count
            errors = self.predictive_loop.fast.failure_count
            error_counts = dict(self.predictive_loop.error_counts)

        schemas = 0
        if self.schema_substrate:
            schemas = self.schema_substrate.memory.count("schema_candidates")

        self.model.update_from_stats(episodes, errors, schemas, error_counts)
        return self.model
