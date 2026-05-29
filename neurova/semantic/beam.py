from __future__ import annotations
from typing import Iterable, List
from ..ir import IRCandidate


class SemanticBeam:
    """Small beam container for semantic parse candidates."""
    def __init__(self, width: int = 20):
        self.width = width

    def prune(self, candidates: Iterable[IRCandidate]) -> List[IRCandidate]:
        # De-duplicate by IR class and salient slots while preserving highest score.
        best = {}
        for c in candidates:
            ir = c.ir
            key = (type(ir).__name__, tuple(sorted((k, str(v)) for k, v in getattr(ir, "__dict__", {}).items() if k not in {"id", "confidence", "evidence_ids"})))
            if key not in best or c.total_score > best[key].total_score:
                best[key] = c
        return sorted(best.values(), key=lambda c: c.total_score, reverse=True)[: self.width]
