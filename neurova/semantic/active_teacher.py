from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List

@dataclass
class ActiveLearningItem:
    text: str
    reason: str
    suggestion: str
    priority: float

class ActiveTeacher:
    """Stores hard parse cases instead of silently hallucinating a parse."""
    def __init__(self):
        self.queue: List[ActiveLearningItem] = []

    def add_failed_parse(self, text: str, reason: str, suggestion: str = "human-or-verifier correction") -> ActiveLearningItem:
        item = ActiveLearningItem(text, reason, suggestion, 1.0 if reason else 0.5)
        self.queue.append(item)
        return item

    def export(self) -> List[Dict]:
        return [asdict(x) for x in self.queue]
