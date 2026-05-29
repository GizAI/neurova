from __future__ import annotations
import json
from typing import Dict, List
from .memory import EvidenceGraphMemory


class StateTransitionWorldModel:
    """Small symbolic world model: state + action -> observed next state candidates."""
    def __init__(self, memory: EvidenceGraphMemory):
        self.memory = memory

    def observe(self, state: Dict, action: str, next_state: Dict, confidence: float = 0.7):
        return self.memory.add_world_transition(state, action, next_state, confidence)

    def predict(self, action: str) -> List[Dict]:
        rows = self.memory.world_predictions(action)
        out = []
        for r in rows:
            out.append({"state": json.loads(r["state_json"]), "action": r["action"], "next_state": json.loads(r["next_state_json"]), "confidence": r["confidence"]})
        return sorted(out, key=lambda x: x["confidence"], reverse=True)
