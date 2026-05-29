from __future__ import annotations
from typing import Iterable, List
from ..ir import CognitiveIR, CompositeIR, IRCandidate

class CandidateAssembler:
    def compose(self, items: Iterable[CognitiveIR], source_text: str, confidence: float = 0.82) -> IRCandidate:
        return IRCandidate(CompositeIR(items=list(items), source_text=source_text), confidence, "candidate_assembler", notes=["IR fragments assembled into CompositeIR"])
