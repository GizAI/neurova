from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .fragment_parser import LearnedSemanticParser

@dataclass
class ParserEvalResult:
    total: int
    type_correct: int
    exact_slots: int

    @property
    def type_accuracy(self) -> float:
        return self.type_correct / max(1, self.total)

    @property
    def slot_exact_accuracy(self) -> float:
        return self.exact_slots / max(1, self.total)

def evaluate_parser(parser: LearnedSemanticParser, rows: Iterable[dict]) -> ParserEvalResult:
    total = type_correct = exact_slots = 0
    for row in rows:
        total += 1
        cands = parser.parse(row["text"])
        if not cands:
            continue
        ir = cands[0].ir
        if type(ir).__name__ == row.get("ir_type"):
            type_correct += 1
        # Simple slot equivalence via object attributes.
        ok = True
        for k, v in (row.get("slots") or {}).items():
            if k == "polarity":
                if getattr(ir, "polarity", None) != v: ok = False
            elif str(getattr(ir, k, "")).lower() != str(v).lower():
                ok = False
        if ok:
            exact_slots += 1
    return ParserEvalResult(total, type_correct, exact_slots)
