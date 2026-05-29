from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple

def toks(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_가-힣]+", text.lower())

@dataclass
class SlotPrediction:
    ir_type: str
    slots: Dict[str, str]
    score: float

class StructuredPerceptronTagger:
    """Tiny structured predictor: IR type classifier + template-backed slot extractor.

    No decoder, no next-token objective, no text generation. It learns which surface
    cues imply which IR type and uses induced slot templates for extraction.
    """
    objective = "IR type classification + slot extraction only; no autoregressive generation"

    def __init__(self):
        self.type_token_weights: Dict[str, Counter[str]] = defaultdict(Counter)
        self.type_bias: Counter[str] = Counter()
        self.templates: List[Tuple[str, re.Pattern, str, Dict[str, str]]] = []
        self.updates = 0

    def fit(self, rows: Iterable[dict]) -> "StructuredPerceptronTagger":
        for row in rows:
            self.observe(row)
        return self

    def observe(self, row: dict, weight: float = 1.0) -> None:
        text = row.get("text", "")
        ir_type = row.get("ir_type", "")
        slots = row.get("slots", {}) or {}
        if not text or not ir_type:
            return
        self.type_bias[ir_type] += weight
        for t in toks(text):
            self.type_token_weights[ir_type][t] += weight
        compiled = self._compile_template(text, ir_type, slots)
        if compiled:
            self.templates.append(compiled)
        self.updates += 1

    def _compile_template(self, text: str, ir_type: str, slots: Dict[str, str]):
        surface = text.strip().lower()
        marker_to_slot: Dict[str, str] = {}
        temp = surface
        for i, (slot, value) in enumerate(sorted(slots.items(), key=lambda kv: len(str(kv[1])), reverse=True)):
            if value is None: continue
            value_s = str(value).strip().lower()
            if not value_s or value_s in {"positive", "negative", "greater_than", "less_than", "is", "can", "has", "means"}:
                continue
            marker = f"§slot{i}§"
            # Replace only first occurrence to avoid overfitting repeated entity names.
            if value_s in temp:
                temp = temp.replace(value_s, marker, 1)
                marker_to_slot[marker] = slot
        if not marker_to_slot:
            return None
        pattern = re.escape(temp)
        for marker, slot in marker_to_slot.items():
            pattern = pattern.replace(re.escape(marker), f"(?P<{slot}>.+?)")
        pattern = r"^" + pattern + r"[?.!]*$"
        try:
            return (surface, re.compile(pattern, re.I), ir_type, dict(slots))
        except re.error:
            return None

    def score_types(self, text: str) -> List[Tuple[str, float]]:
        tokens = toks(text)
        rows = []
        for typ, bias in self.type_bias.items():
            score = 0.05 * bias + sum(self.type_token_weights[typ][t] for t in tokens) / max(1, len(tokens))
            rows.append((typ, score))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def predict(self, text: str, max_predictions: int = 8) -> List[SlotPrediction]:
        low = text.strip().lower()
        out: List[SlotPrediction] = []
        for _, pattern, ir_type, defaults in self.templates:
            m = pattern.match(low)
            if not m:
                continue
            slots = dict(defaults)
            for k, v in m.groupdict().items():
                if v is not None:
                    slots[k] = v.strip(" ?.!,'\"")
            # Confidence from IR-type classifier plus template exactness.
            type_score = dict(self.score_types(text)).get(ir_type, 0.0)
            out.append(SlotPrediction(ir_type, slots, 0.65 + min(0.25, 0.02 * type_score)))
        # De-duplicate by type/slots.
        seen = set(); uniq = []
        for p in sorted(out, key=lambda x: x.score, reverse=True):
            key = (p.ir_type, tuple(sorted(p.slots.items())))
            if key in seen: continue
            seen.add(key); uniq.append(p)
        return uniq[:max_predictions]
