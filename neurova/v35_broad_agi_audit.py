from __future__ import annotations

"""V35 broad-coverage audit.

This is an honest, non-official audit for the new layers: broad chart parsing,
semantic perception/retrieval, social dialogue, world-state grounding, and
standard-benchmark readiness. It avoids claiming official benchmark saturation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import json
from pathlib import Path

from .agent import FinalCognitiveOS
from .semantic_encoder import DeepSemanticEncoder, SemanticMemoryIndex


@dataclass
class AuditCase:
    category: str
    input: str
    expect_any: List[str]
    reject_any: List[str] | None = None


@dataclass
class AuditResult:
    total: int
    passed: int
    accuracy: float
    cases: List[Dict[str, Any]]
    semantic_encoder_backend: str
    official_benchmark_status: Dict[str, Any]


class V35BroadAGIAudit:
    def __init__(self, root: str | Path = "/tmp/brainos_v35_audit"):
        self.root = Path(root)
        self.os = FinalCognitiveOS(root=self.root / "state", auto_seed=False)

    def _teach(self) -> None:
        lessons = [
            'When I say "A zorbles B", it means A is greater than B.',
            'Mina zorbles Taro.',
            'Actually, "A splarns B" means A causes B.',
            'Heat splarns expansion.',
            'When A ferries B from C to D, it means A moves B from C to D, and after that B is located at D.',
            'Kibo is rover.',
            'rover is robot.',
            'robot is machine.',
            'Bob believes that Alice is CEO.',
            'Lena gave Omar a map in Busan yesterday.',
            'Mina was principal from 2020 through 2022.',
            'Mina was not principal during 2021.',
            'all birds can fly',
            'Although ostriches are birds, they cannot usually fly.',
        ]
        for lesson in lessons:
            self.os.observe(lesson)

    def run(self) -> Dict[str, Any]:
        self._teach()
        cases = [
            AuditCase("construction_question", "Does Mina zorble Taro?", ["yes", "mina greater_than taro"], ["cannot prove", "fallback"]),
            AuditCase("construction_did", "Did Mina zorble Taro?", ["yes", "mina greater_than taro"], ["cannot prove", "fallback"]),
            AuditCase("construction_passive", "Taro was zorbled by Mina.", ["mina greater_than taro"], ["fallback"]),
            AuditCase("construction_negation", "Mina does not zorble Taro.", ["mina less_than taro"], ["fallback"]),
            AuditCase("causal_question", "Does heat splarn expansion?", ["yes", "heat causes expansion"], ["cannot prove", "fallback"]),
            AuditCase("causal_passive", "Expansion is splarned by heat.", ["heat causes expansion"], ["fallback"]),
            AuditCase("taxonomy_paraphrase", "Would it be fair to call Kibo a machine?", ["yes", "kibo is machine"], ["research", "cannot prove"]),
            AuditCase("taxonomy_category", "Could Kibo be considered part of the machine category?", ["yes", "kibo is machine"], ["cannot prove"]),
            AuditCase("event_frame_learning", "Eve ferried the crate from Oslo to Lima.", ["event ir", "world_effects=1"], ["fallback"]),
            AuditCase("world_state_query", "Where is crate?", ["crate located_at lima", "yes"], ["oslo", "cannot prove"]),
            AuditCase("coreference_event", "Does he have it?", ["omar has map", "yes"], ["bob has", "cannot prove"]),
            AuditCase("belief_coreference", "Does he believe she is CEO?", ["yes", "bob believes alice is ceo"], ["cannot prove"]),
            AuditCase("temporal_contradiction", "Who served as principal during 2021?", ["inconsistent", "both positive and negative"], []),
            AuditCase("exception_plural", "Can ostriches fly?", ["blocked", "exception"], ["yes"]),
            AuditCase("support_dialogue", "I had a rough day.", ["i can help", "next step"], ["research"]),
            AuditCase("smalltalk", "This is hilarious.", ["glad it's landing", "test next"], ["research"]),
            AuditCase("korean_question", "철수가 영희보다 우세하다고 봐도 되나?", ["철수 greater_than 영희"], ["fallback"]),
        ]
        passed = 0
        rows: List[Dict[str, Any]] = []
        for c in cases:
            resp = self.os.observe(c.input).response
            low = resp.lower()
            ok = all(e.lower() in low for e in c.expect_any)
            if c.reject_any:
                ok = ok and not any(r.lower() in low for r in c.reject_any)
            passed += int(ok)
            rows.append({"category": c.category, "input": c.input, "passed": ok, "response": resp, "expect_any": c.expect_any, "reject_any": c.reject_any or []})
        encoder = DeepSemanticEncoder()
        # Official benchmark status: no files are bundled, so this reports readiness honestly.
        official = {
            "scan_loader": "available_requires_official_file",
            "babi_loader": "available_requires_official_file",
            "clutrr_loader": "available_requires_official_csv",
            "claim": "No official full benchmark score is claimed without external files/checksums.",
        }
        result = AuditResult(len(cases), passed, passed / len(cases), rows, encoder.backend + ("+torch_available" if encoder.torch_available else "+cpu"), official)
        return asdict(result)


def run_v35_broad_agi_audit(root: str | Path = "/tmp/brainos_v35_audit") -> Dict[str, Any]:
    return V35BroadAGIAudit(root).run()
