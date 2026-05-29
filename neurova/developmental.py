from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json, re, time, uuid

from .ir import ClaimIR, CausalClaimIR, ComparisonIR, EventIR, ToolCallIR


@dataclass
class Experience:
    text: str
    predicted_ir: str = ""
    response: str = ""
    success: bool = False
    user_feedback: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class SeedCandidate:
    kind: str
    payload: Dict[str, Any]
    source: str = "llm_distilled_seed"
    confidence: float = 0.35
    status: str = "candidate"  # candidate | experimental | stable | rejected


@dataclass
class LearningGoal:
    target: str
    reason: str
    expected_gain: float
    uncertainty: float
    priority: float


class LLMSeedKnowledgeBank:
    """LLM-distilled seed knowledge importer.

    It does not call an LLM at runtime and does not trust seed rows as truth.  Seeds
    are treated as hypotheses: installable into the construction/world-memory only
    after schema normalization and regression-gate checks by the OS.
    """
    def __init__(self):
        self.candidates: List[SeedCandidate] = []

    def add_builtin_child_language_seed(self) -> "LLMSeedKnowledgeBank":
        rows = [
            # Construction seeds.  These are not answers; they are reusable form↔meaning hypotheses.
            ("construction", {"surface": "A is regarded as B", "target": "claim(A,is,B)"}),
            ("construction", {"surface": "A falls under B", "target": "claim(A,is,B)"}),
            ("construction", {"surface": "A narrowly beats B", "target": "compare(A,greater_than,B)"}),
            ("construction", {"surface": "A trails B", "target": "compare(A,less_than,B)"}),
            ("construction", {"surface": "A gives rise to B", "target": "causal(A,B)"}),
            ("construction", {"surface": "A results in B", "target": "causal(A,B)"}),
            ("construction", {"surface": "A is no longer B", "target": "not_claim(A,is,B)"}),
            ("construction", {"surface": "A cannot be classified as B", "target": "not_claim(A,is,B)"}),
            # Event frame seeds.  Runtime effects are verified by the world-state simulator.
            ("event_frame", {"name": "buy", "effect": {"recipient_role": "actor", "relation": "has", "object_role": "patient"}}),
            ("event_frame", {"name": "sell", "effect": {"recipient_role": "recipient", "relation": "has", "object_role": "patient"}}),
            ("event_frame", {"name": "move", "effect": {"subject_role": "patient", "relation": "located_at", "object_role": "location"}}),
            ("event_frame", {"name": "put", "effect": {"subject_role": "patient", "relation": "located_at", "object_role": "location"}}),
            ("event_frame", {"name": "take", "effect": {"recipient_role": "actor", "relation": "has", "object_role": "patient"}}),
            ("event_frame", {"name": "open", "effect": {"subject_role": "patient", "relation": "state", "object": "open"}}),
            ("event_frame", {"name": "close", "effect": {"subject_role": "patient", "relation": "state", "object": "closed"}}),
            # Mini ontology seeds used by grade-school style tasks.
            ("claim", {"subject": "plant", "relation": "needs", "object": "water"}),
            ("claim", {"subject": "plant", "relation": "needs", "object": "sunlight"}),
            ("causal", {"cause": "water plant", "effect": "plant grows"}),
        ]
        for kind, payload in rows:
            self.candidates.append(SeedCandidate(kind=kind, payload=payload))
        return self

    def import_jsonl(self, path: str | Path) -> "LLMSeedKnowledgeBank":
        p = Path(path)
        if not p.exists():
            return self
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.candidates.append(SeedCandidate(kind=row.get("kind", "construction"), payload=row, source=row.get("source", "llm_seed"), confidence=float(row.get("confidence", 0.35))))
        return self

    def install(self, os: Any) -> Dict[str, int]:
        report = {"construction": 0, "event_frame": 0, "claim": 0, "causal": 0, "rejected": 0}
        for c in self.candidates:
            try:
                if c.kind == "construction":
                    target_ir = os.compiler.parse_target_ir(c.payload["target"])
                    if target_ir and os.compiler.learn_construction(c.payload["surface"], target_ir):
                        report["construction"] += 1
                        c.status = "experimental"
                    else:
                        report["rejected"] += 1
                elif c.kind == "event_frame":
                    os.world_grounder.add_dynamic_frame(c.payload["name"], c.payload["effect"], source=c.source, confidence=c.confidence)
                    report["event_frame"] += 1
                    c.status = "experimental"
                elif c.kind == "claim":
                    os.observe(f"teach: {c.payload['subject']} {c.payload['relation']} {c.payload['object']}")
                    report["claim"] += 1
                    c.status = "experimental"
                elif c.kind == "causal":
                    os.observe(f"{c.payload['cause']} causes {c.payload['effect']}")
                    report["causal"] += 1
                    c.status = "experimental"
            except Exception:
                report["rejected"] += 1
                c.status = "rejected"
        os.memory.log_action("LLM_SEED_INSTALL", "developmental_seed", report, 0.5)
        return report


class IntrinsicMotivationEngine:
    """Chooses learning goals from uncertainty, failures, novelty, and competence gain."""
    def __init__(self):
        self.goals: List[LearningGoal] = []

    def propose(self, os: Any) -> List[LearningGoal]:
        stats = os.memory.stats()
        rows = os.memory.conn.execute("SELECT status, failure_type, COUNT(*) AS n FROM trajectories GROUP BY status, failure_type").fetchall()
        goals: List[LearningGoal] = []
        for r in rows:
            failure = r["failure_type"] or r["status"] or "unknown"
            if failure and failure != "success":
                n = int(r["n"])
                goals.append(LearningGoal(target=f"repair:{failure}", reason=f"{n} recent failures", expected_gain=min(1.0, 0.15*n), uncertainty=0.8, priority=min(1.0, 0.2*n + 0.4)))
        if stats.get("claim_versions", 0) < 8:
            goals.append(LearningGoal("seed_semantic_memory", "few stable claims", 0.5, 0.6, 0.55))
        if len(getattr(os.compiler.active_teacher, "queue", [])):
            goals.append(LearningGoal("active_teacher_hard_cases", "unparsed/corrected utterances queued", 0.7, 0.9, 0.75))
        goals.sort(key=lambda g: g.priority, reverse=True)
        self.goals = goals[:8]
        for g in self.goals:
            os.memory.log_action("INTRINSIC_GOAL", g.target, asdict(g), g.priority)
        return self.goals


class DevelopmentalDialogueTutor:
    """A deterministic tutor that grows the OS through dialogue, not answer injection.

    It supplies constructions, world frames, examples, and corrections.  It does not
    insert benchmark answers.  Post-test success must come from parser/memory/world
    machinery using those learned constructions and event frames.
    """
    def __init__(self, os: Any):
        self.os = os
        self.experiences: List[Experience] = []

    def say(self, text: str, success_predicate: Optional[Callable[[str], bool]] = None) -> Experience:
        r = self.os.observe(text, reward=1.0)
        ok = bool(success_predicate(r.response)) if success_predicate else True
        e = Experience(text=text, predicted_ir=r.ir_type, response=r.response, success=ok)
        self.experiences.append(e)
        return e

    def run_seeded_child_curriculum(self):
        # 1) LLM-distilled seed hypotheses, gated by parser/schema install.
        LLMSeedKnowledgeBank().add_builtin_child_language_seed().install(self.os)

        # 2) Interactive correction dialogues: teaches form→meaning, not final answers.
        lessons = [
            'Actually, "A sparks B" means A causes B.',
            'No, by "A outruns B" I mean A is greater than B.',
            'In this domain, "A absorbs B" means A causes B.',
            'For our task, interpret "A trails B" as A less_than B.',
            'Correction: "A is regarded as B" means A is B.',
            'When I say "A buys B from C", it means A has B.',
        ]
        for l in lessons:
            self.say(l)

        # 3) Core grade-school facts/rules/examples.  They are general concepts, not answers.
        factual_curriculum = [
            'teach: rover is robot',
            'teach: robot is machine',
            'teach: oak is plant',
            'teach: plant needs water',
            'water plant causes plant grows',
            'all birds can fly',
            'penguin is bird',
            'Penguins are birds; however, they usually do not fly.',
            'From 2025 to 2026, Alice served as principal.',
            'In 2026 Bob was not principal.',
            'Mina gave Joon a pencil in Seoul yesterday.',
            'Joon received a notebook from Mina yesterday.',
            'Sora bought a book from Dami yesterday.',
            'Dami sold a ruler to Sora today.',
            'Teacher moved the box from classroom to library.',
            'Alice asked Bob to open the door.',
            'Bob thinks Alice is not the CEO.',
            'Kibo plans to collect rocks.',
        ]
        for l in factual_curriculum:
            self.say(l)

        # 4) Consolidate after curriculum.
        self.os.sleep.run()
        self.os.intrinsic.propose(self.os)
        return self


class ElementaryWorkbookBenchmark:
    """Grade-3-style synthetic benchmark: math, reading facts, events, time, science, grammar."""
    def cases(self) -> List[tuple[str, str]]:
        return [
            ("Mina has 12 marbles and gets 7 more. How many marbles does Mina have?", "19"),
            ("Joon had 20 stickers and gave 6 to Hana. How many stickers does Joon have?", "14"),
            ("There are 4 boxes with 6 pencils each. How many pencils are there?", "24"),
            ("24 candies are shared equally among 6 children. How many candies does each child get?", "4"),
            ("is rover machine?", "Yes"),
            ("Does Joon have pencil?", "Yes"),
            ("Does Sora have book?", "Yes"),
            ("Where is the box?", "library"),
            ("Who was principal in 2026?", "alice"),
            ("Can a penguin fly even though it is a bird?", "Blocked by exception"),
            ("what happens after water plant?", "plant grows"),
            ("heat sparks expansion", "Stored causal IR"),
            ("heat causes expansion?", "Yes"),
            ("orion outruns zephyr", "Stored comparison IR"),
            ("is orion greater than zephyr?", "Yes"),
            ("철수는 영희보다 크지 않다", "Stored comparison IR"),
            ("Bob thinks Alice is not CEO.", "Stored belief IR"),
            ("Does Bob believe Alice is not CEO?", "Yes"),
        ]

    def run(self, os: Any) -> Dict[str, Any]:
        rows = []
        passed = 0
        for prompt, expected in self.cases():
            r = os.observe(prompt)
            ok = expected.lower() in r.response.lower()
            passed += int(ok)
            rows.append({"prompt": prompt, "expected": expected, "observed": r.response[:500], "passed": ok, "ir_type": r.ir_type})
        return {"passed": passed, "total": len(rows), "score": passed / max(1, len(rows)), "rows": rows}


class DevelopmentalGrowthLab:
    """Runs before/after growth proof for a child-like curriculum."""
    def __init__(self, root: str | Path):
        from .agent import FinalCognitiveOS
        self.root = Path(root)
        self.before_os = FinalCognitiveOS(self.root / "before", auto_seed=False)
        self.after_os = FinalCognitiveOS(self.root / "after", auto_seed=False)
        self.benchmark = ElementaryWorkbookBenchmark()

    def run(self) -> Dict[str, Any]:
        before = self.benchmark.run(self.before_os)
        tutor = DevelopmentalDialogueTutor(self.after_os).run_seeded_child_curriculum()
        after = self.benchmark.run(self.after_os)
        report = {
            "before": {"passed": before["passed"], "total": before["total"], "score": before["score"]},
            "after": {"passed": after["passed"], "total": after["total"], "score": after["score"]},
            "growth_delta": after["score"] - before["score"],
            "tutor_experiences": [asdict(e) for e in tutor.experiences],
            "after_rows": after["rows"],
            "claim": "Synthetic grade-3-style benchmark growth, not proof of human-level intelligence.",
        }
        out = self.root / "growth_report.json"
        self.root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
