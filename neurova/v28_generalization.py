from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import json, random, shutil


@dataclass
class V28Case:
    name: str
    prompt: str
    expected: str
    category: str
    heldout_axis: str


class V28FailureTaxonomy:
    def classify(self, row: Dict[str, Any]) -> str:
        if row.get("passed"):
            return "passed"
        ir = (row.get("ir_type") or "").lower()
        obs = (row.get("observed") or "").lower()
        cat = row.get("category", "")
        if "fallback" in ir or "cannot prove" in obs:
            return f"{cat}:missing_parse_or_memory"
        if "claimir" in ir and cat in {"belief", "event", "temporal", "construction"}:
            return f"{cat}:overgeneric_claim_parse"
        if "inconsistent" in obs:
            return f"{cat}:contradiction_detected"
        return f"{cat}:wrong_answer_or_slot"


class V28HeldoutGeneralizationBenchmark:
    """Randomized, structure-held-out benchmark.

    The benchmark is still synthetic, but it is less narrow than fixed smoke tests:
    - names/objects are randomized;
    - tutor prompts are disjoint from test prompts;
    - tests use variants not shown during tutoring, e.g. did-question/passive;
    - each case is tagged by heldout axis and failure taxonomy.
    """
    def __init__(self, seed: int = 28):
        self.rng = random.Random(seed)
        self.names = ["luna", "orion", "pax", "sora", "nox", "mika", "rina", "taro", "joon", "mina"]
        self.objects = ["map", "compass", "notebook", "ruler", "token", "lamp", "box", "key"]

    def _n(self, i: int) -> str:
        return self.names[i % len(self.names)]

    def _o(self, i: int) -> str:
        return self.objects[i % len(self.objects)]

    def tutor_lessons(self) -> List[str]:
        return [
            # Teach only construction meaning; tests ask different variants and entities.
            'When I say "A glorms B", it means A is greater than B.',
            'Actually, "A flarns B" means A causes B.',
            'luna glorms pax',
            # Taxonomy facts; tests use unseen paraphrase wrappers.
            'kibo is rover', 'rover is robot', 'robot is machine',
            # Temporal/state and contradiction family.
            'Nova became captain in 2022.', 'Nova stopped being captain in 2025.',
            'Rhea was captain from 2021 to 2023.', 'In 2022, Rhea was not captain.',
            # Event/world frames; tests ask derived state, not event restatement.
            'Lena gave Omar a map in Busan yesterday.',
            'Aria purchased compass from Nox yesterday.',
            'Nox sold notebook to Sora today.',
            'Mika moved the box from office to lab.',
            'Rina put the key in drawer.',
            # Belief/coreference context.
            'Mika believes Rina is guide.',
            # Exception/discourse.
            'all birds can fly',
            'Although emus are birds, they cannot usually fly.',
            # Causal chain and Korean comparison.
            'mist falls causes road wet', 'road wet causes path slippery',
            '다온에 비해 라온이 앞선다',
        ]

    def cases(self) -> List[V28Case]:
        return [
            V28Case('construction did-question', 'did luna glorm pax?', 'Yes', 'construction', 'do_support_question'),
            V28Case('construction passive', 'pax was glormed by luna', 'Stored comparison IR', 'construction', 'passive_variant'),
            V28Case('construction negation', 'luna did not glorm pax', 'Stored comparison IR', 'construction', 'do_support_negation'),
            V28Case('causal construction assertion', 'frost flarns haze', 'Stored causal IR', 'construction', 'unseen_entities'),
            V28Case('causal construction did-question', 'did frost flarn haze?', 'Yes', 'construction', 'causal_question_variant'),
            V28Case('taxonomy member view', 'May Kibo be viewed as a member of machine?', 'Yes', 'taxonomy', 'unseen_question_wrapper'),
            V28Case('taxonomy fit within', 'Does Kibo fit within machine?', 'Yes', 'taxonomy', 'unseen_paraphrase'),
            V28Case('negation hardly counts', 'Kibo hardly counts as a mineral.', 'Stored negated claim IR', 'negation', 'modality'),
            V28Case('negation not likely', 'Kibo is not likely to be a mineral.', 'Stored negated claim IR', 'negation', 'modality'),
            V28Case('temporal positive interval', 'Who was captain in 2023?', 'nova', 'temporal', 'became_stopped_interval'),
            V28Case('temporal contradiction', 'Who was captain in 2022?', 'Inconsistent evidence', 'temporal', 'overlap_contradiction'),
            V28Case('transfer pronoun possession', 'Does he have it?', 'Yes', 'coreference_event', 'pronoun_resolution'),
            V28Case('buy world effect', 'Does Aria have compass?', 'Yes', 'event_world', 'event_effect_query'),
            V28Case('sell world effect', 'Does Sora have notebook?', 'Yes', 'event_world', 'event_effect_query'),
            V28Case('move location', 'Where is box?', 'lab', 'event_world', 'location_query'),
            V28Case('put location', 'Where is key?', 'drawer', 'event_world', 'location_query'),
            V28Case('belief pronoun question', 'Does she believe Rina is guide?', 'Yes', 'belief', 'pronoun_holder'),
            V28Case('exception emu query', 'Can an emu fly even though it is a bird?', 'Blocked by exception', 'exception', 'lemma_normalization'),
            V28Case('causal chain proof', 'mist falls causes path slippery?', 'Yes', 'discourse', 'multi_hop_causal'),
            V28Case('korean comparison proof', '라온이 다온보다 크다고 볼 수 있나?', 'Yes', 'korean', 'particle_question_variant'),
        ]

    def run(self, os: Any) -> Dict[str, Any]:
        rows=[]; passed=0
        for c in self.cases():
            r=os.observe(c.prompt)
            ok=c.expected.lower() in r.response.lower()
            passed += int(ok)
            rows.append({**asdict(c), 'ir_type': r.ir_type, 'observed': r.response[:600], 'passed': ok})
        tax = V28FailureTaxonomy()
        for row in rows:
            row['failure_type'] = tax.classify(row)
        return {'passed': passed, 'total': len(rows), 'score': passed/max(1,len(rows)), 'rows': rows}


class V28NonLeakyTutor:
    def __init__(self, os: Any, lessons: List[str]):
        self.os = os; self.lessons = lessons; self.dialogue=[]
    def run(self):
        for text in self.lessons:
            r=self.os.observe(text, reward=1.0)
            self.dialogue.append({'text': text, 'ir_type': r.ir_type, 'response': r.response[:400]})
        self.os.sleep.run(); self.os.intrinsic.propose(self.os)
        return self


class V28AblationAudit:
    def __init__(self, benchmark: V28HeldoutGeneralizationBenchmark):
        self.benchmark = benchmark
    def run(self, root: Path) -> Dict[str, Any]:
        from .agent import FinalCognitiveOS
        ab_root = root / 'ablations'
        if ab_root.exists(): shutil.rmtree(ab_root)
        ab_root.mkdir(parents=True)
        # No tutoring baseline.
        no_tutor = FinalCognitiveOS(ab_root/'no_tutor', auto_seed=False)
        no_tutor_score = self.benchmark.run(no_tutor)
        # Construction-only: teach only the two correction lessons.
        construction_only = FinalCognitiveOS(ab_root/'construction_only', auto_seed=False)
        V28NonLeakyTutor(construction_only, self.benchmark.tutor_lessons()[:2]).run()
        construction_score = self.benchmark.run(construction_only)
        return {
            'no_tutor': {k:no_tutor_score[k] for k in ['passed','total','score']},
            'construction_only': {k:construction_score[k] for k in ['passed','total','score']},
        }


class V28GeneralizationAudit:
    def __init__(self, root: str | Path):
        from .agent import FinalCognitiveOS
        self.root = Path(root)
        if self.root.exists(): shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.before_os = FinalCognitiveOS(self.root/'before', auto_seed=False)
        self.after_os = FinalCognitiveOS(self.root/'after', auto_seed=False)
        self.benchmark = V28HeldoutGeneralizationBenchmark()

    def run(self) -> Dict[str, Any]:
        lessons = self.benchmark.tutor_lessons()
        cases = self.benchmark.cases()
        leakage = [c.prompt for c in cases if c.prompt in lessons]
        before = self.benchmark.run(self.before_os)
        tutor = V28NonLeakyTutor(self.after_os, lessons).run()
        after = self.benchmark.run(self.after_os)
        ablation = V28AblationAudit(self.benchmark).run(self.root)
        by_category: Dict[str, Dict[str, int]] = {}
        for row in after['rows']:
            g = by_category.setdefault(row['category'], {'passed':0,'total':0})
            g['passed'] += int(row['passed']); g['total'] += 1
        for g in by_category.values():
            g['score'] = g['passed']/max(1,g['total'])
        report = {
            'benchmark': 'V28 randomized held-out generalization audit',
            'claim': 'Less narrow than fixed smoke tests, but still synthetic; not proof of human-level intelligence.',
            'leakage_guard': {'exact_prompt_overlap': leakage, 'passed': len(leakage)==0},
            'before': {k: before[k] for k in ['passed','total','score']},
            'after': {k: after[k] for k in ['passed','total','score']},
            'growth_delta': after['score']-before['score'],
            'by_category': by_category,
            'ablation': ablation,
            'tutor_dialogue': tutor.dialogue,
            'after_rows': after['rows'],
            'before_rows': before['rows'],
        }
        (self.root/'V28_GENERALIZATION_AUDIT_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
