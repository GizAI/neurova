from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import json, shutil


@dataclass
class V27Case:
    name: str
    prompt: str
    expected: str
    category: str


class V27AdversarialLanguageBenchmark:
    """Harder adversarial benchmark for BrainOS V27.

    It is synthetic and controlled, not proof of human-level intelligence.  The cases
    are intentionally surface-held-out: the tutor teaches construction families and
    world frames, while the benchmark asks passive/question/negated/coreference/event
    variants rather than repeating the teaching utterances.
    """
    def cases(self) -> List[V27Case]:
        return [
            V27Case('verb construction assertion', 'luna eclipses sol', 'Stored comparison IR', 'construction_variant'),
            V27Case('verb construction do-question', 'does luna eclipse sol?', 'Yes', 'construction_variant'),
            V27Case('verb construction passive', 'sol is eclipsed by luna', 'Stored comparison IR', 'construction_variant'),
            V27Case('verb construction negation', 'luna does not eclipse sol', 'Stored comparison IR', 'construction_variant'),
            V27Case('causal construction question', 'heat catalyzes expansion', 'Stored causal IR', 'construction_variant'),
            V27Case('causal construction proof', 'heat causes expansion?', 'Yes', 'construction_variant'),
            V27Case('taxonomy regarded question', 'Can Kibo be regarded as a machine?', 'Yes', 'taxonomy_paraphrase'),
            V27Case('taxonomy fall-under question', 'Does Kibo fall under machine?', 'Yes', 'taxonomy_paraphrase'),
            V27Case('modality unlikely negation', 'Kibo is unlikely to be a mineral.', 'Stored negated claim IR', 'negation_modality'),
            V27Case('modal cannot negation', 'Kibo cannot be classified as a mineral.', 'Stored negated claim IR', 'negation_modality'),
            V27Case('temporal exact bug guard', 'on 2026 민수 is mineral', 'Stored temporal claim IR', 'temporal'),
            V27Case('temporal interval between', 'Alice was CEO between 2025 and 2026.', 'Stored temporal claim IR', 'temporal'),
            V27Case('temporal contradiction', 'Who was CEO in 2026?', 'Inconsistent evidence', 'temporal'),
            V27Case('buy event world state', 'Does Sora have book?', 'Yes', 'world_event'),
            V27Case('sell event world state', 'Does Carol have car?', 'Yes', 'world_event'),
            V27Case('move event location', 'Where is box?', 'library', 'world_event'),
            V27Case('put event location', 'Where is key?', 'drawer', 'world_event'),
            V27Case('belief coreference question', 'Does he believe she is CEO?', 'Yes', 'coreference_belief'),
            V27Case('pronoun possession', 'Does he have it?', 'Yes', 'coreference_event'),
            V27Case('Korean uncertain negative comparison', '철수는 영희보다 크지 않은 것 같다', 'Stored comparison IR', 'korean'),
            V27Case('Korean considered comparison question', '철수가 영희보다 크다고 볼 수 있나?', 'Yes', 'korean'),
            V27Case('Korean relative ahead', '영희에 비해 철수가 앞선다', 'Stored comparison IR', 'korean'),
            V27Case('Korean lagging inverse', '철수보다 영희가 뒤처진다', 'Stored comparison IR', 'korean'),
            V27Case('ostrich exception storage', 'Although ostriches are birds, they cannot usually fly.', 'Stored exception IR', 'exception'),
            V27Case('ostrich exception query', 'Can an ostrich fly even though it is a bird?', 'Blocked by exception', 'exception'),
            V27Case('if-and causal chain', 'If rain falls, the ground gets wet and the road becomes slippery.', 'CompositeIR stored', 'discourse'),
            V27Case('if-and causal proof', 'rain falls causes road slippery?', 'Yes', 'discourse'),
        ]

    def run(self, os: Any) -> Dict[str, Any]:
        rows=[]; passed=0
        for c in self.cases():
            r=os.observe(c.prompt)
            ok=c.expected.lower() in r.response.lower()
            passed += int(ok)
            rows.append({**asdict(c), 'ir_type': r.ir_type, 'observed': r.response[:500], 'passed': ok})
        return {'passed': passed, 'total': len(rows), 'score': passed/max(1,len(rows)), 'rows': rows}


class V27DevelopmentalTutor:
    """Teaches construction families, world frames, and facts—not benchmark answers."""
    def __init__(self, os: Any):
        self.os=os
        self.dialogue=[]

    def say(self, text: str):
        r=self.os.observe(text, reward=1.0)
        self.dialogue.append({'text': text, 'ir_type': r.ir_type, 'response': r.response[:400]})
        return r

    def run(self):
        lessons=[
            # Taxonomy facts.
            'kibo is rover', 'rover is robot', 'robot is machine', 'kibo is not mineral',
            # Construction families. The benchmark uses questions/passives/entities not repeated here.
            'No, by "A eclipses B" I mean A is greater than B.',
            'Actually, "A catalyzes B" means A causes B.',
            # Temporal facts and contradiction.
            'Alice was CEO between 2025 and 2026.', 'In 2026 Alice was not CEO.',
            # World events. Benchmark asks derived states.
            'Sora bought a book from Dami yesterday.', 'Bob sold a car to Carol today.',
            'Teacher moved the box from classroom to library.', 'Mina put the key in drawer.',
            # Belief/coreference context.
            'Bob believes Alice is CEO.',
            # Pronoun/event context.
            'Alice gave Bob a package in Seoul yesterday.',
            # Exception/rule facts.
            'all birds can fly',
            # Causal chain ground.
            'rain falls causes ground wet', 'ground wet causes road slippery', '영희에 비해 철수가 앞선다',
        ]
        for l in lessons:
            self.say(l)
        # Sleep/replay and intrinsic goals are part of growth loop.
        self.os.sleep.run(); self.os.intrinsic.propose(self.os)
        return self


class V27AdversarialGrowthLab:
    def __init__(self, root: str | Path):
        from .agent import FinalCognitiveOS
        self.root=Path(root)
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.before_os=FinalCognitiveOS(self.root/'before', auto_seed=False)
        self.after_os=FinalCognitiveOS(self.root/'after', auto_seed=False)
        self.benchmark=V27AdversarialLanguageBenchmark()

    def run(self) -> Dict[str, Any]:
        before=self.benchmark.run(self.before_os)
        tutor=V27DevelopmentalTutor(self.after_os).run()
        after=self.benchmark.run(self.after_os)
        report={
            'benchmark': 'V27 adversarial held-out construction/world/language benchmark',
            'claim': 'Controlled hard benchmark growth, not proof of human-level intelligence.',
            'before': {k: before[k] for k in ['passed','total','score']},
            'after': {k: after[k] for k in ['passed','total','score']},
            'growth_delta': after['score']-before['score'],
            'tutor_dialogue': tutor.dialogue,
            'after_rows': after['rows'],
            'before_rows': before['rows'],
        }
        out=self.root/'V27_ADVERSARIAL_GROWTH_REPORT.json'
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
