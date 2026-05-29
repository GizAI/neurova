from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import json, shutil


@dataclass
class V29Case:
    name: str
    prompt: str
    expected: str
    category: str
    checks: str = "substring"


class V29SystemicBenchmark:
    """Harder systemic benchmark for grammar operations rather than memorized sentences.

    It checks whether one learned construction/schema is reused under tense, do-support,
    passive, dialogue-act, event-frame, temporal-schema, and meta-memory variants.
    It is still synthetic, but it is intentionally not a single smoke list: tutor prompts
    and test prompts are disjoint, and cases target grammar operations.
    """
    def tutor_lessons(self) -> List[str]:
        return [
            'When I say "A glarns B", it means A is greater than B.',
            'Actually, "A flarns B" means A causes B.',
            'When A carries B from C to D, it means A moves B from C to D, and after that B is located at D.',
            '"Who served as ROLE during T" means who is ROLE in T',
            'Mira glarns Taro',
            'heat flarns expansion',
            'Eve carried the box from Berlin to Rome.',
            'Mia served as mayor during 2024',
            'orion is robot',
            'Bob believes that Alice is CEO.',
        ]

    def cases(self) -> List[V29Case]:
        return [
            V29Case('wrapper did-question', 'did Mira glarn Taro?', 'Yes', 'wrapper_operation'),
            V29Case('passive construction', 'Taro was glarned by Mira', 'Stored comparison IR', 'voice_operation'),
            V29Case('do-support negation', 'Mira does not glarn Taro', 'Stored comparison IR', 'polarity_operation'),
            V29Case('would-you-say wrapper', 'Would you say Mira glarns Taro?', 'Yes', 'wrapper_operation'),
            V29Case('causal do-question', 'does heat flarn expansion?', 'Yes', 'wrapper_operation'),
            V29Case('event frame query', 'Where is box?', 'rome', 'event_frame'),
            V29Case('temporal schema query', 'Who served as mayor during 2024?', 'mia', 'temporal_schema'),
            V29Case('meta-memory dialogue act', 'What did we just learn about Orion?', 'robot', 'dialogue_act'),
            V29Case('support dialogue act', 'I feel stuck and confused. Can you help me think this through?', 'state the goal', 'dialogue_act'),
            V29Case('belief complementizer that', 'Does Bob believe that Alice is CEO?', 'Yes', 'scope_complementizer'),
            V29Case('false friend almost must abstain', 'Zed almost glarns Rex', 'No relevant sources', 'scope_guard'),
        ]

    def run(self, os: Any) -> Dict[str, Any]:
        rows=[]; passed=0
        for c in self.cases():
            r=os.observe(c.prompt)
            ok=c.expected.lower() in r.response.lower()
            rows.append({**asdict(c), 'observed': r.response[:800], 'ir_type': r.ir_type, 'passed': ok})
            passed += int(ok)
        return {'passed': passed, 'total': len(rows), 'score': passed / max(1, len(rows)), 'rows': rows}


class V29SystemicTutor:
    def __init__(self, os: Any, lessons: List[str]):
        self.os=os; self.lessons=lessons; self.dialogue=[]
    def run(self):
        for lesson in self.lessons:
            r=self.os.observe(lesson, reward=1.0)
            self.dialogue.append({'text': lesson, 'ir_type': r.ir_type, 'response': r.response[:500]})
        self.os.sleep.run(); self.os.intrinsic.propose(self.os)
        return self


class V29SystemicLearningAudit:
    def __init__(self, root: str | Path):
        from .agent import FinalCognitiveOS
        self.root=Path(root)
        if self.root.exists(): shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.before_os=FinalCognitiveOS(self.root/'before', auto_seed=False)
        self.after_os=FinalCognitiveOS(self.root/'after', auto_seed=False)
        self.benchmark=V29SystemicBenchmark()

    def run(self) -> Dict[str, Any]:
        lessons=self.benchmark.tutor_lessons(); cases=self.benchmark.cases()
        leakage=[c.prompt for c in cases if c.prompt in lessons]
        before=self.benchmark.run(self.before_os)
        tutor=V29SystemicTutor(self.after_os, lessons).run()
        after=self.benchmark.run(self.after_os)
        by_cat: Dict[str, Dict[str, int]]={}
        for row in after['rows']:
            g=by_cat.setdefault(row['category'], {'passed':0,'total':0})
            g['passed'] += int(row['passed']); g['total'] += 1
        for g in by_cat.values(): g['score']=g['passed']/max(1,g['total'])
        report={
            'benchmark':'V29 systemic grammar-operation/event-frame/dialogue-act audit',
            'claim':'Tests grammar operations and schemas, still synthetic; not human-level proof.',
            'leakage_guard': {'exact_prompt_overlap': leakage, 'passed': not leakage},
            'before': {k: before[k] for k in ['passed','total','score']},
            'after': {k: after[k] for k in ['passed','total','score']},
            'growth_delta': after['score'] - before['score'],
            'by_category': by_cat,
            'tutor_dialogue': tutor.dialogue,
            'after_rows': after['rows'],
            'before_rows': before['rows'],
        }
        (self.root/'V29_SYSTEMIC_LEARNING_AUDIT_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
