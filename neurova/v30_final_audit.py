from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from .agent import FinalCognitiveOS

@dataclass
class AuditRow:
    name: str
    prompt: str
    expected: str
    before: str = ''
    after: str = ''
    passed: bool = False

@dataclass
class V30AuditReport:
    before_passed: int
    after_passed: int
    total: int
    rows: List[AuditRow]
    checklist: Dict[str, bool] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return (self.after_passed - self.before_passed) / max(1, self.total)

class V30FinalAnswerAudit:
    """Cohesive final-structure audit.

    This audit focuses on the failures called out in the V29 critiques: wrapper
    operations should compose with any learned construction, world state should be
    fluent/current, temporal contradiction must block unsafe answers, dialogue acts
    should not become ClaimIR pollution, and Korean/social/belief cases should route
    through typed operations rather than one-off sentence memorization.
    """
    def _seed(self, os: FinalCognitiveOS):
        tutor = [
            'When I say "A outclasses B", it means A is greater than B.',
            'Nova outclasses Mira.',
            'Actually, "A triggers B" means A causes B.',
            'Heat triggers expansion.',
            'When A carries B from C to D, it means A moves B from C to D, and after that B is located at D.',
            'Mina was principal from 2020 through 2022.',
            'Mina was not principal during 2021.',
            'Bob thinks Alice believes Kibo is machine.',
            'Eve carried the box from Berlin to Rome.',
            'Mira moved the box from Rome to Oslo.',
        ]
        for t in tutor:
            os.observe(t)

    def cases(self):
        return [
            AuditRow('wrapper modal over learned relation', 'Would you say Nova outclasses Mira?', 'Yes'),
            AuditRow('did-support over learned relation', 'Did Nova outclass Mira?', 'Yes'),
            AuditRow('passive over learned relation', 'Mira was outclassed by Nova.', 'Stored comparison IR'),
            AuditRow('negation over learned relation', 'Nova does not outclass Mira.', 'Stored comparison IR: nova less_than mira'),
            AuditRow('causal do-support', 'Does heat trigger expansion?', 'Yes'),
            AuditRow('causal passive', 'Expansion is triggered by heat.', 'Stored causal IR'),
            AuditRow('causal negation scoped', 'Heat does not trigger expansion.', 'Stored negated claim IR'),
            AuditRow('fluent latest state', 'Where is box?', 'oslo'),
            AuditRow('temporal contradiction', 'Who served as principal during 2021?', 'Inconsistent evidence'),
            AuditRow('nested belief question', 'Does Bob think Alice believes Kibo is machine?', 'Yes'),
            AuditRow('belief complementizer protection', 'Does Bob believe that Alice is CEO?', 'cannot prove'),
            AuditRow('smalltalk dialogue act', "haha that's wild", 'pretty wild'),
            AuditRow('support dialogue act', "I'm worried I'm doing this wrong.", 'Let’s make it concrete'),
            AuditRow('korean modality comparison', '철수가 영희보다 우세하다고 봐도 되나?', 'greater_than'),
            AuditRow('korean uncertain negation', '철수는 영희보다 크지 않은 것 같다', 'less_than'),
            AuditRow('meta memory', 'What did we just learn about Nova?', 'nova greater_than mira'),
        ]

    def run(self, root: str | Path = '/mnt/data/v30_audit_state') -> V30AuditReport:
        before = FinalCognitiveOS(root=Path(str(root) + '_before'))
        after = FinalCognitiveOS(root=Path(str(root) + '_after'))
        self._seed(after)
        rows = self.cases()
        before_passed = after_passed = 0
        for row in rows:
            rb = before.observe(row.prompt)
            ra = after.observe(row.prompt)
            row.before = rb.response[:500]
            row.after = ra.response[:500]
            if row.expected.lower() in row.before.lower():
                before_passed += 1
            if row.expected.lower() in row.after.lower():
                after_passed += 1
                row.passed = True
        checklist = {
            'wrapper_first_decomposition': all(r.passed for r in rows[:4]),
            'causal_family_composition': all(r.passed for r in rows[4:7]),
            'fluent_world_latest_state': rows[7].passed,
            'temporal_interval_contradiction': rows[8].passed,
            'nested_belief_and_coreference': rows[9].passed,
            'dialogue_action_selector': rows[11].passed and rows[12].passed,
            'korean_operation_family': rows[13].passed and rows[14].passed,
            'meta_memory_dialogue_act': rows[15].passed,
        }
        return V30AuditReport(before_passed, after_passed, len(rows), rows, checklist)
