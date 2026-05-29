"""LLM Teacher Society -- role-separated interfaces for developmental learning.

The LLM is NOT the runtime answer engine. It serves as:
- Mother: easy curriculum generation, developmental stage examples.
- Teacher: error correction, schema candidate proposal.
- Critic: overgeneralization detection, dangerous schema rejection.
- Examiner: hidden holdout tests, adversarial paraphrase generation.
- Linguist: construction grammar analysis, Korean morpho-semantic analysis.
- SafetyJudge: prompt injection detection, source contamination review.

Each role is an abstract interface. Concrete implementations can use
any frontier LLM (GPT-4, Claude, etc.) or even rule-based fallbacks.
BrainOS verifier always has final say -- LLM proposals are hypotheses.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json


# ---------------------------------------------------------------------------
# Role interfaces
# ---------------------------------------------------------------------------

class TeacherRole(ABC):
    """Proposes schema candidates from failure clusters."""

    @abstractmethod
    def propose_schema(
        self, error_type: str, failing_texts: List[str], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Given an error type and sample failures, propose a schema candidate.

        Returns a dict with at least:
            forms: List[str]
            meaning: str
            slot_constraints: Dict[str, str]
        Or None if unable to propose.
        """
        ...


class CriticRole(ABC):
    """Generates counterexamples to prevent schema overgeneralization."""

    @abstractmethod
    def generate_counterexamples(
        self, schema: Dict[str, Any], n: int = 5
    ) -> List[str]:
        """Return sentences that should NOT match the proposed schema."""
        ...


class ExaminerRole(ABC):
    """Creates hidden holdout and adversarial tests."""

    @abstractmethod
    def generate_holdout_tests(
        self, schema: Dict[str, Any], n: int = 5
    ) -> List[Dict[str, Any]]:
        """Return test cases: [{input, expected_ir_type, expected_answer}]."""
        ...

    @abstractmethod
    def generate_adversarial_paraphrases(
        self, text: str, n: int = 5
    ) -> List[str]:
        """Return paraphrases that should map to the same IR."""
        ...


class MotherRole(ABC):
    """Generates developmentally-appropriate curriculum."""

    @abstractmethod
    def generate_curriculum(
        self, difficulty: str, domain: str, n: int = 10
    ) -> List[str]:
        """Return a list of teaching utterances appropriate for the difficulty level."""
        ...


class LinguistRole(ABC):
    """Provides construction grammar and morphological analysis."""

    @abstractmethod
    def analyze_construction(self, text: str) -> Dict[str, Any]:
        """Return construction grammar analysis: {pattern, slots, meaning, variants}."""
        ...

    @abstractmethod
    def analyze_korean_morphemes(self, text: str) -> List[Dict[str, str]]:
        """Return morpheme analysis: [{surface, stem, pos, role}]."""
        ...


class SafetyJudgeRole(ABC):
    """Reviews schemas and data for safety risks."""

    @abstractmethod
    def review_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return {safe: bool, risks: List[str], recommendation: str}."""
        ...


# ---------------------------------------------------------------------------
# Default implementations (rule-based, no actual LLM call)
# ---------------------------------------------------------------------------

class RuleBasedTeacher(TeacherRole):
    """Fallback teacher using pattern matching, no LLM."""

    def propose_schema(
        self, error_type: str, failing_texts: List[str], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not failing_texts:
            return None
        # Simple heuristic: extract common structure.
        return {
            "forms": failing_texts[:3],
            "meaning": f"AutoLearned_{error_type}",
            "slot_constraints": {},
            "source": "rule_based_teacher",
            "confidence": 0.3,
        }


class RuleBasedCritic(CriticRole):
    """Fallback critic using simple negation heuristics."""

    def generate_counterexamples(
        self, schema: Dict[str, Any], n: int = 5
    ) -> List[str]:
        forms = schema.get("forms", [])
        counterexamples = []
        for form in forms[:n]:
            # Insert "almost" to create near-miss.
            words = form.split()
            if len(words) >= 3:
                counterexamples.append(" ".join(words[:1] + ["almost"] + words[1:]))
        return counterexamples


class RuleBasedExaminer(ExaminerRole):
    """Fallback examiner."""

    def generate_holdout_tests(
        self, schema: Dict[str, Any], n: int = 5
    ) -> List[Dict[str, Any]]:
        return []

    def generate_adversarial_paraphrases(self, text: str, n: int = 5) -> List[str]:
        return []


class RuleBasedMother(MotherRole):
    """Fallback curriculum generator."""

    def generate_curriculum(
        self, difficulty: str, domain: str, n: int = 10
    ) -> List[str]:
        return []


class RuleBasedLinguist(LinguistRole):
    """Fallback linguistic analyzer."""

    def analyze_construction(self, text: str) -> Dict[str, Any]:
        return {"pattern": text, "slots": {}, "meaning": "unknown", "variants": []}

    def analyze_korean_morphemes(self, text: str) -> List[Dict[str, str]]:
        return []


class RuleBasedSafetyJudge(SafetyJudgeRole):
    """Fallback safety reviewer."""

    def review_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {"safe": True, "risks": [], "recommendation": "ok"}


# ---------------------------------------------------------------------------
# Teacher Society: unified facade
# ---------------------------------------------------------------------------

@dataclass
class TeacherSociety:
    """Unified facade holding all teacher roles.

    Default: rule-based fallbacks. Swap in LLM-backed implementations
    by assigning to the role fields.
    """

    teacher: TeacherRole = field(default_factory=RuleBasedTeacher)
    critic: CriticRole = field(default_factory=RuleBasedCritic)
    examiner: ExaminerRole = field(default_factory=RuleBasedExaminer)
    mother: MotherRole = field(default_factory=RuleBasedMother)
    linguist: LinguistRole = field(default_factory=RuleBasedLinguist)
    safety_judge: SafetyJudgeRole = field(default_factory=RuleBasedSafetyJudge)

    def propose_and_validate(
        self,
        error_type: str,
        failing_texts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: teacher proposes, critic challenges, safety reviews."""
        context = context or {}

        # 1. Teacher proposes.
        proposal = self.teacher.propose_schema(error_type, failing_texts, context)
        if not proposal:
            return {"status": "no_proposal", "error_type": error_type}

        # 2. Critic generates counterexamples.
        counterexamples = self.critic.generate_counterexamples(proposal)
        proposal["counterexamples"] = counterexamples

        # 3. Safety check.
        safety = self.safety_judge.review_schema(proposal)
        if not safety.get("safe", True):
            return {
                "status": "safety_rejected",
                "proposal": proposal,
                "safety": safety,
            }

        # 4. Examiner generates holdout tests.
        holdout = self.examiner.generate_holdout_tests(proposal)

        return {
            "status": "proposed",
            "proposal": proposal,
            "counterexamples": counterexamples,
            "holdout_tests": holdout,
            "safety": safety,
        }
