"""Semantic Evaluator -- replaces substring-based audit with rich IR evaluation.

Each test case specifies:
- expected_ir_type
- expected_slots (subject, relation, object, etc.)
- expected_polarity
- expected_time
- expected_world_state
- expected_answer_mode
- forbidden_strings
- proof_required / source_required

Four benchmark categories:
1. Curriculum: official teaching/learning cases.
2. Hidden holdout: never seen during training.
3. Adversarial paraphrase: meaning-preserving surface variation.
4. Long-term regression: must never break.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import re


@dataclass
class SemanticExpectation:
    expected_ir_type: Optional[str] = None
    expected_slots: Dict[str, str] = field(default_factory=dict)
    expected_polarity: Optional[str] = None
    expected_time: Optional[str] = None
    expected_world_state: Dict[str, str] = field(default_factory=dict)
    expected_answer_mode: Optional[str] = None
    forbidden_strings: List[str] = field(default_factory=list)
    required_strings: List[str] = field(default_factory=list)
    proof_required: bool = False
    source_required: bool = False


@dataclass
class EvalCase:
    input_text: str
    expectation: SemanticExpectation = field(default_factory=SemanticExpectation)
    expected_answer: str = ""  # legacy substring match
    category: str = "curriculum"  # curriculum | holdout | adversarial | regression
    difficulty: str = "normal"
    name: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    observed_response: str = ""
    observed_ir_type: str = ""
    observed_confidence: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)


class SemanticEvaluator:
    """Evaluates BrainOS responses against rich semantic expectations.

    Replaces simple substring matching with structured verification.
    """

    def evaluate_single(
        self,
        case: EvalCase,
        response: str,
        ir_type: str = "",
        confidence: float = 0.0,
        ir_data: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        failures: List[str] = []
        exp = case.expectation
        ir_data = ir_data or {}

        # 1. IR type check.
        if exp.expected_ir_type and ir_type:
            if exp.expected_ir_type.lower() != ir_type.lower():
                failures.append(f"ir_type: expected={exp.expected_ir_type}, got={ir_type}")

        # 2. Slot checks.
        for slot_name, expected_value in exp.expected_slots.items():
            actual = str(ir_data.get(slot_name, "")).lower()
            if expected_value.lower() not in actual:
                failures.append(f"slot:{slot_name}: expected={expected_value}, got={actual}")

        # 3. Polarity check.
        if exp.expected_polarity:
            actual_polarity = ir_data.get("polarity", "")
            if actual_polarity and actual_polarity != exp.expected_polarity:
                failures.append(f"polarity: expected={exp.expected_polarity}, got={actual_polarity}")

        # 4. Forbidden strings.
        low_response = response.lower()
        for forbidden in exp.forbidden_strings:
            if forbidden.lower() in low_response:
                failures.append(f"forbidden string found: '{forbidden}'")

        # 5. Required strings.
        for required in exp.required_strings:
            if required.lower() not in low_response:
                failures.append(f"required string missing: '{required}'")

        # 6. Legacy substring match (backward compat).
        if case.expected_answer and case.expected_answer.lower() not in low_response:
            failures.append(f"expected_answer '{case.expected_answer}' not in response")

        passed = len(failures) == 0
        return EvalResult(
            case=case,
            passed=passed,
            observed_response=response[:500],
            observed_ir_type=ir_type,
            observed_confidence=confidence,
            failure_reasons=failures,
        )

    def evaluate_batch(
        self,
        cases: List[EvalCase],
        runner: Callable[[str], tuple],
    ) -> Dict[str, Any]:
        """Run a batch of cases. runner(text) -> (response, ir_type, confidence, ir_data)."""
        results: List[EvalResult] = []
        by_category: Dict[str, List[EvalResult]] = {}

        for case in cases:
            try:
                response, ir_type, confidence, ir_data = runner(case.input_text)
            except Exception as e:
                response, ir_type, confidence, ir_data = str(e), "Error", 0.0, {}

            result = self.evaluate_single(case, response, ir_type, confidence, ir_data)
            results.append(result)

            cat = case.category
            by_category.setdefault(cat, []).append(result)

        total = len(results)
        passed = sum(1 for r in results if r.passed)

        category_scores = {}
        for cat, cat_results in by_category.items():
            cat_total = len(cat_results)
            cat_passed = sum(1 for r in cat_results if r.passed)
            category_scores[cat] = {
                "passed": cat_passed,
                "total": cat_total,
                "accuracy": round(cat_passed / max(1, cat_total), 3),
            }

        failed_cases = [
            {
                "name": r.case.name or r.case.input_text[:50],
                "input": r.case.input_text,
                "expected": r.case.expected_answer,
                "observed": r.observed_response[:200],
                "ir_type": r.observed_ir_type,
                "failures": r.failure_reasons,
            }
            for r in results
            if not r.passed
        ]

        return {
            "passed": passed,
            "total": total,
            "accuracy": round(passed / max(1, total), 3),
            "category_scores": category_scores,
            "failed_cases": failed_cases,
        }


# -- Built-in curriculum cases -----------------------------------------------

def build_v36_curriculum() -> List[EvalCase]:
    """Core curriculum cases with semantic expectations."""
    cases = [
        EvalCase(
            input_text="Would you classify Kibo as a machine?",
            expected_answer="Yes",
            expectation=SemanticExpectation(
                expected_ir_type="QuestionIR",
                forbidden_strings=["cannot prove", "fallback", "unknown"],
            ),
            category="curriculum",
            name="taxonomy_classify_question",
        ),
        EvalCase(
            input_text="Could Kibo be considered part of the machine category?",
            expected_answer="Yes",
            expectation=SemanticExpectation(
                expected_ir_type="QuestionIR",
                forbidden_strings=["cannot prove", "fallback"],
            ),
            category="adversarial",
            name="taxonomy_paraphrase_considered",
        ),
        EvalCase(
            input_text="Dana borrows laptop from Omar.",
            expected_answer="Stored event IR",
            expectation=SemanticExpectation(
                expected_ir_type="EventIR",
                expected_slots={"actor": "dana", "patient": "laptop"},
            ),
            category="curriculum",
            name="event_borrow",
        ),
        EvalCase(
            input_text="Does Dana have laptop?",
            expected_answer="Yes",
            expectation=SemanticExpectation(
                expected_ir_type="QuestionIR",
                forbidden_strings=["cannot prove", "unknown"],
            ),
            category="curriculum",
            name="event_borrow_effect",
        ),
        EvalCase(
            input_text="I had a rough day.",
            expected_answer="",
            expectation=SemanticExpectation(
                expected_ir_type="SpeechActIR",
                forbidden_strings=["Stored claim"],
            ),
            category="curriculum",
            name="social_emotion_disclosure",
        ),
        EvalCase(
            input_text="철수가 영희를 압도한다",
            expected_answer="Stored comparison IR",
            expectation=SemanticExpectation(expected_ir_type="ComparisonIR"),
            category="curriculum",
            name="korean_dominance",
        ),
    ]
    return cases
