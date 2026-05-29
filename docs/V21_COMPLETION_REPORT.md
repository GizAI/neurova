# V21 Completion Report — Predictive Semantic Core

## Core thesis

A stronger no-LLM language system should not add more ad-hoc modules. It should learn **constructions**: reusable mappings from surface forms to meaning operations.

```text
surface pattern + slots -> typed IR operation
```

This gives BrainOS a simple generalization mechanism: one corrected example can create a new parser rule for future entities.

## Implemented

- `brainos/semantic/construction.py`
  - `ConstructionLearner`
  - `SemanticConstruction`
  - one-shot abstraction from supervised NL→IR pairs
- `HybridSemanticCompiler`
  - construction learner inserted before learned parser and regex fallback
  - `learn_construction()` API
  - `parse_target_ir()` helper
- `FinalCognitiveOS`
  - command: `learn construction: <surface> => <target-ir>`
  - smoke coverage for one-shot parse and proof
- tests
  - `tests/test_v21_construction_learning.py`

## Evidence

```text
AST parse: all Python files passed
pytest: 16 passed
run_smoke: 37/37 passed, 100.0%
zip re-extract validation: passed
```

## Honest limitation

Construction learning is not open-domain human-level language understanding. It is a compact mechanism for increasing language coverage without returning to next-token LMs or endless handwritten regexes.
