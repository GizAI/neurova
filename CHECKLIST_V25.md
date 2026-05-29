# BrainOS V25 Completion Checklist

Goal: combine no-autoregressive neural semantic perception with symbolic cognition: FCG-style construction grammar, interactive correction-driven parser update, event/world grounding, temporal/exception/contradiction-aware memory, large structured corpora, and regression-gated continual learning.

## Mandatory checklist

- [x] Neural semantic encoder/perception without autoregressive generation
  - `brainos/semantic/neural_perception.py`
  - Structured IR-label scorer, no decoder, no next-token objective.
- [x] FCG-style construction grammar/generalization engine
  - `brainos/semantic/grammar_engine.py`
  - FeatureConstruction, slots, constraints, variants, unification-style binding.
- [x] Natural-language correction based parser update
  - `brainos/semantic/v25_parsers.py::V25InteractiveSemanticFeedbackParser`
  - Handles `When I say "X", it means Y`, `X should be understood as Y`, Korean-style correction scaffolds.
- [x] Event/world-state grounding
  - `brainos/world_frames.py`
  - transfer/open/close/move/collect/speech-act frame effects to world-state claims.
- [x] Temporal/exception/contradiction-aware memory
  - versioned claim memory from earlier versions
  - V25 stop-event closure for open temporal intervals
  - exception question blocking and contradiction status.
- [x] Large text↔IR / correction / event corpus
  - `brainos/datasets.py::generate_v25_multitask_corpus`
  - generated artifact: `data/v25_text_ir_correction_event_corpus_8000.jsonl`.
- [x] Regression + benchmark based continual learning
  - `brainos/continual.py`
  - `ContinualLearningGate` and smoke integration.
- [x] Tests
  - `tests/test_v25_integrated_language_learning.py`
  - Total test files pass individually: 44 tests.
- [x] Extended smoke benchmark
  - `run_smoke`: 85 / 85 passed.

## Research alignment

- FCG/PyFCG: construction as form-meaning pairings with unification-like slot constraints.
- Interactive semantic parsing: user natural-language feedback becomes parser edits, not one-off regex patches.
- Grounded language learning: EventIR must yield world-state effects, not just log events.
- Continual learning: parser updates are regression-gated before promotion.
