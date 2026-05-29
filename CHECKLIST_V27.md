# BrainOS V27 Completion Checklist

This checklist is based on the external critique of V25/V26 and the requested target:
LLM-distilled seed knowledge + FCG-style construction grammar + small neural semantic perception + interactive correction learning + event/world grounding + intrinsic motivation/active learning + sleep consolidation + regression-gated promotion, while preserving no-autoregressive/no-next-token runtime principles.

## Implemented

- [x] Preserve no-autoregressive / no-next-token runtime core.
- [x] Treat LLM seed knowledge as candidate/hypothesis, not oracle truth.
- [x] Add stronger natural-language correction parsing with discourse-prefix stripping: `No,`, `Actually,`, `In this domain`, `For this task`, `Correction:`.
- [x] Strengthen FCG-style construction family generalization:
  - [x] declarative variant
  - [x] do-support question variant
  - [x] do-support negation variant
  - [x] passive variant for transitive verb constructions
  - [x] inverse variants for comparison families
- [x] Add V27 general language parser for taxonomy paraphrases, negation/modality, belief questions, event frames, temporal intervals, exception discourse, Korean comparison variants, and causal chains.
- [x] Fix temporal split bug: `on 2026 민수 is mineral` now preserves subject=`민수`, time=`2026`.
- [x] Add event/world frames for buy/sell/move/put/open/close and possession/location effects.
- [x] Improve lightweight coreference stack for he/she/it/that, with protection for expletive `it` in exception/question frames.
- [x] Add stronger exception normalization for plural subjects such as ostriches → ostrich and adverb-cleaned actions.
- [x] Add adversarial hard benchmark that measures before/after developmental tutoring.
- [x] Add V27 corpus generator output: `data/v27_text_ir_correction_event_corpus_20000.jsonl`.
- [x] Add V27 regression tests.
- [x] Add completion report and growth report.

## Validation Results

- AST parse: 56 Python files passed.
- pytest per-file total: 53 tests passed.
- run_smoke: 85 / 85 passed.
- V27 adversarial growth benchmark:
  - before tutoring: 11 / 27 = 40.7%
  - after tutoring: 27 / 27 = 100.0%
  - delta: +59.3 percentage points

## Honest Scope

This is a controlled synthetic hard benchmark and a no-LLM developmental semantic learner prototype. It is not proof of human-level general intelligence. It demonstrates that the runtime can improve through correction/construction/event/world tutoring and then solve held-out variants using learned machinery.
