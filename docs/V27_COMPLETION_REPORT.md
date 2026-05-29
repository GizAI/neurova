# BrainOS V27 Completion Report

V27 upgrades V26 from a developmental semantic scaffold into a stronger no-LLM developmental language-learning prototype.

## Core Changes

1. **Correction-driven construction learning**
   - Added robust natural-language correction parsing for meta-prefixes such as `No,`, `Actually,`, `In this domain`, and `Correction:`.
   - These corrections create construction patches rather than direct answers.

2. **FCG-style construction family variants**
   - Learned constructions now generate do-support questions, do-support negations, and passive variants for simple transitive verbs.
   - Example: `A eclipses B` generalizes to `does A eclipse B?`, `A does not eclipse B`, and `B is eclipsed by A`.

3. **World/event grounding**
   - Added stronger frames for buy/sell/move/put/open/close plus previous transfer frames.
   - Event effects become state claims such as `Sora has book` or `box located_at library`.

4. **Temporal, exception, and coreference fixes**
   - Fixed year-splitting bug in temporal parser.
   - Improved exception lemma normalization.
   - Added a small discourse entity stack for pronoun resolution.

5. **Hard growth benchmark**
   - Added `V27AdversarialGrowthLab` with before/after evaluation.
   - The benchmark uses held-out variants of constructions and world events rather than repeating training sentences.

## Validation

```text
AST parse: 56 Python files passed
pytest per-file total: 53 tests passed
run_smoke: 85 / 85 passed
V27 adversarial growth: 11/27 → 27/27
```

## Limits

V27 still does not prove human-level intelligence. It remains a controlled neuro-symbolic semantic learner, not an open-domain LLM. Its strength is explicit, inspectable, correction-driven growth.
