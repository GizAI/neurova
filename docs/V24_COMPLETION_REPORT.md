# V24 Completion Report

V24 upgrades V23 from high-priority adaptive regex patches toward a small, dependency-free cognitive construction grammar engine.

## Core module added

`brainos/semantic/grammar_engine.py`

- `FeatureConstruction`
- `ConstructionVariant`
- `FeatureConstraint`
- `CognitiveConstructionGrammar`

The grammar engine stores explicit form-meaning pairings and compiles user corrections into reusable constructions. Each construction can produce multiple parse variants and instantiates typed IR only after lightweight slot unification succeeds.

## New parser modules

`brainos/semantic/v24_parsers.py`

- `V24InteractiveCorrectionParser`
- `V24TaxonomyQuestionParser`
- `V24KoreanGrammarParser`
- `V24TemporalIntervalParser`
- `V24EventFrameParser`
- `V24ExceptionDiscourseParser`

These are not the main generalization mechanism; they are high-precision front-end recognizers for feedback, temporal/event/exception structures, and Korean particle patterns. Generalization is delegated to `CognitiveConstructionGrammar`.

## Reasoning/memory fixes

- Temporal contradiction now respects overlapping time intervals instead of treating all positive/negative versions as the same conflict.
- Temporal role queries return `inconsistent` when overlapping positive and negative evidence exists.
- Direct exception blocking now works for queries such as `Can a penguin fly even though it is a bird?` after exception discourse has been stored.

## Verification summary

- Python AST parse: 45 files passed.
- Pytest: 37 passed.
- Smoke: 72/72 passed.
