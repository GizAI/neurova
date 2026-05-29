# BrainOS V28 Completion Report

V28 focuses on the critique that earlier growth results were too synthetic and narrow. It adds a held-out generalization audit rather than only increasing smoke-test coverage.

## What changed

1. `brainos/semantic/v28_parsers.py`
   - `V28InteractiveFeedbackParser`: richer correction forms.
   - `V28GeneralizationParser`: broader language front-end for class membership, modality/negation, belief questions, temporal intervals, event frames, exception discourse, causal chains, and Korean comparison variants.

2. `brainos/v28_generalization.py`
   - `V28HeldoutGeneralizationBenchmark`: randomized, structure-held-out cases.
   - `V28NonLeakyTutor`: teaches construction/world/fact families, not exact benchmark prompts.
   - `V28AblationAudit`: no-tutor and construction-only baselines.
   - `V28FailureTaxonomy`: failure categorization.
   - `V28GeneralizationAudit`: full before/after report.

3. `brainos/semantic/grammar_engine.py`
   - Adds did-question, did-negation, and past-passive variants to learned constructions.

4. `brainos/agent.py`
   - Adds `run_v28_generalization_audit()`.
   - Improves targeted coreference for belief and event questions.

## Verified result

On the V28 held-out audit:

```text
before: 2 / 20 = 10.0%
after:  20 / 20 = 100.0%
delta:  +90.0 percentage points
exact prompt leakage: 0 overlaps
```

This is a controlled synthetic benchmark. It is harder and less leaky than fixed smoke tests, but it is still not a proof of human-level open-domain intelligence.

## Design principle

No autoregressive language model was added. The runtime remains a no-next-token system: neural/perceptual components score IR candidates, while cognition is represented as typed IR, construction grammar, memory, reasoner, world-state effects, and regression gates.
