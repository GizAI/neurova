# V28 Generalization Upgrade Checklist

Goal: address the attached critique that evaluation was too synthetic/narrow by adding a less-leaky, held-out, adversarial generalization audit and by improving language generalization mechanisms rather than only adding fixed smoke tests.

## Evaluation / proof upgrades
- [x] Add randomized held-out benchmark with generated entities/objects.
- [x] Add exact prompt leakage guard: tutor prompts must not equal test prompts.
- [x] Separate before/after growth measurement.
- [x] Add ablation audit: no tutor and construction-only baselines.
- [x] Add failure taxonomy for parser/memory/world failures.
- [x] Add category-level scores.
- [x] Keep explicit disclaimer: benchmark is harder and less narrow, but still synthetic and not human-level proof.

## Language / learning upgrades
- [x] V28 interactive feedback parser for richer natural-language corrections.
- [x] V28 generalization parser for taxonomy wrappers, negation/modality, belief questions, temporal intervals, event frames, world-state queries, exceptions, causal chains, and Korean comparison variants.
- [x] Extend FCG-style construction variants with did-question, did-negation, and past passive.
- [x] Improve parser priority for V28 candidates.
- [x] Improve targeted coreference resolution for belief/event contexts.
- [x] Keep no-autoregressive / no-next-token runtime principle.

## Regression / packaging
- [x] Add `tests/test_v28_generalization_audit.py`.
- [x] Preserve previous V26/V27 focused tests.
- [x] Run AST parse.
- [x] Run V28 tests and smoke.
- [x] Zip and re-test from extracted package.
