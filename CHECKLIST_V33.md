# BrainOS V33 Schema-Learning Substrate Checklist

Goal: stop adding sentence-specific parser patches. New language ability must enter as experience, prediction error, schema candidate, tests/counterexamples, and gated promotion.

## Completed

- [x] Experience recorder for dialogue/correction/failure episodes (`episodes`).
- [x] Prediction-error recorder with typed error families (`prediction_errors`).
- [x] Schema candidate memory (`schema_candidates`).
- [x] Schema tests and counterexamples (`schema_tests`).
- [x] Regression/promotion result logging (`schema_eval_results`).
- [x] Slow stable schema store (`stable_schemas`).
- [x] Natural-language correction interpreter.
- [x] Construction schema induction.
- [x] Wrapper schema induction.
- [x] Event-frame schema induction.
- [x] Dialogue-act schema induction.
- [x] Schema executor used by runtime before legacy parser cascade.
- [x] Inner-clause schema compilation for wrapper operations.
- [x] Hardcode detector to guard new substrate/runtime code from benchmark-shaped string patches.
- [x] V33 audit proving before/after growth via schemas.
- [x] No autoregressive / no next-token objective preserved.
- [x] Legacy V30/V31/V32 regression selected tests preserved.

## Still not claimed

- [ ] Human-level AGI.
- [ ] Official full external benchmark saturation.
- [ ] GPU/deep semantic encoder.
- [ ] Full chart parser with typed candidate lattice.
