# BrainOS V23 Full TODO / Completion Checklist

Goal: strengthen true language learning without LLM imitation by moving from template parsing to cognitive construction grammar, natural-language correction, discourse frames, Korean particle parsing, and grounded benchmark gates.

## A. Research-to-Implementation Mapping

- [x] Map V21/V22 gaps against SHRDLU / Cyc-SCG / FCG / semantic parsing / interactive correction / grounded language learning.
- [x] Keep no-LLM constraint: no next-token objective, no autoregressive text generation, no `generate()` language model loop.
- [x] Treat language as UI and IR graph as thought.
- [x] Treat new surface forms as learnable form-meaning constructions.

## B. Cognitive Construction Grammar

- [x] Add natural-language correction parser.
- [x] Support: `When I say "A outruns B", I mean A is faster than B.`
- [x] Support: `By "A sparks B" I mean A causes B.`
- [x] Support: `"A ranks above B" means A greater_than B.`
- [x] Convert user correction into `ToolCallIR(learn_construction)`.
- [x] Let learned construction outrank generic claim parsing.
- [x] Generalize corrected construction to new entities immediately.
- [x] Add regression tests for correction-to-construction generalization.

## C. Interactive Semantic Parsing

- [x] Add user-friendly natural-language correction instead of requiring formal `learn construction: ... => ...` DSL.
- [x] Convert natural meaning phrases to canonical IR targets.
- [x] Support comparison, causality, positive claim, and negated claim corrections.
- [x] Preserve existing DSL as explicit expert path.
- [x] Log construction learning through memory actions.

## D. Korean Particle / Word-Order Layer

- [x] Add Korean reverse comparison parsing.
- [x] Support `영희보다 철수가 더 크다`.
- [x] Support `철수가 영희를 앞선다` / `철수가 영희를 능가한다`.
- [x] Support Korean causal chain shortcut for `비가 오면 ... 젖어서 ... 미끄러워질 수 있다`.
- [x] Support Korean giving event with `에게/을/를/에서` particles.
- [x] Support basic Korean belief and goal frames.

## E. Event / Belief / Goal / SpeechAct Frames

- [x] Add discourse-frame parser for speech acts.
- [x] Support `Alice asked Bob to open the door` as `SpeechActIR`.
- [x] Support `Alice handed Bob a package in Seoul yesterday` as `EventIR`.
- [x] Support `Bob thinks Alice is not the CEO` as `BeliefIR` with embedded `NegatedClaimIR`.
- [x] Preserve EventIR-derived possession claim for give/hand/send events.
- [x] Add tests for event, belief, speech act, and discourse frames.

## F. Compound Discourse / Grounding

- [x] Support `because/since` causal discourse frames.
- [x] Support `if/when X, Y` as causal/composite reasoning scaffold.
- [x] Support natural exception discourse pattern.
- [x] Store CompositeIR items through existing memory/reasoner path.
- [x] Keep grounded verifier and world transition memory from V20/V22.

## G. Evaluation and Regression Gate

- [x] Add `tests/test_v23_cognitive_construction_grammar.py`.
- [x] Extend smoke benchmark from 48 to 57 rows.
- [x] Add tests for natural correction, construction generalization, causal construction, discourse frames, Korean variants, and smoke integration.
- [x] Run AST compile checks.
- [x] Run pytest.
- [x] Run `run_smoke()`.
- [x] Repack ZIP.
- [x] Re-extract ZIP and rerun AST / pytest / smoke.

## H. Remaining Honest Limits

- [ ] Full FCG-style feature-structure unification is not complete; V23 adds a practical construction layer but not full FCG.
- [ ] Open-domain free-text understanding is still outside scope.
- [ ] External semantic parsing benchmarks are not bundled.
- [ ] Neural structured parser remains tiny/scaffold-level, not a trained broad semantic encoder.
- [ ] Grounded world/action understanding is still mini-world/event-frame level, not robotics-grade.

