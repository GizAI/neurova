# BrainOS V32 Predictive Developmental Runtime

# BrainOS V32 note

Runtime remains no-autoregressive and no-next-token. V32 adds predictive developmental learning infrastructure, not a GPT-style language model.

Official benchmark claim policy:

- `brainos/official_benchmark_loaders.py` evaluates user-supplied official SCAN/bAbI/CLUTRR-like files.
- If files are absent, it reports `loaded=false` and does not claim a score.
- Generated compatible audits are labeled as compatible generated subsets only.


---

# BrainOS V29 Systemic Learning

V29 focuses on learning **grammar operations, event frames, dialogue acts, and temporal schemas** instead of memorizing more sentences. It keeps the no-autoregressive/no-next-token rule: neural components only score semantic candidates; they do not generate language.

Core V29 additions:
- `WrapperConstructionIR`: learns operations such as `would you say [P]? -> QuestionIR(P)`.
- `EventFrameIR`: learns multi-slot event frames such as `A carries B from C to D -> located_at(B,D)`.
- `TemporalQuerySchemaIR`: represents temporal question schemas such as `Who served as ROLE during T?`.
- `MetaMemoryQuestionIR` and `SupportRequestIR`: dialogue-act layer for meta-memory and help requests.
- `V29GrammarOperationParser`: converts natural corrections into operation/frame/schema candidates.
- `V29SystemicLearningAudit`: evaluates before/after growth on held-out grammar-operation cases with leakage guard.

Validation summary:
```text
AST parse: 59 Python files passed
per-file pytest: 57 tests passed
run_smoke: 85 / 85 passed
V29 systemic audit: before 1/11, after 11/11, exact prompt leakage 0
```

Artifacts:
- `CHECKLIST_V29.md`
- `docs/V29_COMPLETION_REPORT.md`
- `docs/V29_SYSTEMIC_LEARNING_AUDIT_REPORT.json`
- `docs/V29_SMOKE_REPORT.json`

# BrainOS V28 Developmental Language Learning

V28 focuses on correction-driven construction learning, FCG-style construction-family generalization, event/world-state grounding, coreference, temporal/exception robustness, and adversarial growth testing. It keeps the no-autoregressive/no-next-token principle: neural components score semantic candidates; they do not generate text.

Artifacts:
- `CHECKLIST_V28.md`
- `docs/V28_COMPLETION_REPORT.md`
- `docs/V28_ADVERSARIAL_GROWTH_REPORT.json`
- `data/v27_text_ir_correction_event_corpus_20000.jsonl`

# BrainOS V26 Developmental Language Learning

This package extends V25 with LLM-distilled seed hypotheses, intrinsic motivation, developmental dialogue tutoring, dynamic event frames, elementary workbook growth tests, and no-autoregressive semantic perception. See `CHECKLIST_V26.md` and `docs/V26_COMPLETION_REPORT.md`.

# BrainOS V25: No-LLM Neuro-Symbolic Language Learning Runtime

# Brainlike Cognitive OS V24 — Cognitive Construction Grammar

# BrainOS V23 Cognitive Construction Grammar

# Brainlike Cognitive OS V23 — Predictive Semantic Core

V23 is a no-LLM Cognitive IR research kernel. The central upgrade over V20 is **construction learning**: instead of adding endless regexes, BrainOS can learn a reusable surface construction from one supervised example and map new utterances into typed IR.

Example:

```text
learn construction: frost brings icy roads => causal(frost, icy roads)
heat brings expansion
```

The second sentence is parsed as:

```text
CausalClaimIR(cause="heat", effect="expansion")
```

This keeps the no-LLM principle:

- no next-token objective
- no autoregressive text generation
- no `generate()` language model
- language is UI; IR is thought
- learning target is `surface construction -> meaning operation`

## Main components

- `HybridSemanticCompiler`: grammar + construction learner + learned parser + fragment parser + regex fallback
- `ConstructionLearner`: one-shot semantic construction abstraction
- `EvidenceGraphMemory`: versioned claim/evidence/proof memory
- `ActiveMemoryReasoner`: taxonomy, contradiction, temporal, causal, comparison, and exception reasoning
- `EpistemicImmuneSystem`: low-confidence/ambiguous/contradictory memory quarantine
- `SleepReplayConsolidator`: failure trajectory consolidation
- `DomainShardRouter`: code/policy/ops/quant/world/conversation routing

## Validation

```text
pytest: 16 passed
run_smoke: 37 / 37 passed, 100.0%
zip re-extract validation: passed
```

V23 is still not human-level general language intelligence. It is a stronger no-LLM research kernel for building it: broadening language coverage through reusable learned constructions rather than patchwork regex or tiny next-token language models.


## V24 note

V24 replaces many checklist-specific regex patches with a dependency-free cognitive construction grammar engine: feature-structure style form-meaning constructions, slot unification, generated question/negation/reverse variants, and natural-language correction-to-construction learning.


## V25 additions

- Neural semantic perception without autoregressive generation.
- FCG-style construction grammar with slot constraints and variants.
- Natural-language correction-driven parser update.
- Event/world-state grounding.
- Temporal/exception/contradiction-aware memory improvements.
- Large structured text↔IR/correction/event corpus generator.
- Continual regression benchmark gate.


## V28 Generalization Audit

Adds randomized held-out adversarial evaluation, leakage guards, ablation checks, and failure taxonomy. The goal is not to claim human-level intelligence, but to make the evaluation less narrow than fixed smoke tests.


## V30 Final Cohesive Structure

V30 adds a coherent front path: wrapper-first decomposition, construction-family composition, fluent world state, temporal interval algebra, and dialogue-action response planning. Older parsers remain only as fallback. Runtime remains no-autoregressive and no-next-token. See `CHECKLIST_V30.md` and `docs/V30_COMPLETION_REPORT.md`.


## V33 schema-learning substrate

V33 adds a durable learning substrate so new language ability can enter as data-backed schemas rather than parser-code patches. It records episodes, prediction errors, schema candidates, schema tests/counterexamples, gated promotion results, and stable schemas. Runtime now consults learned schemas before the legacy parser cascade. This preserves the no-autoregressive/no-next-token principle while making future growth depend on experience, correction, verification, and consolidation.


## V35 note — Broad Developmental Parser

V35 adds a broad typed chart/lattice front path, a no-generation semantic perception/retrieval layer, dialogue/social act handling, stronger event/world-state grounding, and an honest benchmark-readiness stance. It does not claim AGI or official full benchmark saturation.
