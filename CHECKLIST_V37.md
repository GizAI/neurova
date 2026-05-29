# BrainOS V37 Checklist — Failure-to-Construction Learning Loop

V37 implements the seven requested items without adding another sentence-pattern parser layer.

| # | Requirement | Status | Evidence |
|---:|---|---:|---|
| 1 | Demote regex extractor to seed candidate generator | ✅ | `RegexSeedCandidateGenerator` marks legacy parse outputs as `seed_candidate` or `rejected`; fallback `ResearchTaskIR` is never promoted as truth. |
| 2 | Store failed sentences as PredictionErrorRecord | ✅ | `V37DevelopmentalLoop.record_result()` writes `prediction_error_records` rows from runtime failures. |
| 3 | Cluster similar failures by embedding | ✅ | `HashEmbeddingSpace` + `FailureClusterer` cluster similar prediction errors without generation. |
| 4 | Generate construction candidates from repeated failures | ✅ | `ConstructionCandidateSynthesizer.candidates_from_cluster()` proposes wrapper/event schema candidates from failure clusters. |
| 5 | Promote constructions via user correction | ✅ | `SchemaLearningSubstrate.learn_from_correction()` remains the promotion gate; V37 audit verifies correction → promoted schema. |
| 6 | EventFrame creation is handled by learned construction | ✅ | EventFrame corrections create `EventFrameSchema`; runtime schema execution updates world-state. |
| 7 | Add 500-case paraphrase stress suite beyond Korea audit | ✅ | `V37ParaphraseStressSuite` generates 500 English/Korean/world/taxonomy/causal/exception cases and reports 500/500. |

## Design line

Regex is no longer treated as cognition. It is only a seed hypothesis generator. Stable behavior must come from schema memory, typed execution, event/world grounding, and prediction-error feedback.
