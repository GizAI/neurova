# BrainOS V34 Typed Chart/Lattice Checklist

V34 target: stop adding sentence-level parser patches; make learned schemas execute through a typed chart parser and candidate lattice.

## Completed

- [x] Added `brainos/chart_lattice.py` with typed lattice data structures:
  - `TokenSpan`
  - `LatticeNode`
  - `LatticeEdge`
  - `TypedCandidate`
  - `TypedCandidateLattice`
- [x] Added `TypedChartParser` as schema-driven chart parser.
- [x] Added wrapper-first decomposition into typed lattice nodes.
- [x] Added inner-clause compilation over learned construction schemas.
- [x] Added do/did question composition over learned constructions.
- [x] Added do/did negation composition over learned constructions.
- [x] Added passive voice composition over learned constructions.
- [x] Added event-frame schema execution via role unification.
- [x] Added dialogue-act schema execution through typed lattice.
- [x] Added `TypedConstraintVerifier` to block false-friend/nonassertive scope overgeneralization.
- [x] Integrated typed chart parser into `SchemaExecutor.apply()` before legacy schema executor fallback.
- [x] Updated schema-correction detection so event-frame corrections such as `When A ...` enter schema memory.
- [x] Added V34 audit and tests.
- [x] Preserved V30/V31/V32/V33 selected regressions.
- [x] Preserved no-autoregressive/no-next-token principle.

## Verification

- AST parse: all Python files pass.
- Selected regression tests: pass.
- V34 typed chart lattice tests: pass.
- `run_smoke()`: 85/85 pass.
- V34 chart-lattice audit: before 0/9, after 9/9, lattice has nodes/edges/candidates.

## Honest limits

- This is not a full broad-coverage natural-language chart parser.
- It is a typed schema chart parser for learned construction/event/dialogue schemas.
- It does not prove human-level language intelligence.
- It removes a major architectural bottleneck: learned schemas now execute through a typed candidate lattice rather than only through scattered parser cascade rules.
