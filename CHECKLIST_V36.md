# BrainOS V36 checklist — language acquisition substrate

- [x] Treat BrainOS as language-acquisition substrate, not a flat parser.
- [x] Add embedding-backed association memory for paraphrase/failure/schema recall.
- [x] Add EventFrame/SituationFrame source-of-truth model; triples are projections only.
- [x] Add object-centric state-space world model with current fluents.
- [x] Add spatial separation frame with separator/direction roles.
- [x] Add temporal-state handling that does not treat years as locations.
- [x] Add coreference-aware situation updates for `it` / `the region` / recent object.
- [x] Add prediction-error records for unparsed situation inputs.
- [x] Add integration path in `FinalCognitiveOS.observe` for situation updates/queries.
- [x] Add V36 audit and tests.
- [x] Keep no-autoregressive / no-next-token principle.

Known non-goals:

- Full open-domain language understanding.
- Official external benchmark saturation.
- Runtime LLM generation.
