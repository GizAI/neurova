# LangBurst Structure Audit

This document is the canonical cleanup map for LangBurst-only fragmentation.
The target pattern is production serving:

```text
thin CLI/API
  -> normalized request/config objects
  -> EngineManager / RuntimeEngine
  -> adapter registry + declared capabilities
  -> model-specific implementation behind adapters
```

## Fixed Fragmentation

### Generation options

Problem:

- Sampling and stop controls were copied through server, batch worker,
  model runner, and scheduler rows.
- Adding a new option required editing several layers.

Current structure:

- `GenerationConfig` is the single source of truth.
- `server.py` normalizes OpenAI-compatible request fields into `GenerationConfig`.
- `BatchGenerationWorker`, `BatchedModelRunner`, and `DecodeRequestState` pass
  `GenerationConfig` as one object.

### Single-engine CLI construction

Problem:

- `generate.py`, `bench_serving.py`, `bench_profiles.py`,
  `correctness.py`, and Qwen profile tools each repeated adapter lookup and
  `RuntimeEngine(...)` construction.

Current structure:

- `cli_features.create_runtime_engine_from_args()` owns single-engine CLI
  construction.
- Server remains manager-owned because it supports multi-model serving policy.

### Greedy generation config

Problem:

- Greedy decode was repeatedly expressed as
  `GenerationConfig(max_new_tokens=..., temperature=0.0, top_k=0, eos_token_ids=())`.

Current structure:

- `GenerationConfig.greedy(...)` is the shared constructor for target-only
  greedy decode.

### Single-model server spec

Problem:

- `server.py` manually built `ModelResourceSpec` from argparse fields.
- It also looked up the adapter descriptor only to recover the default model
  name.

Current structure:

- `ModelResourceSpec.from_args(...)` owns single-model resource declaration.
- Adapter default model-name resolution lives inside `ModelResourceSpec`, not
  in the server surface.
- `load_model_specs(...)` remains the multi-model JSON path.

### Chat completion execution path

Problem:

- The server still had a direct non-batched completion path guarded by an
  environment flag.
- That path bypassed batch workers, prefix-cache metrics, stop-sequence
  handling, and stateful-session accounting.

Current structure:

- `/v1/chat/completions` always submits through `BatchGenerationWorker`.
- Streaming and non-streaming requests share `_submit_completion(...)`.
- Direct generation remains only in lower-level runtime, correctness, and
  research harnesses.

### Runtime and research feature grouping

Problem:

- Runtime toggles and research toggles appeared as one flat CLI group.

Current structure:

- `CORE_BOOL_FEATURE_KEYS` contains production serving toggles.
- `RESEARCH_FEATURE_KEYS` contains `infinite_streaming`, `episodic_memory`,
  and `ttt_sidecar`.
- CLI displays them as separate groups.

### Local cache noise

Problem:

- Python bytecode and test-cache directories were present in the tree scan.

Current structure:

- They are removed during cleanup and are not part of the source structure.

## Intentional Boundaries

### `RuntimeEngine(...)` direct calls

Allowed only in:

- `cli_features.create_runtime_engine_from_args`: single-engine CLI factory.
- `EngineManager`: multi-model lifecycle owner.
- `engines/native_impl/conformance.py`: native adapter conformance fixture.
- `research/qwen_nextn_bench.py`: Qwen-specific speculative research harness.

### Qwen-specific code

Allowed only in:

- `adapters/qwen36.py`
- `adapters/qwen36_impl/`
- `adapters/qwen36_mtp.py`
- `adapters/qwen36_tools/`
- `research/qwen_nextn_bench.py`

Generic runtime/server/CLI code must not import `qwen36_impl`.

### Fallback paths

Fallback wording remains only where it describes correctness or research
behavior:

- adapter/model compatibility paths,
- explicit performance-log history,
- research-only demos,
- runtime fallback stats such as `fallback_reason`.

Product serving should still enter through `EngineManager`, `RuntimeEngine`,
and `BatchedModelRunner`.

## Verification

Use this after structure cleanup:

```bash
python -m compileall -q langburst/langburst langburst/tests
pytest -q langburst/tests
find langburst -type d \( -name '__py*' -o -name '.pytest*' \) -prune -exec rm -rf {} +
git diff --check
```
