# QwenBurst Design

## Scope

QwenBurst started as a narrow Qwen3.6-27B text runtime for one 16GB Ada GPU.
The target architecture is now a model-independent low-bit/stateful inference
engine with Qwen3.6 as the first adapter.  Qwen-specific math stays isolated;
server and runtime policy must remain adapter-neutral.

## Runtime Split

```text
qwenburst target path
  - q4 Marlin fused checkpoint
  - GPU-resident target weights
  - Qwen3.6 Gated DeltaNet recurrence
  - fused GDN gate/decay and depthwise conv kernels
  - qwenburst-server OpenAI-compatible API
```

The runtime follows the same separation used by mature editor/server runtimes
such as VS Code extensions and vLLM model executors, but compressed for this
single-user engine:

```text
ModelAdapter       owns architecture-specific config, weights, tokenizer, state
RuntimeEngine      owns prefill/decode/streaming/generation orchestration
EngineManager      owns model residency, resource admission, and model status
RequestScheduler   owns request admission, queue limits, and counters
RuntimeFeatures    owns runtime capability defaults
RuntimeFeatureOverride owns per-call/CLI/bench overrides
RuntimeCapabilities owns adapter-declared support
RuntimePlan        owns effective per-request execution policy
EngineResourcePolicy owns host limits such as loaded models, request slots,
                     state-pool retention, and VRAM reserve
```

`generate.py`, `server.py`, and `bench_profiles.py` must stay thin consumers of
`RuntimeEngine`; they should not reimplement prefill, decode, model-family
policy, or capability checks.

## Current Architecture Decision

The vLLM/TensorRT-LLM pattern is the inspiration: core runtime separated from
model executors and capability-specific kernels.  QwenBurst keeps the simpler
shape needed here:

```text
adapter registry
  -> adapter descriptor + capabilities
  -> RuntimePlan
  -> EngineManager / RequestScheduler
  -> RuntimeEngine
  -> model-specific forward/state implementation
```

This is intentionally smaller than vLLM: it has request admission and
multi-model residency, but not continuous batching, paged KV allocation, or an
iteration-level prefill/decode scheduler yet. Those should be added behind the
same manager/scheduler/plan boundary rather than as Qwen-specific branches.

The current accepted serving boundary is:

```text
FastAPI request
  -> EngineManager model lookup / lazy load / LRU unload
  -> RequestScheduler admission lease
  -> RuntimeEngine pooled DecodeState
  -> adapter model forward
  -> health/status via EngineManager.health()
```

Do not add ad hoc semaphores, model globals, or per-endpoint state caches. OOM
recovery, pool cleanup, and model residency are manager responsibilities.
`RuntimeEngine` may own concrete pooled state objects, but pool size is set by
`EngineResourcePolicy`.

## Current Bottleneck

The current champion is target-only Q4 Marlin fused projection decoding. It is
correct, all-GPU, and avoids CPU offload. Remaining speed work must preserve
the exact target-model output distribution and the exact continuation state.

Latest dmc8 measurement, Qwen3.6-27B Q4 Marlin fused target:

```text
128-token English: 29.63 tok/s
512-token English: 34.03 tok/s
```

The per-change performance history is recorded in `PERFORMANCE_LOG.md`.

The remaining target-only cost is dominated by Marlin projection work. A short
CUDA profile after the attention and gate fixes showed MLP `gate_up` and `down`
as the largest categories; `lm_head` is no longer the primary bottleneck.

Fast raw block is a research path until real-model state trajectory parity is
complete. It can be tested with `QWENBURST_FAST_RAW_BLOCK=1`:

```text
fast raw block:
  default off.
  final logits can match while GDN state, conv state, attention KV, or
  continuation state still diverges.

batched Marlin:
  useful as a layer-level [T, D] primitive.
  not accepted as an lm_head verifier shortcut.

speculative decoding:
  runtime uses a generic SpeculativeProposer verifier contract.
  Qwen3.6 Native MTP1 is implemented but default off after dmc8 benchmarking.
  EAGLE/Medusa must implement the same proposer contract and pass measured
  speed gates before default serving.
```

## Correctness Policy

No speed claim is valid until all numbers include:

- backend and model path,
- quantization level,
- prompt and max token count,
- end-to-end tok/s from the serving API,
- sanity output text.
