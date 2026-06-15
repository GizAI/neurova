# QwenBurst Design

## Scope

QwenBurst is a narrow Qwen3.6-27B text runtime for one 16GB Ada GPU. It keeps
the custom runtime focused on target-model correctness, low-bit weights, GDN
state, fused projection checkpoints, and OpenAI-compatible serving.

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
RuntimeFeatures    owns runtime capability defaults
RuntimeFeatureOverride owns per-call/CLI/bench overrides
```

`generate.py`, `server.py`, and `bench_profiles.py` must stay thin consumers of
`RuntimeEngine`; they should not reimplement prefill or decode loops.

## Current Bottleneck

The current champion is target-only Q4 Marlin fused projection decoding. It is
correct, all-GPU, and avoids CPU offload. Remaining speed work must preserve
the exact target-model output distribution.

Latest dmc8 measurement, Qwen3.6-27B Q4 Marlin fused target:

```text
128-token English: 29.63 tok/s
512-token English: 34.03 tok/s
```

The per-change performance history is recorded in `PERFORMANCE_LOG.md`.

The remaining target-only cost is dominated by Marlin projection work. A short
CUDA profile after the attention and gate fixes showed MLP `gate_up` and `down`
as the largest categories; `lm_head` is no longer the primary bottleneck. The
model includes native MTP weights, but they are not enabled until an exact
accept/verify contract is implemented and shown to preserve greedy target
output.

## Correctness Policy

No speed claim is valid until all numbers include:

- backend and model path,
- quantization level,
- prompt and max token count,
- end-to-end tok/s from the serving API,
- sanity output text.
