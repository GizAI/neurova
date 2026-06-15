# QwenBurst Design

## Scope

QwenBurst is a narrow Qwen3.6-27B text runtime for one 16GB Ada GPU. It keeps
the custom runtime focused on target-model correctness, low-bit weights, GDN
state, and OpenAI-compatible serving.

Speculative acceleration must keep qwenburst as the target runtime. DFlash can
only be added as a draft adapter feeding candidates into qwenburst verification.
Do not switch the target runtime to vLLM/SGLang.

## Runtime Split

```text
qwenburst target path
  - q3/q4 low-bit checkpoint
  - custom CUDA lowbit GEMV
  - Qwen3.6 Gated DeltaNet recurrence
  - qwenburst-server OpenAI-compatible API

DFlash speculative option
  - qwenburst remains the target verifier
  - trained z-lab Qwen3.6-27B DFlash draft model proposes candidates
  - qwenburst verifies and commits accepted tokens
  - draft weights must be compact, low-bit, or target-state-reusing on 16GB
```

## DFlash Memory Policy

The qwenburst target model owns the correctness path and should remain GPU
resident as the q3/q4 checkpoint. A DFlash adapter is only valid if its own
weights fit the remaining VRAM without forcing target offload. If the published
DFlash safetensors file is a full fp16 draft network, convert it to qwenburst
low-bit draft tensors before enabling `--speculative-backend dflash`.

The current public Qwen3.6-27B DFlash artifact is a BF16 drafter component, not
a normal target CausalLM. Treat it as a side model with its own memory budget;
for 16GB deployment the default path is q3 draft conversion.

The adapter contract is:

```text
target forward collects DFlash target_layer_ids hidden taps
converted DFlash draft predicts a block of candidates
qwenburst target verifies candidates on a forked DecodeState
accepted tokens are committed to the real DecodeState
```

## Current Bottleneck

The GDN recurrent kernel is not the limiting path. The main target-runtime cost
is low-bit projection, especially MLP `gate_proj`, `up_proj`, and `down_proj`.

The current qwenburst kernel path is useful for correctness and controlled
experiments, but 100+ emitted tok/s needs either:

1. DFlash draft adapter accepting multiple tokens per qwenburst target verification, or
2. a much stronger low-bit dequant+MMA projection kernel.

## Correctness Policy

No speed claim is valid until all numbers include:

- backend and model path,
- quantization level,
- prompt and max token count,
- accepted draft tokens per target step if using DFlash,
- end-to-end tok/s from the serving API,
- sanity output text.
