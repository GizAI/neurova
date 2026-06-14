# SaneFlow From-Scratch Plan

## Goal

Build a separate, simple, high-efficiency causal language model that can produce
general English before adding byte-patching, slots, diffusion, or speculative
decoding. This line should not be a GPT clone.

## Architecture V1: SaneFlowLM

SaneFlowLM is deliberately not a Transformer/GPT. It uses no full token-token
attention matrix.

```text
self-trained 16K byte-level BPE tokenizer
token embedding
pre-norm SaneFlow blocks
  causal depthwise local conv
  gated multi-timescale state mixer
  SwiGLU MLP
RMSNorm
tied LM head
```

Initial trainable sizes:

```text
base:   d_model=384, layers=8,  state_heads=6,  seq=256
larger: d_model=512, layers=10, state_heads=8,  seq=512
```

The tokenizer is trained locally with the `tokenizers` package only. The model
does not use the `transformers` library.

## Data Order

1. FineWeb-Edu `sample-10BT` bounded subset.
2. Cosmopedia v2 / filtered explanation prose.
3. Tiny answer-only SFT with base replay.
4. Verifiable QA/MCQ/copy/code only after the base can answer basic prompts.

Forbidden in base training:

- ChatML-only training.
- `Answer:`-heavy templated data as the majority.
- Memory/copy/slot-proof data.
- Benchmark eval data.

## Promotion Gates

A checkpoint can be promoted to a speaking baseline only if greedy or low-temp
sampling answers these probes with readable English:

```text
Hi. Who are you?
Tell me a short story about a robot and a garden.
Explain what a computer is in simple English.
What is the capital of France?
Write one sentence about the moon.
```

Loss alone is not a gate.
