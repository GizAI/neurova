# QwenBurst Adapter Architecture

QwenBurst is being split into a model-independent runtime plus model adapters.
The goal is to keep the current Qwen3.6 Q4 Marlin champion intact while making
Gemma-style targets possible without adding architecture-specific branches to
the server or decode loop.

## Canonical Shape

```text
qwenburst/
  core/
    adapter.py      # ModelAdapter protocol and AdapterRegistry
    runtime.py      # prefill, decode, sampling, OpenAI-server generation path

  adapters/
    qwen36.py       # Qwen3.6 hybrid GDN adapter
    gemma4*.py      # future Gemma adapters, not mixed into Qwen code

  model.py          # current Qwen3.6 math implementation
  server.py         # thin OpenAI-compatible API over RuntimeEngine
  generate.py       # thin CLI over RuntimeEngine
```

## Adapter Contract

Every model family owns:

- config import from the HF directory
- tokenizer and chat template
- checkpoint tensor mapping
- model construction
- decode state allocation
- EOS handling

The common runtime owns:

- prefill
- token-by-token decode
- greedy GPU sampling path
- server generation
- lock/state lifetime per request

This prevents Gemma, Qwen, MoE, or future MTP logic from forking the server
loop.

## Current Adapter

`qwen36` wraps the existing Qwen3.6 implementation:

- `Qwen36_27B_TextConfig`
- `QwenBurstModel`
- `DecodeState` ring KV/GDN state
- Qwen chat template with `enable_thinking=False`
- Q4 Marlin fused checkpoint defaults

No Qwen math or CUDA kernel behavior is changed by the adapter split.

## Gemma Path

Gemma support should be added as a new adapter, not by editing Qwen branches:

```text
Gemma4DenseAdapter:
  E2B / E4B / 12B text-only first
  local/global attention policy inside adapter/model implementation
  PLE handling isolated from Qwen
  HF parity tests before speed work

Gemma4MoEAdapter:
  26B-A4B later
  router/expert placement as adapter-owned model logic
```

Required gates before a Gemma adapter is considered working:

- weight coverage test
- tokenizer chat-template parity
- first-token HF top-k parity on BF16 or reference path
- token-by-token vs one-shot KV equivalence
- low-bit error audit after the BF16/reference path is correct
- 16GB memory-fit test
