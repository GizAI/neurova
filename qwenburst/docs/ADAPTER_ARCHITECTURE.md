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
    features.py     # RuntimeFeatures, RuntimeCapabilities, RuntimePlan
  runtime.py      # prefill, decode, sampling, OpenAI-server generation path
    manager.py      # multi-model residency, resource admission, status
    scheduler.py    # request admission and queue counters

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
- runtime capability declaration

The common runtime owns:

- prefill
- token-by-token decode
- greedy GPU sampling path
- accepted optimization features: block prefill, state pool, GPU sampling
- server generation
- lock/state lifetime per request
- capability resolution through `RuntimePlan`

This prevents Gemma, Qwen, MoE, or future speculative proposer logic from
forking the server loop.  A new adapter can decline ring KV, snapshots, TTT,
CUDA Graph, or speculative decoding; the runtime sees only the resolved
effective plan.

## Current Adapter

`qwen36` wraps the existing Qwen3.6 implementation:

- `Qwen36_27B_TextConfig`
- `QwenBurstModel`
- `DecodeState` ring KV/GDN state
- Qwen chat template with `enable_thinking=False`
- Q4 Marlin fused checkpoint defaults
- `RuntimeCapabilities.qwen_hybrid_gdn()`

No Qwen math or CUDA kernel behavior is changed by the adapter split.

## Extension Seam

The extension seam is one registry entry:

```python
adapter_registry.register(MyAdapter())
```

Each adapter exposes:

```text
AdapterDescriptor
  adapter_id
  family
  default_model_name
  capabilities
```

The server, CLI, benchmarks, and runtime must not branch on the model family.
They resolve `RuntimePlan` from descriptor capabilities and requested features.

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
- capability-resolution test showing unsupported Qwen-only features are disabled

## Serving Roadmap

The vLLM/TensorRT-LLM pattern has four layers:

```text
model registry
  -> request scheduler
  -> KV/resource manager
  -> model executor kernels
```

QwenBurst now has the first production-shaped layer:

- `ModelResourceSpec`: declarative model entry.
- `EngineManager`: lazy engine load, bounded loaded-model residency, LRU eviction.
- `EngineResourcePolicy`: host-level model/request/queue/memory admission policy.
- `RequestScheduler`: central request admission and stats boundary.
- `RuntimePlan`: accepted/gated optimization feature resolution.
- `RuntimeEngine.pooled_state`: state/KV object reuse behind a feature flag.
- GPU greedy sampling and chunked block prefill as accepted runtime features.
- server endpoints backed by the manager instead of a hardcoded single engine.
- `--models-json`: declarative multi-model resource config.
- model runtime status, explicit unload, queue rejection, and admission timeout.
- prompt/generation token admission before DecodeState allocation.

Example model config:

```json
{
  "models": [
    {
      "model_name": "qwenburst-qwen3.6-27b-q4-marlin",
      "adapter": "qwen36",
      "hf_model": "/home/user/models/Qwen3.6-27B",
      "qb_model": "/home/user/models/Qwen3.6-27B-qb4-marlin-fused",
      "device": "cuda",
      "recent_window": 2048,
      "runtime_profile": "stateful",
      "estimated_vram_mib": 14000
    }
  ]
}
```

Still required before calling it vLLM-class serving:

- iteration-level scheduler that separates prefill and decode work;
- paged/ring KV allocator shared across requests;
- prefix/state cache with copy-on-write forks;
- batchable executor interface for dense Transformer, Qwen GDN, and Gemma;
- device-counter `GraphDecodeState` before CUDA Graph can be enabled;
- native MTP/NEXTN or EAGLE proposer behind the shared verifier contract;
- async cancellation of in-flight generation;
- model warmup/health transitions beyond loaded/loading/failed/unloaded.

## Validation Policy

Use targeted validation by default:

```text
Python-only structure changes:
  py_compile + touched CPU tests

Runtime/server policy changes:
  scheduler/manager/server config tests

CUDA/model math changes:
  targeted CUDA test + qwenburst-correctness

Release candidate:
  full qwenburst tests + one dmc8 real-model gate
```
