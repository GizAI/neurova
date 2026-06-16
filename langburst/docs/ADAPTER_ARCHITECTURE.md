# LangBurst Adapter Architecture

LangBurst is being split into a model-independent runtime plus model adapters.
The goal is to keep the current Qwen3.6 Q4 Marlin champion intact while making
Gemma-style targets possible without adding architecture-specific branches to
the server or decode loop.

## Canonical Shape

```text
langburst/
  core/
    adapter.py      # ModelAdapter protocol and AdapterRegistry
    features.py     # RuntimeFeatures, RuntimeCapabilities, RuntimePlan
    runtime.py      # prefill, decode, sampling, OpenAI-server generation path
    manager.py      # multi-model residency, resource admission, status
    scheduler.py    # request admission and queue counters

  adapters/
    qwen36.py       # Qwen3.6 hybrid GDN adapter
    qwen36_mtp.py   # Qwen-only native NEXTN/MTP proposer
    hf_causal.py    # generic Transformers-backed causal adapter + Gemma4 path
    qwen36_impl/    # Qwen-only config/model/state implementation

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
forking the server loop.  A new adapter can decline ring KV, snapshots, CUDA
Graph, or speculative decoding; the runtime sees only the resolved effective
plan. Sidecar memory features such as infinite streaming, episodic memory, and
TTT are also resolved through the same plan instead of a separate flag object.

## Current Adapter

`qwen36` and `qwen36-a3b` wrap the Qwen3.6 hybrid implementation:

- `Qwen36_27B_TextConfig`
- `Qwen36Model`
- `DecodeState` ring KV/GDN state
- Qwen chat template with `enable_thinking=False`
- Q4 Marlin fused checkpoint defaults
- `RuntimeCapabilities.stateful_hybrid()`
- optional Qwen-native NEXTN/MTP proposer through `create_speculative_proposer`

No Qwen math or CUDA kernel behavior is changed by the adapter split.
The generic runtime does not import Qwen model/config/state or Qwen MTP code.
Those modules live under `langburst.adapters.qwen36_impl` and
`langburst.adapters.qwen36_mtp`; they are implementation details behind the
`qwen36`/`qwen36-a3b` adapters and Qwen-specific tools (`langburst-qwen-quantize`,
`langburst-qwen-audit`, `langburst-qwen-profile`, `langburst-qwen-smoke`).

## Extension Seam

The extension seam is one registry entry or a package entry point:

```python
adapter_registry.register(MyAdapter())
```

Third-party packages can also expose adapters through the `langburst.adapters`
entry-point group.  Runtime/CLI surfaces call the registry and do not hardcode
adapter ids.

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

Gemma support is exposed through `gemma4`, currently backed by the generic
Transformers causal wrapper. This is a correctness/conformance path, not the
final 16GB optimized low-bit path. Production Gemma should replace only the
adapter-owned model/state implementation while keeping the same runtime
contract.

```text
Gemma4Adapter:
  adapter_id = gemma4
  family = gemma4-transformer
  tokenizer/chat template isolated from Qwen
  state = HF past_key_values in the initial adapter
  low-bit optimized model/state can replace the wrapper later
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

The production inference-runtime pattern has four layers:

```text
model registry
  -> request scheduler
  -> KV/resource manager
  -> model executor kernels
```

LangBurst now has the first production-shaped layer:

- `ModelResourceSpec`: declarative model entry.
- `EngineManager`: lazy engine load, bounded loaded-model residency, LRU eviction.
- `EngineResourcePolicy`: host-level model/request/queue/memory admission policy.
- `AdmissionController`: central request admission and stats boundary.
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
      "model_name": "langburst-qwen3.6-27b-q4-marlin",
      "adapter": "qwen36",
      "hf_model": "/path/to/hf-model",
      "qb_model": "/path/to/converted-runtime-model",
      "device": "cuda",
      "recent_window": 16384,
      "runtime_profile": "stateful",
      "estimated_vram_mib": 14000
    }
  ]
}
```

`adapter` or `adapter_id` is required in model config.  The manager does not
guess a default adapter from registry order because third-party adapters can be
installed through entry points.

Still required before calling it production-class serving:

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
  targeted CUDA test + langburst-correctness

Release candidate:
  full langburst tests + one dmc8 real-model gate
```
