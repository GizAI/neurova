# LangBurst Engine Architecture

LangBurst now follows a provider-host shape inspired by VS Code's extension
host: a small core owns policy and routing, while engine providers own actual
serving execution. The default provider is LangBurst Native. vLLM, SGLang, and
EXL3 use the same provider seam as optional engines.

## Canonical Shape

```text
langburst/
  engines/
    base.py       # EngineDescriptor, EngineCapabilities, EngineBackend protocol
    registry.py   # EngineRegistry, the single extension seam
    native/       # default provider, public facade, native runtime implementation
      __init__.py # public native facade
      provider.py # NativeBackend / NativeProvider
      runtime.py
      manager.py
      scheduler.py
      ...
    vllm/         # optional vLLM provider and bridge implementation
      __init__.py # VLLMProvider facade
      provider.py
      bridge.py
      lowbit.py
      plugins.py
      qwen36.py
    unavailable.py# registered SGLang/EXL3 placeholders until their packages are wired

  core/
    features.py   # LangBurst-only feature policy and capability vocabulary
    adapter.py    # legacy native-model adapter registry

  adapters/
    qwen36.py
    qwen36_impl/
    qwen36_mtp.py

  server.py       # thin OpenAI-compatible facade over EngineBackend
  generate.py     # thin CLI over EngineBackend
```

The new single source of truth is:

```text
EngineDescriptor
  engine_id
  display_name
  capabilities

EngineModelSpec
  model
  served_model_name
  tokenizer
  dtype / quantization / max_model_len / engine extras

EngineBackend
  list_models
  health
  generate_chat
  stream_chat
```

Server and CLI code resolve an engine once through `engine_registry` and do not
branch on vLLM, SGLang, EXL3, or native execution details.

## Engine Providers

### Native

`native` is the default engine and the reference implementation for the current
Qwen3.6/GDN low-bit path:

- Qwen3.6 hybrid GDN state
- LangBurst q3/q4 low-bit checkpoint format
- GDN recurrence CUDA op
- native continuous batching / paged KV / prefix cache
- native MTP/NEXTN experiments and production gates

Server, correctness, and benchmark surfaces import native runtime types from
`engines/native/__init__.py` via `langburst.engines.native`. Runtime-specific
tests may target native submodules, but product surfaces should not depend on
the internal file layout.

### vLLM

`vllm` is an optional engine. It owns its own standard serving machinery:

- OpenAI-compatible chat semantics
- continuous batching
- PagedAttention / paged KV
- prefix caching
- chunked prefill
- CUDA graph support
- standard quantization backends
- speculative decoding
- LangBurst feature bridge for Qwen3.6/GDN custom-model integration

The vLLM provider should not call or wrap LangBurst's native runtime. It should
use vLLM modules directly where vLLM already provides the capability.

LangBurst-only features are resolved once into `EngineFeaturePlan` and then
translated by `engines/vllm/bridge.py`:

- `stateful_sessions`: host-side conversation state for vLLM's stateless
  request model.
- `ring_kv` and `infinite_context`: vLLM prefix caching/PagedAttention are
  enabled and the exact LangBurst policy is forwarded through `hf_overrides`.
- `qwen36_lowbit`: the converted LangBurst checkpoint path and 4-bit metadata
  are forwarded to a custom vLLM model implementation. The custom model must be
  registered through vLLM's out-of-tree model/plugin mechanism; LangBurst passes
  the architecture name through `hf_overrides["architectures"]`.
- `recurrent_state`: marked as custom-model bridge metadata for Qwen3.6/GDN.
- `episodic_memory` and `ttt_sidecar`: exposed as sidecar metadata so request
  processors or custom model code can consume the same feature request.

This keeps the server and CLI out of vLLM-specific branches. vLLM owns its
generic execution substrate; LangBurst owns only the bridge contract for
features that vLLM does not provide natively.

The vLLM Qwen3.6 path must not call or wrap the native serving runtime. These
native runtime pieces are explicitly excluded from the vLLM path:

- `RuntimeEngine`
- `EngineManager`
- `BatchGenerationWorker` / `BatchedModelRunner`
- `ContinuousBatchScheduler`
- `KVBlockTable` / `RadixPrefixCache`
- `GenerationConfig` / `sample_next`

The only LangBurst-owned Qwen3.6 pieces allowed in the vLLM path are the model
semantics and checkpoint/kernel pieces that vLLM cannot replace:

- Qwen3.6 config/model-family mapping
- LangBurst 4-bit/low-bit checkpoint loader metadata
- GDN block/recurrent state logic
- `gdn_recurrent` custom kernel bridge
- episodic memory / TTT sidecar metadata

Everything else in the vLLM provider goes through vLLM: request serving,
scheduling, batching, paged KV, prefix caching, sampling, and standard HF
loading.

### SGLang

`sglang` is registered as an optional provider target. It should be wired for
structured generation, prefix-heavy multi-turn workloads, and constrained
decoding when SGLang is installed.

### EXL3

`exl3` is registered as an optional provider target for ExLlamaV3/TabbyAPI-style
local EXL3 deployments. It should stay behind the same `EngineBackend` contract.

### Native

`native` remains the default unless the operator explicitly chooses another
engine.

## Adapter Contract In Native

The old adapter contract remains inside the native provider boundary:

- config import from HF directory
- tokenizer/chat template
- checkpoint tensor mapping
- model construction
- decode state allocation
- native runtime capability declaration

This is the top-level LangBurst serving architecture for the native default.
External engines implement the same `EngineBackend` contract beside it.

## Extension Seam

Third-party engines register through the `langburst.engines` entry-point group
or directly:

```python
from langburst.engines import engine_registry

engine_registry.register(MyEngineProvider())
```

Each provider exposes:

```text
EngineDescriptor
  engine_id
  display_name
  module
  capabilities
```

The server and CLI consume only `EngineBackend`.

## Validation Policy

Use targeted validation by default:

```text
Engine registry/provider changes:
  py_compile + test_engine_registry_cpu

Server/CLI routing changes:
  server config tests + help smoke

Native Qwen36 math changes:
  targeted CUDA test + langburst-correctness

vLLM deployment gate:
  install vLLM on target host, run one real model chat/server smoke
```
