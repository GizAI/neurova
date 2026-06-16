# LangBurst Engine Architecture

LangBurst now follows a provider-host shape inspired by VS Code's extension
host: a small core owns policy and routing, while engine providers own actual
serving execution. The default provider is vLLM. SGLang, EXL3, and the legacy
LangBurst native runtime use the same provider seam.

## Canonical Shape

```text
langburst/
  engines/
    base.py       # EngineDescriptor, EngineCapabilities, EngineBackend protocol
    registry.py   # EngineRegistry, the single extension seam
    vllm.py       # default provider
    vllm_bridge.py# LangBurst feature requests -> vLLM kwargs/metadata/session bridge
    native.py     # legacy LangBurst runtime wrapped as a provider
    native_impl/  # native runtime/scheduler/KV/batch implementation
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

### vLLM

`vllm` is the default engine. It owns standard serving machinery:

- OpenAI-compatible chat semantics
- continuous batching
- PagedAttention / paged KV
- prefix caching
- chunked prefill
- CUDA graph support
- standard quantization backends
- speculative decoding
- LangBurst feature bridge for Qwen3.6/GDN custom-model integration

LangBurst should not reimplement these generic serving features.

LangBurst-only features are resolved once into `EngineFeaturePlan` and then
translated by `engines/vllm_bridge.py`:

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

This keeps the server and CLI out of feature-specific branches. vLLM owns the
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

Everything else goes through vLLM: request serving, scheduling, batching, paged
KV, prefix caching, sampling, and standard HF loading.

### SGLang

`sglang` is registered as an optional provider target. It should be wired for
structured generation, prefix-heavy multi-turn workloads, and constrained
decoding when SGLang is installed.

### EXL3

`exl3` is registered as an optional provider target for ExLlamaV3/TabbyAPI-style
local EXL3 deployments. It should stay behind the same `EngineBackend` contract.

### Native

`native` wraps the existing LangBurst in-process engine. It exists only for
Qwen3.6/GDN custom-kernel work and for validating the same semantics before or
beside their vLLM bridge:

- Qwen3.6 hybrid GDN state
- LangBurst low-bit checkpoint format
- GDN recurrence CUDA op
- native Qwen NEXTN/MTP experiments

Generic HF/Gemma/Llama serving should use vLLM instead of the native runtime.

## Adapter Contract In Native

The old adapter contract remains inside the native provider boundary:

- config import from HF directory
- tokenizer/chat template
- checkpoint tensor mapping
- model construction
- decode state allocation
- native runtime capability declaration

This is no longer the top-level LangBurst serving architecture. It is a plugin
implementation detail of `--engine native`.

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
