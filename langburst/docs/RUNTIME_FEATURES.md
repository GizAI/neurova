# LangBurst Runtime Features

LangBurst has one request contract and one execution contract:

- `RuntimeFeatures`: what a caller requested.
- `RuntimeCapabilities`: what the active adapter can execute.
- `RuntimePlan`: the resolved effective behavior used by runtime/server code.

CLI args, server requests, and benchmark profile overrides all normalize
through `RuntimeFeatureOverride`, then `RuntimePlan` disables unsupported
features centrally.  Server, CLI, and benchmarks must consume the effective
plan instead of re-deciding adapter policy.

## Profiles

| Profile | Intent | KV policy | Stateful chat | State pool | GPU sampling | Block prefill | Snapshots | Speculative / Graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `original` | Run closest to ordinary adapter decode. No stateful extras. | `error` | off | on | on | on | off | native MTP/NEXTN off, graph off |
| `stateful` | Default runtime. Bounded exact KV plus recurrent state. | `ring` | on | on | on | on | off | native MTP/NEXTN on, graph off |
| `research` | Production runtime profile with snapshots enabled for experiments. | `ring` | on | on | on | on | on | native MTP/NEXTN on, graph off |

`state_pool`, `gpu_sampling`, public `block_prefill`, and native MTP/NEXTN with
adaptive fallback are accepted runtime optimizations in the default stateful
profile. Infinite streaming, episodic memory, and TTT sidecar are common
runtime feature gates now; adapters decide support through `RuntimeCapabilities`
and the actual implementations can still live under `langburst.research` until
they graduate. Fast raw block internals remain explicit research toggles until
they show state-safe repeatable speed wins. EAGLE and Medusa remain future
proposer options behind the same verifier contract. CUDA Graph still needs
`GraphDecodeState`.

## Resolution Flow

```text
RuntimeFeatureOverride
  <- CLI args
  <- OpenAI-compatible request fields
  <- benchmark profile overrides

RuntimeFeatures
  <- profile defaults + override validation

RuntimeCapabilities
  <- adapter-declared support matrix

RuntimePlan
  <- effective features after capability resolution

RuntimeEngine
  <- one owner for prefill, decode, verification, and generation results
```

Do not add new runtime flags directly in `server.py`, `bench_profiles.py`, or
`generate.py` without adding them to the feature contract and capability
resolver first.

## Adapter Capabilities

The Qwen3.6 adapter declares `RuntimeCapabilities.stateful_hybrid()`:

- `kv_window_policies=("error", "shift", "ring")`
- stateful chat supported
- state pool, GPU greedy sampling, and block prefill supported
- snapshots supported for research profile state persistence
- infinite streaming, episodic memory, and TTT sidecar supported as adapter
  sidecar gates
- native MTP1 speculative proposer supported through the shared verifier
- CUDA Graph not accepted as default
- block prefill supported as the public API

Future Gemma adapters should declare their own capabilities instead of adding
Gemma-specific branches to the server or runtime. If an adapter does not support
ring KV, snapshots, sidecar memory, or speculation, `RuntimePlan` disables
those requested features before state allocation.

## CLI

Use the same options on `langburst-chat`, `langburst-server`, and
`langburst-qwen-profile`:

```bash
langburst-chat \
  --runtime-profile original \
  --block-prefill on \
  --prefill-chunk-size 64 \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --prompt "Say hello." \
  --max-new-tokens 64 \
  --temperature 0
```

Server execution-plan introspection:

```bash
curl http://127.0.0.1:8008/v1/langburst/features
```

Per-request override is supported by the OpenAI-compatible chat endpoint:

```json
{
  "model": "langburst-qwen3.6-27b-q4-marlin",
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "temperature": 0,
  "runtime_profile": "original",
  "kv_window_policy": "ring",
  "stateful_chat": true,
  "state_pool": true,
  "gpu_sampling": true,
  "block_prefill": true,
  "infinite_streaming": false,
  "episodic_memory": false,
  "ttt_sidecar": false,
  "prefill_chunk_size": 64
}
```

This changes only runtime state allocation and helper behavior for that request.
It does not change weights or logits math.

## TensorRT-Style Optimization Contract

LangBurst exposes TensorRT/vLLM-style optimizations through one feature plan,
not scattered server flags:

| Feature | Status | Runtime owner | Default |
| --- | --- | --- | --- |
| Chunked block prefill | accepted | `RuntimeEngine.prefill` | on |
| State/KV object reuse | accepted, explicit | `RuntimeEngine.pooled_state` | off by host default |
| State pool retention limit | accepted, explicit | `EngineResourcePolicy.max_state_pool_size` | 0 by host default |
| GPU greedy sampling | accepted | `RuntimeEngine.generate_decode_result` | on |
| Request admission / queue limits | accepted | `AdmissionController` | host policy |
| Lazy multi-model residency / LRU unload | accepted | `EngineManager` | host policy |
| VRAM load admission reserve | accepted | `EngineManager` | host policy |
| Prompt / generation token admission | accepted | `EngineManager` | host policy |
| Health / OOM pool cleanup | accepted | `EngineManager` + server boundary | on |
| Native MTP/NEXTN speculative decoding | accepted with adaptive fallback | `RuntimeEngine.generate_decode_result` + `RuntimeEngine.verify_nextn_tokens` + `NativeNextNVerifier` | on |
| Infinite streaming / episodic memory / TTT sidecar | adapter-gated | `RuntimePlan` + `langburst.research` implementations | off unless research profile/request enables |
| EAGLE / Medusa proposer | gated | `SpeculativeProposer` | off |
| CUDA Graph decode | gated | adapter capability | off |
| Paged KV / continuous batching | partial | `ContinuousBatchScheduler` + `BatchedModelRunner` + state arena | on for greedy batch worker |

Accepted features may be enabled by default. Gated features must stay disabled
until parity and state-safety tests prove they preserve the target model.
Admission, OOM cleanup, model residency, and state-pool limits are host resource
policy, not model-family logic; keep them centralized in `EngineManager` and
`EngineResourcePolicy`.

## Correctness Contract

`block_prefill` is enabled by default through the exact public block API. Fast
raw block internals are disabled by default until state and continuation parity
are complete. Enable with `LANGBURST_FAST_RAW_BLOCK=1` only for research.

The current conclusion from dmc8 testing:

```text
fast raw block:
  disabled by default.
  final-logit parity is insufficient; state trajectory parity is required.

batched Marlin:
  keep as a layer-level primitive candidate.
  do not use as an lm_head/verifier shortcut.
  only enable larger M after continuation-state parity.

current default:
  Q4 Marlin target path with native MTP/NEXTN enabled.
  Adaptive gate falls back to plain target decode when acceptance is poor.
  public block prefill API with exact target state updates.
```

Accepted gate:

```text
langburst-correctness --require-block-prefill-parity
  ok=true
  input_tokens=114
  argmax_match=true
  max_abs_logit_diff=0.0
  gdn_state_max_abs_diff=0.0
  conv_state_max_abs_diff=0.0
  attention_kv_max_abs_diff=0.0
  continuation_argmax_match=true
  exact recall filler0 passed
```

## Current Measurement

After removing unnecessary Marlin output-buffer zeroing:

```text
same include-prefill profile row:
  before: 2.464s, 6.49 generated tok/s
  after:  2.399s, 6.67 generated tok/s
  gain:   about 2.7%
```

The remaining accepted-path bottleneck is projection-heavy decode/prefill:
`mlp_gate_up`, `mlp_down`, GDN projection, and GDN output projection dominate
the profile.  Any future speed path must preserve continuation state, not only
the final prompt logits.

## Benchmarking Profiles

When the GPU is free, compare profiles with one model load:

```bash
langburst-bench-profiles \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --recent-window 256 \
  --max-new-tokens 128 \
  --profiles original,stateful,research
```

The CSV output reports effective features and disabled capability fields, so a
Gemma or other future adapter cannot silently pretend to support Qwen-only
runtime features.
