# QwenBurst Runtime Features

QwenBurst has one request contract and one execution contract:

- `RuntimeFeatures`: what a caller requested.
- `RuntimeCapabilities`: what the active adapter can execute.
- `RuntimePlan`: the resolved effective behavior used by runtime/server code.

CLI args, server requests, and benchmark profile overrides all normalize
through `RuntimeFeatureOverride`, then `RuntimePlan` disables unsupported
features centrally.  Server, CLI, and benchmarks must consume the effective
plan instead of re-deciding adapter policy.

## Profiles

| Profile | Intent | KV policy | Stateful chat | State pool | GPU sampling | Infinite streaming | Block prefill | Snapshots | Episodic / TTT | Speculative / Graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `original` | Run closest to ordinary Qwen decode. No long-stream extras. | `error` | off | on | on | off | on | off | off | native MTP1 off, graph off |
| `stateful` | Default QwenBurst runtime. Bounded exact KV plus recurrent state. | `ring` | on | on | on | on | on | off | off | native MTP1 off, graph off |
| `research` | Turn on research memory scaffolds for experiments. | `ring` | on | on | on | on | on | on | on | native MTP1 off, graph off |

`state_pool`, `gpu_sampling`, and public `block_prefill` are accepted runtime
optimizations. Fast raw block internals and Qwen3.6 Native MTP1 are explicit
research toggles until they show state-safe repeatable speed wins. EAGLE and
Medusa remain future proposer options behind the same verifier contract. CUDA
Graph still needs `GraphDecodeState`.

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
  <- one owner for prefill, decode, streaming, and greedy GPU generation
```

Do not add new runtime flags directly in `server.py`, `bench_profiles.py`, or
`generate.py` without adding them to the feature contract and capability
resolver first.

## Adapter Capabilities

The Qwen3.6 adapter declares `RuntimeCapabilities.qwen_hybrid_gdn()`:

- `kv_window_policies=("error", "shift", "ring")`
- stateful chat and infinite streaming supported
- state pool, GPU greedy sampling, and block prefill supported
- snapshots, episodic memory, and TTT sidecar available as research scaffolds
- native MTP1 speculative proposer supported through the shared verifier
- CUDA Graph not accepted as default
- block prefill supported as the public API

Future Gemma adapters should declare their own capabilities instead of adding
Gemma-specific branches to the server or runtime. If an adapter does not support
ring KV, snapshots, stateful streaming, or TTT, `RuntimePlan` disables those
requested features before state allocation.

## CLI

Use the same options on `qwenburst-chat`, `qwenburst-server`, and
`qwenburst-profile`:

```bash
qwenburst-chat \
  --runtime-profile original \
  --block-prefill on \
  --prefill-chunk-size 64 \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --prompt "Say hello." \
  --max-new-tokens 64 \
  --temperature 0
```

Server execution-plan introspection:

```bash
curl http://127.0.0.1:8008/v1/qwenburst/features
```

Per-request override is supported by the OpenAI-compatible chat endpoint:

```json
{
  "model": "qwenburst-qwen3.6-27b-q4-marlin",
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "temperature": 0,
  "runtime_profile": "original",
  "kv_window_policy": "ring",
  "stateful_chat": true,
  "state_pool": true,
  "gpu_sampling": true,
  "block_prefill": true,
  "prefill_chunk_size": 64
}
```

This changes only runtime state allocation and helper behavior for that request.
It does not change weights or logits math.

## TensorRT-Style Optimization Contract

QwenBurst exposes TensorRT/vLLM-style optimizations through one feature plan,
not scattered server flags:

| Feature | Status | Runtime owner | Default |
| --- | --- | --- | --- |
| Chunked block prefill | accepted | `RuntimeEngine.prefill` | on |
| State/KV object reuse | accepted | `RuntimeEngine.pooled_state` | on |
| State pool retention limit | accepted | `EngineResourcePolicy.max_state_pool_size` | host policy |
| GPU greedy sampling | accepted | `RuntimeEngine.generate_ids_greedy_gpu` | on |
| Request admission / queue limits | accepted | `RequestScheduler` | host policy |
| Lazy multi-model residency / LRU unload | accepted | `EngineManager` | host policy |
| VRAM load admission reserve | accepted | `EngineManager` | host policy |
| Prompt / generation token admission | accepted | `EngineManager` | host policy |
| Health / OOM pool cleanup | accepted | `EngineManager` + server boundary | on |
| Native MTP1 speculative decoding | gated | `RuntimeEngine` + `SpeculativeProposer` | off |
| EAGLE / Medusa proposer | gated | `SpeculativeProposer` | off |
| CUDA Graph decode | gated | adapter capability | off |
| Paged KV / continuous batching | roadmap | future resource manager | off |

Accepted features may be enabled by default. Gated features must stay disabled
until parity and state-safety tests prove they preserve the target model.
Admission, OOM cleanup, model residency, and state-pool limits are host resource
policy, not model-family logic; keep them centralized in `EngineManager` and
`EngineResourcePolicy`.

## Correctness Contract

`block_prefill` is enabled by default through the exact public block API. Fast
raw block internals are disabled by default until state and continuation parity
are complete. Enable with `QWENBURST_FAST_RAW_BLOCK=1` only for research.

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
  Q4 Marlin target-only.
  public block prefill API with exact target state updates.
  Native MTP1 available only with explicit speculative_decoding override.
```

Accepted gate:

```text
qwenburst-correctness --require-block-prefill-parity
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
qwenburst-bench-profiles \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --recent-window 256 \
  --max-new-tokens 128 \
  --profiles original,stateful,research
```

The CSV output reports effective features and disabled capability fields, so a
Gemma or other future adapter cannot silently pretend to support Qwen-only
runtime features.
