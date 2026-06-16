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

| Profile | Intent | KV policy | Stateful chat | State pool | GPU sampling | Block prefill | Prefix cache | Snapshots | Speculative / Graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `original` | Run closest to ordinary adapter decode. No stateful extras. | `error` | off | on | on | on | off | off | native MTP/NEXTN off, graph off |
| `stateful` | Default runtime. Bounded exact KV plus recurrent state. | `ring` | on | on | on | on | on | off | native MTP/NEXTN on, graph off |
| `research` | Production runtime profile with snapshots enabled for experiments. | `ring` | on | on | on | on | on | on | native MTP/NEXTN on, graph off |

`state_pool`, `gpu_sampling`, public `block_prefill`, automatic prefix cache,
and native MTP/NEXTN with adaptive fallback are accepted runtime optimizations
in the default stateful profile. Infinite streaming, episodic memory, and TTT sidecar are common
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
- automatic prefix cache supported through shared runner/cache/block-table
  boundaries
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
  "kv_cache_dtype": "int4_bdr",
  "stateful_chat": true,
  "state_pool": true,
  "gpu_sampling": true,
  "block_prefill": true,
  "prefix_cache": true,
  "infinite_streaming": false,
  "episodic_memory": false,
  "ttt_sidecar": false,
  "prefill_chunk_size": 64
}
```

This changes only runtime state allocation and helper behavior for that request.
It does not change weights or logits math.

`kv_cache_dtype` is a storage/runtime option, not a model conversion option.
Supported values are `fp16`, `fp8_e4m3`, `int4`, and `int4_bdr`. `int4` and
`int4_bdr` use packed UINT4 K/V plus per-token/head scale+zero tensors.
`int4_bdr` adds K-only block-diagonal Hadamard rotation following the
SAW-INT4 serving-compatible path; V rotation remains disabled until a separate
inverse-output transform and parity gate are accepted.
Support is adapter-gated through `RuntimeCapabilities.kv_cache_dtypes`.
`KVCacheSpec` defines the requested storage format and `KVCacheLayout` owns the
model-agnostic tensor shapes, metadata allocation, and byte accounting. If a
generic adapter such as an HF/Gemma wrapper cannot execute a requested KV dtype,
`RuntimePlan` disables that request and falls back to the adapter's safe dtype
before allocation.

## Production Runtime Optimization Contract

LangBurst exposes accelerated runtime/continuous-serving optimizations through one feature plan,
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
| Automatic prefix cache / Radix-style trie | accepted | `RadixPrefixCache` + `BatchedModelRunner` + `KVBlockTable` | on for stateful adapters |
| OpenAI-style usage / cached-token accounting | accepted | `RequestUsage` + `BatchGenerationHandle.metrics` + server response builder | on |
| OpenAI/HF-style generation options | accepted | `GenerationConfig` + `sample_next` + `BatchGenerationWorker` | on |
| Explicit stateful sessions in continuous batching | accepted | `SessionStateStore` + `BatchGenerationWorker` external state attach | opt-in per request |
| Native MTP/NEXTN speculative decoding | accepted with adaptive fallback | `RuntimeEngine.generate_decode_result` + `RuntimeEngine.verify_nextn_tokens` + `NativeNextNVerifier` | on |
| Infinite streaming / episodic memory / TTT sidecar | adapter-gated | `RuntimePlan` + `langburst.research` implementations | off unless research profile/request enables |
| EAGLE / Medusa proposer | gated | `SpeculativeProposer` | off |
| CUDA Graph decode | gated | adapter capability | off |
| Paged KV / continuous batching | accepted | `ContinuousBatchScheduler` + `BatchedModelRunner` + `KVBlockTable` + state arena | on for greedy batch worker |
| FP8 / INT4 / BDR KV storage | adapter-gated | `RuntimeCapabilities.kv_cache_dtypes` + `KVCacheSpec` + `KVCacheLayout` + paged attention kernels | fp16 unless requested |

Accepted features may be enabled by default. Gated features must stay disabled
until parity and state-safety tests prove they preserve the target model.
Admission, OOM cleanup, model residency, and state-pool limits are host resource
policy, not model-family logic; keep them centralized in `EngineManager` and
`EngineResourcePolicy`.

### Automatic Prefix Cache

LangBurst's prefix cache is model-agnostic. `RadixPrefixCache` indexes token
prefixes in a trie and stores adapter state snapshots plus optional immutable
paged KV block IDs. `BatchedModelRunner` is the only execution owner: it looks
up a reusable prefix before scheduling, skips already-computed prompt tokens,
and stores new cache entries after prefill chunks complete.

Paged KV sharing is refcounted in `KVBlockTable`. Cache entries pin only full
block boundaries, so partial prompt blocks are never shared. A hit reuses
physical KV blocks when available and copies only non-attention state; adapters
without physical KV sharing fall back to cloned attention state snapshots.
Full-prompt prefix hits still schedule the last uncached token because the cache
does not store final logits. `prompt_cache_key` partitions the namespace for
tenant/workload isolation and better routing. Cache hits are surfaced as
`usage.prompt_tokens_details.cached_tokens`; they are accounting and latency
signals, not semantic conversation memory.

### API Usage Accounting

`RequestUsage` is the single server-side accounting object. Non-streaming
responses always include `usage`; streaming responses include it in the final
chunk when `stream_options.include_usage=true`.

```json
{
  "usage": {
    "prompt_tokens": 2048,
    "completion_tokens": 128,
    "total_tokens": 2176,
    "prompt_tokens_details": {
      "cached_tokens": 1024,
      "uncached_tokens": 1024
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "accepted_prediction_tokens": 3,
      "rejected_prediction_tokens": 1
    },
    "performance": {
      "queue_wait_s": 0.001,
      "ttft_s": 0.04,
      "e2e_s": 1.2,
      "decode_tok_s": 110.0
    }
  }
}
```

`reasoning.effort`, `reasoning_effort`, `text.verbosity`, `verbosity`,
`previous_response_id`, `user`, and `metadata` are accepted at the API boundary
so clients can use standard request shapes. The current low-level runtime does
not synthesize hidden reasoning tokens, so `reasoning_tokens` remains 0 unless
an adapter explicitly reports them. Logprob fields fail fast until a model path
can return exact scores.

### Generation Options

`GenerationConfig` is the single source of truth for request-level decoding
behavior. The server normalizes OpenAI/HF/continuous-serving request fields into this
object, the batch worker stores those values on the scheduled request row, and
`sample_next` applies the actual constraints.

Implemented end-to-end:

```text
temperature
top_p
top_k
min_p
seed
max_tokens / max_completion_tokens / max_new_tokens
min_tokens / min_new_tokens
stop / stop_sequences
stop_token_ids
ignore_eos
include_stop_str_in_output
repetition_penalty
presence_penalty
frequency_penalty
no_repeat_ngram_size
logit_bias
bad_words_ids
suppress_tokens / begin_suppress_tokens
```

Not implemented yet, intentionally fail-fast or no-op compatibility only:

```text
beam_search / best_of / n > 1
logprobs / top_logprobs / prompt_logprobs
non-text structured response_format / grammar / regex constrained decoding
tools / function_call / parallel_tool_calls
true hidden reasoning token generation
```

### Explicit Stateful Sessions

OpenAI-compatible `/v1/chat/completions` remains stateless by default: callers
send the messages they want the model to consider, and LangBurst may reuse
pooled state objects or prefix-cache blocks only as internal optimizations.
Those optimizations must not create semantic memory between unrelated requests.

Persistent model state is opt-in. A request enters the stateful path only when
it provides `session_id` or sets `stateful_session=true`. The session store keeps
the adapter DecodeState/KV/recurrent state under a per-session lock and keeps it
until TTL, explicit delete, model unload, OOM recovery, or LRU capacity
eviction.

Session requests now use the same continuous-batching worker as stateless
requests. At admission, the worker attaches the preserved DecodeState to the
scheduled row instead of allocating a resettable pooled state. On finish it
commits the final generated token, records prompt/completion counters, detaches
the state, and releases the per-session lock. This keeps the execution path
batchable/speculative-capable while preserving true low-level state continuity.

```json
{
  "model": "langburst-...",
  "messages": [{"role": "user", "content": "Continue from here."}],
  "session_id": "sess-...",
  "max_tokens": 128
}
```

Session API:

```text
POST   /v1/langburst/sessions
GET    /v1/langburst/sessions
DELETE /v1/langburst/sessions/{session_id}
```

For session requests, `messages` should be the next delta turn, not the whole
conversation transcript, unless the caller intentionally wants to re-ingest that
transcript into the native state.

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
