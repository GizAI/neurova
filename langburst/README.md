# LangBurst

LangBurst is a production-class serving engine target for low-bit, stateful, and
resource-constrained model serving. It is not limited to one 16GB GPU shape:
the design goal is a generic adapter-based runtime that can serve different
model families, scale down to modest GPUs with quantization/offload policies,
and expose stateful features such as ring KV, snapshots, infinite streaming,
episodic memory, and TTT sidecars through one execution contract.

The current champion measured path is Qwen3.6-27B text inference on a
16GB-class RTX 4080/4090 using flexible groupwise low-bit weights, Qwen Gated
DeltaNet kernels, ring-KV attention, paged/state-arena serving work, and
streaming decode state.

This is not yet a complete external serving engine replacement. The runtime is split into a common
`RuntimeEngine`, `EngineManager`, `AdmissionController`,
`ContinuousBatchScheduler`, `RuntimePlan`, and model adapters so future
Gemma/Llama/MoE-style targets do not fork the server/decode loop. The current
canonical adapter contract is:

- Adapter: `qwen36`.
- HF source model: Qwen3.6-27B / Qwen3.5-style text model.
- Champion converted model: Q4 Marlin fused projection checkpoint.
- Quantized tensor kinds: `lowbit_marlin_groupwise`,
  `lowbit_symmetric_groupwise`, and small `fp16_raw` gate projections.
- Supported rowwise weight bits: 2 through 8, selected by checkpoint metadata.
- Runtime entrypoints: `langburst-qwen-quantize`, `langburst-qwen-audit`,
  `langburst-chat`, `langburst-server`, `langburst-doctor`,
  `langburst-qwen-nextn-bench`, `langburst-qwen-graph-audit`, `langburst-qwen-profile`.
- Generic serving features: lazy multi-model residency, LRU unload, bounded
  request admission, queue timeout/reject counters, prompt/generation token
  admission, state pooling, OpenAI-compatible streaming, health/model/feature
  introspection, and OOM pool cleanup.
- Stateful/long-context features: `kv_window_policy=error|shift|ring`, ring KV,
  snapshots, boundary decay, `kv_cache_dtype=fp16|fp8_e4m3|int4|int4_bdr`,
  infinite streaming gate, episodic memory gate, and TTT sidecar gate through
  `RuntimeFeatures -> RuntimeCapabilities -> RuntimePlan`.
- continuous-serving execution work: continuous-batching scheduler, slot-indexed state
  arena, paged KV/block table, automatic prefix cache with Radix-style token
  trie and shared immutable KV blocks, batch-state CUDA kernels, native
  MTP/NEXTN speculative decoding with adaptive fallback, and CUDA Graph bucket
  scaffolding.
- CUDA extension symbols: Marlin W4A16 GEMM, rowwise low-bit fallback,
  RMSNorm, GDN recurrence, fp16/fp8/int4 paged attention decode, and sampling
  helpers.

The adapter split is documented in `docs/ADAPTER_ARCHITECTURE.md`.
Runtime feature profiles are documented in `docs/RUNTIME_FEATURES.md`.
Current duplicate/fragmentation boundaries are documented in `docs/STRUCTURE_AUDIT.md`.

## Environment

On `ml-dmc8` the standard environment is:

```bash
cd /home/user/workspace/neurova/langburst
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst
```

Build and validate CUDA:

```bash
./scripts/cuda_compile_and_test.sh
```

CPU-only tools can be installed with:

```bash
LANGBURST_SKIP_CUDA_EXT=1 python -m pip install -e .
```

## Convert

Convert a checkpoint with the desired bit width:

```bash
langburst-qwen-quantize /path/to/hf-model /path/to/converted-runtime-model --bits 4 --group-size 128
langburst-qwen-audit /path/to/converted-runtime-model --hf-model /path/to/hf-model
```

For a smaller or more constrained GPU target:

```bash
langburst-qwen-quantize /path/to/hf-model /path/to/converted-runtime-model --bits 3 --group-size 128
langburst-qwen-audit /path/to/converted-runtime-model --hf-model /path/to/hf-model
```

## Chat

The default `--weight-device auto` keeps Marlin checkpoints GPU-resident. CPU
offload is only a fallback for checkpoints that cannot fit in VRAM.

```bash
langburst-chat \
  --adapter qwen36 \
  --runtime-profile stateful \
  --block-prefill on \
  --prefill-chunk-size 64 \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --prompt "안녕. 너는 누구야?" \
  --recent-window "${LANGBURST_CONTEXT_WINDOW:-8192}" \
  --max-new-tokens 96 \
  --temperature 0 \
  --stream \
  --stats
```

Force the GPU-resident path:

```bash
langburst-chat \
  --adapter qwen36 \
  --runtime-profile original \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --weight-device cuda \
  --prompt "Say hello." \
  --max-new-tokens 32 \
  --temperature 0 \
  --stats
```

## OpenAI-Compatible Server

Run once and keep the model resident:

```bash
LANGBURST_LOWBIT_ROWS_PER_CTA=8 \
LANGBURST_CONTEXT_WINDOW=8192 \
langburst-server \
  --adapter qwen36 \
  --runtime-profile stateful \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --host 0.0.0.0 \
  --port 8008 \
  --recent-window "$LANGBURST_CONTEXT_WINDOW"
```

Smoke:

```bash
curl -sS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"langburst-qwen3.6-27b-q3","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
```

SSE streaming is supported with `"stream": true`. Add
`"stream_options": {"include_usage": true}` to receive the final usage object in
the last stream chunk.

Inspect the resolved runtime plan:

```bash
curl http://127.0.0.1:8008/v1/langburst/features
```

The response includes requested features, adapter capabilities, effective
features, and unsupported fields disabled by `RuntimePlan`. The
OpenAI-compatible chat endpoint also accepts per-request runtime overrides such
as `"runtime_profile": "original"`, `"kv_window_policy": "ring"`,
`"kv_cache_dtype": "fp8_e4m3"`, `"kv_cache_dtype": "int4_bdr"`,
`"state_pool": true`, `"gpu_sampling": true`, `"block_prefill": true`, or
`"prefix_cache": true`, or `"prefill_chunk_size": 64`.
`block_prefill` is the default serving path. Its accepted implementation uses
exact decode semantics internally; faster batched internals must clear final
logit parity, continuation-state parity, and recall gates before becoming
default.

The chat endpoint accepts common modern LLM API controls. These are wired into
the shared generation contract:

- Sampling/decoding: `temperature`, `top_p`, `top_k`, `min_p`, `seed`.
- Length/termination: `max_tokens`, `max_completion_tokens`,
  `max_new_tokens`, `min_tokens`, `min_new_tokens`, `stop`,
  `stop_sequences`, `stop_token_ids`, `ignore_eos`,
  `include_stop_str_in_output`.
- Repetition/logit controls: `repetition_penalty`, `presence_penalty`,
  `frequency_penalty`, `no_repeat_ngram_size`, `logit_bias`,
  `bad_words_ids`, `suppress_tokens`, `begin_suppress_tokens`.
- State/cache/observability: `session_id`, `stateful_session`,
  `reset_session`, `prompt_cache_key`, `prompt_cache_retention`,
  `stream_options.include_usage`, `previous_response_id`, `user`,
  `metadata`.
- Reasoning/text compatibility fields: `reasoning.effort`,
  `reasoning_effort`, `text.verbosity`, `verbosity`.

Unsupported scoring, structured output, and tool-calling options such as
`logprobs`, non-text `response_format`, and `tools` fail fast instead of
silently returning misleading data.

Responses include OpenAI-style usage accounting plus local performance metrics:

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
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
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

### Automatic Prefix Cache

The stateful serving path includes automatic prefix caching for repeated system
prompts, shared chat templates, and repeated document prefixes. LangBurst uses a
shared `RadixPrefixCache` over token IDs and stores adapter state snapshots plus
optional pinned paged KV block IDs. Cache hits skip already-computed prompt
tokens and reuse immutable full KV blocks through `KVBlockTable` refcounts.
Partial blocks are intentionally not shared, and unsupported adapters have the
feature disabled by `RuntimePlan`.

`prompt_cache_key` partitions the prefix-cache namespace. Use the same key for
traffic with a stable shared prefix and different keys for tenants or workloads
that must not share cache state. Cache hits are reported as
`usage.prompt_tokens_details.cached_tokens`; they are a latency/cost accounting
signal, not persistent conversation memory.

### Explicit Stateful Sessions

The OpenAI-compatible chat endpoint is stateless by default. Pooled states and
prefix cache are internal speed optimizations and do not imply persistent memory
between unrelated requests. To keep native model state across turns, create or
provide a session id:

```bash
curl -sS http://127.0.0.1:8008/v1/langburst/sessions \
  -H 'Content-Type: application/json' \
  -d '{"model":"langburst-qwen3.6-27b-q3"}'
```

Then send only the next turn delta with `session_id`:

```bash
curl -sS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"langburst-qwen3.6-27b-q3","session_id":"sess-...","messages":[{"role":"user","content":"Continue."}],"max_tokens":64}'
```

Sessions are bounded by `--max-sessions` and `--session-ttl-s`, and can be
deleted with `DELETE /v1/langburst/sessions/{session_id}`. Session requests use
the same continuous-batching worker as stateless requests: the worker attaches
the preserved DecodeState for the lifetime of the request, locks the session
row, and commits the final generated token before releasing it so the next turn
sees the real low-level model state.

### KV Cache Dtypes

LangBurst keeps KV storage behind one model-agnostic `KVCacheSpec` contract.
`RuntimeCapabilities.kv_cache_dtypes` declares what each adapter can execute
without changing model semantics; unsupported requests are resolved back to the
adapter's safe dtype by `RuntimePlan` before state allocation. Physical shapes,
metadata tensors, and byte accounting live in the shared `KVCacheLayout` helper,
not in individual model adapters.

| dtype | Storage | Intended use |
| --- | --- | --- |
| `fp16` | fp16 K/V | Highest compatibility baseline. |
| `fp8_e4m3` | fp8 K/V with fixed scale | Existing long-context memory saver for 16K-class serving. |
| `int4` | packed UINT4 K/V + per-token/head scale+zero | SAW/serving-compatible 4-bit KV memory reduction without rotation. |
| `int4_bdr` | packed UINT4 K/V + per-token/head scale+zero + K-only block Hadamard rotation | SAW-INT4 default accuracy-preserving path. |

`int4_bdr` follows the SAW-INT4 paper's serving-compatible path: K is rotated
with a block-diagonal Hadamard transform before token-wise INT4 write, and Q is
rotated inside decode before the dot product. V rotation is intentionally not
enabled in the default LangBurst path because it requires an inverse transform
on the attention output and needs separate parity/speed gates. The source paper
and implementation snapshot used for this port live under
`../papers/2604.19157.pdf` and `../third_party/research/saw-int4/`.

Multi-model serving uses a declarative resource file instead of server-side
branches:

```json
{
  "models": [
    {
      "model_name": "langburst-qwen3.6-27b-q3",
      "adapter": "qwen36",
      "hf_model": "/path/to/hf-model",
      "qb_model": "/path/to/converted-runtime-model",
      "device": "cuda",
      "recent_window": 8192,
      "runtime_profile": "stateful",
      "estimated_vram_mib": 14000
    }
  ]
}
```

Run with bounded model residency, bounded request admission, and VRAM reserve:

```bash
langburst-server \
  --models-json /path/to/langburst-models.json \
  --max-loaded-models 1 \
  --max-active-requests 1 \
  --max-queued-requests 2 \
  --admission-timeout-s 30 \
  --reserve-free-vram-mib 512 \
  --max-state-pool-size 1 \
  --max-prompt-tokens 4096 \
  --max-generation-tokens 1024
```

Inspect loaded models, scheduler counters, and CUDA resource state:

```bash
curl http://127.0.0.1:8008/v1/langburst/models
curl http://127.0.0.1:8008/v1/langburst/health
```

`/v1/langburst/health` is the operational endpoint for model load state,
request admission counters, CUDA memory, and pooled decode-state residency.
If generation hits CUDA OOM, the server clears the affected model's runtime
state pool and returns a 503 instead of silently leaving stale pooled state.
Prompts and generation lengths are admitted before runtime state is allocated;
oversized requests fail fast instead of pushing a constrained server into OOM.

Current serving status is intentionally conservative: the server has lazy
multi-model residency, LRU unload, bounded request admission, pooled
DecodeState reuse, and a partial continuous-serving greedy batch worker. Further
continuous-batching work must stay behind the same
`EngineManager`/`AdmissionController` boundary.
Serving defaults such as recent window, VRAM reserve, state-pool size, prompt
token limit, and generation token limit live in `langburst.core.defaults`.

## Runtime Tuning

The CUDA extension compiles low-bit GEMV variants once and selects at runtime:

```bash
LANGBURST_LOWBIT_ROWS_PER_CTA=4 langburst-chat ...
LANGBURST_LOWBIT_ROWS_PER_CTA=8 langburst-chat ...
LANGBURST_LOWBIT_ROWS_PER_CTA=16 langburst-chat ...
```

Marlin direct batch defaults to `4` after T=4 state/continuation parity passed.
Use this only as an emergency bisect override:

```bash
LANGBURST_MARLIN_DIRECT_MAX_BATCH=1 langburst-chat ...
```

Benchmark without rebuilding:

```bash
python benchmarks/bench_kernels.py --bits 3 --rows-per-cta 8
```

CUDA Graph is gated by audit, not enabled by default:

```bash
langburst-qwen-graph-audit --static
```

The current blocker is architectural: decode state still uses Python `pos` /
`kv_len` counters and Python ring-KV logical views. Do not count CUDA Graph as a
speed feature until a real device-counter `GraphDecodeState` path lands.

Long prompt prefill uses chunked `forward_block` by default in every runtime
profile. This preserves target-model math while avoiding the old token-by-token
prefill loop. Disable it only for regression bisects:

```bash
langburst-chat --block-prefill off ...
```

Current dmc8 result for q3 5120x5120 GEMV:

```text
rows_per_cta=4  : ~57.8 us
rows_per_cta=8  : ~56.8 us
rows_per_cta=16 : ~119.0 us
```

The rowwise fallback champion is `8`. The current speed path is Q4 Marlin, so
rowwise GEMV tuning is only relevant for fallback tensors and embeddings.

## Current Champion

The current measured Qwen champion path is target-only Q4 Marlin with fused projections:

```bash
langburst-qwen-quantize \
  /path/to/hf-model \
  /path/to/converted-runtime-model \
  --bits 4 \
  --layout marlin \
  --group-size 128 \
  --fuse-projections
```

Run it with all target weights GPU-resident:

```bash
python -m langburst.generate \
  --adapter qwen36 \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --device cuda \
  --recent-window 256 \
  --max-new-tokens 512 \
  --stats \
  --prompt "Write a concise technical note about quantized LLM inference."
```

Latest dmc8 result:

```text
128-token English: 29.63 tok/s
512-token English: 34.03 tok/s
```

Detailed per-change speed history is in `docs/PERFORMANCE_LOG.md`.
State/runtime feature coverage is in `docs/V05_FEATURE_TEST_MATRIX.md`.
Native MTP speculative decoding notes are in `docs/SPECULATIVE_RESEARCH.md`.
Structure cleanup boundaries are in `docs/STRUCTURE_AUDIT.md`.

Break down decode bottlenecks without changing the serving path:

```bash
langburst-qwen-profile \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --max-new-tokens 16
```

Compare feature profiles when the GPU is free:

```bash
langburst-bench-profiles \
  --hf-model /path/to/hf-model \
  --qb-model /path/to/converted-runtime-model \
  --profiles original,stateful,research \
  --max-new-tokens 128
```

## dmc8 One-Shot

```bash
MODEL_DIR=/path/to/hf-model \
QB_DIR=/path/to/converted-runtime-model \
BITS=3 \
./scripts/dmc8_reconvert_and_chat.sh
```

## Validation

CPU validation:

```bash
LANGBURST_SKIP_CUDA_EXT=1 python -m pytest -q \
  tests/test_quant_lowbit_cpu.py \
  tests/test_gdn_reference_cpu.py \
  tests/test_v04_correctness_cpu.py \
  tests/test_v05_runtime_cpu.py \
  tests/test_state_streaming_cpu.py \
  tests/test_memory_ttt_cpu.py
```

CUDA validation:

```bash
python -m pytest -q \
  tests/test_v05_cuda_kernels.py \
  tests/test_lowbit_gemv_cuda.py \
  tests/test_sampling_cuda.py \
  tests/test_gdn_parity_cuda.py
```

## Current Speed Boundary

The langburst kernel path is currently dominated by target-model Marlin
projection work, especially MLP `gate_up` and `down` projections. Reaching 100
emitted tok/s requires a high-acceptance speculative proposer such as native
MTP/EAGLE/Medusa behind the shared verifier contract, or a deeper fused target
layer path that preserves logits and state trajectory. Qwen3.6 Native MTP1 is
implemented as the only built-in proposer because the checkpoint includes MTP
weights, but it is not the serving default until it shows a repeatable suite
speed win. Learned external proposers such as EAGLE/Medusa stay gated by the
same verifier contract.
