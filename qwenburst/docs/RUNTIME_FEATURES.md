# QwenBurst Runtime Features

QwenBurst has one feature contract: `RuntimeFeatures`.  Use profiles for common
setups, then override individual flags only when needed. CLI args, server
requests, and benchmark profile overrides all normalize through
`RuntimeFeatureOverride` before they touch `RuntimeFeatures`.

## Profiles

| Profile | Intent | KV policy | Stateful chat | Infinite streaming | Block prefill | Snapshots | Episodic / TTT | Speculative / Graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `original` | Run closest to ordinary Qwen decode. No long-stream extras. | `error` | off | off | on | off | off | off |
| `stateful` | Default QwenBurst runtime. Bounded exact KV plus recurrent state. | `ring` | on | on | on | off | off | off |
| `research` | Turn on research memory scaffolds for experiments. | `ring` | on | on | on | on | on | off |

`speculative_mtp` and `cuda_graph` remain off by default because the current
native MTP1 path is prompt-dependent and CUDA Graph still needs
`GraphDecodeState`.

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

Override one flag without changing the whole profile:

```bash
qwenburst-chat \
  --runtime-profile original \
  --kv-window-policy ring \
  --stateful-chat on \
  --prompt "Say hello." \
  ...
```

Server feature introspection:

```bash
curl http://127.0.0.1:8008/v1/qwenburst/features
```

Per-request override is also supported by the OpenAI-compatible chat endpoint:

```json
{
  "model": "qwenburst-qwen3.6-27b-q4-marlin",
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "temperature": 0,
  "runtime_profile": "original",
  "kv_window_policy": "ring",
  "stateful_chat": true,
  "block_prefill": true,
  "prefill_chunk_size": 64
}
```

This changes only runtime state allocation and helper behavior for that request.
It does not change weights or logits math. `block_prefill` is enabled for every
profile because it preserves the same target-model math while avoiding the slow
token-by-token prefill loop. Use `--block-prefill off` only for regression
bisects.

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

This command only changes runtime feature state allocation and helper flags; it
does not change model weights or logits math.

## Code Boundary

```text
RuntimeFeatureOverride
  <- CLI args
  <- OpenAI-compatible request fields
  <- benchmark profile overrides

RuntimeFeatures
  <- profile defaults + override validation

RuntimeEngine
  <- one owner for prefill, decode, streaming, and greedy GPU generation
```

Do not add new runtime flags directly in `server.py`, `bench_profiles.py`, or
`generate.py` without adding them to the feature contract first.

## Current Measurement

```text
server model: qwenburst-qwen3.6-27b-q4-marlin
profile: stateful default
block_prefill: true
prefill_chunk_size: 64
checkpoint path: /home/user/models/Qwen3.6-27B-qb4-marlin-fused
decode-only after 6612-token prefill: 40.45 tok/s
```

For short prompts, `original` and `stateful` execute the same target math and
the same non-wrapped direct KV path. For long prompts below the KV window, both
profiles now use block prefill and SDPA causal prefill.

## Longer Context Comparison

| Scenario | Expected behavior | Measurement status |
| --- | --- | --- |
| Short prompt + 128 decode, no KV wrap | `original` and `stateful` use the same target math and direct KV path. | about 31 tok/s decode |
| Long prompt below `recent_window=8192` | `original` and `stateful` should be speed-equivalent except for negligible policy overhead. | 4984 input tokens prefill in about 4.4s, about 1130 tok/s |
| Stream exceeds `recent_window=8192` | `original` with `kv_window_policy=error` must fail at the finite KV limit; `stateful` with ring KV continues with bounded exact recent window. | functional difference, not apples-to-apples speed |

Long-context measurement:

```text
stateful, max_new_tokens=128

input_tokens=784:
  elapsed_s=4.660
  generated_tok_s_full=27.47
  total_tok_s=195.70

input_tokens=2074:
  elapsed_s=4.851
  generated_tok_s_full=26.39
  total_tok_s=453.97

input_tokens=6612:
  elapsed_s=9.129
  generated_tok_s_full=14.02
  total_tok_s=738.27

decode_after_6612:
  elapsed_s=3.164 for 128 tokens
  decode_only_tok_s=40.45
```

The older live-server row, `4962` input tokens plus an `82` token response in
`245.003s`, measured the obsolete token-loop prefill process and should not be
used as the current performance baseline.

To measure literal profile rows without disturbing OpenWebUI, run the benchmark
on a free GPU or after intentionally stopping the server:

```bash
qwenburst-bench-profiles \
  --recent-window 8192 \
  --prefill-chunk-size 64 \
  --max-new-tokens 512 \
  --profiles original,stateful,research \
  --prompt "$(python - <<'PY'
print('Quantized inference note. ' * 2000)
PY
)"
```
