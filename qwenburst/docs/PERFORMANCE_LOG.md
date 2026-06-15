# QwenBurst Performance Log

This document records measured speed changes for the QwenBurst Qwen3.6-27B
runtime on `ml-dmc8` RTX 4080 16GB. Only end-to-end generated-token speed is
counted unless explicitly marked as profiler data.

## Measurement Contract

Canonical target path:

```text
HF model:        /home/user/models/Qwen3.6-27B
QwenBurst model: /home/user/models/Qwen3.6-27B-qb4-marlin-fused
Runtime:         target-only Q4 Marlin, GPU-resident weights
Device:          ml-dmc8 RTX 4080 16GB
Decoding:        greedy temperature=0
```

Important interpretation rules:

- Compare only rows with the same prompt class and token count.
- `tok/s` is emitted tokens divided by full prefill+decode wall time reported by
  `qwenburst.generate --stats`.
- Output sanity was checked on every accepted benchmark row. UTF-8 and natural
  English generation remained valid.
- Native MTP/NEXTN is not enabled. Current speed is pure target-model speed.

## Improvement Timeline

### 0. Starting Champion Before This Session

Inherited baseline from the existing Q4 Marlin fused target path:

```text
128-token English: 21.75 tok/s
512-token English: 22.18 tok/s
```

This already included earlier work before this session:

- Q4 Marlin W4A16 projection path.
- GPU-resident Q4 target weights.
- Fused checkpoint projections:
  - `mlp.gate_up_proj`
  - `linear_attn.in_proj_qkvz`
  - `self_attn.qkv_proj`
- Fused GDN gate/decay kernel.
- Fused GDN depthwise-conv update kernel.
- DFlash removed from the champion path.

### 1. GPU Greedy Token Path

Change:

- Added non-streaming greedy GPU token path in `qwenburst.generate`.
- Kept next-token argmax as a CUDA tensor during decode.
- Avoided per-token CPU `.item()` synchronization in the non-streaming benchmark
  path.

Measured result:

```text
128-token sky/math prompt:
  before: 21.75 tok/s
  after:  22.64 tok/s
  gain:   +0.89 tok/s, +4.1%
```

Notes:

- This is a small but clean latency improvement.
- Streaming mode still uses token-by-token CPU-visible output by design.
- Quality is unchanged for greedy decoding; the selected token is still exact
  argmax.

### 2. Attention Decode QK Reduction Fix

Problem:

The previous `attention_decode_kernel<256>` recomputed the same QK dot product
independently for every output dimension thread. With `head_dim=256`, that made
the attention score path do roughly 256x redundant dot-product work inside each
query-head block.

Change:

- Rewrote the kernel so one CUDA block owns one query head.
- Threads compute one QK product each, reduce it once in shared memory, then each
  thread updates its output dimension with the shared score.
- No model math or logits contract changed.

Measured result:

```text
128-token sky/math prompt:
  before: 22.64 tok/s
  after:  27.70 tok/s
  gain:   +5.06 tok/s, +22.3%

512-token technical-note prompt:
  before: 17.89 tok/s
  after:  30.89 tok/s
  gain:   +13.00 tok/s, +72.7%
```

Notes:

- The 512-token improvement is much larger because attention cache scan cost
  grows with generated length.
- This was the highest-value target-only fix in this session.
- Output remained natural and valid after the kernel change.

### 3. Fused FP16 `in_proj_ba` Gate Projection

Problem:

CUDA profiling after the attention fix showed the old low-bit pair kernel for
GDN `in_proj_a` and `in_proj_b` was still the second-largest remaining kernel
category:

```text
lowbit_gemv_pair: 24.766 ms over 8 profiled tokens, 11.43% of CUDA time
```

`in_proj_a` and `in_proj_b` are only 48 rows each. Their fused 96-row projection
does not satisfy Marlin's `N % 256 == 0` layout contract, so forcing Marlin is
not appropriate.

Change:

- Added fused `linear_attn.in_proj_ba.weight = cat(in_proj_b, in_proj_a)`.
- Stored this small fused tensor as FP16 raw in Marlin checkpoints.
- Updated the converter so `--fuse-projections --layout marlin` creates this
  canonical FP16 fused gate tensor automatically.
- Existing model loader already preferred `in_proj_ba` when present.

Measured result:

```text
128-token sky/math prompt:
  before: 27.70 tok/s
  after:  29.63 tok/s
  gain:   +1.93 tok/s, +7.0%

512-token technical-note prompt:
  before: 30.89 tok/s
  after:  34.03 tok/s
  gain:   +3.14 tok/s, +10.2%
```

Profiler confirmation after the change:

```text
lowbit_gemv_pair disappeared from the hot path.
Marlin projection kernels became 86.04% of CUDA time.
```

Top remaining projection categories over 8 profiled tokens:

```text
mlp_gate_up  total_ms=77.244  calls=512  avg_us=150.87
mlp_down     total_ms=37.836  calls=512  avg_us=73.90
gdn_qkvz     total_ms=26.798  calls=384  avg_us=69.79
gdn_out      total_ms=15.361  calls=384  avg_us=40.00
attn_qkv     total_ms=7.926   calls=128  avg_us=61.92
lm_head      total_ms=7.907   calls=8    avg_us=988.42
attn_o       total_ms=5.101   calls=128  avg_us=39.85
```

### 4. Runtime-Tunable Marlin `max_par`

Change:

- Added `QWENBURST_MARLIN_MAX_PAR`.
- Default remains `16`, preserving the existing champion behavior.
- This removes a hardcoded runtime knob so Marlin parallelism can be tuned
  without rebuilding CUDA.

Measured result:

```text
Not yet accepted as a speed change.
```

Notes:

- This is infrastructure for narrow tuning, not a claimed performance win yet.
- Any future accepted value must be recorded here with the same prompt and token
  count.

### 5. Chunked Block Prefill

Problem:

Long prompts were extremely slow because `RuntimeEngine.prefill()` advanced the
target model through `forward_one()` once per input token. A live 4962-token
prompt on the already-running server took 245.003s wall time including an
82-token response, so generation appeared to crawl even though short-prompt
decode was healthy.

Change:

- Added `RuntimeFeatures.block_prefill` and `prefill_chunk_size`.
- Enabled block prefill in `original`, `stateful`, and `research` profiles.
- Routed runtime prefill through chunked `model.forward_block(..., commit=True)`.
- Added `logits_mode="last"` so the final chunk computes the giant vocabulary
  projection only for the last prefill token.
- Kept `--block-prefill off` for exact regression bisects against the old token
  loop.

Measured result after the first chunked block connection:

```text
160-token prefill microprofile:
  token_loop:      7.538s,  21.23 tok/s
  block_chunk_32:  2.531s,  63.21 tok/s
```

Notes:

- This proved the runtime dispatch problem but was still not normal prefill
  speed because attention layers and GDN conv still contained token-level loops.

### 6. Attention Block Projection/MLP Prefill

Problem:

After chunked prefill, attention layers still ran qkv projection, output
projection, and MLP per token. In a 160-token profile, `attn_qkv` and `attn_o`
still had 2560 calls, and attention-layer MLP calls remained token-count
dependent.

Change:

- Batched attention-layer qkv projection for the whole chunk.
- Applied block RoPE for contiguous token positions.
- Kept causal attention state order correct.
- Batched attention output projection and post-attention MLP.

Measured result:

```text
160-token prefill:
  before attention batch path, chunk=32:  2.531s,  63.21 tok/s
  after attention batch path,  chunk=32:  0.692s, 231.24 tok/s
  after attention batch path,  chunk=64:  0.688s, 232.56 tok/s
```

Remaining bottleneck at this stage:

```text
gdn_conv: 7680 calls over 160 prefill tokens
attention_decode: 2560 calls over 160 prefill tokens
```

### 7. GDN Conv Scan Kernel

Problem:

`depthwise_conv_update_block()` still called the single-token CUDA conv kernel
once per token per GDN layer. That caused 7680 kernel launches for a 160-token
prefill.

Change:

- Added `depthwise_conv_update_scan` CUDA op.
- The kernel scans the chunk time axis per channel and updates the same conv
  state in the same order as the single-token loop.
- Added CUDA parity test against the old single-token loop.

Measured result:

```text
160-token prefill after attention batch path:
  before conv scan, chunk=64: 0.688s, 232.56 tok/s
  after conv scan,  chunk=64: 0.342s, 467.58 tok/s
```

### 8. SDPA Block Causal Prefill

Problem:

After conv scan, `attention_decode` became the dominant prefill bottleneck:
the attention layers still verified each chunk by calling the single-token
decode attention kernel once per token.

Change:

- For non-wrapped prefill, write the chunk KV by slice and use PyTorch SDPA for
  chunk causal attention.
- Keep the old token-loop attention fallback for ring-wrap and unusual state
  boundaries.
- Raised the default `prefill_chunk_size` from 32 to 64 after measurement.

Measured result:

```text
160-token prefill:
  token_loop baseline:                  7.457s,   21.46 tok/s
  block + attention batch + conv scan:  0.342s,  467.58 tok/s
  block + SDPA causal prefill:          0.154s, 1039.92 tok/s
```

Long-context prefill measurement:

```text
input_tokens=784:
  original prefill: 0.928s,  844.56 tok/s
  stateful prefill: 0.661s, 1186.67 tok/s

input_tokens=2074:
  original prefill: 1.742s, 1190.44 tok/s
  stateful prefill: 1.740s, 1191.96 tok/s

input_tokens=4984:
  original prefill: 4.392s, 1134.80 tok/s
  stateful prefill: 4.422s, 1127.16 tok/s
```

This replaces the earlier live-server long-context row where 4962 input tokens
plus an 82-token response took 245.003s on the old token-loop prefill process.

### 9. Long-KV Decode SDPA Path

Problem:

After normalizing prefill, long-context decode became the next bottleneck. At
about 5K live KV length, the custom single-token attention decode kernel was
about 65% of profiled decode time:

```text
decode_after_5k before SDPA path:
  8 tokens in 0.578s, 13.84 tok/s
  attention_decode: 349.453 ms over 128 calls, 64.92%
```

Microbench:

```text
KV length 5000, H=32, KVH=8, D=256:
  custom attention_decode_fp16:     2.335 ms/call
  SDPA with repeat_interleave:      0.598 ms/call
```

Change:

- Added a decode attention dispatch that uses SDPA when `length >= 1024`.
- Uses native SDPA GQA for decode (`enable_gqa=True`) without materializing
  repeated KV heads.
- Kept the existing custom CUDA kernel for shorter KV lengths.
- Kept block prefill on explicit repeated KV heads because native GQA SDPA
  OOMed in long block-prefill tests on 16GB.

Measured result:

```text
decode_after_5k after SDPA path:
  16 tokens in 0.807s, 19.83 tok/s

Native GQA SDPA microbench:
  KV length 5000, H=32, KVH=8, D=256:
    SDPA with repeated KV heads: 0.598 ms/call
    native enable_gqa=True view: 0.050 ms/call
```

End-to-end with 64-token decode:

```text
input_tokens=4984:
  original: prefill 4.392s, decode 2.071s, total 6.463s, total 781.11 tok/s
  stateful: prefill 4.422s, decode 2.071s, total 6.492s, total 777.53 tok/s
```

### 10. GDN Norm-Gate Hot Path

Problem:

After SDPA decode, profiling showed `gdn_norm_gate` as a misleadingly large
hot path because the decode path went through the general shape-dispatch helper.
That helper preserved the correct math but added unnecessary hot-path shape and
layout work before the CUDA RMSNorm+SiLU gate kernel.

Change:

- Added `gdn_norm_silu_gate_2d`, which uses the checkpoint norm width as the
  RMSNorm hidden size and calls the CUDA fused RMSNorm+SiLU gate directly.
- Decode still flattens the resulting GDN core before `out_proj`, preserving
  the original projection contract.
- Block prefill uses the same helper over chunk rows.

Measured result on `ml-dmc8`, 6.6K-token prompt, 32 decode tokens:

```text
before:
  generated=32 elapsed_s=1.508 tok_s=21.22
  gdn_norm_gate: 203.738 ms over 1536 calls, 132.64 us/call

after:
  generated=32 elapsed_s=1.315 tok_s=24.34
  gdn_norm_gate: 34.644 ms over 1536 calls, 22.55 us/call
```

Unprofiled target-only speed after this change:

```text
input_tokens=784,  generated=128:
  elapsed_s=4.660, generated_tok_s_full=27.47, total_tok_s=195.70

input_tokens=2074, generated=128:
  elapsed_s=4.851, generated_tok_s_full=26.39, total_tok_s=453.97

input_tokens=6612, generated=128:
  elapsed_s=9.129, generated_tok_s_full=14.02, total_tok_s=738.27

decode_after_6612:
  128 tokens in 3.164s, decode_only_tok_s=40.45
```

## Net Result

Best current measured target-only speed:

```text
128-token sky/math prompt:          29.63 tok/s
512-token technical-note prompt:    34.03 tok/s
```

Comparable net improvements inside this session:

```text
128-token chain:
  21.75 -> 22.64 -> 27.70 -> 29.63 tok/s
  total gain: +7.88 tok/s, +36.2%

512-token chain on the technical-note prompt:
  17.89 -> 30.89 -> 34.03 tok/s
  measured session gain: +16.14 tok/s, +90.2%
```

Using the inherited 512-token champion reference as a rough cross-prompt
reference:

```text
22.18 -> 34.03 tok/s
rough gain: +11.85 tok/s, +53.4%
```

## Current Bottleneck

After the accepted fixes, the bottleneck is no longer DFlash, CPU offload,
attention dot-product duplication, or tiny GDN `a/b` projections. The current
target-only path is dominated by Marlin projection work, especially MLP
`gate_up` and `down`.

For long prompts, the previous live-server bottleneck was prefill because that
process still used token-by-token prefill. Current code now has chunked block
prefill; it needs a fresh process benchmark before the long-context row can be
accepted as a measured improvement.

That means:

- `lm_head` fused top-k alone cannot get this runtime to 100 tok/s.
- More rowwise low-bit GEMV tuning is not the main path for the champion Q4
  Marlin checkpoint.
- CUDA Graph may reduce launch overhead, but it cannot remove the dominant MLP
  projection work.
- 100 tok/s likely requires exact native MTP/NEXTN acceptance or a deeper fused
  target-layer path that preserves target logits.

## MTP/NEXTN Status

The checkpoint contains native MTP weights:

```text
mtp_num_hidden_layers=1
mtp_use_dedicated_embeddings=False
15 mtp.* tensors including mtp.fc, mtp.layers.0.*, and mtp.norm
```

The installed Transformers `qwen3_next` implementation ignores them with:

```text
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
```

QwenBurst now has a single research probe:

```bash
qwenburst-speculative native-mtp1 --steps 16
```

Latest dmc8 result:

```text
probe accepted=15 / total=16
probe accept_rate=0.938
```

Exact verifier benchmark with fair timing initially failed the speed gate because
accepted tokens needed raw hidden replay. That replay is now removed by returning
raw hiddens from `forward_block`.

Latest adaptive MTP1 suite:

```text
bench-suite-mtp1 --adaptive --min-verified 2 --accept-threshold 0.8 --recent-window 64

sky 32:       19.38 -> 24.11 tok/s, speedup 1.244, keep true
math 32:      18.43 -> 18.55 tok/s, speedup 1.006, keep false
technical 64: 28.28 -> 28.60 tok/s, speedup 1.011, keep false
average speedup: 1.087
all outputs identical: true
```

Therefore MTP is still not enabled in the default champion server path. It is
kept as an explicit research/tuning CLI because the gain is prompt-dependent.
The acceptance and speed criteria for default MTP serving are:

```text
candidate tokens must match target greedy tokens exactly before state commit
end-to-end tok/s must beat target-only by at least 3%
```

Only after that can MTP speedups be counted as quality-preserving.

## CUDA Graph Decode1 Audit

Phase 2 is now gated by an explicit audit instead of a partial graph path:

```bash
qwenburst-graph-audit --static
```

Current static result is intentionally not graph-ready:

```text
graph_ready=false
greedy_argmax_device_safe=true
marlin_workspace_preallocated=true
device_position_counters=false
ring_kv_device_indexing=false
no_python_state_mutation=false
```

Interpretation:

- Greedy sampling already has a graph-friendly device argmax path.
- Marlin tensors already cache output/workspace buffers after warmup.
- The real blockers are still `DecodeState.pos` / `kv_len` as Python counters,
  Python ring-KV logical view materialization, and `forward_one()` mutating
  Python-visible state during the captured decode step.

The next graph work must therefore be a real `GraphDecodeState` with CUDA
counter tensors plus ring-KV device indexing. Capturing the current Python
decode loop would be misleading and is not counted as a speed improvement.

## Decode Bottleneck Decomposition

QwenBurst now has a dedicated profiling entrypoint:

```bash
qwenburst-profile \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --recent-window 256 \
  --max-new-tokens 16 \
  --prompt "Write a concise technical note about quantized LLM inference."
```

The default profiler excludes prompt prefill and profiles generated-token
decode only, after prefill has warmed model caches.

Latest dmc8 decode-only profile:

```text
generated=32 elapsed_s=1.315 tok_s=24.34

category          calls  total_ms  avg_us  measured_pct
mlp_gate_up       2048   311.961   152.32  35.63
mlp_down          2048   152.030    74.23  17.37
gdn_qkvz          1536   107.277    69.84  12.25
gdn_out           1536    59.094    38.47   6.75
rmsnorm           4128    47.574    11.52   5.43
sdpa_attention     512    47.178    92.15   5.39
gdn_norm_gate     1536    34.644    22.55   3.96
attn_qkv           512    31.766    62.04   3.63
lm_head             32    31.714   991.08   3.62
gdn_recurrent     1536    23.981    15.61   2.74
attn_o             512    22.197    43.35   2.54
gdn_conv          1536     5.906     3.85   0.67
embedding           32     0.130     4.05   0.01
```

Interpretation:

- The first 70%+ is now MLP/GDN projection work, not sampling or KV.
- `lm_head` is expensive per call but only about 3% of measured decode time, so
  fused top-k is not the next major lever.
- CUDA Graph can help launch overhead but cannot by itself reach 100 tok/s
  while `mlp_gate_up` and `mlp_down` dominate.
- The two credible 100 tok/s paths are still:
  1. exact native MTP/NEXTN with high acceptance, or
  2. a deeper fused target-layer path that reduces projection and GDN norm-gate
     kernel count without changing logits.

Rejected experiment:

- Tried preallocated `_out` variants for RMSNorm and GDN norm-gate to remove
  output allocation overhead.
- CUDA correctness passed, but decode-only profile stayed effectively flat and
  128-token generation did not beat the champion.
- The change was removed. Do not reintroduce this path unless a lower-level
  profiler proves allocation is the bottleneck.

Additional rejected/neutral tuning:

- Tried shape-based RMSNorm launch dispatch using larger 512/1024-thread blocks
  for hidden=5120/6144.
- CUDA parity stayed exact for tested tensors, but decode-only speed regressed:
  `gdn_norm_gate` moved from about 129 us/call to about 134 us/call and
  end-to-end 16-token decode dropped from about 21.4 tok/s to 20.6 tok/s.
- Reverted to the original 256-thread RMSNorm kernel.

Marlin `max_par` sweep on the 128-token technical-note prompt:

```text
max_par=4   31.29 tok/s
max_par=8   31.21 tok/s
max_par=12  29.93 tok/s
max_par=16  31.28 tok/s
```

No value beat the default materially. Keep the default `16`; do not add another
runtime branch for this.

## 100 tok/s Gap Analysis

The current repeatable 128-token target-only speed is roughly 31 tok/s on the
technical-note prompt. 100 tok/s therefore requires about:

```text
100 / 31.3 = 3.19x emitted-token throughput
```

This rules out several tempting but insufficient paths:

```text
CUDA Graph only:
  Can reduce launch gaps, but cannot remove the projection-dominated 75%+ hot
  path. It is useful after state is graph-safe, not enough by itself.

lm_head fused top-k only:
  lm_head is about 3% of measured decode time. Even perfect elimination is not
  a 100 tok/s path.

native MTP1 only:
  num_spec=1 can emit at most two tokens per target verification pass. Even a
  perfect 2.0x acceptance multiplier is below the required 3.19x.

RMSNorm block-size or allocation tweaks:
  Tested and rejected. They do not move the dominant projection wall.
```

The remaining viable speed paths are:

```text
1. Native NEXTN/MTP with more than one accepted speculative token per target
   pass, or a trained Medusa/EAGLE-style head verified by the exact target.

2. A real fused target-layer path:
   - keep Q4 Marlin projections
   - reduce MLP/GDN projection launches
   - fuse or partially fuse GDN recurrent output with norm-gate without changing
     RMSNorm scope or logits
   - keep all state commits exact

3. CUDA Graph decode1/verifyN after GraphDecodeState lands:
   - device pos/kv counters
   - device ring-KV indexing
   - no Python state mutation inside captured decode
```

Priority is therefore:

```text
P0: preserve champion Q4 Marlin target-only serving and OpenWebUI
P1: larger exact speculative suite; prove accepted_tokens_per_target_pass > 3.2
P2: target-layer fusion research around MLP/GDN hot path
P3: GraphDecodeState and CUDA Graph only after P1/P2 has a speed-positive path
```

## Long-Context Prefill Status

The old OpenWebUI row with 4962 input tokens in 245.003s is obsolete. It measured
the removed token-loop prefill path. Current chunked block prefill is the
accepted baseline.

Current remaining long-context issue:

```text
prefill is now about 1.1K tokens/s on 2K-6K prompts.
decode-only after 6.6K live KV is about 40 tok/s.
full generated-token tok/s falls on long prompts because prefill wall time is
included in the end-to-end denominator.
```
