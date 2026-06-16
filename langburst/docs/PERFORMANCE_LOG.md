# LangBurst Performance Log

## 2026-06-16 external serving engine MTP Comparison and Verifier Contract

Current accepted conclusion:

```text
fast raw block:
  default off.
  final-logit parity was not enough; real-model state parity still showed
  GDN/conv/KV differences and speculative technical output drift.
  keep LANGBURST_FAST_RAW_BLOCK=1 only as an explicit research override.

speculative decoding:
  stateful default on with adaptive fallback to target-only decode.
  Native MTP/NEXTN remains the only built-in proposer.
  prompt-lookup/n-gram has been removed from the runtime path.

EAGLE / Medusa:
  future proposer implementations behind the same verifier interface.
  not enabled until they show sequence identity, state identity, and a measured
  speed win on a prompt suite.
```

Follow-up external serving engine gap check:

```text
external serving engine Qwen3NextMTP pattern:
  - MTP layer is a real full_attention decoder layer.
  - Draft prefill/decode keeps draft hidden state and slot mappings.
  - Multi-step speed comes from target block verification, not from MTP1 alone.

LangBurst measured limit:
  - MTP proposer cost: about 1.76 ms.
  - Target step cost: about 24.1 ms.
  - Exact MTP1 still needs the same target forward count as greedy decode.
  - Therefore exact MTP1 cannot produce a large speedup without a target block
    verifier.

Fast raw verifier experiment with LANGBURST_FAST_RAW_BLOCK=1:
  sky 128:       target 41.16 tok/s, rawspec 44.88 tok/s, speedup 1.09, identical true
  technical 128: target 41.38 tok/s, rawspec 40.41 tok/s, speedup 0.98, identical true
  interpretation: this is the right direction for production-like speed, but it is
  prompt-dependent and still lacks full state trajectory parity.

Kernel/primitive checks:
  - depthwise_conv_update_scan matches the single-token loop.
  - gdn_recurrent_ab_scan now has a CUDA parity test against the single-token
    loop.
  - Real-model GDN block still differs slightly because block projection and
    layer composition are not exactly the same state trajectory as token-loop
    target decode.
```

external serving engine comparison pass:

```text
source checked: /tmp/vllm-qwen-mtp, latest main fast-forwarded on 2026-06-16
relevant pattern:
  - Qwen3Next MTP uses current target hidden + sampled first token.
  - Speculative proposer avoids prompt-history reconstruction.
  - Hot candidate path should not preserve full vocab logits when only argmax is needed.

LangBurst fixes kept:
  - Native MTP proposer now receives empty history because it only consumes model signals.
  - MTP candidate argmax no longer clones the full 248K-vocab lm_head output.
  - Verifier logits/raw_hidden are cloned because Marlin lm_head reuses output
    buffers; without this, rejected candidates corrupted target-second checks.
  - q/k RMSNorm now goes through the shared CUDA RMSNorm path instead of ad hoc
    torch ops.
  - Verifier now commits the first target token on the live state, compares the
    MTP candidate to the target second-token argmax, and commits the accepted or
    corrected second token directly on the live target state. This removes the
    accepted-path fork/copy and candidate CPU sync.

LangBurst fix rejected:
  - Reusing one branch DecodeState was measured and removed. It was no faster
    because copy_from_ still copies the full GDN/conv state every speculative step.
  - Fast raw block as speculative verifier. It is not lossless yet.

Latest dmc8 128-token suite after kept fixes:
  sky:        speedup 1.048, accept_rate 0.909, identical true
  math:       speedup 0.994, accept_rate 0.800, identical true
  technical:  speedup 0.993, accept_rate 0.667, identical true
  average speedup: 1.012
  policy sweep best: min_verified=1, accept_threshold=0.70, max_rejections=1
  historical decision at that point: keep as research path.
  superseded by the current stateful default-on native MTP/NEXTN path with
  adaptive fallback.
```

Native NEXTN follow-up after matching the reference runtime's recurrent MTP proposer loop:

```text
implementation:
  - QwenNativeMTP1.argmax_sequence now reuses the checkpoint's single MTP layer
    for multiple draft steps, matching the reference runtime's Qwen3NextMTP decode pattern.
  - RuntimeEngine and langburst.research.qwen_nextn_bench now share the same max_draft
    candidate loop.
  - n-gram/prompt lookup remains absent from the runtime path.

dmc8, Qwen3.6-27B-qb4-marlin-fused, recent_window=2048, 128-token suite:

max_draft=2 adaptive min_verified=1 accept_threshold=0.70:
  sky:        target 33.67 tok/s, speculative 35.21 tok/s, speedup 1.046, accept 0.500, identical true
  math:       target 32.55 tok/s, speculative 32.34 tok/s, speedup 0.993, accept 0.667, identical true
  technical:  target 35.35 tok/s, speculative 35.11 tok/s, speedup 0.993, accept 0.667, identical true
  average speedup: 1.011

max_draft=4 adaptive min_verified=1 accept_threshold=0.70:
  sky:        target 33.63 tok/s, speculative 35.14 tok/s, speedup 1.045, accept 0.500, identical true
  math:       target 32.49 tok/s, speculative 32.20 tok/s, speedup 0.991, accept 0.857, identical true
  technical:  target 35.29 tok/s, speculative 35.04 tok/s, speedup 0.993, accept 0.667, identical true
  average speedup: 1.010

decision:
  Native NEXTN is enabled in the stateful default path with adaptive fallback.
  Larger `max_draft` policies still need measured positive speed before they
  should become a recommended serving policy.

next real speed boundary:
  the reference runtime's proposer loop is not enough. The missing piece is a state-safe target
  block verifier with commit-able per-token GDN/conv/KV trajectory, so accepted
  speculative tokens reduce target work instead of only adding MTP proposal
  overhead.
```

Latest dmc8 fast raw block parity gate after adding gdn_recurrent_ab_scan:

```text
LANGBURST_FAST_RAW_BLOCK=1
input_tokens=88
argmax_match=true
max_abs_logit_diff=0.0
mean_abs_logit_diff=0.0
pos_match=true
kv_len_match=true
gdn_state_max_abs_diff=0.017456 to 0.033325
conv_state_max_abs_diff=0.109375 to 0.131836
attention_kv_max_abs_diff=1.092529 to 2.088379
continuation_argmax_match=true
continuation_max_abs_logit_diff=0.0
```

Interpretation: fast raw block still needs full state trajectory parity before
it can be a default path.  The fused `gdn_recurrent_ab_scan` reduced one known
source of mismatch but did not complete the fix.

Removed prompt-lookup check that motivated the native-MTP-only policy:

```text
normal explanation prompt, 96 generated tokens:
  speculative off: 30.09 tok/s
  speculative on:  30.11 tok/s
  output: identical

repeated pattern prompt, 72 generated tokens:
  speculative off: 20.81 tok/s
  speculative on:  20.78 tok/s
  output: identical
```

Conclusion: prompt-lookup speculation was correct but not useful enough for
this Qwen3.6 runtime, so it was removed instead of kept as a parallel path.

Speculative research basis from current literature:

```text
EAGLE-3 is the highest-priority learned proposer family.
MTP is attractive when the target checkpoint has native heads.
N-gram/prompt lookup is not part of langburst's Qwen3.6 runtime path.
All proposers must share the verifier/commit interface.
```

## 2026-06-16 Runtime Plan and Fast-Block Audit

Architecture cleanup:

- Added `RuntimeCapabilities` and `RuntimePlan`.
- `RuntimeFeatures` now represents requested behavior only.
- `RuntimePlan` resolves requested features against adapter-declared
  capabilities before runtime/server/bench code uses them.
- `/v1/langburst/features` now reports the execution plan rather than only raw
  requested defaults.

Earlier audit conclusion before the full state/continuation parity gate:

```text
fast raw block:
  keep as a research/internal path.
  do not promote from final-logit parity alone.
  it needs state trajectory + continuation parity.

batched Marlin:
  keep as a layer-level [T, D] primitive candidate.
  do not use as an lm_head/verifier shortcut.
  enable larger M only after continuation-state parity.

current default:
  Q4 Marlin target-only.
  public block prefill API with exact state updates.
```

Regression shield added:

```text
langburst-correctness --require-block-prefill-parity
  now compares continuation logits after prefill, not only final prefill logits.
```

Accepted micro-optimization:

```text
Removed unnecessary Marlin output-buffer zeroing.
Correctness gate passed.
Same include-prefill profile row:
  before: 2.464s, 6.49 generated tok/s
  after:  2.399s, 6.67 generated tok/s
  gain:   about 2.7%
```

Rejected/default-off during audit:

```text
fast raw block with batched Marlin:
  final prompt logits matched on a short parity row,
  but long continuation recall failed on filler32 UUID.
  not default.
```

Latest accepted-path profile, same include-prefill prompt:

```text
generated=16 elapsed_s=2.435 tok_s=6.57
mlp_gate_up:      36.95%
mlp_down:         16.86%
gdn_qkvz:         12.06%
gdn_out:           8.97%
rmsnorm:           5.60%
gdn_norm_gate:     4.17%
attn_qkv:          3.55%
gdn_recurrent:     3.11%
attn_o:            3.00%
gdn_conv:          2.83%
attention_decode:  1.54%
lm_head:           1.16%
```

Interpretation:

```text
single-request tok/s:
  dominated by layer projections, especially MLP and GDN projections.

multi-user throughput:
  next architecture layer is request scheduling + paged/ring KV resource
  management, not more server-side branching.
```

This document records measured speed changes for the LangBurst Qwen3.6-27B
runtime on `ml-dmc8` RTX 4080 16GB. Only end-to-end generated-token speed is
counted unless explicitly marked as profiler data.

## Measurement Contract

Canonical target path:

```text
HF model:        /home/user/models/Qwen3.6-27B
LangBurst model: /home/user/models/Qwen3.6-27B-qb4-marlin-fused
Runtime:         target-only Q4 Marlin, GPU-resident weights
Device:          ml-dmc8 RTX 4080 16GB
Decoding:        greedy temperature=0
```

Important interpretation rules:

- Compare only rows with the same prompt class and token count.
- `tok/s` is emitted tokens divided by full prefill+decode wall time reported by
  `langburst.generate --stats`.
- Output sanity was checked on every accepted benchmark row. UTF-8 and natural
  English generation remained valid.
- Historical rows in this section predate the current default-on adaptive
  Native MTP/NEXTN path. Treat them as
  pure target-model speed unless a row explicitly says speculative decoding is
  enabled.

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

- Added non-streaming greedy GPU token path in `langburst.generate`.
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

- Added `LANGBURST_MARLIN_MAX_PAR`.
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
- Added block prefill as the default runtime switch.
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

Correctness status update:

```text
The original fast block internals were not accepted:
  1. batched Marlin M>1 projection was not deterministic on dmc8;
  2. the vectorized block path did not match token-loop logits.

Block prefill remains enabled as the public/default API, but it now uses exact
single-token decode semantics internally. LowBitMarlinTensor.gemm also routes
M>1 calls through repeated stable M=1 Marlin calls until a batched kernel passes
repeated logits parity.

Accepted dmc8 gate:

langburst-correctness --require-block-prefill-parity
  ok=true
  input_tokens=270
  token_loop_argmax=41874
  block_prefill_argmax=41874
  max_abs_logit_diff=0.0
  exact recall filler0/filler8 passed
```

Current accepted speed after the correctness fix:

```text
input_tokens=20, generated=32:
  elapsed_s=1.499
  generated_tok_s_full=21.35
  total_tok_s=34.70

input_tokens=76, generated=32:
  elapsed_s=2.653
  generated_tok_s_full=12.06
  total_tok_s=40.71
```

Historical rejected/default-off acceleration candidates checked after that fix:

```text
native MTP1 adaptive suite:
  sky:       speedup 1.146, keep true
  math:      speedup 0.929, keep false
  technical: speedup 0.956, keep false
  avg_speedup 1.010, all_identical true
Historical decision: keep that earlier implementation as research CLI only.
This is superseded by the current default-on native MTP/NEXTN runtime path with
adaptive fallback.

CUDA Graph static audit:
  graph_ready=false
  blockers: Python pos/kv_len counters, Python ring-KV logical view, Python state mutation.
Decision: not default.

FlashInfer:
  not installed in langburst.
Decision: not default.

FP8:
  torch FP8 dtypes available, but no KV/GDN parity path yet.
Decision: not default.
```

Accepted stability change:

```text
RuntimeEngine now reuses a single pooled DecodeState per feature contract for
server completion paths. This reduces repeated 8192-window state allocation and
fragmentation risk without changing logits.
```

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

LangBurst now has a single research probe:

```bash
langburst-qwen-nextn-bench native-mtp1 --steps 16
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

Historical decision at that point: MTP was not enabled in the default champion
server path because the gain was prompt-dependent. Current stateful runtime
keeps native MTP/NEXTN enabled with adaptive fallback. The acceptance and speed
criteria remain:

```text
candidate tokens must match target greedy tokens exactly before state commit
end-to-end tok/s must beat target-only by at least 3%
```

Only after that can MTP speedups be counted as quality-preserving.

## CUDA Graph Decode1 Audit

Phase 2 is now gated by an explicit audit instead of a partial graph path:

```bash
langburst-qwen-graph-audit --static
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

LangBurst now has a dedicated profiling entrypoint:

```bash
langburst-qwen-profile \
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

The old OpenWebUI row with 4962 input tokens in 245.003s is obsolete as a speed
target. Current correctness baseline is block prefill with exact decode
semantics.

Current remaining long-context issue:

```text
experimental batched-Marlin block prefill measured about 1.1K tokens/s on 2K-6K prompts.
decode-only after 6.6K live KV measured about 40 tok/s.
these rows are not default-serving claims after the M>1 Marlin correctness fix;
fresh speed rows are required.
```

## 2026-06-16 Runtime Loop Cleanup

Change:

```text
generate_ids / generate_ids_greedy_gpu:
  stop computing the unused next-token logits after the last emitted token.

profile manual decode loop:
  same fix, so profiler call counts match emitted-token semantics.

RuntimeEngine:
  cache forward_block capability inspection at engine construction instead of
  calling inspect.signature during every prefill.
```

Correctness:

```text
local targeted:
  17 passed in 0.73s

dmc8 targeted:
  17 passed in 1.01s
```

dmc8 Q4 Marlin target-only actual generation:

```text
prompt:
  "Write a concise technical note about quantized LLM inference."

settings:
  LANGBURST_LOWBIT_ROWS_PER_CTA=8
  runtime_profile=stateful
  weight_device=cuda
  recent_window=256
  max_new_tokens=256
  temperature=0

result:
  generated=256
  elapsed=7.243s
  tok/s=35.34
  output sanity: normal English technical note
```

Short decode bottleneck profile after the loop cleanup:

```text
prompt_tokens=24 generated=64 max_new_tokens=64 include_prefill=False
elapsed_s=2.702 tok_s=23.68

category            calls   total_ms   avg_us   pct_measured
mlp_gate_up          4032    658.052   163.21   36.12
mlp_down             4032    302.112    74.93   16.58
gdn_qkvz             3024    214.607    70.97   11.78
gdn_out              3024    156.594    51.78    8.60
rmsnorm              8127     99.007    12.18    5.43
gdn_norm_gate        3024     71.576    23.67    3.93
attn_qkv             1008     63.534    63.03    3.49
lm_head                63     62.472   991.63    3.43
gdn_recurrent        3024     55.217    18.26    3.03
attn_o               1008     53.420    53.00    2.93
gdn_conv             3024     42.994    14.22    2.36
attention_decode     1008     41.859    41.53    2.30
```

Interpretation:

```text
The cleanup removes pure wasted work and improves short-generation fairness.
For long fixed-length generations the speed impact is bounded to roughly one
target forward per request, so the 256-token tok/s stays near the previous
champion range.

The next material optimization is still not lm_head or sampling. The largest
wall is MLP/GDN projection work, especially mlp_gate_up + mlp_down.
```

## 2026-06-16 MLP Activation Fusion And MTP Gate Recheck

Change:

```text
Added langburst_cuda.silu_mul(gate, up):
  fused FP16 SiLU(gate) * up for MLP activation.

Qwen36MLP:
  uses silu_mul instead of torch F.silu(gate) * up when CUDA FP16 is available.

SpeculativeBenchmarkResult.keep:
  now requires identical output, speedup > 1.03, and accept_rate > 0.
  This prevents noisy adaptive fallback rows from being marked as keep.
```

Build / correctness:

```text
dmc8 build:
  CUDA 13 nvcc from nvidia/cu13 package was required.
  System /usr/bin/nvcc is CUDA 12.0 and mismatches torch CUDA 13.0.

dmc8 CUDA targeted:
  tests/test_v05_cuda_kernels.py: 5 passed in 1.21s

local CPU targeted:
  18 passed in 0.80s
```

dmc8 actual generation after `silu_mul`:

```text
same 256-token technical-note prompt:
  before: 35.34 tok/s
  after:  35.54 tok/s
  gain:   about 0.6%
  output sanity: normal English technical note
```

Decode profile after `silu_mul`:

```text
prompt_tokens=24 generated=64 max_new_tokens=64 include_prefill=False
elapsed_s=2.658 tok_s=24.08

category            calls   total_ms   avg_us   pct_measured
mlp_gate_up          4032    658.157   163.23   36.55
mlp_down             4032    302.242    74.96   16.78
gdn_qkvz             3024    214.339    70.88   11.90
gdn_out              3024    156.026    51.60    8.66
rmsnorm              8127     97.799    12.03    5.43
gdn_norm_gate        3024     71.745    23.73    3.98
attn_qkv             1008     63.557    63.05    3.53
lm_head                63     62.384   990.22    3.46
gdn_recurrent        3024     55.072    18.21    3.06
attn_o               1008     53.522    53.10    2.97
attention_decode     1008     41.926    41.59    2.33
gdn_conv             3024     23.719     7.84    1.32
```

Interpretation:

```text
silu_mul is a small positive cleanup, not a main 100 tok/s lever.
The projection wall remains unchanged: mlp_gate_up + mlp_down + gdn_qkvz + gdn_out.
```

Historical Native MTP1 adaptive recheck before the external serving engine-shift, Marlin aliasing,
state-copy, and GDN block-scan fixes:

```text
bench-suite-mtp1 --adaptive --min-verified 2 --accept-threshold 0.8

sky:
  target 21.10 tok/s, speculative 22.97 tok/s, speedup 1.089,
  accept_rate 0.000, identical true

math:
  target 19.79 tok/s, speculative 18.58 tok/s, speedup 0.939,
  accept_rate 0.000, identical true

technical:
  target 30.64 tok/s, speculative 29.07 tok/s, speedup 0.949,
  accept_rate 0.000, identical true

summary:
  avg_speedup 0.992
```

Historical decision at that point:

```text
Do not enable that earlier MTP1 path in serving.
This result has been superseded by the current Native MTP1 default described at
the top of this file and in SPECULATIVE_RESEARCH.md.
```

## 2026-06-16 Batched Marlin T=4 Gate

Context:

```text
fast raw block and batched Marlin are still required for a real verifier, but
they must be gated by state trajectory parity, not final logits only.

The current accepted public forward_block still uses exact sequential state
updates. This section only changes the Marlin direct batch limit used by
accepted block prefill attention/MLP projections.
```

Marlin micro correctness:

```text
lowbit_marlin_gemm direct M=1:  rel=0.000350 pass
lowbit_marlin_gemm direct M=2:  rel=0.000334 pass
lowbit_marlin_gemm direct M=4:  rel=0.000347 pass
lowbit_marlin_gemm direct M=8:  rel=0.000342 pass
lowbit_marlin_gemm direct M=16: rel=0.000344 pass
```

Model-level strengthened parity:

```text
LANGBURST_MARLIN_DIRECT_MAX_BATCH=4
prefill_chunk_size=4
input_tokens=88

ok=true
argmax_match=true
max_abs_logit_diff=0.0
pos_match=true
kv_len_match=true
gdn_state_max_abs_diff=0.0
conv_state_max_abs_diff=0.0
attention_kv_max_abs_diff=0.0
continuation_argmax_match=true
continuation_max_abs_logit_diff=0.0
recall filler0 passed
```

Long-ish prompt include-prefill comparison:

```text
prompt_tokens=3618
generated=64
prefill_chunk_size=4

direct_max_batch=1:
  elapsed=92.818s
  tok/s=0.69

direct_max_batch=4:
  elapsed=91.844s
  tok/s=0.70

gain:
  about 1.1% wall-clock on this long prompt
```

Decision:

```text
Set LANGBURST_MARLIN_DIRECT_MAX_BATCH default to 4.
It is a small but state-parity-positive improvement and remains overridable by
environment variable if a future checkpoint exposes a regression.
```

## 2026-06-16 Serving Resource Boundary Cleanup

Accepted structural changes:

```text
EngineResourcePolicy.max_state_pool_size:
  host-level cap for retained DecodeState objects.

RuntimeEngine.state_pool_summary:
  exposes pooled-state residency without endpoint-specific introspection.

EngineManager.health:
  one operational status payload for model states, scheduler counters, CUDA
  resource state, and pooled-state residency.

EngineManager.validate_generation_request:
  rejects over-limit prompt and generation requests before DecodeState
  allocation or CUDA work starts.

server OOM handling:
  CUDA OOM during generation clears the affected model runtime pool and returns
  503 instead of leaving stale pooled state resident.
```

Performance interpretation:

```text
This is not a tok/s optimization and does not change logits math.
It is kept because it removes a real serving failure mode on 16GB GPUs:
oversized requests, request-scoped state allocation, and stale pooled state
after OOM.
```

Targeted verification:

```text
pytest -q langburst/tests/test_adapter_runtime_cpu.py \
          langburst/tests/test_engine_manager_cpu.py \
          langburst/tests/test_server_config_cpu.py \
          langburst/tests/test_scheduler_cpu.py

21 passed in 1.00s
```

Default decision:

```text
Keep state pooling off by default with max_state_pool_size=0; enable it only
for explicit arena/state-pool validation runs until CUDA arena parity is fixed.
Keep server default recent_window=16384 via langburst.core.defaults.
Keep max_prompt_tokens=16384 and max_generation_tokens=1024 as admission
defaults for 16K-context serving; larger context should be an explicit operator choice.
Keep request admission conservative.
Do not claim continuous batching until a real forward_batch/paged-KV executor
exists and shows measured throughput benefit.
```

## 2026-06-16 Native NEXTN Auto-Adopt Sweep

Structural changes:

```text
SpeculativeDecodePolicy.max_draft:
  raised from 4 to 10 because QwenNativeMTP1 can reuse the native MTP layer for
  NEXTN-style longer draft proposals.

python -m langburst.research.qwen_nextn_bench bench-auto-nextn:
  single command for draft/context/batch sweeps.
  Writes JSONL rows plus a champion JSON.

LANGBURST_MTP_AUTOTUNE_JSON:
  runtime reads this file only when keep=true.
  explicit environment variables still override the file.
```

Smoke command:

```bash
python -m langburst.research.qwen_nextn_bench bench-auto-nextn \
  --context-values 256 \
  --batch-values 1 \
  --draft-values 4,6,8,10 \
  --max-new-tokens 16 \
  --recent-window 256 \
  --output-json runs/langburst_nextn_autotune_draft_4_10_smoke.json \
  --jsonl runs/langburst_nextn_autotune_draft_4_10_smoke.jsonl
```

dmc8 decode-only result:

```text
target-only: 40.53 tok/s
draft=4:    17.38 tok/s, speedup 0.429, accept_rate 0.273, identity true
draft=6:    13.57 tok/s, speedup 0.335, accept_rate 0.179, identity true
draft=8:    11.62 tok/s, speedup 0.287, accept_rate 0.139, identity true
draft=10:   10.28 tok/s, speedup 0.254, accept_rate 0.116, identity true
champion:   none
```

Long-context smoke:

```text
context=65K, batch=1, draft=4, max_new_tokens=8:
  timed out at 120s before prefill completed.
```

Decision:

```text
Do not auto-enable draft 4/6/8/10.
Keep auto-adopt gated by champion keep=true only.
Use decode-only measurements for speculative policy decisions; prefill time is
recorded separately.
Next speed work is not larger drafts. It is long prefill optimization and true
batch=2 NEXTN generation through the shared batch verifier.
```

## 2026-06-16 Batch-State GDN Kernel Parity Fix

Root cause:

```text
The batch-state conv kernel was parity-safe and faster.
The batch-state GDN recurrent kernel updated state correctly but produced wrong
output for the real Qwen shape:
  kv_heads=16
  v_heads=48

The old small CUDA test used v_heads=8 and did not cover this shape.
```

Fix:

```text
gdn_recurrent_ab_batch:
  keep the shared-memory state update path
  compute output from the updated state using the original q row directly

canonical KV sync:
  paged attention now also writes the canonical arena KV view so snapshot,
  fork, fallback, and parity paths do not observe stale KV.

LANGBURST_BATCH_STATE_KERNELS:
  default changed to ON after deterministic batch runner parity passed.
  Split debug flags remain available:
    LANGBURST_BATCH_CONV_KERNELS
    LANGBURST_BATCH_GDN_KERNELS
```

dmc8 deterministic batch=2 runner:

```text
base eager arena path:       17.46 tok/s
conv batch kernel only:      20.11 tok/s, token-identical
GDN batch kernel only:       20.11 tok/s, token-identical
conv + GDN batch kernels:    20.35 tok/s, token-identical
```

Decision:

```text
Promote batch-state CUDA kernels to the default decode batch path.
Next bottleneck is P4 query_len > 1 / batch prefill and P8 verify-N on the same
batch model path.
```

## 2026-06-16 Serving Metrics and Multi-User Throughput Split

external serving engine comparison point:

```text
external serving engine tracks request scheduled_ts, first_token_ts, last_token_ts, TTFT, ITL,
scheduled token counts, and prefill/decode work separately. LangBurst numbers
were previously mixed between:
  - direct decode-only target path
  - worker end-to-end serving path
  - multi-user aggregate decode window
```

Fix:

```text
Added langburst-bench-serving.
BatchGenerationHandle now records:
  queue_wait_s
  ttft_s
  e2e_s
  decode_s
  mean_itl_s
  per-request e2e/decode tok/s

The benchmark reports both:
  aggregate_output_tok_s: includes prompt prefill and full request latency
  aggregate_decode_tok_s: output tokens divided by shared decode wall time
```

Scheduler bug fixed:

```text
Before:
  _drain_pending used a wall-clock deadline even when requests were already
  queued. First state allocation could consume max_wait_s and split a ready
  batch. In batch=4, request 4 waited about 4.35s before admission.

After:
  already-queued requests are admitted up to max_num_requests before a model
  step, then max_wait_s is only used for extra late arrivals.
```

dmc8, Qwen3.6-27B Q4 Marlin, prompt=256 tokens, max_new=128:

```text
recent_window=256, paged KV ON:
  batch=1: aggregate_output 11.97 tok/s, aggregate_decode 28.26 tok/s, TTFT 6.16s
  batch=2: aggregate_output 15.25 tok/s, aggregate_decode 51.92 tok/s, TTFT 11.86s
  batch=4: aggregate_output 17.85 tok/s, aggregate_decode 102.47 tok/s, TTFT 23.69s

recent_window=2048, paged KV ON:
  batch=1: aggregate_output 12.00 tok/s, aggregate_decode 28.25 tok/s, TTFT 6.14s
  batch=2: aggregate_output 15.28 tok/s, aggregate_decode 51.93 tok/s, TTFT 11.82s
  batch=4: aggregate_output 17.89 tok/s, aggregate_decode 102.82 tok/s, TTFT 23.65s
```

Paged KV comparison, same dmc8 setup:

```text
paged KV ON:
  batch=1 decode: 28.07 tok/s
  batch=4 decode: 102.32 tok/s

paged KV OFF:
  batch=1 decode: 29.57 tok/s
  batch=4 decode: 78.41 tok/s
```

Decision:

```text
Keep paged KV enabled for multi-user serving. It is slightly slower for
single-request decode, but materially faster at batch=4.

Current 16GB champion:
  max stable high-throughput batch = 4
  aggregate decode throughput ~= 102 tok/s

batch=6:
  runs but drops to 40.18 aggregate decode tok/s, likely memory pressure and/or
  less efficient high-B hot-path behavior.

batch=8:
  OOM with arena + paged KV on 16GB.
```

Why "40 tok/s became 20 tok/s":

```text
The 40 tok/s number was direct target-only decode after prefill, measured by
bench-auto-nextn:
  context=256, max_new=128 -> target-only 39.67 tok/s, prefill_s=5.924

The 20-30 tok/s numbers are worker serving decode per request. They include
continuous-batching state arena/paged path and request handling. End-to-end
serving numbers are lower because prompt prefill dominates TTFT.

For multi-user serving the right throughput number is aggregate_decode_tok_s:
  batch=4 reaches about 102 tok/s after prefill.
```

## 2026-06-16 TTFT / Prefill Fast Path

dmc8, Qwen3.6-27B Q4 Marlin, prompt=256, max_new=32, batch=4,
recent_window=256, prefill_chunk_size=64:

```text
paged KV ON, existing safe prefill:
  aggregate_output: 5.09 tok/s
  aggregate_decode: 104.47 tok/s
  mean TTFT: 23.91s

paged KV ON, timestep batch prefill through paged attention:
  aggregate_output: 12.04 tok/s
  aggregate_decode: 106.98 tok/s
  mean TTFT: 9.44s
  decision: rejected as default because deterministic continuation diverged.

paged KV ON, guarded default:
  output parity with existing paged path: pass on prompt=128, requests=2,
  max_new=16.

paged KV OFF, timestep batch prefill:
  OFF: aggregate_output 5.19 tok/s, aggregate_decode 79.45 tok/s, TTFT 23.06s
  ON:  aggregate_output 9.02 tok/s, aggregate_decode 80.31 tok/s, TTFT 12.59s
  deterministic output parity: pass on prompt=128, requests=2, max_new=16.
```

Root cause:

```text
Timestep batch prefill is safe against the canonical/no-paged path, but using
paged KV during prefill changes continuation. The shipped guard only enables
the timestep batch prefill route when the batch plan has no block table or slot
mapping. Paged serving still uses the existing safe block prefill and publishes
KV pages after the chunk.
```

Next TTFT target:

```text
Paged serving TTFT should be attacked through prefix/state cache and paged
prefill parity, not by enabling the current paged timestep prefill route.
```
