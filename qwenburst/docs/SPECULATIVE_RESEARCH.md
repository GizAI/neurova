# Speculative Decoding Research Plan Without DFlash

QwenBurst no longer carries DFlash runtime code. Future acceleration must keep
the QwenBurst target model as the verifier and must preserve the target output
contract.

## Non-Negotiable Acceptance Contract

For greedy decoding:

```text
accepted speculative tokens == tokens produced by target greedy one-by-one
```

For sampled decoding:

```text
accept/reject must preserve the target distribution
```

Any proposed path that cannot prove one of these contracts stays outside the
champion runtime.

## Candidate Methods

### 1. Native MTP / NEXTN

Qwen3.6 checkpoints contain `mtp.*` weights, but the currently installed
Transformers `qwen3_next` implementation ignores them:

```text
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
```

This is still the most promising path because it can avoid a separate draft
model and may reuse target hidden state. Required work:

```text
1. Reverse/verify exact MTP computation from checkpoint tensors.
2. Produce candidate tokens from target hidden state.
3. Verify with qwenburst target on a forked DecodeState.
4. Commit only accepted prefix.
5. Prove greedy parity against one-token target decode.
```

Status: not enabled.

Current qwenburst implementation:

```bash
qwenburst-speculative native-mtp1 \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --steps 16
```

The probe follows the vLLM Qwen3Next MTP dataflow:

```text
norm(next-token embedding), norm(previous hidden)
→ concat [embedding, hidden]
→ mtp.fc
→ one MTP full-attention decoder layer
→ mtp.norm
→ target lm_head
```

The probe is intentionally not wired into serving.  It only becomes eligible
for runtime integration if:

```text
native_mtp1 accept_rate >= 0.55 on a prompt suite
target verifier output remains byte-identical to target-only greedy
accepted tokens reduce target passes enough to beat target-only tok/s
```

Latest dmc8 probe:

```text
native_mtp1 probe: accepted=15 / total=16, accept_rate=0.938, viable=true
```

Exact verifier benchmark with prefill included, after adding raw hidden returns
from `forward_block` and accepted-branch state promotion:

```text
bench-suite-mtp1 --adaptive --min-verified 2 --accept-threshold 0.8 --recent-window 64

sky 32:
  target-only: 19.38 tok/s
  adaptive MTP1: 24.11 tok/s
  speedup: 1.244
  identical: true
  keep: true

math 32:
  target-only: 18.43 tok/s
  adaptive MTP1: 18.55 tok/s
  speedup: 1.006
  identical: true
  keep: false

technical 64:
  target-only: 28.28 tok/s
  adaptive MTP1: 28.60 tok/s
  speedup: 1.011
  identical: true
  keep: false

summary:
  all_identical: true
  avg_speedup: 1.087
```

Conclusion: native MTP1 can produce real speedup after raw-hidden replay removal,
but the gain is prompt-dependent. It remains a research/tuning CLI only until a
larger prompt suite shows no regressions and a consistent speed gate win.

### 2. Medusa-Style Heads

Medusa adds multiple decoding heads to predict several future tokens and uses
tree verification. Medusa-1 can keep the backbone frozen, which is attractive
for preserving target quality.

Fit for QwenBurst:

```text
Pros:
  - no external draft model at runtime
  - small additional heads
  - exact target verification possible

Cons:
  - needs head training or self-distillation
  - tree verifier still needs block target path for real speed
```

Recommended experiment:

```text
train small heads on frozen hidden states
verify top candidate chain with qwenburst forward_block/fork
measure accepted tokens per target pass
```

### 3. EAGLE-Style Feature Drafter

EAGLE drafts at the second-to-top feature level and then verifies with the
target model. It is attractive because feature dynamics can be easier than
token dynamics.

Fit for QwenBurst:

```text
Pros:
  - potentially higher acceptance than token-only small heads
  - can preserve output distribution with verification

Cons:
  - requires reliable hidden-state taps
  - needs a trained feature extrapolator
  - qwenburst must expose exact hidden states cheaply
```

Recommended experiment:

```text
collect final/near-final hidden taps from qwenburst
train tiny feature predictor
project through lm_head for candidates
verify exact target prefix before commit
```

### 4. Lookahead / Jacobi Decoding

Lookahead decoding can be exact and does not require a draft model, but it
trades extra parallel target computation for fewer sequential steps.

Fit for QwenBurst:

```text
Pros:
  - no extra model weights
  - exact target-compatible direction

Cons:
  - qwenburst is batch-1 and Marlin projection dominated
  - recurrent GDN state makes parallel candidate state handling harder
  - likely needs true block verifier before it helps
```

Recommended status: research-only until `forward_block` is a real fast block
verifier.

## Current Ranking

```text
1. Native MTP/NEXTN using existing checkpoint tensors, only if exact verifier speed clears the gate
2. Medusa-style frozen-backbone heads
3. EAGLE-style feature drafter
4. Lookahead/Jacobi exact decoding
```

## First Implementation Target

Do not start with CUDA Graph or lm_head top-k. The current profiler says MLP
Marlin projection dominates. A useful speculative path must reduce target
passes per emitted token.

Minimal proof:

```text
prompt set: 50 deterministic greedy prompts
baseline: qwenburst one-token target decode
candidate path: MTP/Medusa/EAGLE candidates + exact target verify
pass condition:
  token sequence identical for all prompts
  DecodeState hash identical after generation
  accepted_tokens_per_target_pass >= 1.8
```

Only after this proof should CUDA/block verifier optimization be attached.

Current adaptive `native_mtp1` is promising but not yet a default runtime path.
The next useful work item is a larger prompt suite and reducing verifier fork
cost; CUDA Graph should attach only after the exact verifier path is consistently
speed-positive.
