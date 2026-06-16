# Native MTP Speculative Decoding

QwenBurst uses Qwen3.6 Native MTP as the only built-in speculative decoding
path.  DFlash and prompt/ngram draft paths are not part of the runtime.

## Contract

```text
draft proposer:
  may propose future tokens
  must not mutate live target state

target verifier:
  verifies candidates with the QwenBurst target path
  commits only accepted target state
  never commits rejected candidate state
```

For greedy decoding, speculative output must be byte-identical to target greedy
one-token decoding.

## Current Implementation

```text
qwenburst.qwen_mtp.QwenNativeMTP1
  loads mtp.* checkpoint tensors
  predicts draft tokens from raw target hidden + first greedy token
  can run a vLLM-style NEXTN loop by feeding each accepted draft token and the
  previous MTP hidden back through the checkpoint's single native MTP layer

qwenburst.qwen_mtp.QwenNativeMTP1Proposer
  implements SpeculativeProposer
  returns DraftProposal(method="native_mtp1", tokens=[...])
  exposes propose_tensor(...) for the CUDA hot path to avoid candidate CPU sync
  exposes propose_tensors(...) for max_draft > 1 experiments

RuntimeEngine.generate_ids_native_mtp1_speculative
  prefill returns logits + raw_hidden
  samples first token from target logits
  asks Native MTP/NEXTN for one or more candidates using the current target raw
  hidden and the first token id, matching vLLM's Qwen3NextMTP shift
  preserves target logits before running MTP because Marlin lm_head buffers are
  reused
  commits the first target token on the live state
  compares MTP candidates against target argmax tokens in order
  commits accepted candidates directly on the live target state
  commits the target-correct token directly on first rejection
  falls back to plain target decode when recent acceptance is too low
```

Native MTP1 is implemented behind the shared proposer/verifier contract, but it
is not enabled by default.  The current RTX 4080 / Q4 Marlin measurements show
identity safety after the verifier clone fix, but not a consistent speed win.
Enable it explicitly only for experiments:

```bash
qwenburst-chat --speculative-decoding on ...
```

## Why Only Native MTP

Qwen3.6 includes native MTP weights.  That makes MTP the best first proposer for
a 27B single-GPU low-VRAM engine because it avoids loading an external draft
model.  EAGLE and Medusa remain future proposer families, but they must plug
into the same `SpeculativeProposer` interface and target verifier.

## Gates

Native MTP serving is considered valid only when:

```text
greedy output identity: target-only ids == native-MTP ids
state safety: draft proposal does not mutate live DecodeState
rollback safety: rejected candidate state is not committed
speed: speculative tok/s beats target-only on the measured prompt suite
```

Current implementation prioritizes target identity and state safety.  The best
tested experimental gate is:

```text
min_verified: 1
accept_threshold: 0.70
max_draft: 1, 2, or 4 as an explicit benchmark axis
```

Latest dmc8 Q4 Marlin fused result:

```text
2026-06-16, 128-token suite, recent_window=2048, after removing candidate CPU
sync and accept-path fork/copy:
  identity: true on all prompts
  sky:        accept_rate 0.909, speedup 1.048
  math:       accept_rate 0.800, speedup 0.994
  technical:  accept_rate 0.667, speedup 0.993
  average speedup: 1.012
  best tested gate: min_verified=1, accept_threshold=0.70, max_rejections=1
  decision: default OFF. Keep implementation for research; do not use it as
            the serving default until the suite shows a larger repeatable win.
```

Latest vLLM-style NEXTN recheck:

```text
2026-06-16, 128-token suite, recent_window=2048:
  max_draft=2 average speedup: 1.011, identity true on all prompts
  max_draft=4 average speedup: 1.010, identity true on all prompts
```

This is not yet a large speculative win.  More draft tokens alone do not help
while the target verifier still advances the target model token by token.  The
next speed work must reduce verifier and proposer overhead without changing the
vLLM-compatible contract:

1. exact fast raw block verifier with state trajectory parity
2. target block verifier that commits only the accepted GDN/conv/KV trajectory
3. lower-overhead MTP lm_head candidate selection
4. verifier CUDA Graph only after shapes are static
5. larger prompt suite before increasing default draft length
```
