# Native MTP Speculative Decoding

LangBurst uses Qwen3.6 Native MTP as the only built-in speculative decoding
path.  DFlash and prompt/ngram draft paths are not part of the runtime.

## Contract

```text
draft proposer:
  may propose future tokens
  must not mutate live target state

target verifier:
  verifies candidates with the LangBurst target path
  commits only accepted target state
  never commits rejected candidate state
```

For greedy decoding, speculative output must be byte-identical to target greedy
one-token decoding.

## Current Implementation

```text
langburst.adapters.qwen36_mtp.QwenNativeMTP1
  loads mtp.* checkpoint tensors
  predicts draft tokens from raw target hidden + first greedy token
  can run a vLLM-style NEXTN loop by feeding each accepted draft token and the
  previous MTP hidden back through the checkpoint's single native MTP layer

langburst.adapters.qwen36_mtp.QwenNativeMTP1Proposer
  implements SpeculativeProposer
  returns DraftProposal(method="native_mtp1", tokens=[...])
  exposes propose_tensor(...) for the CUDA hot path to avoid candidate CPU sync
  exposes propose_tensors(...) for max_draft > 1 experiments

RuntimeEngine.generate_decode_result
  is the single decode-result entry point for greedy, sampling fallback, and
  native MTP/NEXTN
  returns ids plus decode stats instead of making callers reconstruct policy

RuntimeEngine.generate_native_nextn_result
  prefill returns logits + raw_hidden
  samples first token from target logits
  asks Native MTP/NEXTN for one or more candidates using the current target raw
  hidden and the first token id, matching vLLM's Qwen3NextMTP shift
  preserves target logits before running MTP because Marlin lm_head buffers are
  reused
  routes target verification through RuntimeEngine.verify_nextn_tokens
  lets NativeNextNVerifier own the accept/reject/rollback contract
  falls back to plain target decode when recent acceptance is too low

RuntimeEngine.verify_nextn_tokens
  is the single batch-verifier boundary for sampled-first + draft tokens
  prefers model.forward_verify_batch when available
  uses the shared DecodeBatchPlan shape for sampled-first + draft rows
  keeps model-specific verify implementations behind the runtime boundary

Qwen36Model.forward_verify_batch
  advances speculative rows by timestep through the normal forward_decode_batch
  layer path
  can use batch-state GDN kernels and paged attention when the active state
  arena and plan expose those buffers
```

Native MTP/NEXTN is implemented behind the shared proposer/verifier contract and
is enabled in the default stateful profile.  The adaptive gate preserves target
identity and falls back to plain target decode when acceptance is poor.  It can
still be disabled explicitly for target-only comparisons:

```bash
langburst-chat --speculative-decoding off ...
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
max_draft: 1..10 as an explicit benchmark axis
auto-adopt: only from a champion JSON with keep=true
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
  decision: default ON with adaptive fallback. Keep larger draft lengths gated
            by identity and measured speed-positive suites.
```

Latest vLLM-style NEXTN recheck:

```text
2026-06-16, 128-token suite, recent_window=2048:
  max_draft=2 average speedup: 1.011, identity true on all prompts
  max_draft=4 average speedup: 1.010, identity true on all prompts
```

Latest decode-only auto-adopt sweep:

```text
2026-06-16, dmc8, context=256, max_new_tokens=16:
  target-only: 40.53 tok/s
  max_draft=4:  17.38 tok/s, speedup 0.429, identity true
  max_draft=6:  13.57 tok/s, speedup 0.335, identity true
  max_draft=8:  11.62 tok/s, speedup 0.287, identity true
  max_draft=10: 10.28 tok/s, speedup 0.254, identity true
  champion: none

2026-06-16, dmc8, context=65K smoke:
  batch=1, max_draft=4 did not finish prefill under a 120s timeout.
```

This is not yet a large speculative win.  More draft tokens alone do not help
while the target verifier still advances the target model token by token.  The
next speed work must reduce verifier and proposer overhead without changing the
vLLM-compatible contract:

1. exact state-trajectory parity audit for the fused verify path on the real model
2. lower-overhead MTP lm_head candidate selection
3. verifier CUDA Graph only after shapes are static
4. long-context prefill optimization before 65K/100K NEXTN sweeps
5. true batch=2 NEXTN generation path before any batch=2 speculative claim
```
