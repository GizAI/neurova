# Mamba-3 Fixed-Memory Research Plan

## Current Rule

The production fast path must remain pure official Mamba-3 SISO recurrent decode with CUDA graph support.
Do not promote ordinary attention-hybrid layers into the fast serving trunk.

## Candidate Order

1. Pure SISO fast core, `d_state=64`.
   - Previous default: `mamba3-siso-fast-0.3b`.
   - Stable fallback.

2. Pure SISO fast core, `d_state=128`.
   - Current default: `mamba3-siso-fast-0.3b-ds128`.
   - Purpose: more recurrent state capacity without variable KV cache.
   - Promotion result: passed quality gate, recurrent parity, 16K context smoke, and improved batched recurrent throughput.

3. SISO-MultiLane.
   - Run several SISO lanes with fixed shape and merge using a gated mixer.
   - It must keep preallocated recurrent state and fixed tensor shapes.
   - Do not implement as general attention or growing KV cache.

4. Fixed local ring memory.
   - Window `W=64/128`.
   - Preallocated circular K/V tensors.
   - Optional use every N layers or N tokens, not every layer by default.

5. Fixed global memory slots.
   - Slots `S=16/32/64`.
   - State is compressed into fixed slots.
   - Shape must not depend on past sequence length.

## Promotion Gates

- Recurrent cache/full-forward argmax parity on the prompt suite.
- CUDA graph decode must work.
- Single-request latency must not regress materially against the current fast default.
- Batched aggregate throughput must remain in the thousands tok/s regime.
- Quality must improve on basic QA, copy/recall, JSON extraction, and long-context retrieval.
- If a candidate improves quality but breaks fast path, keep it as research-only.

## Current Evidence

- `mamba3-siso-fast-0.3b-ds128` reaches around 200 tok/s on repeated single-request fast QA.
- It improved batched recurrent CUDA graph throughput to about 8.3K aggregate tok/s at batch 64.
- State-compiled prefix reuse has correct first-token parity in the current benchmark, but it is not a speed win yet:
  - 4096-token prefix compile took about 5.6s,
  - repeated full prefill after warmup took about 0.045s,
  - compiled question replay took about 0.20s because the official recurrent path must consume question tokens one by one after `seqlen_offset > 0`.
  - Keep state compilation as a memory/session feature, not a promoted latency feature, until a graph-captured question replay path exists.
- 2026-06-14 retest on `neuromamba/runs/mamba3_current_training_chat/model.pt`:
  - `state-roundtrip` passed restored-cache/full-forward argmax parity.
  - prefix 512: first-token parity passed, but warm full prefill was about 0.053s and compiled question replay about 0.24s.
  - prefix 4096: first-token parity passed, but warm full prefill was about 0.044s and compiled question replay about 0.21s.
  - Direct `./neurova.sh` chat path stayed fast at about 236-242 tok/s on basic QA.
  - Verdict unchanged: state compile is useful for persisted recurrent state, not yet for lower latency.
- Batched CUDA graph recurrent decode reaches thousands tok/s aggregate on RTX 4080:
  - batch 16: about 2.6K tok/s,
  - batch 32: about 4.6K tok/s,
  - batch 64: about 6.6K tok/s.
- 16K input smoke passes, but long-context reasoning quality is not solved by context size alone.
