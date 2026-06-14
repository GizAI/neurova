# LUMA Final Structure

LUMA should not become a pile of tricks around a weak decoder. The final target
is a two-layer model with one hard contract:

```text
input bytes/tokens
  -> event encoder
  -> local mixer
  -> slot memory core
  -> evidence ledger
  -> language decoder
  -> answer with proof state
```

## Canonical Roles

- `slot memory core`: stores and updates working facts. It must pass ablation:
  `normal` clearly beats `no_slots` and `random_slot_keys`.
- `evidence ledger`: append-only raw byte spans plus pointers. It is not a
  replacement for slots; it is proof storage.
- `language decoder`: carries conversation quality and world/language prior.
  A small from-scratch decoder is acceptable for memory proof, but not for a
  high-quality chat model.
- `promotion gate`: no run becomes `runs/luma_current` unless both memory proof
  and chat sanity pass.

## Training Order

1. `memory_proof`
   - Objective: make slots useful.
   - Data: chunk-gap memory tasks only.
   - Pass condition: exact memory QA improves with slots and collapses when
     slots are removed or randomized.

2. `mixed_chat`
   - Objective: keep proven slots while adding language behavior.
   - Data: raw continuation + short chat SFT + answer-only memory + slot-proof
     stream.
   - Pass condition: chat probes are readable and memory ablation separation is
     preserved.

3. `chat_candidate`
   - Objective: candidate for `runs/luma_current`.
   - Required gates: readable hi/self-intro/simple QA, copy exact > 50%,
     recall > 60%, json_field > 70%, no repetition collapse, and ablation
     separation.

## Non-Goals

- Do not promote a run because loss is low.
- Do not hide slot failure with local attention, copy heads, or larger models.
- Do not call Qwen/bytepatch a true dual front-end until both front-ends share
  an aligned event space and a hidden-slot alignment loss.
- Do not call ledger memory solved until evidence is stored as raw byte spans
  and generation predicts or retrieves proof pointers.

## Current Verdict

`runs/luma_mixed_v7_slotproof_fast` is not a chat model and not a proven memory
model. Its training loss decreased, but online probes showed `no_slots` and
`random_slot_keys` had lower LM loss than `normal`, and exact memory QA remained
0%. It is preserved as a negative control, not promoted.
