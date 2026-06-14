# SaneFlow Research Actions

This file turns the current research direction into concrete actions for the
non-Transformer, non-Mamba-clone line.

## Current Direction

```text
local 16K byte-BPE
causal depthwise local conv
gated multi-timescale recurrent state
v2 erase/write decoupled state mixer
SwiGLU
tied LM head
```

It does not use the `transformers` library, full softmax attention, GPT decoder
blocks, or Mamba kernels.

## Immediate Experiments

1. `saneflow_fineweb_edu_base_v1`
   - data: FineWeb-Edu `sample-10BT` bounded subset
   - tokenizer: local 16K byte-BPE trained on the same subset
   - model: d_model 384, layers 8, heads 6, d_ff 1152, seq 256
   - success: readable generic continuation and simple explanation prompts

2. If the base passes:
   - add Cosmopedia v2 / filtered explanation prose
   - keep plain QA at 3 percent or below
   - do not add ChatML yet

3. If the base fails:
   - compare `v2` state mixer with local-conv-only
   - inspect tokenization fertility and generated token distribution
   - reduce data noise before changing architecture

## Separate Architecture Research Track

The active `saneflow_fineweb_edu_base_v1` run is preserved. Higher-risk
architecture experiments run in a separate track and are merged into the main
track only after measured benefit.

Primary candidates:

```text
baseline:
  state_mixer_version=v2
  syntax_mix_version=v1

stable_vector:
  state_mixer_version=v2
  state_clip=8.0
  state_zoneout=0.02

delta_matrix:
  state_mixer_version=delta_matrix
  state_clip=8.0

delta_matrix_syntax_v2:
  state_mixer_version=delta_matrix
  state_clip=8.0
  syntax_mix_version=v2
  syntax_kernels=3,7,15
```

Promotion requirements:

- lower validation loss than `state_v2` under the same tokenizer/data/steps
- no cache/full argmax parity regression
- no speed collapse in `saneflow_quality_gate.py`
- no worse repeated-ngram/empty/invalid-output gate
- readable prompt samples improve or at least do not regress

Current implementation status:

- `delta_matrix` exists behind `state_mixer_version=delta_matrix`.
- `state_clip` and `state_zoneout` exist as config/CLI options.
- `scripts/saneflow_ablation_grid.py` includes `state_v2_stable`,
  `delta_matrix`, and `delta_matrix_syntax_v2`.

Neurova-R-specific implementation, experiments, failed attempts, and promotion
boundaries are tracked in `docs/neurova_r_research_report.md`. Treat that line
as a separate high-risk research track until it beats the current SaneFlow
baseline on loss, generation quality, reasoning/copy gates, and speed.
- These are research candidates, not the main line.

## Deferred

- Bytepatch: later front-end or alignment experiment.
- Slot memory: later, only with ablation proof.
- MTP: after next-token generation quality is stable.
- Chunk/global summaries: finite-state summaries, not Transformer attention.

## Hard Promotion Gate

No checkpoint is promoted unless it can answer:

```text
Hi. Who are you?
Explain what a computer is in simple words:
What is the capital of France?
Write one sentence about the moon:
```

with readable English and no repeated template collapse.

## HERA Proposal Review

Reference: `docs/HERA_LM_ARCHITECTURE_PROPOSAL.md`.

HERA is directionally compatible with SaneFlow: both avoid full token-token
attention, separate local syntax from recurrent memory, and prefer explicit
erase/write control. But applying the full HERA stack now would repeat the LUMA
failure mode: too much memory machinery before the base language model is
stable.

### Apply Soon

1. Multi-kernel SyntaxMix
   - HERA proposes parallel causal depthwise kernels such as 3/7/15.
   - SaneFlow currently uses one causal depthwise kernel.
   - This is the safest HERA-derived upgrade because it targets grammar, local
     phrase shape, and repetition control without adding attention or slots.
   - Test behind `syntax_mix_version=v2` against the current single-kernel
     baseline.

2. Factorized embedding/head
   - Useful if SaneFlow moves beyond the current 16K BPE vocabulary.
   - Do not add it to the current 16K baseline unless embeddings become a clear
     bottleneck.
   - Add before any 64K+ tokenizer experiment.

3. DeltaFlow-style matrix memory as an ablation
   - SaneFlow v2 has vector recurrent state with independent erase/write.
   - HERA's DeltaFlow uses grouped matrix memory and outer-product updates.
   - This could improve key/value binding, but naive PyTorch may be slower.
   - Test only after the current vector-state model passes stronger language
     gates. Keep it behind `state_mixer_version=delta_matrix`.

### Defer

1. FactBoard slots
   - Better specified than LUMA slots because it includes RMSNorm, clipping,
     weak memory scale, and chunk-boundary writes.
   - Still not needed for the first speaking model.
   - Add only after explanation/chat gates pass, and require ablation separation:
     `normal > no_slots` and `normal > random_slot_keys`.

2. Memory head / copy head
   - HERA correctly says memory logits must be tiny or off at first.
   - Keep them off until memory ablations prove value.

3. Router over syntax/delta/slot paths
   - Useful only once all paths exist and are individually proven.
   - Dangerous now because there is no proven slot path yet.

4. Bytepatch alignment
   - Consistent with the existing plan.
   - Defer until the BPE route speaks reliably.

### Reject For Current SaneFlow

1. Qwen tokenizer as default
   - HERA suggests Qwen/Length-MAX tokenizers, but current SaneFlow is
     intentionally self-contained with local 16K byte-BPE.
   - Qwen's large vocabulary would force factorized embeddings and complicate
     the clean baseline.

2. Teacher distillation as Stage 1
   - HERA recommends teacher distillation for cheap sentence quality.
   - The current SaneFlow goal is from-scratch architecture validation.
   - Teacher data may be used later for SFT/MCQ skill, not for judging base
     architecture viability.

3. Korean QA during speech-prior stage
   - The active gate is English sentence formation first.
   - Add Korean after English continuation and explanation behavior are stable.

### Revised Experiment Order

```text
v2 current:
  single-kernel SyntaxMix
  vector recurrent state
  clean base data

v3 safe architecture:
  multi-kernel SyntaxMix
  same vector recurrent state
  same tokenizer and data

v4 capacity/efficiency:
  optional factorized embedding/head if vocab grows
  larger d_model/layers only after v3 gate

v5 memory ablation:
  DeltaFlow matrix memory candidate
  no FactBoard slots yet

v6 memory proof:
  FactBoard slots + weak memory head
  only after chat/explanation gate
```

The strongest immediate HERA-derived action is therefore not FactBoard or
teacher distillation. It is a small, measurable SyntaxMix upgrade that improves
local language modeling while preserving the clean SaneFlow baseline.

### Decode Policy

Default interactive and one-shot generation uses recurrent cache decode, not
full-prefix recomputation.

Implementation:

- `SaneFlowLM.forward_step()`
- `SaneFlowBlock.step()`
- `GatedStateMixer.step()`
- `CausalDepthwiseConv.step()`
- `MultiKernelSyntaxMix.step()`
- `scripts/saneflow_generate.py --decode cache`
- `scripts/saneflow_chat.py --decode cache`
- `scripts/saneflow_eval_prompts.py --decode cache`
- `./neurova.sh saneflow` passes `--decode ${NEUROVA_SANEFLOW_DECODE_MODE:-cache}`

Quality gate on `runs/saneflow_tinystories_v2/model.pt`:

```text
artifact: runs/saneflow_tinystories_v2/quality_gate_cache_v2.json
valid_loss: 3.00390625
cache average total speed: 196.85 tok/s
cache average decode-only speed: 225.59 tok/s
full-prefix short generation speed: 56.14 tok/s
cache/full argmax parity: 100%
cache/full max_abs logit diff: 0.0625
empty output rate: 0.0
invalid output rate: 0.0
EOS rate: 28.57%
max repeated 4-gram: 3
mean distinct-1: 0.6793
mean distinct-2: 0.9478
```

Interpretation:

- Cache decode is the canonical default for interactive SaneFlow.
- Full-prefix decode remains only as a diagnostic fallback.
- The current parity target is argmax equality, because bf16 full-scan and
  incremental recurrence can differ slightly at logit-value level while
  preserving next-token choice.
- Context probe generation speed must be read with prefill/decode separated;
  total speed falls on long prompts because prefill grows with context.

### SyntaxMix v2 Ablation Result

Implementation:

- `SaneFlowConfig.syntax_mix_version`
- `v1`: original single causal depthwise conv, default kernel 5
- `v2`: HERA-style multi-kernel causal depthwise conv, default kernels 3/7/15
- training CLI: `scripts/saneflow_train.py --syntax-mix-version v1|v2`

Fair short-run comparison on `ml-dmc8`:

```text
data: data/saneflow/tinystories/train.jsonl
valid: data/saneflow/tinystories/valid.jsonl
tokenizer: tokenizers/saneflow_tinystories_16k
steps: 600
batch: 16
seq_len: 128
d_model: 256
layers: 6
heads: 4
d_ff: 768
state_mixer_version: v2
```

Results:

```text
v1 single-kernel:
  run: runs/saneflow_ablation_syntax_v1_600
  params: 10,819,346
  valid_loss step 200: 4.0879
  valid_loss step 400: 3.6191
  valid_loss step 600: 3.5820
  prompt eval avg speed: 29.0 tok/s

v2 multi-kernel:
  run: runs/saneflow_ablation_syntax_v2_600
  params: 10,853,138
  valid_loss step 200: 4.1250
  valid_loss step 400: 3.6348
  valid_loss step 600: 3.5586
  prompt eval avg speed: 29.1 tok/s
```

Interpretation:

- v2 improved final short-run validation loss by about `0.0234`.
- v2 did not slow generation in this small test.
- Generation quality at 600 steps remains rough for both models; explanation
  prompts still fall into TinyStories-style continuation.
- Therefore v2 is a promising candidate, but not yet the default promoted
  architecture. It should be tested in the next larger run with the same
  promotion gate.
