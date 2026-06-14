# Neurova / SaneFlow Practical Retraining Plan

This is the current training plan for turning SaneFlow into a practical
conversation model. The current small runs are sanity runs, not final model
training.

## Target

```text
model line: SaneFlowLM
scale: about 100M parameters first
tokenizer: local 16K byte-level BPE
format after pretrain: standard ChatML
SFT loss: assistant-only ChatML labels
architecture constraint: no Transformer block, no Mamba dependency
promotion rule: never promote on loss alone
```

## Current Baseline Status

Prepared pretrain corpus:

```text
data/corpus/sources/fineweb_edu_sample10bt/train.jsonl
  rows: 39,608
  size: about 191 MiB

data/corpus/sources/fineweb_edu_sample10bt/valid.jsonl
  rows: 1,016
  size: about 4.8 MiB
```

Observed token exposure:

```text
dmc9 completed base: about 50.7M training tokens
dmc8 planned base: about 138M training tokens
```

Assessment:

```text
good for: architecture, optimizer, tokenizer, ChatML SFT sanity
not enough for: practical 100M from-scratch conversation quality
```

The current dmc8/dmc9 checkpoints should be treated as diagnostic checkpoints.
They can prove that the pipeline is healthy, but they are not enough language
pretraining for the final practical model.

## Canonical Data Layout

Pretrain sources:

```text
data/corpus/sources/<dataset_name>/train.jsonl
data/corpus/sources/<dataset_name>/valid.jsonl
```

Normalized SFT source adapters:

```text
data/corpus/sft_sources/*.jsonl
```

Canonical SFT mix:

```text
configs/saneflow_chatml_sft_recipe.json
data/corpus/mixes/saneflow_chatml_sft_train_v1.jsonl
data/corpus/mixes/saneflow_chatml_sft_valid_v1.jsonl
data/corpus/mixes/saneflow_chatml_sft_manifest_v1.json
```

Active builders:

```bash
python scripts/saneflow_prepare_chatml_sources.py
python scripts/saneflow_build_chatml_sft.py
```

No direct training from raw parquet/csv/json. Raw files are adapter inputs only.

## Stage 0: Sanity Completion

Purpose: confirm that the current code path works end to end.

Run:

```text
dmc8-base-100m
dmc8-chatml-sft-100m
dmc9-sparse-chatml-sft
```

Pass criteria:

```text
no NaN or repeated-token collapse
valid loss does not explode
ChatML stop token works
basic prompts produce readable sentences
```

Required probes:

```text
Hi. Who are you?
Explain what a computer is in simple English.
한국어로 짧게 자기소개해.
What is 12 + 7?
Write a Python function that adds two numbers.
```

If these fail badly, fix model/training/generation before scaling data.

Current R-version decision:

```text
run: dmc9-neurova-r-full
checkpoint: runs/saneflow_neurova_r_full_chunked/delta_sparse_thought_landmark_d384_l8_h32_b96_s256_tc64_eval50/model.pt
status: do not run SFT yet
reason: generation gate failed before SFT
observed behavior: ChatML prompts produce HTML/control-like fragments; raw prompts repeat shallow web phrases
next action: keep as architecture research only, do not spend SFT budget until a stronger base pretrain passes plain generation gates
```

R-version optimization plan:

```text
goal: isolate which R module helps before any SFT
budget: short 800-step pretrain ablation on the same FineWeb-Edu subset
launcher: scripts/saneflow_r_ablation_dmc9.sh
report: scripts/saneflow_r_ablation_report.py
champion launcher: scripts/saneflow_r_champion_dmc9.sh
output root: runs/saneflow_r_ablation/
```

R ablation profiles:

```text
A dmc9-r-a-delta-only          delta_matrix only
B dmc9-r-b-delta-sparse-attn   delta_matrix + sparse attention
C dmc9-r-c-delta-thought-late  delta_matrix + 4 thought slots on last 2 layers only
D dmc9-r-d-delta-landmark      delta_matrix + landmark memory
E dmc9-r-e-full-lite           B + C + D, with late thought slots
```

Decision rule:

```text
choose the champion by best valid loss plus readable generation probes
do not choose a module just because it is novel
if all ablations fail generation, scale practical base data before adding R modules
only the champion may receive a longer pretrain and later ChatML SFT
```

Current R champion:

```text
winner: D dmc9-r-d-delta-landmark
reason: best ablation valid loss among A-E
continuation profile: dmc9-r-champion-delta-landmark-long
continuation output: runs/saneflow_r_champion/d_delta_landmark_long
rule: continue pretrain only; do not SFT until generation gate passes
```

Important data distinction:

```text
Current R champion live pretrain:
  FineWeb-Edu sample only
  path: data/corpus/sources/fineweb_edu_sample10bt/train.jsonl
  role: architecture diagnostic continuation

Practical base pretrain:
  FineWeb-Edu + DCLM-Baseline + GneissWeb + FineWeb2-HQ EN/KO
  path: data/corpus/mixes/saneflow_practical_pretrain_v1
  role: real language-prior improvement before SFT
```

Do not describe the R champion run as using DCLM/Gneiss unless its command line
actually points at `saneflow_practical_pretrain_v1`.

## Stage 1: Practical Base Pretrain V1

Goal: build enough language prior for a real SFT to work.

Target exposure:

```text
minimum: 500M tokens
preferred first practical run: 1B tokens
stretch if stable: 2B tokens
```

Recommended mix:

```text
FineWeb-Edu / clean educational web       45-60%
DCLM-Baseline / DataComp-LM filtered web  20-30%
GneissWeb / IBM high-quality web          10-20%
FineWeb2-HQ English                        5-10%
FineWeb2-HQ Korean                         3-8%
```

Practical pretrain sources to prepare:

```text
data/corpus/sources/dclm_baseline_v1
data/corpus/sources/gneissweb_v1
data/corpus/sources/fineweb2_hq_en
data/corpus/sources/fineweb2_hq_ko
```

Builders:

```bash
python scripts/saneflow_download_practical_pretrain.py
python scripts/saneflow_build_practical_pretrain_mix.py
```

First practical mix:

```text
config: configs/saneflow_practical_pretrain_mix.json
output: data/corpus/mixes/saneflow_practical_pretrain_v1
profile: dmc9-practical-base-100m
```

Speak-base bridge before ChatML SFT:

```text
builder: scripts/saneflow_build_speak_pretrain_v1.py
output: data/corpus/mixes/saneflow_speak_pretrain_v1
profile: dmc8-speak-base-v1
purpose: teach stable natural continuation before assistant-only SFT
```

Standard assistant SFT:

```text
builder: scripts/saneflow_prepare_chatml_sources.py
builder: scripts/saneflow_build_chatml_sft.py
output: data/corpus/mixes/saneflow_chatml_sft_train_v1.jsonl
profile: dmc8-chatml-sft-v9
```

Rules:

```text
no ChatML in base pretrain
no answer-only curriculum as majority
no benchmark eval data
no collapsed model outputs
dedup exact and near duplicates before training
keep train/valid split deterministic
```

Why:

The model currently lacks enough broad continuation prior. More SFT cannot
replace base pretraining. A 100M model needs hundreds of millions to billions of
clean tokens before instruction tuning can reliably create natural dialogue.

## Stage 2: Practical Base Pretrain V2

Run only if V1 passes basic generation gates.

Target exposure:

```text
additional 1B-2B tokens
```

Mix adjustment:

```text
increase code if code probes are weak
increase Korean if Korean sentence quality is weak
increase explanation prose if simple QA is shallow
reduce synthetic data if responses become patterned
```

Keep the same tokenizer unless UTF-8, Korean, or code fragmentation is clearly
damaging generation.

## Stage 3: Standard ChatML SFT

Use one ChatML format only.

Current target recipe:

```text
Tulu3-SFT                 25%
SmolTalk2-SFT             20%
Reasoning/OpenR1/Stratos  25%
OpenCodeInstruct          15%
ToolACE/APIGen             7%
Korean/KIT-19              5%
Safety/PolyGuard           3%
```

Scale plan:

```text
22k samples: current sanity run
100k samples: first practical SFT
200k-300k samples: practical broad SFT if 100k passes
```

SFT rules:

```text
assistant-only loss
short, clean answers preferred for 100M scale
clip very long reasoning traces
dedup exact and near duplicates
remove benchmark contamination names and direct eval samples
keep refusal/safety data small but present
do not train direct open_sft files as targets
```

The SFT builder must stay recipe-driven. Add new sources through
`data/corpus/sft_sources/*.jsonl` and the recipe, not new launch scripts.

## Stage 4: Verification SFT

Add only after the model speaks naturally.

Targets:

```text
copy exact
JSON field exact
short arithmetic
small no-cheat MCQ
simple code generation with executable checks
tool-call JSON validity
```

Keep this small and measured. If it hurts ordinary conversation, reduce it and
move the skill into a later pass.

## Stage 5: Final Polish

Use the best base + SFT checkpoint.

Train a small final mix:

```text
clean short chat             40-50%
reasoning/code/tool          25-35%
Korean and bilingual QA      10-15%
safety/refusal                3-5%
base continuation replay      5-15%
```

Purpose:

```text
reduce repetition
restore natural tone
preserve base language prior
prevent overfitting to tool/reasoning templates
```

## Promotion Gates

A checkpoint can become `runs/luma_current` or the default `./neurova.sh` target
only after passing generation gates, not just loss.

Conversation:

```text
hi / who are you: natural self-introduction
simple QA: direct sentence answer
Korean prompt: valid Korean sentence
unknown prompt: says it does not know
no repeated word or repeated character collapse
```

Reasoning:

```text
short arithmetic works more often than not
simple multi-step questions stay on task
answer is not just copied from prompt
```

Format:

```text
ChatML control tokens are not shown to the user
streaming stops cleanly at <|im_end|>
JSON/tool outputs are syntactically valid when requested
```

Minimum practical gate:

```text
chat sanity: pass
repeat collapse: fail rate near zero
copy exact: above 50%
simple recall/QA: above 60%
JSON field: above 70%
```

## Operating Plan

1. Finish current sanity runs and evaluate generation.
2. If sanity passes, build a larger pretrain corpus targeting 500M tokens.
3. Train SaneFlow 100M base from scratch on that corpus.
4. Evaluate plain continuation and basic prompts before any SFT.
5. Run 100k ChatML SFT from the standard recipe.
6. Evaluate conversation, reasoning, Korean, code, tool, and repetition.
7. Scale to 200k-300k SFT only if the 100k run improves quality.
8. Promote only a checkpoint that passes the gates above.

## Non-Goals

```text
do not resume LUMA as the main path
do not add one-off SFT launchers
do not mix raw and ChatML pretrain
do not use SFT to compensate for missing base pretraining
do not promote a checkpoint because loss is low
```
