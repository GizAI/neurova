# SaneFlow Standardization

This document defines the active SaneFlow layout. The final SFT path is single
format ChatML with assistant-only labels. Old `open_sft`, `sft/current`
symlinks, and full-sequence ChatML loss are not active paths.

The practical retraining plan lives in:

```text
docs/neurova_training_data_master_plan.md
```

The autonomous research program SSOT lives in:

```text
configs/saneflow_research_program.json
```

## Canonical Data

Base pretraining:

```text
data/corpus/sources/fineweb_edu_sample10bt/train.jsonl
data/corpus/sources/fineweb_edu_sample10bt/valid.jsonl
```

Practical base pretraining:

```text
data/corpus/sources/fineweb_edu_sample10bt/{train,valid}.jsonl
data/corpus/sources/dclm_baseline_v1/{train,valid}.jsonl
data/corpus/sources/gneissweb_v1/{train,valid}.jsonl
data/corpus/sources/fineweb2_hq_en/{train,valid}.jsonl
data/corpus/sources/fineweb2_hq_ko/{train,valid}.jsonl
data/corpus/mixes/saneflow_practical_pretrain_v1/{train,valid}.jsonl
```

Current R diagnostic pretrain is intentionally narrower:

```text
runs/saneflow_r_champion/d_delta_landmark_long
  train-data: data/corpus/sources/fineweb_edu_sample10bt/train.jsonl
```

That FineWeb-Edu file contains many domains inside one source, but it is not the
same as the explicit practical mix. DCLM-Baseline and GneissWeb become active
only through `saneflow_practical_pretrain_v1`.

Speak-base continuation before SFT:

```text
data/corpus/mixes/saneflow_speak_pretrain_v1/{train,valid}.jsonl
```

Chat SFT:

```text
data/corpus/sft_sources/*.jsonl
data/corpus/mixes/saneflow_chatml_sft_train_v1.jsonl
data/corpus/mixes/saneflow_chatml_sft_valid_v1.jsonl
data/corpus/mixes/saneflow_chatml_sft_manifest_v1.json
```

Prepare raw HF/cache files once, then build SFT data with:

```bash
python scripts/saneflow_prepare_chatml_sources.py
python scripts/saneflow_build_chatml_sft.py
```

Recipe:

```text
configs/saneflow_chatml_sft_recipe.json
```

The recipe is the SFT data SSOT. `saneflow_prepare_chatml_sources.py` is the
only raw-source adapter layer. `saneflow_build_chatml_sft.py` reads normalized
JSONL sources only, applies clipping, benchmark-name filtering, exact/near
dedup, and writes the single standard ChatML train/valid files above.

Target SFT mix:

```text
Tulu3-SFT                 25%
SmolTalk2-SFT             20%
Reasoning/OpenR1/Stratos  25%
OpenCodeInstruct          15%
ToolACE/APIGen             7%
Korean/KIT-19              5%
Safety/PolyGuard           3%
```

`Salesforce/xlam-function-calling-60k` and `allenai/wildguardmix` require gated
HF access in the current environment, so the standard recipe uses public
ToolACE/APIGen and PolyGuardMix inputs for those buckets.

## ChatML Contract

All SFT rows use:

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

Training uses:

```text
--loss-mode chatml_assistant
```

Only assistant answer tokens, including the closing `<|im_end|>`, contribute to
loss. System/user tokens are context only.

## Canonical Launchers

Registry and profile launcher:

```text
scripts/saneflow_run.py
```

dmc8 after-base waiter:

```text
scripts/saneflow_after_base_dmc8.sh
scripts/saneflow_researchctl_dmc8.sh
profiles: dmc8-speak-base-v1, dmc8-chatml-sft-v9
```

dmc9 research control:

```text
scripts/saneflow_researchctl_dmc9.sh
profiles: dmc9-practical-base-100m, dmc9-r-champion-delta-landmark-long
```

Autonomous host loops:

```text
scripts/saneflow_autoresearch_loop.sh dmc8
scripts/saneflow_autoresearch_loop.sh dmc9
scripts/saneflow_fleetctl.sh status
scripts/saneflow_fleetctl.sh active
scripts/saneflow_fleetctl.sh start
```

`saneflow_autoresearch_loop.sh` is a persistent state machine by default
(`SANEFLOW_LOOP_FOREVER=1`). It does not own long-running training processes.
Each cycle only checks whether a profile is running or complete, starts missing
active work once, runs quality gates when checkpoints appear, and promotes only
after the gate.

## Active Profiles

```text
dmc8 line A:
  dmc8-speak-base-v1
  dmc8-chatml-sft-v9

dmc9 line B:
  dmc9-practical-base-100m

dmc9 line C:
  dmc9-r-champion-delta-landmark-long
  dmc9-r-champion-practical-cont

legacy/diagnostic:
dmc8-base-100m
dmc9-sparse-chatml-sft
dmc9-neurova-r-full
```

Default status commands should focus on the active lines. Legacy/diagnostic
runs can be inspected with explicit `status-all` commands, but they are not
allowed to drive deployment or promotion.

The active research lines are separate by design:

```text
Line A, speak/chat:
  stable natural continuation first, gated ChatML SFT second
  host: ml-dmc8
  promotion: only after quality gate

Line B, practical base:
  best 100M-class base from explicit source mix
  host: ml-dmc9 GPU0
  data: FineWeb-Edu + DCLM-Baseline, plus verified future sources

Line C, R architecture:
  DeltaMatrix/landmark recurrent architecture research
  host: ml-dmc9 GPU1
  phase 1: FineWeb-Edu diagnostic continuation
  phase 2: practical-mix continuation
  status: research-only until generation gate passes
```

## Autonomous Policy

The current default policy is:

```text
1. dmc8 Line A:
   build speak-pretrain + ChatML data
   run dmc8-speak-base-v1
   quality-gate speak-base
   only then run dmc8-chatml-sft-v9
   quality-gate and promote to runs/saneflow_current/model.pt

2. dmc9 Line B:
   build practical pretrain mix from verified source config
   run dmc9-practical-base-100m on GPU0
   keep it base-pretrain only until language quality is good enough

3. dmc9 Line C:
   continue dmc9-r-champion-delta-landmark-long on GPU1
   after it finishes, continue dmc9-r-champion-practical-cont
   keep it research-only until it passes generation/reasoning gates
```

Meaningless SFT jobs should be stopped or left as legacy artifacts. New SFT
should be scheduled only through `dmc8-chatml-sft-v9` or a future profile that
explicitly initializes from a gated base checkpoint.

## Inference

`./neurova.sh` defaults to SaneFlow chat and passes `--chatml`. The generator and
streaming chat both stop on `<|im_end|>` and strip ChatML control tokens before
showing output. Runtime decoding supports:

```text
--repetition-penalty
--no-repeat-ngram-size
```

Default shell values:

```text
NEUROVA_SANEFLOW_REPETITION_PENALTY=1.08
NEUROVA_SANEFLOW_NO_REPEAT_NGRAM_SIZE=4
```

These are inference controls only. A checkpoint that only becomes usable because
of penalties still fails the model-quality gate.

## Removed Paths

Do not recreate these as active launch paths:

```text
scripts/saneflow_sftctl_dmc9.sh
scripts/saneflow_build_open_sft_mix.py
data/corpus/sft/current
runs/**/open_sft*
```

Existing broad SFT files under `data/corpus/mixes/saneflow_open_sft_v2` are
allowed only as builder inputs, never as direct training targets.

Raw parquet/csv/json files are allowed only under `data/corpus/raw_hf_sft_v3`.
They are never direct training targets and are never read by the remote training
launcher.
