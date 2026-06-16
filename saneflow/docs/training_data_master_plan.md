# Neurova / SaneFlow Training Data Master Plan

This is the active data plan for SaneFlow. It intentionally excludes previous
one-off SFT and ablation launch paths.

## Target

```text
model: SaneFlowLM
dense tokenizer: saneflow/tokenizers/neurova_spm_unigram_64k
legacy recurrent tokenizer: saneflow/tokenizers/saneflow_fineweb_edu_16k
profile registry: saneflow/configs/saneflow_profiles.json
active program: saneflow/configs/saneflow_research_program.json
pretrain format: raw document continuation
SFT format: ChatML
SFT loss: assistant-only
promotion: quality gate, not loss alone
```

## Active Data Flow

```text
Line A, dmc8 speak/chat:
  saneflow/configs/saneflow_speak_pretrain_mix.json
    -> saneflow/data/corpus/mixes/saneflow_speak_pretrain_v1/{train,valid}.jsonl
  saneflow/configs/saneflow_chatml_sft_recipe.json
    -> saneflow/data/corpus/mixes/saneflow_chatml_sft_{train,valid}_v1.jsonl

Line B, dmc9 practical base:
  saneflow/configs/saneflow_practical_pretrain_sources.json
    -> saneflow/data/corpus/sources/*/{train,valid}.jsonl
  saneflow/configs/saneflow_practical_pretrain_mix.json
    -> optional DoReMi ratio override:
       saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/doremi_ratios.json
    -> saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/{train,valid}.jsonl

Line C, dmc9 dense Transformer baseline:
  saneflow/tokenizers/neurova_spm_unigram_64k
  saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/{train,valid}.jsonl
```

## Source Policy

All source ingestion must be explicit and reproducible:

```text
download/adapter script
source JSON config
normalized JSONL source output
ratio-controlled mix builder
DoReMi-style proxy Group DRO ratio update when available
manifest with counts and missing_or_empty reporting
```

Do not hardcode dataset paths inside training loops or launch scripts. Training
profiles live in `saneflow/configs/saneflow_profiles.json` and may point only at
normalized source or mix JSONL files.

## Base Pretrain Policy

Base pretraining must stay continuation-first:

```text
no ChatML majority in base
no answer-only majority in base
no benchmark eval data
no collapsed model generations
dedup exact and near duplicates where builders support it
strong document quality filter before mixing
source weights from DoReMi proxy Group DRO, not downstream benchmark tuning
```

## DoReMi Data Mixing Contract

The practical mix follows the DoReMi paper structure, scaled down for the local
hardware:

```text
1. Train or choose a small reference checkpoint on prior/default source ratios.
2. Train a small proxy with Group DRO over source domains.
3. Track adversarial domain weights from proxy excess loss
   proxy_loss(domain) - reference_loss(domain).
4. Write saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/doremi_ratios.json.
5. Rebuild the practical pretrain mix with that ratio override.
6. Train the larger active model on the rebuilt mix.
```

Canonical commands:

```bash
python saneflow/scripts/saneflow_train_tokenizer.py \
  --kind sentencepiece_unigram \
  --vocab-size 65536 \
  --character-coverage 0.99995 \
  --out saneflow/tokenizers/neurova_spm_unigram_64k \
  --input saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/train.jsonl \
          saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/valid.jsonl

python saneflow/scripts/saneflow_doremi_pipeline.py \
  --tokenizer-path saneflow/tokenizers/neurova_spm_unigram_64k \
  --reference-steps 300 \
  --steps 500 \
  --seq-len 512 \
  --batch-size 2 \
  --tf32 --activation-checkpointing
```

If no reference checkpoint is supplied, `saneflow_doremi_pipeline.py` first
trains a same-tokenizer prior-ratio reference proxy, then trains the excess-loss
DRO proxy and rebuilds the mix.

The practical target is hundreds of millions to billions of clean tokens before
large SFT investment. Small runs are pipeline and architecture diagnostics.

## ChatML SFT Policy

The SFT recipe is:

```text
saneflow/configs/saneflow_chatml_sft_recipe.json
```

The builder:

```bash
python saneflow/scripts/saneflow_prepare_chatml_sources.py
python saneflow/scripts/saneflow_build_chatml_sft.py
```

The model trains with:

```text
--loss-mode chatml_assistant
```

Only assistant response tokens contribute to loss. System/user turns are context
only.

## Promotion Gates

Before promotion to `saneflow/runs/saneflow_current/model.pt`, a checkpoint must pass:

```text
empty_output_rate <= 0.05
invalid_output_rate <= 0.02
max_repeated_4gram <= 4
mean_distinct_1 >= 0.22
mean_distinct_2 >= 0.45
```

Reasoning-oriented checkpoints also need the reasoning gate before being treated
as anything other than research artifacts.

## Training Efficiency Contract

The active dense baseline uses the subset that is stable on one 16GB GPU:

```text
attention: torch SDPA with Flash/mem-efficient backends enabled
precision: bf16
optimizer: Muon for matrix parameters, AdamW for embedding/norm/scalar params
checkpointing: activation checkpointing enabled
data path: dataset tensor cached on GPU when VRAM allows
dense tokenizer: SentencePiece Unigram 64K, byte_fallback, identity normalization
```

Available but not mainline:

```text
fp8:
  dtype support is detected, but training is disabled until a separate fp8
  scaling smoke test passes.

FSDP:
  available through torch.distributed.fsdp, but useful only with a proper
  multi-process/multi-GPU launcher.

ZeRO:
  requires DeepSpeed. Config files exist, but the package is not assumed.

GaLore:
  implemented as built-in low-rank AdamW for memory-reduction experiments.
  It is not mixed with Muon on the same matrix parameters.
```
