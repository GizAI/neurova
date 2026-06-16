# SaneFlow Standardization

This document is the current SaneFlow contract. Anything not listed here is not
an active SaneFlow launch path.

## Single Source Of Truth

```text
saneflow/configs/saneflow_research_program.json
saneflow/configs/saneflow_profiles.json
saneflow/scripts/saneflow_run.py
saneflow/scripts/saneflow_autoresearch_loop.sh
```

`saneflow/configs/saneflow_profiles.json` is the canonical source for profile runtime,
data paths, output paths, optimizer settings, and checkpoint initialization.
`saneflow/configs/saneflow_research_program.json` is the canonical source for which
profiles are active. `saneflow/scripts/saneflow_run.py` is only a thin launcher over
those configs; it must not grow hardcoded model lines.

## Active Lines

```text
Line A, speak/chat on ml-dmc8 GPU0:
  dmc8-speak-base-v1
  dmc8-chatml-sft-v9

Line B, practical base on ml-dmc9 GPU0:
  dmc9-practical-base-100m

Line C, dense Transformer baseline on ml-dmc9 GPU1:
  dmc9-dense-0.3b-v1
```

Line A trains stable natural continuation first. It runs assistant-only ChatML
SFT only after the speak-base quality gate passes.

Line B is base pretraining only. It exists to improve the language prior from
verified continuation data before any SFT is considered.

Line C is a direct PyTorch decoder-only Transformer baseline: RMSNorm, bias-free
GQA attention, RoPE, QK-Norm, bias-free SwiGLU, bf16, SDPA/Flash-style attention
backend, activation checkpointing, and Muon. It is research-only until it passes
both generation and reasoning gates.

Line C uses `saneflow/tokenizers/neurova_spm_unigram_64k`: SentencePiece Unigram, 64K
target vocab, byte fallback enabled, identity normalization, character coverage
0.99995, and the canonical chat/tool special tokens:
`<pad>`, `<bos>`, `<eos>`, `<unk>`, `<|user|>`, `<|assistant|>`,
`<|system|>`, `<|tool|>`, `<|im_start|>`, `<|im_end|>`. The tokenizer is
trained from the same practical pretraining mix ratio used by the active dense
line.

System optimization contract:

```text
FlashAttention path:
  Use torch.scaled_dot_product_attention. On CUDA, training enables PyTorch
  Flash SDP and memory-efficient SDP backends. The external flash_attn package
  is optional and not required by the active profile.

Precision:
  Active training uses bf16. fp8 is not enabled by default just because the dtype
  exists; it needs an explicit smoke test and scaling recipe before use.

Optimizer:
  Active profile uses Muon for 2D matrix parameters and AdamW for embedding,
  norm, and scalar parameters. Built-in GaLoreAdamW is available as an
  optimizer-state memory experiment, not mixed into the Muon mainline.

FSDP/ZeRO:
  Single-GPU profiles do not use FSDP/ZeRO. Torch FSDP can be used by a future
  multi-process launcher. ZeRO requires DeepSpeed to be installed and is
  currently represented only by config files.

Checkpointing:
  Dense 0.3B enables activation checkpointing.
  Active checkpoints keep the existing `latest.pt` and `model.pt` filenames and
  include `config`, `model`, `step`, `global_step`, `train_state`, and optimizer
  state by default. Older weight-only checkpoints still load, but they resume
  with a fresh optimizer once and write full resume checkpoints on the next save.
```

## Canonical Data

```text
Base/practical sources:
  saneflow/data/corpus/sources/*/{train,valid}.jsonl

Speak bridge:
  saneflow/configs/saneflow_speak_pretrain_mix.json
  saneflow/data/corpus/mixes/saneflow_speak_pretrain_v1/{train,valid}.jsonl

Practical pretrain:
  saneflow/configs/saneflow_practical_pretrain_sources.json
  saneflow/configs/saneflow_practical_pretrain_mix.json
  saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/{train,valid}.jsonl

ChatML SFT:
  saneflow/configs/saneflow_chatml_sft_recipe.json
  saneflow/data/corpus/sft_sources/*.jsonl
  saneflow/data/corpus/mixes/saneflow_chatml_sft_{train,valid}_v1.jsonl
```

Raw HF parquet/csv/json files are adapter inputs only. Training profiles never
point directly at raw files.

## Canonical Commands

```bash
python saneflow/scripts/saneflow_prepare_chatml_sources.py
python saneflow/scripts/saneflow_build_chatml_sft.py
python saneflow/scripts/saneflow_build_speak_pretrain_v1.py
python saneflow/scripts/saneflow_build_practical_pretrain_mix.py
python saneflow/scripts/saneflow_train_tokenizer.py
python saneflow/scripts/saneflow_doremi_pipeline.py
python saneflow/scripts/saneflow_system_capabilities.py

bash saneflow/scripts/saneflow_fleetctl.sh active
bash saneflow/scripts/saneflow_fleetctl.sh start
bash saneflow/scripts/saneflow_researchctl_dmc8.sh status
bash saneflow/scripts/saneflow_researchctl_dmc9.sh status
```

`saneflow_autoresearch_loop.sh` is persistent by default. It does not own
training processes; it checks profile status, starts missing active work once,
runs quality gates, and promotes only after the gate.

## Inference

`./neurova.sh saneflow` starts the SaneFlow chat path. Runtime decoding uses
the shared decoder controls in `saneflow/decoding.py`:

```text
NEUROVA_SANEFLOW_REPETITION_PENALTY=1.08
NEUROVA_SANEFLOW_NO_REPEAT_NGRAM_SIZE=4
NEUROVA_SANEFLOW_DECODE_MODE=cache
```

These are runtime controls only. A checkpoint that is usable only because of
penalties still fails the model-quality gate.

## Removed Legacy

Old one-off SaneFlow SFT builders, ablation launchers, speed sweeps, and
pipeline wrappers were removed. New experiments must be added as either:

```text
1. a new profile in saneflow/configs/saneflow_profiles.json, plus
2. a matching entry in saneflow/configs/saneflow_research_program.json, plus
3. a quality-gate rule before promotion.
```

Do not reintroduce direct training from broad ad-hoc SFT mixes or raw downloaded
files. Add data through the source adapters and recipe files above.
