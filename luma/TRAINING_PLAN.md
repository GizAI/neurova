# LUMA Training Data And Run Plan

Current status: LUMA is not the active speaking-model training line. The
canonical Neurova training/data policy is now
`docs/neurova_training_data_master_plan.md`; keep this file as the LUMA-specific
slot-memory research and archive reference.

This is the canonical local plan. Do not promote a checkpoint to
`runs/luma_current` from any stage unless the final gates pass.

## Data Contract

- Raw continuation is not ChatML.
- Natural pretraining is not ChatML; use plain prose and plain `Question:/Answer:` text.
- Natural pretraining uses two streams: document continuation as packed raw
  loss, and `natural_speak_raw` as plain QA answer-only loss.
- Chat, reasoning, and memory SFT are strict ChatML.
- Legacy `Instruction:/Answer:` data may be converted only by the offline data
  builder. Runtime datasets stay strict and do not carry compatibility branches.
- Slot-proof data remains a separate memory-ablation stream, not a substitute
  for normal chat or raw language training.

## Canonical Files

Build:

```bash
python -m luma.build_training_data --version v2
python -m luma.analyze_training_data \
  data/luma_stage_raw_cont_v2.jsonl \
  data/luma_stage_chatml_dialogue_v2.jsonl \
  data/luma_stage_chatml_slotproof_v2.jsonl \
  --out data/luma_training_data_analysis_v2.json
```

Outputs:

- `data/luma_stage_raw_cont_v2.jsonl`
- `data/luma_stage_natural_raw_v2.jsonl`
- `data/luma_stage_natural_speak_raw_v2.jsonl`
- `data/luma_stage_chatml_sft_v1.jsonl`
- `data/luma_stage_chatml_reasoning_v1.jsonl`
- `data/luma_stage_chatml_memory_v1.jsonl`
- `data/luma_stage_chatml_dialogue_v2.jsonl`
- `data/luma_stage_chatml_slotproof_v2.jsonl`
- `data/luma_training_data_manifest_v1.json`
- `data/luma_training_data_manifest_v2.json`
- `data/luma_training_data_analysis_v2.json`

## Source Policy

Use now:

- `data/english_bootstrap.txt` and `data/english_completion_bootstrap.txt` for
  raw continuation.
- `data/train_dialogues.txt`, `data/english_instruction_bootstrap.txt`, and
  `data/governed_instruction_sample.jsonl` for small clean chat behavior.
- `data/deepseek_no_cheat_mcq_sft_v1_pilot.jsonl` and
  `data/rlvr_verifier_bootstrap.jsonl` for reasoning/verifiable answers.
- `data/mamba3_programmatic_curriculum.jsonl`,
  `data/luma_memory_curriculum_v1.jsonl`, and v25-v27 IR corpora for memory
  and event-slot extraction.
- `data/luma_stage_chatml_dialogue_v2.jsonl` for the main dialogue route.
- `data/luma_stage_chatml_slotproof_v2.jsonl` for the clean slot-proof route.

Do not use as direct chat SFT:

- raw IR corpora without conversion
- narrow old `neurova_chat_sft_v1` style data
- any checkpoint output that showed repetition collapse
- MCQ answer-letter corpora as default dialogue distribution
- synthetic IR/event-slot corpora as default dialogue distribution

## Canonical Training Stages

This is a from-scratch LUMA route. It may use the Qwen tokenizer contract, but
it does not use Qwen model weights, Transformer blocks, or Mamba blocks. Local
attention, slots, and copy are configurable research options; the default
language-prior path keeps them off until the base model can produce readable
sentences.

### Stage 1: Natural Language Prior

Train from scratch on non-ChatML text with the Qwen tokenizer. Do not use ChatML
here. Keep document continuation and plain QA as separate streams so the model
learns language flow from raw text and direct answers from answer-only QA.

```bash
RUN_DIR=runs/luma_stage1_qwen_natural_pre_v1 \
RECIPE=custom \
RAW_DATA=data/luma_stage_raw_cont_v2.jsonl \
QA_DATA=data/luma_stage_natural_speak_raw_v2.jsonl \
CHAT_DATA= \
MEMORY_DATA= \
RAW_WEIGHT=0.75 \
QA_WEIGHT=0.25 \
CHAT_WEIGHT=0.0 \
MEMORY_WEIGHT=0.0 \
SLOT_PROOF_WEIGHT=0.0 \
TOKENIZER_BACKEND=qwen \
RAW_DATASET_MODE=packed \
RAW_ANSWER_ONLY=0 \
USE_SLOTS=0 \
USE_LOCAL_ATTENTION=0 \
LOCAL_HEADS=0 \
STEPS=1500 \
SEQ_LEN=256 \
BATCH_SIZE=6 \
D_MODEL=768 \
LAYERS=10 \
SLOTS=0 \
TOPK=1 \
COPY_WINDOW=0 \
RUN_GENERATE=0 \
RUN_MEMORY_EVAL=0 \
./scripts/luma_train_dmc8.sh
```

Gate:

```bash
python -m luma.eval_natural_sanity \
  --ckpt runs/luma_stage1_qwen_natural_pre_v1/model.pt \
  --out runs/luma_stage1_qwen_natural_pre_v1/natural_sanity.json \
  --device cuda \
  --dtype bf16
```

### Stage 2: Clean ChatML SFT

Continue from Stage 1. The objective is readable assistant behavior, not memory
proof. Keep a small raw stream in the mix so the narrow local chat set does not
overwrite the language prior learned in Stage 1.

```bash
RUN_DIR=runs/luma_stage2_qwen_chat_sft_v1 \
INIT_FROM=runs/luma_stage1_qwen_natural_pre_v1/model.pt \
RECIPE=custom \
RAW_DATA=data/luma_stage_raw_cont_v2.jsonl \
QA_DATA=data/luma_stage_natural_speak_raw_v2.jsonl \
CHAT_DATA=data/luma_stage_chatml_dialogue_v2.jsonl \
MEMORY_DATA= \
RAW_WEIGHT=0.15 \
QA_WEIGHT=0.15 \
CHAT_WEIGHT=0.70 \
MEMORY_WEIGHT=0.0 \
SLOT_PROOF_WEIGHT=0.0 \
TOKENIZER_BACKEND=qwen \
USE_SLOTS=0 \
USE_LOCAL_ATTENTION=0 \
LOCAL_HEADS=0 \
STEPS=1000 \
SEQ_LEN=256 \
BATCH_SIZE=6 \
D_MODEL=768 \
LAYERS=10 \
SLOTS=0 \
TOPK=1 \
COPY_WINDOW=0 \
RUN_GENERATE=0 \
RUN_MEMORY_EVAL=0 \
./scripts/luma_train_dmc8.sh
```

### Stage 3: Clean Slot-Proof

Continue from Stage 2. The objective is slot usefulness. Loss alone is not a
pass.

```bash
RUN_DIR=runs/luma_stage3_qwen_slotproof_v1 \
INIT_FROM=runs/luma_stage2_qwen_chat_sft_v1/model.pt \
RECIPE=custom \
RAW_DATA=data/luma_stage_raw_cont_v2.jsonl \
QA_DATA=data/luma_stage_natural_speak_raw_v2.jsonl \
CHAT_DATA=data/luma_stage_chatml_dialogue_v2.jsonl \
MEMORY_DATA=data/luma_stage_chatml_slotproof_v2.jsonl \
RAW_WEIGHT=0.15 \
QA_WEIGHT=0.10 \
CHAT_WEIGHT=0.35 \
MEMORY_WEIGHT=0.25 \
SLOT_PROOF_WEIGHT=0.10 \
AB_MARGIN_WEIGHT=0.25 \
MEMORY_LOGIT_WEIGHT=0.0 \
TOKENIZER_BACKEND=qwen \
USE_SLOTS=1 \
USE_LOCAL_ATTENTION=0 \
LOCAL_HEADS=0 \
STEPS=3000 \
SEQ_LEN=256 \
BATCH_SIZE=4 \
D_MODEL=768 \
LAYERS=10 \
SLOTS=256 \
COPY_WINDOW=0 \
./scripts/luma_train_dmc8.sh
```

## Gates

- Chat sanity: `hi`, `who are you?`, simple ML question, simple Korean QA.
- Repetition: no token/phrase collapse in greedy or low-temperature sampling.
- Memory: copy exact > 50%, recall > 60%, json_field > 70%.
- Slot proof: `normal` must beat `no_slots` and `random_slot_keys`; otherwise
  slots are noise.
- Promotion: only a run that passes all gates can become `runs/luma_current`.
  Use `gate_summary.json` as the source of truth; do not promote from raw loss,
  chat loss, or one-off manual probes.
