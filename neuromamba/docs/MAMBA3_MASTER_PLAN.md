# Neurova Mamba-3 Master Plan

This is the canonical Mamba-3 path for Neurova. Older `MAMBA3_*` notes are supporting records; this file is the operating contract.

## Single Source Of Truth

Keep the project deliberately non-fragmented:

- `MAMBA3_MASTER_PLAN.md` defines the operating contract and promotion rules.
- `MAMBA3_TODO.md` tracks executable work only.
- `neuromamba/configs/mamba3_english_top_intelligence_recipe.json` is the only full long-term data/post-training recipe. Do not paste that recipe into more docs.
- `neuromamba/scripts/mamba3_chat_autoloop.sh` is the canonical 24-hour Neurova-chat improvement loop.
- `neuromamba/scripts/mamba3_chat_autoloopctl.sh` is the canonical operator entrypoint for that loop.
- `neuromamba/scripts/mamba3_moe24_trainctl.sh` is the canonical operator entrypoint for the active 2.4B MoE run.
- `neuromamba/scripts/mamba3_post100m_decision.py` is the canonical post-block decision report. It turns loss/collapse evidence into the next action without relying on memory.
- `neuromamba/scripts/mamba3_research_autopilot.sh` is the canonical top-level autonomous research supervisor. It keeps the 2.4B training watchdog active, applies the GPU policy, writes status snapshots, and records the post-block decision report.
- `neuromamba/scripts/mamba3_research_autopilotctl.sh` is the canonical operator entrypoint for that supervisor.

Older `MAMBA3_*` markdown files are evidence logs or design notes. They are not allowed to redefine the operating path. If they disagree with this file, this file wins.

## Goal

Build an English-first Mamba-3 conversational model that is reproducible, measurable, and promotable only when it can actually talk without collapse.

Immediate operating priority: keep the user-facing fast SISO checkpoint usable when serving is the priority, but when the user explicitly asks for full autonomous training/research, switch to the `research autopilot` and let it own the GPU. The 2.4B from-scratch MoE run remains a diagnostic pretraining artifact, not a SOTA claim.

The 24-hour autonomous loop is:

```text
seed checkpoint
-> deterministic governed chat SFT corpus
-> answer-only masked SFT
-> validation answer loss
-> chat quality gate
-> decode tune
-> promote if gate passes and score is >= current best
-> start next candidate from the best promoted checkpoint
```

The current default user CLI resolves checkpoints in this order:

```text
neuromamba/runs/mamba3_current/model.pt
-> neuromamba/runs/mamba3_neurova_chat_v1/chat.pt
-> neuromamba/runs/mamba3_neurova_speak_v1/sft.pt
```

This keeps `/neurova.sh` usable while research continues in the background.

The autonomous research loop is:

```text
research autopilot
-> apply GPU policy
-> if train_priority: stop chat servers to avoid 16GB OOM
-> ensure 2.4B watchdog is active
-> resume the 100M-token 2.4B block from the latest weight checkpoint
-> write status snapshot
-> write 100M decision report
-> repeat until stopped
```

Operator commands:

```bash
./neurova.sh mamba3 research-start
./neurova.sh mamba3 research-status
./neurova.sh mamba3 research-stop
./neurova.sh mamba3 status
```

The project optimizes for the best attainable quality under local 16GB constraints, but it must not confuse total sparse-MoE parameters with dense active compute. The current `mimo-r4-moe-2.4b` run is a maximum-parameter diagnostic path, not proof of dense 2.4B intelligence. SOTA-oriented escalation requires active-compute scale, high-quality deduplicated data scale, optimizer-state discipline, and broad evaluation.

For dense-2.4B-class or SOTA-proximate intelligence, the next system must be active-compute driven: either a trainable dense/dense-ish 1.3B-2.7B path, or a top-2/top-4 MoE path with load-balancing auxiliary loss, router entropy monitoring, and expert-usage control. A top-1 sparse MoE with 2.4B total parameters is useful as a 16GB diagnostic and parameter-scaling experiment, but it is not equivalent to a dense 2.4B model.

The runtime target is:

- Core architecture: Mamba-3 MIMO, starting with r=2 and escalating to r=4 only after stability gates pass.
- Recall-hybrid research path: Mamba-3 + SwiGLU with a small number of GQA attention layers every 4-6 Mamba blocks, plus optional learnable meta tokens.
- Tokenizer: Llama-3.1 tokenizer.
- Features to preserve: exponential-trapezoidal recurrence, complex/data-dependent state tracking, B/C bias, BC/QK Norm, pre-norm, SwiGLU interleave, no short convolution.
- Deployment target: 16GB RTX 4080 for inference and local canaries.
- Training target: the largest model that proves multi-step backward on RTX 4080. The current maximum-parameter local target is sparse-SwiGLU `mimo-r4-moe-2.4b` at seq_len=2048 batch_size=1. The faster throughput target remains `mimo-r4-moe-1.3b` at seq_len=2048 batch_size=3. Dense paper-scale Mamba cores are blocked locally by TileLang shared-memory limits.

## Non-Negotiable Rules

1. Do not train the active chat checkpoint directly on raw web text.
2. Base pretraining, instruction SFT, quality evaluation, speed benchmarking, and promotion are separate stages.
3. A checkpoint is not a product checkpoint until `quality-gate` and decode benchmark pass.
4. Failed or exploratory checkpoints stay under their run directory and must not replace `neuromamba/runs/mamba3_current/model.pt`.
5. All corpus records used for governed training must include provenance metadata.
6. Early pretraining is raw document continuation, not chat or QA. Use explicit document boundaries such as `<doc ...>...</doc>` plus tokenizer EOS; reserve `Instruction:/Answer:` and user/assistant templates for SFT only.
7. Recall/copy/routing tasks are mid-training completion curricula. They must not be used as a substitute for a stable base language model.

## Canonical File Map

- `neuromamba/model.py`: model presets and pure Mamba-3 architecture contract.
- `neuromamba/contract.py`: hard architecture checks.
- `neuromamba/cli.py`: training, inference, speed tests, quality gates, state tests.
- `neuromamba/configs/mamba3_scientific_training_plan.json`: machine-readable staged plan.
- `neuromamba/configs/mamba3_english_top_intelligence_recipe.json`: long-term data and post-training recipe.
- `MAMBA3_TODO.md`: execution TODO with dense train-max, data, sparse scaling, and long-state phases.
- `MAMBA3_OPTIMIZATION_RESEARCH.md`: evidence-backed optimization notes from local tests, papers, and open-source docs.
- `MAMBA3_KERNEL_BACKWARD_NOTES.md`: source-level decision record for official Mamba-3 kernel backward recomputation and TileLang limits.
- `neuromamba/scripts/mamba3_prepare_corpora.sh`: governed corpus download entrypoint.
- `neuromamba/scripts/mamba3_expand_governed_base.sh`: increases governed FineWeb-Edu/DCLM sample budgets and builds an expanded base split.
- `neuromamba/scripts/mamba3_build_doc_continuation_corpus.py`: converts governed records into raw `<doc source="..." domain="...">...</doc>` continuation samples for early base pretraining.
- `neuromamba/scripts/mamba3_find_16gb_train_target.sh`: probes the largest model that actually trains on 16GB.
- `neuromamba/scripts/mamba3_probe_train_grid.sh`: multi-step train/backward grid for mode, context length, and batch size.
- `neuromamba/scripts/mamba3_train_clean_doc_base.sh`: reproducible clean raw-document base pretraining run with overfit sanity and held-out loss.
- `neuromamba/scripts/mamba3_continue_clean_doc_base.sh`: standard weight-only continuation loop for the 900M clean-doc base, with eval/decode artifacts.
- `neuromamba/scripts/mamba3_train_max_moe_base.sh`: maximum-parameter 16GB raw-document base run for `mimo-r4-moe-2.4b`.
- `neuromamba/scripts/mamba3_continue_max_moe_base.sh`: maximum-parameter 2.4B weight-only continuation wrapper.
- `neuromamba/scripts/mamba3_train_max_moe_until_gate.sh`: maximum-parameter 2.4B repeated training loop with the shared loss/decode/collapse gate.
- `neuromamba/scripts/mamba3_train_max_moe_100m_block.sh`: first sufficient local base-training block for the 2.4B checkpoint; defaults to 100M planned tokens in 2,000-step evaluated rounds.
- `neuromamba/scripts/mamba3_launch_max_moe_100m_background.sh`: robust detached launcher for the 100M-token 2.4B block; writes command, log, and PID artifacts.
- `neuromamba/scripts/mamba3_status_max_moe_100m.sh`: status helper for the active 100M-token 2.4B block; prints process, GPU, latest log, and gate summary.
- `neuromamba/scripts/mamba3_moe24_trainctl.sh`: managed training controller for the 2.4B block. It supports `start`, `resume`, `adopt`, `status`, `decision`, `router-diagnostics`, `tail`, `stop`, `logs`, `watchdog-start`, `watchdog-status`, and `watchdog-stop`; control artifacts stay under `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/control`.
- `neuromamba/scripts/mamba3_post100m_decision.py`: reads the gate summary and prints the non-interrupting active-compute/SOTA decision report.
- `neuromamba/scripts/mamba3_moe_router_diagnostics.py`: collects per-layer expert counts, router entropy, expert skew, and active-parameter estimates after the active training process is idle.
- `neuromamba/scripts/mamba3_research_autopilot.sh`: top-level autonomous research supervisor; default policy is `GPU_POLICY=train_priority`.
- `neuromamba/scripts/mamba3_research_autopilotctl.sh`: starts, stops, tails, and reports the autonomous research supervisor.
- `neuromamba/scripts/mamba3_generate_neurova_speak_sft.py`: deterministic no-teacher identity/basic QA/Korean-English speaking SFT generator.
- `neuromamba/scripts/mamba3_train_neurova_speak_v1.sh`: fast speaking-model training path starting from the best existing speak checkpoint.
- `neuromamba/scripts/mamba3_speak_trainctl.sh`: managed background controller for same-day speaking-model training.
- `neuromamba/scripts/mamba3_generate_neurova_chat_sft.py`: deterministic no-teacher chat SFT corpus generator.
- `neuromamba/scripts/mamba3_train_neurova_chat_v1.sh`: single chat candidate training/eval/tune script.
- `neuromamba/scripts/mamba3_chat_quality_gate.py`: chat-specific gate for identity, basic QA, unknown handling, Korean, simple reasoning, and anti-repetition.
- `neuromamba/scripts/mamba3_chat_trainctl.sh`: managed controller for one chat candidate.
- `neuromamba/scripts/mamba3_chat_autoloop.sh`: autonomous multi-trial chat improvement loop.
- `neuromamba/scripts/mamba3_chat_autoloopctl.sh`: managed controller for the autonomous loop.
- `neuromamba/scripts/mamba3_eval_raw_decode_set.py`: multi-prompt raw continuation collapse probe for science/history/code/math document starts.
- `neuromamba/scripts/mamba3_train_scientific_tiny.sh`: reproducible base-pretrain then SFT canary.
- `neuromamba/scripts/mamba3_continue_base_training.sh`: resume a governed base checkpoint for longer base-only training before any recall curriculum.
- `neuromamba/scripts/mamba3_run_arch_context_compare.sh`: equal-token-budget architecture/context comparison for pure Mamba, MIMO, and recall-hybrid candidates.
- `neuromamba/scripts/mamba3_promote_if_pass.sh`: only path that promotes a checkpoint to the default runtime.
- `neurova.sh`: workspace router; Mamba-3 runtime is reached through `./neurova.sh mamba3 ...` or `neuromamba/scripts/run.sh ...`.

## Scientific Training Order

### Stage 0: Kernel and Runtime Gate

Run:

```bash
neuromamba/scripts/mamba3_run_gates.sh
```

Required:

- MIMO-R4 tiny forward is finite.
- Recurrent step decode matches full forward on the next token.
- Decode speed is measured with warmup excluded.
- English I/O does not collapse.

### Stage 1: Governed Corpus

Run:

```bash
neuromamba/scripts/mamba3_prepare_corpora.sh
```

Required:

- FineWeb-Edu and DCLM samples validate.
- Dolma and Nemotron-CC remain blocked until raw-compatible downloaders are implemented.
- `neuromamba/data/mamba3_corpus_manifest.json` states whether the corpus is bootstrap-only or real-pretraining scale.
- Early base data is built as document continuation:

```bash
python neuromamba/scripts/mamba3_build_doc_continuation_corpus.py \
  --inputs neuromamba/data/governed_fineweb_edu_sample.jsonl neuromamba/data/governed_dclm_sample.jsonl \
           neuromamba/data/governed_open_web_math_sample.jsonl neuromamba/data/governed_arxiv_abstracts_sample.jsonl \
  --out neuromamba/data/base_doc_continuation_v1.jsonl
```

Do not include clean SFT, basic QA, chat, or `Instruction:/Answer:` records in this stage. Those are later SFT/post-training artifacts.

### Stage 2: Tiny Scientific Canary

Run:

```bash
BASE_STEPS=1000 SFT_STEPS=1000 RUN_DIR=neuromamba/runs/mamba3_scientific_v1 neuromamba/scripts/mamba3_train_scientific_tiny.sh
```

This trains from scratch on governed base text, then performs instruction SFT. The script logs to `neuromamba/runs/.../train.log` and fails if the SFT checkpoint does not pass `quality-gate`.

### Stage 2B: 16GB Train-Max Selection

Run:

```bash
neuromamba/scripts/mamba3_find_16gb_train_target.sh
neuromamba/scripts/mamba3_probe_train_grid.sh
```

The training target is not chosen by parameter-count ambition. It is chosen by evidence:

- forward finite,
- recurrent decode finite,
- backward succeeds,
- peak VRAM fits 16GB with room for optimizer state and data pipeline overhead.

Current candidate order:

```text
mimo-r4-tiny
mimo-r4-moe-260m
mimo-r4-moe-520m
mimo-r4-moe-900m
mimo-r4-16gb-120m
mimo-r4-16gb-180m
mimo-r4-paper-180m
mimo-r4-440m
mimo-r4-880m
mimo-r4-1.5b
```

Current decision:

- Dense `mimo-r4-440m`, `mimo-r4-880m`, and `mimo-r4-1.5b` fail backward on RTX 4080 at seq_len=128 batch_size=1 because the Mamba-3 TileLang backward kernel requests dynamic shared memory size `223904`.
- This is not ordinary VRAM pressure; shrinking batch or context does not solve it for those dense cores.
- The maximum-parameter local path is sparse SwiGLU scaling with a smaller Mamba core.
- Current maximum-parameter RTX 4080 config: `mimo-r4-moe-2.4b`, 2,397,810,688 parameters, seq_len=2048, batch_size=1, grad_accum=1, bf16, AdamW8bit, weight-only checkpoints. It uses about 14.1GB-14.3GB peak VRAM in training.
- Current faster-throughput RTX 4080 config: `mimo-r4-moe-1.3b`, 1,290,469,376 parameters, seq_len=2048, batch_size=3. Use this when the objective is tokens/day rather than maximum parameter count.
- `mimo-r4-moe-2.5b` fails Mamba-3 MIMO backward with CUDA OOM, and `mimo-r4-moe-2.9b` fails AdamW8bit optimizer allocation.

If paper-scale dense MIMO-R4 backward fails on RTX 4080 due to dynamic shared memory, it remains an inference/probe target, not a local training target.

### Stage 2B-1: Clean Raw-Document Base

Run:

```bash
RUN_DIR=neuromamba/runs/mamba3_clean_doc_base_moe900_v1 \
  OVERFIT_STEPS=120 \
  BASE_STEPS=1200 \
  SAVE_EVERY=200 \
  EVAL_BATCHES=32 \
  neuromamba/scripts/mamba3_train_clean_doc_base.sh
```

Continuation from the base checkpoint, preserving the 16GB max batch, uses a weight-only restart because full optimizer-state resume OOMs at batch_size=3:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m neuromamba.cli train-packed \
  --mode mimo-r4-moe-900m \
  --tokenizer llama31 \
  --checkpoint neuromamba/runs/mamba3_clean_doc_base_moe900_v1/base.pt \
  --data neuromamba/data/splits/base_doc_cont_train.txt \
  --steps 2000 \
  --lr 3e-5 \
  --save-every 500 \
  --grad-accum-steps 1 \
  --optimizer adamw8bit \
  --seq-len 2048 \
  --batch-size 3 \
  --device cuda \
  --dtype bf16 \
  --no-resume
```

This stage is still base pretraining only. Do not run SFT, QA templates, or chat templates until the base passes raw continuation gates.

### Stage 2B-2: Maximum-Parameter Sufficient Base Block

The 2.4B model is from-scratch raw document continuation. It should not be expected to answer QA until it first stops collapsing on raw continuation.

Current operational status: paused and preserved. Do not resume this line until the speaking v1 path has a usable checkpoint and the MoE diagnostics are reviewed.

### Stage 2B-0: Neurova Speak v1 Today

Use this path when the objective is a usable CLI/chat checkpoint today:

```bash
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
neuromamba/scripts/mamba3_speak_trainctl.sh start
neuromamba/scripts/mamba3_speak_trainctl.sh status
```

Rules:

- Start from `neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt` unless a better speaking checkpoint is proven.
- Save to `neuromamba/runs/mamba3_neurova_speak_v1/sft.pt`; never overwrite 2.4B research checkpoints.
- Train deterministic identity/basic QA/simple reasoning/Korean-English answers with answer-only loss.
- Use a larger local batch by default (`batch_size=32`) to use the free 16GB GPU more efficiently for the small speaking checkpoint.
- Promote only if sample generations and `quality-gate` avoid repetition collapse.
- This path is for “speaks today”; the SOTA/base-pretrain path remains separate.
- Treat this as an interim SFT deliverable, not a substitute for the canonical from-scratch intelligence recipe in `neuromamba/configs/mamba3_english_top_intelligence_recipe.json`. The full recipe still requires governed large-scale base pretraining, continued pretraining, reasoning midtraining, SFT, preference/RLVR, and long-context adaptation.

Current evidence:

- 500-step rounds are only about 1.024M tokens each at seq_len=2048 and batch_size=1.
- The 2.4B trajectory is improving, but greedy raw decode still collapses to repeated high-frequency words. Treat the live controller status and `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/until_gate/summary.jsonl` as the numeric source of truth.
- The multi-prompt probe on science/history/code/math document starts collapsed on all 4 prompts, so this is not a QA-template issue.

First sufficient local block:

```bash
TARGET_TOKENS=100000000 LR=8e-6 EVAL_BATCHES=32 \
  neuromamba/scripts/mamba3_train_max_moe_100m_block.sh
```

Detached run on `ml-dmc8`:

```bash
ssh ml-dmc8 'cd /home/user/workspace/neurova && neuromamba/scripts/mamba3_moe24_trainctl.sh start'
ssh ml-dmc8 'cd /home/user/workspace/neurova && neuromamba/scripts/mamba3_moe24_trainctl.sh status'
ssh ml-dmc8 'cd /home/user/workspace/neurova && neuromamba/scripts/mamba3_moe24_trainctl.sh tail'
ssh ml-dmc8 'cd /home/user/workspace/neurova && neuromamba/scripts/mamba3_moe24_trainctl.sh decision'
```

If a run was started before the controller existed, register it without interrupting training:

```bash
ssh ml-dmc8 'cd /home/user/workspace/neurova && neuromamba/scripts/mamba3_moe24_trainctl.sh adopt'
```

If training stops unexpectedly, `resume` starts a detached continuation from the latest weight checkpoint:

```bash
ssh ml-dmc8 'cd /home/user/workspace/neurova && neuromamba/scripts/mamba3_moe24_trainctl.sh resume'
```

This plans about 50,000 optimizer steps, split into 2,000-step rounds. At observed training throughput, this is the first serious local base-training block, not a finished LLM. A checkpoint can move to recall curriculum or SFT only after:

- held-out raw document loss is below `5.0`,
- continuation-only decode is not collapsed,
- multi-prompt raw continuation passes without long repeated-word runs,
- QA/chat templates are still absent from the base data.

The detailed post-100M decision matrix lives in `neuromamba/configs/mamba3_english_top_intelligence_recipe.json` under `post_100m_local_decision_gate`. The automated report is:

```bash
neuromamba/scripts/mamba3_moe24_trainctl.sh decision
```

This report is advisory only. It must not interrupt the active 100M-token run, promote checkpoints, or start SFT.

### Stage 2C: No-Teacher Intelligence-Defect Ladder

Run:

```bash
neuromamba/scripts/mamba3_train_stability_ladder.sh
```

The purpose is to reduce Mamba-style state-compression defects without external LLM teachers or judges.

Compare:

```text
siso
mimo-r2
mimo-r2-attn-tiny
mamba3-recall-r2-tiny
mimo-r4-tiny
mimo-r4-attn-tiny
mamba3-recall-r4-tiny
```

Required measurements:

- governed LM validation loss,
- programmatic exact-match accuracy,
- English quality gate,
- decode speed,
- recurrent decode parity,
- peak VRAM.

Hybrid attention candidates are allowed as research candidates because modern SSM/linear-attention work consistently shows that compressed recurrent state is efficient but weak at exact recall. The current local implementation includes GQA attention every 4 blocks and optional learnable meta tokens. A hybrid candidate must beat simpler Mamba-only stages and pass runtime parity before promotion.

SFT is intentionally kept answer-only by default. Programmatic copy/recall/routing examples are trained and evaluated as a separate curriculum phase; mixing them directly into tiny SFT caused marker/repetition collapse in the v3/v4 ladder canaries.

### Stage 3: Promotion

Run:

```bash
CHECKPOINT=neuromamba/runs/mamba3_scientific_v1/sft.pt neuromamba/scripts/mamba3_promote_if_pass.sh
```

Only this script may update:

```text
neuromamba/runs/mamba3_current/model.pt
neuromamba/runs/mamba3_current/metadata.json
```

`./neurova.sh mamba3` uses `neuromamba/runs/mamba3_current/model.pt` first and falls back to the last known stable tiny chat checkpoint only when current is missing.

### Stage 4: Paper-Scale Escalation

Before moving from tiny to 180M/440M:

```bash
python -m neuromamba.cli probe-kernel --mode mimo-r4-paper-180m --tokenizer llama31 --checkpoint /tmp/nonexistent.pt --seq-len 128 --batch-size 1
```

If backward fails due to dynamic shared memory, do not start paper-scale training on that GPU. The current RTX 4080 path is suitable for inference, decode optimization, data governance, and tiny canaries.

## Quality Gate

Run:

```bash
python -m neuromamba.cli quality-gate \
  --mode mimo-r4-tiny \
  --tokenizer llama31 \
  --checkpoint neuromamba/runs/mamba3_current/model.pt \
  --seq-len 128 \
  --device cuda \
  --dtype bf16 \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --cuda-graph
```

The gate checks:

- Required semantic terms for four English prompts.
- Minimum output length.
- Repeated-token collapse.
- Decode speed metrics.

This is deliberately simple and strict enough to prevent the known failure mode where web continuation training destroys instruction behavior.

## Runtime Parity Contract

`mimo-r4-tiny` is served through `decode_mode=safe` by default.

This is the canonical contract for the current checkpoint:

```bash
python neuromamba/scripts/mamba3_recurrent_parity.py \
  --mode mimo-r4-tiny \
  --tokenizer llama31 \
  --checkpoint neuromamba/runs/mamba3_current/model.pt \
  --seq-len 128 \
  --steps 12 \
  --exact-cache
```

The required exact-cache audit result is `ok: true` with `max_abs: 0.0` for every checked step. Normal interactive serving uses the faster full-forward safe path because it preserves the same quality contract without advancing the currently unsafe recurrent MIMO cache. Unsafe recurrent-cache decode is research-only for this checkpoint. It may only become the default after it passes full-forward/cache-step argmax parity on the promoted checkpoint and prompt suite.

Why this rule exists:

- Official Mamba-3 fast decode uses a recurrent `step()` kernel after prefill.
- The local MIMO step kernel is explicitly marked as H100-tested in the source and the official MIMO step tests use approximate tolerances, not bit-exact LM logits.
- The local `mimo-r4-tiny` shape is a small RTX 4080 validation shape, not the paper-scale H100 benchmark shape.
- A real implementation mismatch was fixed locally: `Trap` is now passed raw in `_preprocess()` so recurrent decode matches the parallel MIMO path's parameterization.
- Remaining unsafe-cache drift is not a dtype issue; bf16 and fp32 both diverge after several generated tokens on the current checkpoint.

Operationally:

- `./neurova.sh mamba3` and `neuromamba/scripts/mamba3_chat_serverctl.sh` default to `NEUROVA_MAMBA3_DECODE_MODE=safe`.
- `NEUROVA_MAMBA3_DECODE_MODE=exact-cache` is retained as a correctness audit path, but it is slower because it verifies exact full-forward logits while maintaining the cache-oriented interface.
- `decode_mode=cache` keeps a short parity guard and falls back when the cache path is unsafe.
- Do not promote a fast recurrent runtime for `mimo-r4-tiny` unless the parity script passes without `--exact-cache`.

## Current Reality

The stable interactive checkpoint is now the pure SISO fast line, not the older MIMO canary. It can stream and answer simple English instruction prompts through the official recurrent `InferenceParams` cache path with CUDA graph enabled. It is still a small instruction-capable local model, not a broad-knowledge SOTA LLM.

Current inference-speed evidence on `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt`:

- Default `./neurova.sh mamba3` uses `mamba3-siso-fast-0.3b-ds128`, `seq_len=128`, recurrent cache decode, CUDA graph on, and checkpoint `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt`.
- v3 recurrent parity passed on the promotion probe.
- Same-condition EOS-limited recurrent benchmark:
  - batch 1: about 159-162 tok/s,
  - batch 64: about 6,580-6,620 aggregate tok/s.
- Actual one-shot server samples after warmup commonly report about 160-250 tok/s on short answers, depending on prompt and generated length.
- The paper-style thousands tok/s number is a batched decode throughput regime. Local single interactive requests must be reported separately.

Official recurrent-path decision:

- The official Mamba-3 MIMO step code is H100-tested and assumes the CUTE step tile shape used by the sample tests.
- Local `mimo-r4-tiny` uses `headdim=32,d_state=32`; it is a trainable canary, not an official fast recurrent runtime.
- The local official-shape preset `mimo-r4-official-tiny` uses `headdim=64,d_state=128,chunk_size=16,mimo_rank=4`. Random-init recurrent parity passed on RTX 4080 with argmax agreement and small logit drift.
- Official-shape MIMO backward failed on RTX 4080 due to dynamic shared memory (`223904` bytes). This is a kernel/hardware limit, not ordinary VRAM pressure.
- Promotion to fast recurrent serving now requires `neuromamba/scripts/mamba3_recurrent_parity.py` without `--exact-cache`; otherwise the checkpoint may only be served through the safe path.

## Current Target Architecture

The 4080 fast-runtime target is now pure `Mamba-3 SISO`, not MIMO-R4 and not the recall-hybrid attention path.

```text
mamba3-siso-fast-0.3b-ds128
  d_model=1024
  n_layer=16
  d_state=128
  headdim=64
  chunk_size=64
  attention_interval=0
  SwiGLU MLP enabled
  official recurrent InferenceParams decode
  CUDA graph decode when quality/parity gates pass
```

The recall-hybrid line remains useful for exact-recall research, but it is not the fastest serving trunk:

```text
mamba3-siso-hybrid-0.3b
  d_model=1024
  n_layer=16
  d_state=64
  headdim=64
  chunk_size=64
  attention_interval=5
  GQA heads=16, KV heads=4
  SwiGLU MLP enabled
```

Scale ladder:

```text
0.3B -> 0.7B -> 1.3B -> 2B
```

Rules:

- Mamba-3 SISO is the speed/state backbone.
- Pure SISO is the fast runtime path.
- GQA attention every 4-6 blocks is the exact-recall research path.
- MIMO r=4 remains H100/kernel-surgery research, not the 4080 mainline.
- A checkpoint may use fast recurrent serving only after recurrent parity passes.
- Current 0.3B fast SISO training run: `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3`.
- Preserved 0.3B SISO-hybrid run: `neuromamba/runs/mamba3_siso_hybrid_0_3b_v1`.
- Published Mamba-3 latency numbers should be interpreted with their benchmark setup: 1.5B model, H100-SXM 80GB, batch size 128. Local RTX 4080 single-request speed is a different regime.

Real knowledge and intelligence require governed pretraining at much larger token scale. On this machine, the correct professional path is to keep the product CLI stable while scaling data and training only through gated, reproducible runs.

Recent local evidence:

- The `mamba3-siso-fast-0.3b-ds128_v3` checkpoint is the current default and passed the basic chat quality gate `15/15`.
- State-memory curriculum has a measurable signal but is incomplete: v3 reached `73/128` on the clean held-out state-memory eval, with JSON/state-summary perfect and copy/routing still at zero.
- The copy/phonebook continuation candidate `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v4_copy` is preserved but not promoted. It improved held-out phonebook exact match from `9/31` to `12/31`, but copy span stayed `0/27`.
- The `seq_len=64` turbo candidate was rejected because it passed quality but was slower than `seq_len=128` in the same benchmark.
- The first HTTP batch endpoint was rejected and removed. Direct batched recurrent decode is fast; the HTTP endpoint did not preserve that throughput.
- `mamba3-recall-r2-tiny` and `mamba3-recall-r4-tiny` both pass forward/backward on RTX 4080 after meta-token padding to the Mamba-3 chunk multiple.
- Both recall-hybrid presets fail recurrent decode-step parity, so they are trainable research candidates, not fast stateful runtime candidates yet.
- v3/v4 tiny SFT runs proved direct programmatic SFT mixing causes marker/repetition collapse.
- v5 answer-only SFT removed `Instruction:` marker collapse, but both `mimo-r2` and `mamba3-recall-r2-tiny` still failed quality and exact-match gates. The bottleneck is now insufficient base training/data and missing separated recall-curriculum training, not merely prompt formatting.
- The separated `base -> recall/copy curriculum -> answer-only SFT` canary proved the curriculum signal is real: exact-match moved from 0/32 to 5/32 and joint recall reached 4/4. It also proved the model is still undertrained: answer-only SFT and clean-English SFT recovery both failed autoregressive quality gates.
- The longer base-first recall run did not produce a promotable model. `mamba3-recall-r2-tiny` trained for 800 governed base steps reached base validation loss 7.1191, but failed English quality. A normal LM recall curriculum then reached curriculum validation loss 1.875 while exact-match stayed 0/64 and English generations became curriculum-contaminated.
- Masked answer-only curriculum is now the default diagnostic for recall data. In the first canary it reduced answer validation loss to 2.9478 and improved exact-match to 9/64, mostly from joint recall and multiple-choice routing. It still scored 0/8 on copy, phonebook, JSON extraction, code tracing, arithmetic, and needle retrieval, and failed English quality. This proves answer-token masking fixes part of the objective mismatch, but not the full intelligence defect.
- Interleaved base plus masked-answer multitask training is implemented as `train-multitask` and wired into `neuromamba/scripts/mamba3_train_base_first_recall.sh` through `CURRICULUM_LOSS=multitask`. The first 300-step comparison from the 800-step base checkpoint reached 11/64 exact-match, again only on joint recall and multiple-choice routing, while English quality still failed.
- Hard-recall shard generation is implemented with `--tasks` and `CURRICULUM_TASKS`. A hard-only multitask run over copy, phonebook, JSON extraction, and needle retrieval reached hard answer validation loss 4.1729 but exact-match was 0/64 on both hard-only and full curriculum evals. It also damaged English quality. This rules out aggressive hard-recall specialization on the current undertrained tiny base.
- Staged easy recall generation is implemented with `--difficulty easy` and `CURRICULUM_DIFFICULTY=easy`. The first easy canary still scored 0/64 exact-match and failed English quality. This means the failure is not just task difficulty; the current 800-step base and recall-hybrid checkpoint cannot yet learn exact copy/lookup reliably.
- A longer base-only continuation from step 800 to step 3200 improved base validation loss from 7.1191 to 6.1875, but still failed English quality. The output became longer and more grammatical than the 800-step base, but remained semantically weak and sometimes produced malformed phrases such as `2th century`.
- The governed base split was expanded from the small repeated bootstrap split to 23,520 train records and 480 validation records: 10,000 FineWeb-Edu documents, 10,000 DCLM documents, and 4,000 deterministic clean-English records. The manifest is about 130.9MB, estimated 32.6M tokens, and remains bootstrap-only rather than real pretraining scale.
- Continuing the same base-only checkpoint from step 3200 to step 5600 on the expanded split improved validation loss again from 6.1875 to 5.6631. It still failed English quality with generic continuations such as `The main idea is the most important thing to do with the right.` This proves data expansion is helping the base loss, but not enough to promote or start recall curriculum again.
- Warmup-excluded decode on the expanded-base checkpoint measured about 103 new tok/s on RTX 4080 bf16 with recurrent step decode. Passing `--cuda-graph` currently falls back to non-graph decode for the recall-hybrid checkpoint, so graph compatibility remains a speed blocker.
- The next expansion added OpenWebMath, deterministic technical bootstrap records, and an arXiv article science shard. Continuing from step 5600 to step 8000 improved validation loss sharply from 5.6631 to 4.3130, so broader math/science/technical data is useful. However, the arXiv article shard leaked preprocessing artifacts into generation, for example `@xmath4-@xmath4`, and the English quality gate still failed. This checkpoint is not promotable.
- Source decision: do not train the default base split on `ccdv/arxiv-summarization` article text. The science source is switched to `gfissore/arxiv-abstracts-2021` abstracts for the next governed split, while artifact-heavy article text remains disabled unless a cleaner is added.
- Rebuilding the expanded split with arXiv abstracts and continuing from step 8000 to step 9200 improved validation loss again from 4.3130 to 4.0815 and removed the visible `@xmath` artifact. It still failed English quality: structured-data field names such as `temperature_c` leaked into general chat, and one science prompt repeated `the most common`. The deterministic technical shard is now proven useful for lowering loss but too strong for early general-English base continuation.
- Data policy update: keep code, JSON/YAML, exact field extraction, and deterministic technical records, but do not let them dominate the early base model. They should be down-weighted or moved into a later recall/copy curriculum after base English passes.
- The English quality gate now rejects known contamination and collapse patterns, including `@xmath`, `@xcite`, structured field names such as `temperature_c`, repeated 3-grams, and known arXiv/template leakage phrases.
- A rebalanced split with lower deterministic technical weight was generated: 10,000 FineWeb-Edu docs, 10,000 DCLM docs, 10,000 OpenWebMath docs, 10,000 arXiv abstracts, 5,000 clean-English records, and 1,200 technical-bootstrap records. The split has 45,276 train records and 924 validation records, estimated at about 119.7M tokens. It is still bootstrap-only.
- Continuing from `base_clean_abstract_long.pt` step 9200 to `base_rebalanced_recovery.pt` step 10800 with `LR=5e-5` failed as a recovery strategy. Validation loss regressed from 4.0815 to 6.0254. The strengthened quality gate still failed, with output such as `The main idea is to be a great deal with a new, and a new, a 1, a 1, 1, 1, ...`. Warmup-excluded decode speed measured 120.98 new tok/s on RTX 4080 bf16 recurrent step decode, but speed is not promotable while generation quality is collapsed.
- Root-cause update: reducing technical records after a checkpoint has already learned bad structured/science patterns is not enough. Continuing through contaminated checkpoints compounds drift. The next base recovery should start from the earlier `base_expanded_long.pt` checkpoint at step 5600, before arXiv article artifacts and structured-data leakage became dominant, and should use a `TECH_RECORDS=0` general-English/math/abstract split.
- The `TECH_RECORDS=0` split path is implemented safely. When technical records are disabled, `neuromamba/scripts/mamba3_expand_governed_base.sh` skips technical generation and removes stale `neuromamba/data/technical_bootstrap_v1.jsonl` before building the split. The generated recovery split has 44,100 train records, 900 validation records, and about 119.4M estimated tokens.
- Continuing from the earlier `base_expanded_long.pt` checkpoint on the `TECH_RECORDS=0` split also failed. `base_general_recovery_from5600.pt` reached step 8000 with validation loss 5.8198, worse than the starting checkpoint's 5.6631. The quality gate still failed: one prompt repeated `a good way to be a good way`, one prompt generated no answer, and the outputs remained generic rather than instruction-following. Warmup-excluded decode speed was 105.65 new tok/s, but the sample stopped after 12 tokens, so it is not a strong long-output speed result.
- Root-cause update: this is no longer just a visible contamination problem. The current tiny recall-hybrid base is under-capacity and under-context for the mixed web/math/abstract recipe, and short `seq_len=128` base continuation is not enough to create a stable conversational LM. More continuation on the same tiny recipe is low leverage.
- Operational rule: do not promote any checkpoint trained mainly by SFT. First make base generation stable, then add recall curriculum, then use SFT only as a light formatting pass.
- Next training rule: do not run more recall curriculum until the base model passes English quality. The exact-copy problem should be revisited only after a larger governed base corpus and another base-only run, or after a corrected high-resolution recall path.
- The canonical next run is a controlled architecture/context comparison, not another blind continuation run. Compare pure Mamba-3 r2/r4, recall-hybrid r2/r4, longer `seq_len` 512 or 1024, source-stratified train/valid splits, and a small Transformer baseline under the same tokenizer and token budget. Do not use recall curriculum or SFT as a shortcut for a weak base.
- `neuromamba/scripts/mamba3_run_arch_context_compare.sh` is the canonical entrypoint for that comparison. It uses approximate equal token budget, not equal step count, then records base validation loss, English quality gate, and decode speed for each candidate.
- The first quick architecture/context comparison is complete. Under `TOKEN_BUDGET=65536`, `seq_len=128` beats `seq_len=512` for all tested candidates. The best base validation loss was `mamba3-recall-r4-tiny seq128` at 8.0286, followed by `mamba3-recall-r2-tiny seq128` at 8.0417, but all candidates failed quality. SISO was fastest at about 198.5 new tok/s but collapsed into punctuation repetition. MIMO and recall-hybrid candidates had lower loss but mostly generated immediate EOS or very short outputs. The first conclusion is that longer context is not the next lever; base stability at short context must be fixed first.
- Next scientific correction: make train/valid splits source-stratified and add a small Transformer baseline. Without these, it is impossible to separate a Mamba-3 architecture defect from a data distribution/token-budget defect.
- Source-stratified split generation is implemented. `neuromamba/scripts/mamba3_make_source_stratified_splits.py` groups JSONL records by `source` and preserves the validation ratio per source. `neuromamba/scripts/mamba3_expand_governed_base.sh` uses this splitter by default. The current stratified recovery split on `ml-dmc8` has 44,100 train and 900 validation records, with each 10,000-record governed shard contributing 9,800 train / 200 valid and clean English contributing 4,900 train / 100 valid.
- A diagnostic `transformer-tiny` baseline is implemented under the same CLI and tokenizer interface. It is not part of the Mamba-3 product path; it exists to determine whether current collapse is specific to Mamba-3/SSM recurrence or caused by tiny token budget and data mixture. `neuromamba/presets.py` is now the lightweight preset source so CLI parsing does not require importing the full Mamba stack.
- Data-format correction: early pretraining is now governed raw document continuation, not QA/chat formatting. `neuromamba/scripts/mamba3_build_doc_continuation_corpus.py` generated `neuromamba/data/base_doc_continuation_v1.jsonl` on `ml-dmc8` with 39,647 wrapped documents and skipped 353 short or instruction-like records. The source-stratified split contains 38,856 train records and 791 validation records.
- A short `mimo-r4-tiny` continuation canary from `neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt` to `neuromamba/runs/mamba3_doc_cont_v1/base_doc.pt` ran for 800 steps on the doc-continuation split. Validation loss improved from 10.9063 to 7.3945, confirming the corrected base objective is learnable. It is still not promotable: both full-forward and recurrent decode produce numeric repetition such as `Photosynthesis is a 1. 1. 1...`. Warmup-excluded recurrent decode on an 80-token sample measured about 108.5 new tok/s.
- QA/chat gates are now SFT-stage diagnostics only. A basic QA answer-only experiment lowered teacher-forced answer loss but did not create reliable generation; it must not be used as evidence of base pretraining quality.
- Train-max update: dense Mamba core scaling is locally blocked by TileLang dynamic shared-memory limits, while sparse SwiGLU MoE scaling trains. The current largest stable RTX 4080 training configuration is `mimo-r4-moe-900m`, seq_len=2048, batch_size=3, AdamW8bit, bf16, with about 12.98GB peak VRAM and about 24K-26K warm training tokens/sec.
- Clean 900M raw-document base run: overfit sanity passed on 64 documents with loss 1.0986 after 120 steps. The from-scratch base run reached held-out doc validation loss 6.4404 after 1200 steps. Weight-only optimizer-reset continuations preserved batch_size=3 and improved held-out validation loss to 6.3330, then 6.3037. The raw decode probe still stops at `The main idea is to be a new`, so the model is not yet language-capable, not SFT-ready, and not promotable.
- The clean-doc continuation path is now scripted as `neuromamba/scripts/mamba3_continue_clean_doc_base.sh`. It uses `--no-resume` and `--no-save-optimizer` by default because full optimizer-state resume OOMs at batch_size=3 and is not useful for the selected max-VRAM setting. A 250-step verification run at `LR=1.5e-5` completed at about 24K-26K tok/s, held peak VRAM at 12.98GB, reduced checkpoint size from about 3.4GB to 1.7GB, and recorded artifacts under `neuromamba/runs/mamba3_clean_doc_base_moe900_v1/continuations/20260613T002911Z_*`. Held-out loss moved from 6.3037 to 6.2969, while the decode probe remained `The main idea is to be a new`.
- The clean raw-doc corpus was expanded to v2 without overwriting v1 evidence: `neuromamba/data/base_doc_continuation_v2.jsonl` contains 79,256 records from 20,000-document governed samples of FineWeb-Edu, DCLM, OpenWebMath, and arXiv abstracts, with 744 short or instruction-like records skipped. The source-stratified v2 split has 77,673 train records and 1,583 validation records. A 250-step 900M continuation on v2 reduced v2 held-out loss from 6.3535 to 6.3311 and preserved the same 12.98GB peak VRAM / 24K-26K tok/s training profile. Decode quality did not improve yet, so continued base pretraining is still required before SFT.
- A gate-driven repeated training loop is now implemented as `neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh`. The first verification round failed the intended gate, with v2 held-out loss 6.3174 and only 4 decoded tokens, and wrote `neuromamba/runs/mamba3_clean_doc_base_moe900_v1/until_gate/summary.jsonl`.
- Root-cause fix: repeated weight-only restarts were re-reading corpus text in the same deterministic order, which over-sampled early records. `train-packed` now supports `--shuffle-texts --data-seed`, `neuromamba/scripts/mamba3_continue_clean_doc_base.sh` defaults to seeded shuffling, and the until-gate loop uses a different deterministic seed per round. A shuffled verification run with `DATA_SEED=777001` preserved the same 12.98GB / 24K-26K tok/s profile, yielded v2 held-out loss 6.3281, and decoded `The main idea is to be a few of the`. This is still not promotable, but the training loop now covers the corpus correctly.
- Root-cause fix: the v1/v2 `.txt` source-stratified splits corrupted document boundaries because document-continuation records contain internal newlines. The writer joined records with newline separators, while the trainer read `.txt` line by line, turning each document body line into a separate sample. The corrected canonical split is now JSONL: `neuromamba/data/splits/base_doc_cont_v3_train.jsonl` and `neuromamba/data/splits/base_doc_cont_v3_valid.jsonl`, with one full `<doc>...</doc>` text per JSONL row. The true v3 held-out loss before recovery was 7.1006, so prior 6.3-range `.txt` losses were line-fragment metrics and must not be used as base objective gates. A 250-step recovery run on v3 improved true held-out loss to 6.9697 and decoded `The main idea is to be a new, but the first time the`; still not promotable, but the objective is now correct.
- Max-parameter update: sparse expert scaling was extended past 900M without increasing the dense Mamba core width. `mimo-r4-moe-2.4b` has 2,397,810,688 estimated parameters with top-1 active experts and passes seq_len=2048 batch_size=1 multi-step backward on RTX 4080, peaking at about 14.21GB. `mimo-r4-moe-2.5b` fails batch_size=1 backward, and `mimo-r4-moe-2.9b` fails optimizer-state allocation, so 2.4B is the current local maximum-parameter training target. A 20-step canary saved `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/base.pt`, reached held-out loss 11.6719 over 4 validation batches, and produced random text as expected for a from-scratch canary.
- Max-parameter continuation update: `neuromamba/scripts/mamba3_continue_max_moe_base.sh` now reuses the standard clean-doc continuation loop for 2.4B. A 250-step weight-only continuation at `LR=2e-5`, `DATA_SEED=991001` reduced held-out doc loss to 10.3281 over 16 validation batches, with 14.205GB peak training VRAM and roughly 6.5K-11K warm training tok/s. Decode now collapses into English frequency tokens (`the the the...`) instead of random multilingual/code fragments. This proves the 2.4B line is learning but still far from a base LM; do not run SFT or promote it.
- Max-parameter gate update: `neuromamba/scripts/mamba3_train_max_moe_until_gate.sh` runs the same repeated clean-doc gate for 2.4B. The shared gate now rejects repetition collapse using `MAX_REPEATED_WORD_RUN` and `MIN_DISTINCT_WORDS`, so a long but degenerate decode cannot pass. A 500-step verification round at `LR=1.5e-5`, `DATA_SEED_BASE=993000` reduced held-out loss to 9.7656, but decode still repeated `the` with a longest repeated-word run of 94, so the gate failed correctly. Continue base training; the model is not SFT-ready.
- Collapse-aware summary verification: a subsequent 500-step 2.4B round at `LR=1.2e-5`, `DATA_SEED_BASE=994000` reduced held-out loss again to 9.5781. The decode still collapsed into `the the the...`; the summary now records `collapsed=true`, `longest_repeated_word_run=96`, and `passed=false`. The current evidence is therefore: the 2.4B line is learning the corpus objective, but greedy generation remains repetition-dominated. Keep training/evaluating as base CLM; do not promote or SFT.
- Repetition-gate correction: the collapse gate now strips the prompt before calculating distinct generated words and repeated-word runs. This prevents prompt words from inflating quality metrics. The next 2.4B round at `LR=1.0e-5`, `DATA_SEED_BASE=995000` reduced held-out loss to 9.5195, but the continuation-only decode had `distinct_words=1` and `longest_repeated_word_run=96`, so the gate failed correctly. The bottleneck is no longer trainability; it is base-LM repetition collapse under greedy decode.

## Sparse Scaling Policy

MoE is allowed as an experiment, not as the first default. The first sparse path should keep Mamba-3 recurrence dense and add sparse SwiGLU experts between Mamba-3 blocks. It must include router load-balancing metrics and must beat the dense 16GB baseline on validation loss or quality at comparable active compute before promotion.

## Efficiency Stack

Use every optimization only when it preserves correctness:

- bf16 activations and weights for training canaries.
- TF32 matmul where PyTorch uses fp32 kernels.
- CUDA graph recurrent decode for streaming inference.
- Warmup-excluded benchmark reporting.
- Top-1 sparse SwiGLU experts to increase total parameters without increasing active Mamba recurrence cost.
- Activation checkpointing and gradient accumulation before increasing dense width.
- Official Mamba kernel-level backward recomputation before any generic checkpointing.
- 8-bit optimizer only after the unquantized optimizer path is known-good.
- Quantized inference only after quality promotion, never during the first correctness gate.
