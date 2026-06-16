# Mamba-3 Execution TODO

## Phase 0: Neurova Speak v1 Today

## Current Autonomous Research Policy

- [x] Add top-level autonomous research supervisor.
  - Script: `neuromamba/scripts/mamba3_research_autopilot.sh`.
  - Controller: `neuromamba/scripts/mamba3_research_autopilotctl.sh`.
  - User entrypoints:
    - `./neurova.sh mamba3 research-start`
    - `./neurova.sh mamba3 research-status`
    - `./neurova.sh mamba3 research-stop`
    - `./neurova.sh mamba3 status`
  - Default policy: `GPU_POLICY=train_priority`.
  - Under `train_priority`, the supervisor stops Mamba chat servers on ports `8765` and `8767` before resuming the 2.4B block, because the 2.4B run peaks near 13GB and the server can push a 16GB RTX 4080 into OOM.
  - It keeps `neuromamba/scripts/mamba3_moe24_trainctl.sh watchdog-start` active and calls `resume` if the training process disappears.
  - It writes `neuromamba/runs/mamba3_research_autopilot/control/latest_status.txt` with process, GPU, training progress, and decision output.
- [x] Fix autopilot/watchdog double-start race.
  - Root cause: the first supervisor version started the watchdog and also called `resume` in the same loop. That launched two 2.4B `train-packed` processes, each loading about 7.5GB, causing CUDA OOM.
  - Fix: when the supervisor starts the watchdog, the watchdog owns the first resume. The supervisor does not directly resume again in that loop.
  - Verification: one watchdog, one `mamba3_train_clean_doc_until_gate.sh`, and one `mimo-r4-moe-2.4b train-packed` process are active; GPU memory is about `13.6GB/16GB`.
- [ ] Let the research autopilot own the GPU when the explicit objective is "all training/autonomous research".
  - Start command: `./neurova.sh mamba3 research-start`.
  - Monitor command: `./neurova.sh mamba3 research-status`.
  - If interactive chat is needed again, stop autopilot first and restart the chat server.

## Current Intelligence Benchmark Gate

- [x] Add a no-cheat multiple-choice benchmark harness.
  - Script: `neuromamba/scripts/mamba3_eval_mcq_bench.py`.
  - It shuffles choices with a fixed seed and reports both letter likelihood and choice-text likelihood, because raw `A/B/C/D` scoring can be dominated by a letter prior.
  - Smoke suite covers science, history, math, logic, ARC-like science, GSM-like arithmetic, code, and JSON.
  - MMLU sample uses HuggingFace `cais/mmlu` when available.
- [x] Add MMLU-Redux as a stricter no-cheat evaluation gate.
  - Suite: `--suite mmlu_redux`.
  - User command: `./neurova.sh mamba3 bench-mmlu-redux`.
  - Default dataset: `edinburgh-dawg/mmlu-redux-2.0`.
  - Default filter: `error_type=ok` only.
  - This benchmark is never used for training, SFT, curriculum generation, or checkpoint selection by memorizing examples; it is only a held-out promotion gate.
- [x] Fix benchmark GPU ownership.
  - Root cause: `neuromamba/scripts/mamba3_infer_guard.sh` used `SIGSTOP`, which pauses compute but keeps the 2.4B training process' CUDA memory allocated. MMLU/MMLU-Redux then OOMed during Triton Mamba kernel autotune.
  - Fix: `neuromamba/scripts/mamba3_exclusive_gpu_guard.sh` stops autopilot/watchdog/training, runs the benchmark with free VRAM, then restarts the previously active loop.
  - Bench commands now use the exclusive guard: `bench-mcq`, `bench-mmlu`, `bench-mmlu-redux`, and `bench-suite`.
- [x] Add a combined benchmark suite.
  - Script: `neuromamba/scripts/mamba3_benchmark_suite.sh`.
  - User command: `./neurova.sh mamba3 bench-suite`.
  - Outputs under `neuromamba/runs/mamba3_benchmarks/<timestamp>/` and writes `neuromamba/runs/mamba3_benchmarks/latest_summary.json`.
- [x] Run the initial real intelligence baseline for the current v3 fast checkpoint.
  - Checkpoint: `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt`.
  - MCQ smoke, shuffled, choice-text scoring: `2/10 = 0.20`.
  - MCQ smoke, shuffled, letter scoring: `3/10 = 0.30`.
  - MMLU sample, `limit=50`, shuffled, choice-text scoring: `15/50 = 0.30`.
  - MMLU sample, `limit=50`, shuffled, letter scoring: `8/50 = 0.16`.
  - MMLU-Redux sample, `limit=100`, shuffled, `error_type=ok`, choice-text scoring: `29/100 = 0.29`.
  - MMLU-Redux sample, `limit=100`, shuffled, `error_type=ok`, letter scoring: `27/100 = 0.27`.
  - Interpretation: current v3 is a fast basic chat checkpoint, not a high-intelligence checkpoint. Do not claim MMLU competence or SOTA.
- [ ] Use benchmark improvement as a promotion gate, not training data.
  - Do not train on MMLU test/validation examples.
  - Increase intelligence through governed base pretraining, verifier-generated synthetic tasks, code/math/JSON/copy curricula, and held-out evals.
  - A future checkpoint must beat the current v3 baseline on MCQ smoke, MMLU sample, programmatic heldout, chat quality, and repetition collapse before promotion.

- [x] Re-scope the 24-hour deliverable to usable `Neurova-chat v1`, not 2B SOTA pretraining.
  - The 2.4B top-1 MoE line is preserved as a diagnostic pretraining artifact and is excluded from the 24-hour chat-deliverable path.
  - GPU priority for the next 24 hours is chat SFT candidates, automatic eval, decode tuning, and promotion/fallback packaging.
- [x] Add deterministic higher-quality Neurova-chat SFT corpus generator.
  - Script: `neuromamba/scripts/mamba3_generate_neurova_chat_sft.py`.
  - Default: 60k no-teacher records across identity, English QA, Korean QA, definitions, unknown/uncertainty behavior, simple reasoning, and anti-repetition prompts.
- [x] Add Neurova-chat v1 training and management scripts.
  - Training script: `neuromamba/scripts/mamba3_train_neurova_chat_v1.sh`.
  - Controller: `neuromamba/scripts/mamba3_chat_trainctl.sh`.
  - Default seed checkpoint: `neuromamba/runs/mamba3_neurova_speak_v1/sft.pt`, falling back to `neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt`.
  - Output checkpoint: `neuromamba/runs/mamba3_neurova_chat_v1/chat.pt`.
- [x] Add chat-specific quality gate.
  - Script: `neuromamba/scripts/mamba3_chat_quality_gate.py`.
  - Gate covers identity, Korea, unknown handling, definitions, math, Korean prompts, and anti-repetition.
- [x] Add 24-hour autonomous chat improvement loop.
  - Loop: `neuromamba/scripts/mamba3_chat_autoloop.sh`.
  - Controller: `neuromamba/scripts/mamba3_chat_autoloopctl.sh`.
  - The loop waits for in-flight chat work, evaluates, promotes passing checkpoints, then starts the next candidate from the best promoted checkpoint.
  - Current user-facing default in `neurova.sh` resolves `neuromamba/runs/mamba3_current/model.pt` first, then chat v1, then speak v1.
- [x] Promote first usable Neurova-chat v1 checkpoint.
  - Source: `neuromamba/runs/mamba3_neurova_chat_v1/chat.pt`.
  - Gate: `pass_rate=0.9333`.
  - Promoted target: `neuromamba/runs/mamba3_current/model.pt`.
- [ ] Let the autonomous chat loop continue through its scheduled trials.
  - Monitor: `./neurova.sh mamba3 status` or `neuromamba/scripts/mamba3_chat_autoloopctl.sh status` on `ml-dmc8`.
  - Do not resume 2.4B MoE pretraining while chat autoloop is using the GPU.
- [x] Pause and preserve the 2.4B max-parameter MoE line instead of continuing blind training.
  - Remote command used: `neuromamba/scripts/mamba3_moe24_trainctl.sh stop` and `watchdog-stop`.
  - Preserved checkpoint: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/base.pt`.
  - Last observed research status: loss improved to about `9.0566`, but decode remained collapsed; this line is not a speaking model.
- [x] Select the fastest viable speaking seed checkpoint.
  - Candidate selected: `neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt`.
  - Evidence: with conda env active, it generated a short English sentence for `Question: Who are you? Answer:` while other checked candidates were worse or mismatched.
- [x] Add deterministic no-teacher speaking SFT data generator.
  - Script: `neuromamba/scripts/mamba3_generate_neurova_speak_sft.py`.
  - Data: identity, basic English QA, simple reasoning, and Korean-English answers.
- [x] Add reproducible speaking-model training script.
  - Script: `neuromamba/scripts/mamba3_train_neurova_speak_v1.sh`.
  - Default output: `neuromamba/runs/mamba3_neurova_speak_v1/sft.pt`.
  - Uses answer-only SFT from the existing speak seed checkpoint, not raw from-scratch pretraining.
  - Default high-throughput local setting: `batch_size=32`, `steps=3000`, `lr=4e-5`, `grad_accum=1`.
  - This is a separate “speaks today” line and must not overwrite the 2.4B research checkpoint.
- [x] Add managed background controller for same-day speaking training.
  - Script: `neuromamba/scripts/mamba3_speak_trainctl.sh`.
  - Commands: `start`, `status`, `tail`, `stop`, `logs`.
  - Logs: `neuromamba/runs/mamba3_neurova_speak_v1/logs/*.log`.
- [ ] Run `neuromamba/scripts/mamba3_train_neurova_speak_v1.sh` on `ml-dmc8`.
- [ ] Test exact input/output samples:
  - `Who are you?`
  - `Where is Korea?`
  - `What is machine learning inference?`
  - `너는 누구야?`
  - `What should you do if you do not know?`
- [ ] Promote to `neuromamba/runs/mamba3_current/model.pt` only if it answers without repetition collapse.
- [ ] After today’s speaking checkpoint is usable, return to the canonical long-term intelligence recipe.
  - Source of truth: `neuromamba/configs/mamba3_english_top_intelligence_recipe.json`.
  - Do not mix same-day SFT success with from-scratch base-model quality claims.
  - Next long-term work: governed corpus expansion, active-compute architecture fork, optimizer-state-preserving base training, broad eval matrix, then staged SFT/preference/RLVR.

## Phase A: 16GB Train-Max

- [x] Keep default Mamba runtime interactive through `./neurova.sh mamba3`.
- [x] Add quality gate so collapsed checkpoints cannot be promoted.
- [x] Add promotion script for `neuromamba/runs/mamba3_current/model.pt`.
- [x] Add 16GB candidate presets:
  - `mimo-r4-tiny`
  - `mimo-r4-moe-260m`
  - `mimo-r4-16gb-120m`
  - `mimo-r4-16gb-180m`
  - `mimo-r4-paper-180m`
  - `mimo-r4-moe-520m`
  - `mimo-r4-moe-900m`
  - `mimo-r4-moe-1.1b`
  - `mimo-r4-moe-1.3b`
  - `mimo-r4-moe-1.7b`
  - `mimo-r4-moe-2.1b`
  - `mimo-r4-moe-2.3b`
  - `mimo-r4-moe-2.4b`
  - `mimo-r4-moe-2.5b`
  - `mimo-r4-moe-2.9b`
- [x] Run `neuromamba/scripts/mamba3_find_16gb_train_target.sh` on `ml-dmc8`.
- [x] Select the largest candidate whose forward and backward pass succeed under 16GB.
  - dense `mimo-r4-16gb-120m` and `mimo-r4-16gb-180m` fail backward on RTX 4080 due to TileLang dynamic shared memory.
  - sparse `mimo-r4-moe-260m` succeeds backward with about 284M total parameters and about 2.7GB peak VRAM in the canary.
- [x] Re-probe maximum trainable scale with true multi-step backward.
  - dense `mimo-r4-440m`, `mimo-r4-880m`, and `mimo-r4-1.5b` fail even at seq_len=128 batch_size=1 with TileLang dynamic shared memory `223904`; this is a kernel shared-memory limit, not ordinary VRAM shortage.
  - `mimo-r4-moe-520m` succeeds backward with 485,130,240 total parameters.
  - `mimo-r4-moe-900m` succeeds backward with 887,799,808 total parameters.
  - `mimo-r4-moe-900m` seq_len=2048 batch_size=4 can pass a one-step probe but OOMs during multi-step warm training.
  - Earlier max stable RTX 4080 config: `mimo-r4-moe-900m`, seq_len=2048, batch_size=3, grad_accum=1, bf16, AdamW8bit, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; peak VRAM is about 12.98GB and warm training throughput is about 24K-26K tok/s.
- [x] Re-probe maximum sparse-MoE parameter count on RTX 4080.
  - `mimo-r4-moe-1.1b`: 1,089,134,592 parameters; seq_len=2048 batch_size=3 passes, peak about 13.74GB, warm throughput about 23.4K tok/s.
  - `mimo-r4-moe-1.3b`: 1,290,469,376 parameters; seq_len=2048 batch_size=3 passes, peak about 14.49GB, warm throughput about 19.7K-21.7K tok/s.
  - `mimo-r4-moe-1.7b`: 1,693,138,944 parameters; batch_size=2 passes at about 12.79GB and 15.6K tok/s; batch_size=3 OOMs during backward.
  - `mimo-r4-moe-2.1b`: 2,095,808,512 parameters; batch_size=1 passes at about 12.51GB and 7.7K tok/s.
  - `mimo-r4-moe-2.3b`: 2,297,143,296 parameters; batch_size=1 passes at about 13.64GB and 7.1K tok/s.
  - `mimo-r4-moe-2.4b`: 2,397,810,688 parameters; batch_size=1 passes at about 14.21GB and 6.9K tok/s.
  - `mimo-r4-moe-2.5b`: 2,498,478,080 parameters; batch_size=1 fails Mamba-3 MIMO backward with CUDA OOM.
  - `mimo-r4-moe-2.9b`: 2,901,147,648 parameters; batch_size=1 fails AdamW8bit optimizer state allocation with CUDA OOM.
  - Current max-parameter RTX 4080 config: `mimo-r4-moe-2.4b`, seq_len=2048, batch_size=1, grad_accum=1, bf16, AdamW8bit, weight-only checkpoints.
  - Current faster throughput config: `mimo-r4-moe-1.3b`, seq_len=2048, batch_size=3.
- [x] Add maximum-size base training entrypoint.
  - Script: `neuromamba/scripts/mamba3_train_max_moe_base.sh`.
  - Default: `MODE=mimo-r4-moe-2.4b`, `BATCH_SIZE=1`, `SEQ_LEN=2048`, `LR=3e-5`, `--no-save-optimizer`.
  - Canary: `BASE_STEPS=20 SAVE_EVERY=0 EVAL_BATCHES=4 LR=3e-5 DATA_SEED=990001 neuromamba/scripts/mamba3_train_max_moe_base.sh`.
  - Result: checkpoint saved at `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/base.pt`; held-out loss 11.6719 over 4 batches; decode is random because this is only a from-scratch canary, not a trained model.
- [x] Add maximum-size weight-only continuation loop.
  - Script: `neuromamba/scripts/mamba3_continue_max_moe_base.sh`.
  - It is a thin wrapper around `neuromamba/scripts/mamba3_continue_clean_doc_base.sh` to avoid duplicating train/eval/decode logging logic.
  - Default: `MODE=mimo-r4-moe-2.4b`, `RUN_DIR=neuromamba/runs/mamba3_clean_doc_base_moe24_v1`, `BATCH_SIZE=1`, `LR=2e-5`, `STEPS=250`, `EVAL_BATCHES=16`, `NO_SAVE_OPTIMIZER=1`.
  - Verification run: `DATA_SEED=991001 STEPS=250 LR=2e-5 EVAL_BATCHES=16 neuromamba/scripts/mamba3_continue_max_moe_base.sh`.
  - Result: final train loss 10.5000, held-out doc loss 10.3281 over 16 batches, peak training VRAM 14.205GB, warm training throughput roughly 6.5K-11K tok/s.
  - Decode moved from random multilingual/code fragments to English frequency-token collapse: `<doc source="probe" domain="science"> The main idea is,,,,,, the the the ...`; this is progress from pure randomness but still not language-capable, not SFT-ready, and not promotable.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/continuations/20260613T010628Z_*`.
- [x] Add maximum-size until-gate loop and repetition-collapse guard.
  - Script: `neuromamba/scripts/mamba3_train_max_moe_until_gate.sh`.
  - It wraps the shared `neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh` with 2.4B defaults.
  - Gate now checks held-out loss, minimum generated token count, maximum repeated-word run, and minimum distinct generated words.
  - Default collapse thresholds: `MAX_REPEATED_WORD_RUN=12`, `MIN_DISTINCT_WORDS=8`.
  - Verification run: `MAX_ROUNDS=1 STEPS_PER_ROUND=500 LR=1.5e-5 EVAL_BATCHES=16 DATA_SEED_BASE=993000 neuromamba/scripts/mamba3_train_max_moe_until_gate.sh`.
  - Result: final train loss 9.6875, held-out doc loss 9.7656 over 16 batches, peak training VRAM 14.064GB, warm throughput roughly 8K-13K tok/s.
  - Decode remained collapsed: `<doc source="probe" domain="science"> The main idea is,, the the the ...`.
  - Collapse check on this decode reports `longest_repeated_word_run=94`, `distinct_words=9`, `collapsed=true`.
  - Gate failed correctly; continue base training only.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/continuations/20260613T011044Z_*` and `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/until_gate/summary.jsonl`.
- [x] Verify collapse-aware summary on the next 2.4B round.
  - Run: `MAX_ROUNDS=1 STEPS_PER_ROUND=500 LR=1.2e-5 EVAL_BATCHES=16 DATA_SEED_BASE=994000 neuromamba/scripts/mamba3_train_max_moe_until_gate.sh`.
  - Result: final train loss 9.2500, held-out doc loss 9.5781 over 16 batches, peak training VRAM 14.134GB, warm throughput roughly 9K-14K tok/s.
  - Decode remained collapsed: `<doc source="probe" domain="science"> The main idea is the the the ...`.
  - Summary now records collapse fields correctly: `collapsed=true`, `longest_repeated_word_run=96`, `distinct_words=9`, `passed=false`.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/continuations/20260613T011617Z_*`.
- [x] Tighten repetition gate to evaluate generated continuation only.
  - Root cause: the first collapse-aware gate counted prompt words such as `<doc source="probe" domain="science"> The main idea is` in `distinct_words`.
  - Fix: `neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh` now strips the prompt before calculating repeated-word runs and distinct generated words.
  - Verification on the previous decode changed the effective generated-word count from prompt-inflated `distinct_words=9` to continuation-only `distinct_words=1`, with `longest_repeated_word_run=96`.
- [x] Verify continuation-only collapse summary on 2.4B.
  - Run: `MAX_ROUNDS=1 STEPS_PER_ROUND=500 LR=1.0e-5 EVAL_BATCHES=16 DATA_SEED_BASE=995000 neuromamba/scripts/mamba3_train_max_moe_until_gate.sh`.
  - Result: final train loss 9.1875, held-out doc loss 9.5195 over 16 batches, peak training VRAM 14.205GB, warm throughput roughly 9K-16K tok/s.
  - Decode remained collapsed: `<doc source="probe" domain="science"> The main idea is the the the ...`.
  - Summary records `repetition_scope=continuation_only`, `distinct_words=1`, `longest_repeated_word_run=96`, `collapsed=true`, `passed=false`.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/continuations/20260613T012058Z_*`.
- [x] Confirm 2.4B collapse is general, not a single-prompt artifact.
  - Run: `python neuromamba/scripts/mamba3_eval_raw_decode_set.py --mode mimo-r4-moe-2.4b --tokenizer llama31 --checkpoint neuromamba/runs/mamba3_clean_doc_base_moe24_v1/base.pt`.
  - Result: science/history/code/math raw document starts all collapsed.
  - Summary: `collapsed_prompts=4`, `passed=false`.
  - Artifact: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/raw_decode_set/20260613T013052Z.jsonl`.
- [x] Add first sufficient maximum-parameter base-training block.
  - Script: `neuromamba/scripts/mamba3_train_max_moe_100m_block.sh`.
  - Detached launcher: `neuromamba/scripts/mamba3_launch_max_moe_100m_background.sh`.
  - Status helper: `neuromamba/scripts/mamba3_status_max_moe_100m.sh`.
  - Status helper reports active PID, GPU usage, current round, trained/planned tokens, progress percentage, last-100-step average token/s, ETA, latest log, gate trend, and gate summary.
  - Managed train controller: `neuromamba/scripts/mamba3_moe24_trainctl.sh`.
  - Controller commands: `start`, `resume`, `adopt`, `status`, `decision`, `router-diagnostics`, `tail`, `stop`, `logs`, `watchdog-start`, `watchdog-status`, `watchdog-stop`.
  - Controller artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/control/current.log`, `current.log.path`, `train.pid`, `wrapper.pid`, `state.json`.
  - Default model: `mimo-r4-moe-2.4b`.
  - Default planned tokens: `100,000,000`.
  - Calculation: `2048 tokens/step * 2000 steps/round * 25 rounds = 102,400,000 planned tokens`.
  - Default optimizer path: bf16, AdamW8bit, batch_size=1, weight-only checkpoints, shuffled v3 JSONL document continuation.
  - Gate remains held-out raw document loss plus continuation-only repetition collapse.
- [ ] Let the active 100M-token 2.4B base block finish on `ml-dmc8`.
  - Started: `20260613T013135Z`.
  - Log: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/long_blocks/20260613T013135Z_100m.log`.
  - Command record: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/long_blocks/20260613T013135Z_100m.cmd`.
  - First observed status: round 1 training was active, peak VRAM about 13.1GB-14.5GB, GPU utilization about 60%+, token throughput mostly 9K-16K tok/s.
  - Round 1 result: final train loss `9.1875`, held-out raw doc loss `9.36328125`, `collapsed=true`, continuation-only `distinct_words=1`, `longest_repeated_word_run=96`; gate failed correctly.
  - Round 2 automatically started with `data_seed=997002`; continue the 100M block.
  - Round 2 observed progress: `5,943,296 / 102,400,000` tokens, `5.80%`, last-100-step average about `12.8K tok/s`, ETA about `2h05m`.
  - Round 2 result: final train loss `8.8750`, held-out raw doc loss `9.3046875`, `collapsed=true`, continuation-only `distinct_words=1`, `longest_repeated_word_run=96`; gate failed correctly.
  - Round 3 automatically started with `data_seed=997003`.
  - Latest observed progress: round `3/25`, `8,593,408 / 102,400,000` tokens, `8.39%`, last-100-step average about `13.4K tok/s`, ETA about `1h56m`.
  - Current loss trend is improving but still far from the base gate: `9.4921875 -> 9.36328125 -> 9.3046875`.
  - Status trend now reports `records=6`, `loss_first=9.765625`, `loss_latest=9.304688`, `loss_best=9.304688`, `loss_delta_first_to_latest=-0.460938`, `passed_records=0`.
  - Found a data-performance issue during round 3: some v3 JSONL document-continuation records are extremely long, up to about `679,695` chars; one tokenizer call warned about `436010 > 131072` tokens and caused a local speed drop.
  - Fix for subsequent rounds: `neuromamba.data.iter_packed_token_batches` now adaptively splits long texts before tokenization, `train-packed` has `--max-text-chars` default `65536` and `--max-text-tokens` default `120000`, and `neuromamba/scripts/mamba3_continue_clean_doc_base.sh` passes `MAX_TEXT_CHARS`/`MAX_TEXT_TOKENS`.
  - This preserves raw document-continuation training while avoiding single giant tokenizer inputs. It does not alter the currently running Python process, but it applies to the next round because each round launches a fresh `train-packed` process.
  - Status helper now reports `tokenizer_length_warnings`; one in-flight round used `--max-text-chars 131072`, but the controller default is now `65536` plus `--max-text-tokens 120000`.
  - Round 3 result: held-out raw doc loss `9.248046875`, `collapsed=true`, continuation-only `distinct_words=1`, `longest_repeated_word_run=96`; gate failed correctly.
  - Round 4 automatically started with `data_seed=997004` and a token-length guard visible in the process command.
  - Current loss trend is still improving but collapse remains: `9.4921875 -> 9.36328125 -> 9.3046875 -> 9.248046875`.
  - The already-running long block has been adopted by `neuromamba/scripts/mamba3_moe24_trainctl.sh adopt` without interrupting training. Current adopted log: `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/long_blocks/20260613T013135Z_100m.log`.
  - Watchdog is running under `neuromamba/scripts/mamba3_moe24_trainctl.sh watchdog-loop`; if wrapper/train processes disappear before the planned block completes, it computes remaining rounds from the log and restarts detached training from the latest weight checkpoint.
  - Round 4 result: held-out raw doc loss `9.193359375`, `collapsed=true`, continuation-only `distinct_words=3`, `longest_repeated_word_run=85`; gate failed correctly but repetition is slightly less degenerate than earlier `distinct_words=1`, run `96`.
  - SOTA-oriented post-100M decision gate is not duplicated here. Source of truth: `neuromamba/configs/mamba3_english_top_intelligence_recipe.json` field `post_100m_local_decision_gate`; operator view: `neuromamba/scripts/mamba3_moe24_trainctl.sh decision`.
  - Do not start SFT or QA training while this item is open.
- [x] Add a non-interrupting post-100M decision report.
  - Script: `neuromamba/scripts/mamba3_post100m_decision.py`.
  - Operator command: `neuromamba/scripts/mamba3_moe24_trainctl.sh decision`.
  - It reads `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/until_gate/summary.jsonl`, estimates 100M progress, and returns one of:
    - `in_progress`: keep the current block running.
    - `base_gate_passed`: extend raw base training to `300M -> 1B` local tokens before SFT.
    - `collapse_free_but_undertrained`: continue base training, not SFT.
    - `loss_down_but_collapse_persists`: after 100M, stop blind continuation and diagnose active compute/router/decode.
    - `loss_plateau_or_unclear`: run the diagnostic matrix before more tokens.
  - This script never stops training, starts training, or promotes a checkpoint.
- [ ] Convert the SOTA/dense-intelligence requirement into an active-compute architecture fork after the current 100M block finishes.
  - Keep the current `2.4B total / top-1 MoE / 100M token` experiment running as the diagnostic baseline.
  - Do not expect dense 2.4B-class intelligence from top-1 sparse total parameters.
  - Candidate fork A: high-active dense-ish `900M/1.3B` throughput baseline with optimizer-state preservation if feasible.
  - Candidate fork B: trainable/offloaded dense `1.3B-2.7B` if official Mamba-3 backward kernel limits can be bypassed by offload, smaller head/state configuration, or kernel changes.
  - Candidate fork C: top-2/top-4 sparse MoE with load-balancing auxiliary loss, router entropy monitoring, and expert-usage histogram gates.
  - Required preflight before any long run: tokenizer boundary check, packing check, label-shift check, full-forward vs recurrent decode parity, CUDA graph/cache parity, and optimizer-state strategy.
  - Required eval matrix: held-out perplexity, multi-prompt continuation, repetition/collapse, MMLU-style knowledge, GSM/math exact answer, code/unit tests, copy/retrieval, JSON/field extraction, long-context needle/passkey, and later chat/tool QA.
  - Dense-class local ambition requires at least `10B-50B` high-quality tokens; SOTA proximity requires multi-GPU and hundreds of billions to trillions of tokens.
- [x] Add post-100M MoE/router diagnostic artifact generation.
  - Script: `neuromamba/scripts/mamba3_moe_router_diagnostics.py`.
  - Operator command: `neuromamba/scripts/mamba3_moe24_trainctl.sh router-diagnostics`.
  - It logs per-layer expert counts, router entropy, top expert share, expert skew, mean top route probability, and estimated active parameters per token.
  - It saves artifacts under `neuromamba/runs/mamba3_clean_doc_base_moe24_v1/router_diagnostics/`.
  - The controller refuses to run this while training is active unless `FORCE_ROUTER_DIAGNOSTICS=1`, to avoid GPU OOM.
- [ ] Add train-time router regularization for sparse MoE if diagnostics show expert collapse.
  - Add z-loss/load-balancing auxiliary loss to the training objective.
  - Gate: if one or a few experts dominate, do not extend sparse top-1 training; switch to load-balanced top-2/top-4 or a higher-active dense-ish baseline.
- [ ] Fix recurrent cache parity and CUDA graph compatibility before promoting sparse MoE for fast interactive inference.
- [x] Train the selected candidate through clean raw-document base sanity gates.
  - Script: `neuromamba/scripts/mamba3_train_clean_doc_base.sh`.
  - Run: `neuromamba/runs/mamba3_clean_doc_base_moe900_v1`.
  - Stage 0 overfit sanity passed: 64-document subset loss reached 1.0986 after 120 steps.
  - Stage 1 raw-document CLM from scratch reached held-out validation loss 6.4404 after 1200 steps.
  - Optimizer-state resume at batch_size=3 OOMed, so the continuation loaded weights with `--no-resume` and reset optimizer state.
  - Optimizer-reset continuation reached checkpoint step 500 and held-out validation loss 6.3330.
  - A second optimizer-reset 500-step continuation at `LR=2e-5` held the same 12.98GB peak and improved held-out validation loss to 6.3037.
  - Decode probe still fails language quality: `<doc source="probe" domain="science"> The main idea is to be a new` then EOS. This is not SFT-ready and not promotable.
- [x] Add standard clean-doc continuation/eval loop.
  - Script: `neuromamba/scripts/mamba3_continue_clean_doc_base.sh`.
  - Default: weight-only restart with `--no-resume` and `--no-save-optimizer`.
  - Reason: optimizer-state resume at the max stable batch_size=3 OOMs; weight-only continuation is the selected 16GB path.
  - Verification run: `STEPS=250 SAVE_EVERY=250 LR=1.5e-5 neuromamba/scripts/mamba3_continue_clean_doc_base.sh`.
  - Result: held-out validation loss 6.2969, peak VRAM 12.98GB, warm training throughput about 24K-26K tok/s.
  - Checkpoint size dropped from about 3.4GB to 1.7GB because optimizer state is no longer saved.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe900_v1/continuations/20260613T002911Z_*`.
  - Decode remains too short: `<doc source="probe" domain="science"> The main idea is to be a new`; still not promotable.
- [x] Expand raw document-continuation corpus without overwriting v1 evidence.
  - Command: `MAX_DOCS=20000 MAX_BYTES=500000000 TECH_RECORDS=0 CLEAN_RECORDS=0 DOC_CORPUS_OUT=neuromamba/data/base_doc_continuation_v2.jsonl TRAIN_OUT=neuromamba/data/splits/base_doc_cont_v2_train.txt VALID_OUT=neuromamba/data/splits/base_doc_cont_v2_valid.txt neuromamba/scripts/mamba3_expand_governed_base.sh`.
  - Sources: FineWeb-Edu 20,000 docs, DCLM 20,000 docs, OpenWebMath 20,000 docs, arXiv abstracts 20,000 docs.
  - Written doc-continuation records: 79,256; skipped short or instruction-like records: 744.
  - Source-stratified split: 77,673 train records, 1,583 validation records.
  - Size: `neuromamba/data/base_doc_continuation_v2.jsonl` about 365MB, train split about 346MB, valid split about 7.5MB.
  - Manifest estimate: about 312.5M bootstrap tokens across local data files; still bootstrap-only, not real 100B+ pretraining scale.
- [x] Verify 900M continuation on v2 corpus.
  - Baseline v2 held-out loss before v2 continuation: 6.3535.
  - Run: `TRAIN_DATA=neuromamba/data/splits/base_doc_cont_v2_train.txt VALID_DATA=neuromamba/data/splits/base_doc_cont_v2_valid.txt STEPS=250 SAVE_EVERY=250 LR=1.5e-5 neuromamba/scripts/mamba3_continue_clean_doc_base.sh`.
  - Result: v2 held-out loss 6.3311, peak VRAM 12.98GB, warm training throughput about 24K-26K tok/s.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe900_v1/continuations/20260613T003427Z_*`.
  - Decode is unchanged and too short: `<doc source="probe" domain="science"> The main idea is to be a new`; still not promotable.
- [x] Add gate-driven repeated clean-doc training loop.
  - Script: `neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh`.
  - Gate: held-out doc loss must be below target and raw decode must produce at least the minimum new-token count.
  - Verification run: `MAX_ROUNDS=1 STEPS_PER_ROUND=500 LR=1.2e-5 TRAIN_DATA=neuromamba/data/splits/base_doc_cont_v2_train.txt VALID_DATA=neuromamba/data/splits/base_doc_cont_v2_valid.txt neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh`.
  - Result: v2 held-out loss 6.3174, new_tokens 4, gate failed as expected.
  - Summary artifact: `neuromamba/runs/mamba3_clean_doc_base_moe900_v1/until_gate/summary.jsonl`.
- [x] Fix repeated weight-only restart data-order bias.
  - Root cause: `train-packed` used deterministic corpus order; because 900M continuation uses `--no-resume`, each run restarted from the beginning of the corpus and over-sampled the same early data.
  - Fix: `neuromamba.data.iter_packed_token_batches` now supports seeded text shuffling.
  - CLI: `train-packed --shuffle-texts --data-seed N`.
  - `neuromamba/scripts/mamba3_continue_clean_doc_base.sh` now defaults to `SHUFFLE_TEXTS=1` and records `data_seed` in metadata.
  - `neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh` uses a different deterministic seed per round.
  - Verification run: `TRAIN_DATA=neuromamba/data/splits/base_doc_cont_v2_train.txt VALID_DATA=neuromamba/data/splits/base_doc_cont_v2_valid.txt STEPS=250 SAVE_EVERY=250 LR=1e-5 DATA_SEED=777001 neuromamba/scripts/mamba3_continue_clean_doc_base.sh`.
  - Result: v2 held-out loss 6.3281, new_tokens 7, peak VRAM 12.98GB. Decode remains too short: `<doc source="probe" domain="science"> The main idea is to be a few of the`; still not promotable.
- [x] Fix document-boundary corruption in split files.
  - Root cause: source-stratified splits wrote raw text records with `"\n".join(lines)`, but doc-continuation records contain internal newlines. The `.txt` train/valid files therefore split one `<doc>...</doc>` sample into many independent line samples.
  - Evidence: `neuromamba/data/splits/base_doc_cont_v2_train.txt` had 77,673 `<doc>` records but 2,561,832 physical lines; the first sample read by the trainer was only `<doc ...>\n` followed by separate body lines.
  - Fix: `neuromamba/scripts/mamba3_make_source_stratified_splits.py` and `neuromamba/scripts/mamba3_make_splits.py` now write JSONL records when the output path ends in `.jsonl`.
  - New canonical split: `neuromamba/data/splits/base_doc_cont_v3_train.jsonl` and `neuromamba/data/splits/base_doc_cont_v3_valid.jsonl`.
  - v3 split preserves one full document per JSONL row: 77,673 train records and 1,583 valid records.
  - True v3 held-out loss before recovery: 7.1006. The old 6.3-range loss was a line-fragment metric and should not be used as the base objective gate.
- [x] Verify recovery on corrected v3 split.
  - Run: `STEPS=250 SAVE_EVERY=250 LR=1e-5 DATA_SEED=880001 neuromamba/scripts/mamba3_continue_clean_doc_base.sh`.
  - Result: true v3 held-out loss improved from 7.1006 to 6.9697.
  - Decode improved but remains too short: `<doc source="probe" domain="science"> The main idea is to be a new, but the first time the`, 11 new tokens.
  - Artifacts: `neuromamba/runs/mamba3_clean_doc_base_moe900_v1/continuations/20260613T004652Z_*`.
  - Status: correct objective is now restored; continue training on v3 only. Do not compare future results against the broken v2 `.txt` line-fragment loss.
- [ ] Continue clean raw-document pretraining until held-out doc loss is below 5.0 and raw continuations stop collapsing.
  - For maximum parameter count, continue `neuromamba/scripts/mamba3_continue_max_moe_base.sh`.
  - For governed repeated runs, use `neuromamba/scripts/mamba3_train_max_moe_until_gate.sh`; a decode with long repeated-word runs must never pass even if it produces many tokens.
  - Current 2.4B trajectory: held-out doc loss `11.6719 -> 10.3281 -> 9.7656 -> 9.5781 -> 9.5195`; still collapsed on greedy raw decode.
  - For faster iteration and more tokens/day, continue `neuromamba/scripts/mamba3_continue_clean_doc_base.sh` on the 900M or run the 1.3B throughput candidate after a clean from-scratch base run.
- [ ] Promote only if `quality-gate` and decode benchmark pass.

## Phase A2: Stability Ladder For Intelligence Defects

- [x] Add SISO -> MIMO r2 -> MIMO r4 stability ladder script.
  - `neuromamba/scripts/mamba3_train_stability_ladder.sh`
- [x] Add hybrid GQA attention presets.
  - `mimo-r2-attn-tiny`
  - `mimo-r4-attn-tiny`
- [x] Add recall-hybrid presets with GQA attention plus learnable meta tokens.
  - `mamba3-recall-r2-tiny`
  - `mamba3-recall-r4-tiny`
- [x] Split SFT formatting from programmatic curriculum formatting.
  - Natural SFT defaults to answer-only text.
  - Programmatic SFT mixing defaults to disabled because direct tiny-SFT mixing caused `Instruction:`/colon repetition collapse.
- [ ] Run the ladder and compare loss, exact-match curriculum behavior, quality gate, speed, and VRAM.
- [x] Run answer-only v5 ladder for `mimo-r2` vs `mamba3-recall-r2-tiny`.
  - SFT split: 58 natural answer-only records, 0 programmatic records.
  - `mimo-r2`: SFT loss 10.59 -> 2.87, validation loss 4.79, quality gate failed, programmatic exact-match 0/32.
  - `mamba3-recall-r2-tiny`: SFT loss 10.27 -> 2.19, validation loss 4.70, quality gate failed, programmatic exact-match 0/32.
  - Recall-hybrid produced two good English continuations but still failed two prompts and cannot be promoted.
- [x] Add separated base -> recall/copy curriculum -> answer-only SFT pipeline.
  - `CURRICULUM_STEPS>0` now writes `base.pt`, `curriculum.pt`, and `sft.pt`.
  - Curriculum exact-match is evaluated before and after SFT.
- [x] Run `mamba3-recall-r2-tiny` separated curriculum canary.
  - Base: 32 steps, loss 11.88 -> 8.63, base validation loss 8.26.
  - Recall/copy curriculum: 80 steps, loss 9.09 -> 3.10, curriculum validation loss 3.13.
  - Curriculum exact-match improved from 0/32 to 5/32; joint recall reached 4/4 and MC routing 1/4.
  - Post answer-only SFT exact-match was 4/32, but English quality gate still failed due to prompt/style contamination.
- [x] Add deterministic clean-English SFT generator.
  - `neuromamba/scripts/mamba3_generate_clean_english_sft.py`
  - Clean recovery SFT reduced teacher-forced loss but still failed autoregressive quality.
- [x] Sweep clean recovery SFT length from the curriculum checkpoint.
  - 20/50/100 saved-step sweep all failed quality gate.
  - Exact-match stayed at 2/16 in the quick sweep, preserving only joint-recall.
  - Conclusion: more SFT templates alone are not enough; the model needs longer governed base training and better interleaved curriculum before SFT.
- [ ] Promote MIMO r4 or hybrid r4 only if it beats the simpler stages without collapse.
- [x] Probe `mimo-r4-attn-tiny`.
  - Forward/backward passes on RTX 4080.
  - Recurrent decode-step parity fails, so it is a trainable research candidate, not a fast runtime candidate.

## Phase B: Data Quality Before More Parameters

- [x] Canonicalize the English-first top-intelligence data/post-training recipe without duplicating it across docs.
  - Single source of truth: `neuromamba/configs/mamba3_english_top_intelligence_recipe.json`.
  - Governance schema: `neuromamba/configs/mamba3_data_governance_schema.json`.
  - Required per-sample fields are represented there: `source`, `license`, `language`, `domain`, `quality_score`, `toxicity_score`, `pii_score`, `dedup_hash`, `benchmark_contamination_flag`, `teacher_model`, `generation_date`.
  - Recipe covers tokenizer target, base-pretrain token budgets for 300M/1B/3B-8B/7B/30B+, Nemotron-CC/DCLM/FineWeb/FineWeb-Edu/Dolma mixtures, continued pretrain, reasoning midtraining, SFT, preference optimization, long-context curriculum, source URLs, and the 100M local decision gate.
  - Do not duplicate the full recipe in this TODO; update the JSON recipe first, then keep TODO items as executable work.
- [ ] Implement governed corpus expansion toward the canonical recipe.
  - Add or complete raw-compatible downloaders/manifests for Nemotron-CC, DCLM, FineWeb/FineWeb-Edu, Dolma, Nemotron-CC-Math, OpenThoughts3, NuminaMath, OpenCodeInstruct, Tulu3, SmolTalk2, Magpie/LongMagpie, HelpSteer2, and Skywork/SynPref where license and access permit.
  - Every produced JSONL must pass `neuromamba/scripts/mamba3_validate_governance.py`.
  - Training samples with `benchmark_contamination_flag=true`, forbidden/unknown license, high PII, or high toxicity remain blocked.
- [ ] Build SOTA-oriented eval matrix before post-training.
  - Required gates: held-out loss/perplexity, multi-prompt continuation, repetition/collapse, copy/retrieval, JSON/field extraction, code unit tests, math exact-answer, long-context needle/passkey, and later chat/tool QA.
  - SFT/DPO/RLVR may start only after a base checkpoint is collapse-free on raw continuation.
- [x] Governed FineWeb-Edu sample.
- [x] Governed DCLM sample.
- [x] Add raw document-continuation corpus builder for early pretraining.
  - Script: `neuromamba/scripts/mamba3_build_doc_continuation_corpus.py`.
  - Format: `<doc source="..." domain="...">...</doc>` with tokenizer EOS at sample boundary.
  - Policy: early base pretraining must not use QA/chat templates or `Instruction:/Answer:` loss masking.
- [x] Wire document-continuation format into governed base expansion.
  - `neuromamba/scripts/mamba3_expand_governed_base.sh` now defaults to `DOC_CONTINUATION=1`.
  - `DOC_CONTINUATION=0` keeps the old mixed split path for diagnostics only.
- [x] Run first raw document-continuation canary on `ml-dmc8`.
  - Corpus: `neuromamba/data/base_doc_continuation_v1.jsonl`.
  - Records: 39,647 written, 353 short or instruction-like records skipped.
  - Split: `neuromamba/data/splits/base_doc_cont_train.txt` 38,856 records, `neuromamba/data/splits/base_doc_cont_valid.txt` 791 records.
  - Run: `neuromamba/runs/mamba3_doc_cont_v1/base_doc.pt`.
  - Training: `mimo-r4-tiny`, Llama-3.1 tokenizer, bf16, AdamW8bit, seq_len=128, grad_accum=4, 800 steps.
  - Validation loss: 10.9063 before -> 7.3945 after.
  - Speed: recurrent decode warmup-excluded about 108.5 new tok/s on RTX 4080.
  - Status: not promotable; generation still collapses into numeric repetition.
- [ ] Scale the corrected base run before SFT.
  - Increase context after base loss stabilizes: 128 -> 512 -> 1024 -> 2048.
  - Increase effective batch/GPU utilization because the canary used only about 0.94GB VRAM.
  - Do not run QA/chat SFT until raw-document continuation samples stop collapsing.
- [x] Add no-teacher programmatic weakness-correction curriculum.
  - copy, phonebook lookup, joint recall, MC routing, JSON field extraction, code variable tracing, arithmetic, needle-in-haystack.
- [x] Add exact-match eval for the no-teacher curriculum.
  - `neuromamba/scripts/mamba3_eval_programmatic.py`
- [x] Run baseline and short correction candidate eval.
  - Stable tiny baseline: 0/16 exact-match.
  - 30-step programmatic correction candidate: loss decreased but still 0/16 exact-match; English quality gate stayed green.
- [x] Identify SFT contamination failure mode.
  - v3 with 120 programmatic samples in tiny SFT produced repeated `Instruction:` output.
  - v4 with 30 programmatic samples still produced colon repetition.
  - Fix: natural SFT answer-only, programmatic curriculum separated from the short SFT gate.
- [ ] Train longer and compare SISO/r2/r4/hybrid on exact-match before any promotion.
- [ ] Run a longer base-first `mamba3-recall-r2-tiny` training pass before any more SFT.
  - Target: at least thousands of base steps on governed English text before recall curriculum.
  - Gate: quality must improve before SFT; do not use SFT to compensate for an unstable base model.
- [x] Run first base-first `mamba3-recall-r2-tiny` governed training pass.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1`.
  - Base: 800 steps on governed FineWeb-Edu sample, DCLM sample, and deterministic clean English.
  - Base validation loss: 7.1191.
  - Base quality gate failed: outputs were grammatical fragments such as `The main idea is a time of the body.`
  - Normal LM recall curriculum: 200 steps, validation loss 1.875, but exact-match stayed 0/64 and English output became curriculum-contaminated.
- [x] Add masked answer-only curriculum loss.
  - CLI: `train-answer`, `eval-answer-loss`.
  - Reproducible script option: `CURRICULUM_LOSS=answer neuromamba/scripts/mamba3_train_base_first_recall.sh`.
  - Manual canary from the base checkpoint reduced answer validation loss to 2.9478.
  - Programmatic exact-match improved to 9/64:
    - joint recall: 7/8
    - multiple-choice routing: 2/8
    - copy, phonebook, JSON extraction, code tracing, arithmetic, needle retrieval: 0/8 each
  - English quality gate still failed. This checkpoint is not promotable.
- [x] Add interleaved base + masked-answer multitask training.
  - CLI: `train-multitask`.
  - Script path: `CURRICULUM_LOSS=multitask neuromamba/scripts/mamba3_train_base_first_recall.sh`.
  - 5-step GPU canary passed with AdamW8bit, bf16, recall-hybrid, base loss plus answer loss in one optimizer step.
  - Root cause addressed: sequential curriculum over-specializes the tiny model and either learns prompt/form tokens under LM loss or damages general English under answer-only loss.
  - Next gate: maintain/improve base English quality while raising copy/phonebook/json/needle exact-match above zero.
- [x] Run first multitask comparison from the 800-step base checkpoint.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/curriculum_multitask.pt`.
  - Setup: base_accum=3, answer_accum=1, 300 steps.
  - Answer validation loss: 3.7903.
  - Base validation loss: 7.0352.
  - Programmatic exact-match: 11/64, from joint recall 7/8 and multiple-choice routing 4/8 only.
  - English quality gate failed with numeric/repetition contamination.
  - Conclusion: multitask preserves base loss better than answer-only but does not fix exact copy/lookup.
- [x] Add hard-recall curriculum shard generation.
  - `neuromamba/scripts/mamba3_generate_programmatic_curriculum.py --tasks copy,phonebook_lookup,json_field_extraction,needle_in_haystack`.
  - `CURRICULUM_TASKS=...` is wired into the base-first script.
- [x] Run hard-recall multitask comparison from the 800-step base checkpoint.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/curriculum_hard_multitask.pt`.
  - Setup: hard tasks only, base_accum=2, answer_accum=4, 300 steps.
  - Hard answer validation loss: 4.1729.
  - Hard-only exact-match: 0/64.
  - Full curriculum exact-match: 0/64.
  - English quality gate failed harder, producing short numeric continuations such as `a 2.`
  - Conclusion: the current tiny base is too weak for aggressive hard-recall specialization; do not promote.
- [x] Add staged recall curriculum levels.
  - `neuromamba/scripts/mamba3_generate_programmatic_curriculum.py --difficulty easy`.
  - Easy variants exist for copy, phonebook lookup, JSON field extraction, and needle retrieval.
  - `CURRICULUM_DIFFICULTY=easy` is wired into the base-first script.
  - Level 1 is now available as short copy, 2-entry phonebook, 2-field JSON, and short needle.
  - Level 2 remains the normal structured curriculum.
  - Level 3 remains the hard-recall shard.
  - Gate each level by exact-match before increasing difficulty.
- [x] Run easy staged recall canary from the 800-step base checkpoint.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/curriculum_easy_multitask.pt`.
  - Setup: easy copy/phonebook/json/needle, base_accum=3, answer_accum=1, 300 steps.
  - Easy answer validation loss: 5.5176.
  - Easy exact-match: 0/64.
  - English quality gate failed with numeric contamination.
  - Conclusion: even easy exact copy/lookup does not work from the current base; the next bottleneck is base undertraining and/or recall-path architecture, not task difficulty alone.
- [ ] Run a longer base-only training pass before the next recall curriculum.
  - Current 800-step base loss and quality are not enough to support exact recall training.
  - Do not run more hard curriculum until base quality passes.
- [x] Add reproducible base-only continuation script.
  - `neuromamba/scripts/mamba3_continue_base_training.sh`.
  - Starts from `START_CHECKPOINT`, writes/resumes `CHECKPOINT`, and runs validation loss plus English quality gate.
  - Default continuation: `EXTRA_STEPS=2400`, `LR=1e-4`, `GRAD_ACCUM_STEPS=4`, AdamW8bit, bf16.
- [x] Run first longer base-only continuation.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/base_long.pt`.
  - Continued from step 800 to step 3200.
  - Validation loss improved from 7.1191 to 6.1875.
  - Quality gate still failed:
    - `The main idea is the first, then checks whether the answer is that is a little bit of the 2th century.`
    - `A good teacher should be a little bit of the same time.`
  - Conclusion: longer base training helps loss but is still not sufficient; do not promote.
- [x] Expand governed base corpus before another long base run.
  - Current base split is only about 15MB and repeats heavily over long training.
  - Add larger governed English shards or more local deterministic clean English before running another base continuation.
- [x] Add reproducible governed base expansion script.
  - `neuromamba/scripts/mamba3_expand_governed_base.sh`.
  - Increases FineWeb-Edu/DCLM streaming budgets, regenerates a clean-English supplement, and writes `neuromamba/data/splits/base_expanded_train.txt` plus `neuromamba/data/splits/base_expanded_valid.txt`.
- [x] Run governed corpus expansion.
  - Command: `MAX_DOCS=10000 MAX_BYTES=250000000 CLEAN_RECORDS=4000 neuromamba/scripts/mamba3_expand_governed_base.sh`.
  - FineWeb-Edu: 10,000 documents, 47,357,610 text bytes.
  - DCLM: 10,000 documents, 55,873,154 text bytes.
  - Clean deterministic English: 4,000 records.
  - Split: `base_expanded_train.txt` 23,520 records, `base_expanded_valid.txt` 480 records.
  - Manifest: 23 files, 130,870,470 bytes, estimated 32,586,056 tokens; still bootstrap-only, not real pretraining scale.
- [x] Run expanded base-only continuation.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/base_expanded_long.pt`.
  - Continued from step 3200 to step 5600 on the expanded governed split.
  - Validation loss improved from 6.1875 to 5.6631.
  - Quality gate still failed:
    - `The main idea is the most important thing to do with the right.`
    - `A good teacher should be a good way to get a lot of the world.`
    - `In simple words, science is a good way to get a lot of the world.`
  - Warmup-excluded decode benchmark: about 103 new tok/s on RTX 4080, bf16, recurrent step decode, `top_k=1`.
  - `--cuda-graph` currently falls back to non-graph decode for this recall-hybrid checkpoint.
  - Conclusion: expanded data improves base loss and grammar, but the checkpoint is still semantically weak and not promotable.
- [x] Add math/science/technical shards to the governed base expansion path.
  - `open-web-math/open-web-math` is enabled as a governed math shard.
  - `neuromamba/scripts/mamba3_generate_technical_bootstrap.py` creates deterministic no-teacher code/math/science/structured-data records.
  - `ccdv/arxiv-summarization` article text was tested and rejected for default training because its `@xmath`/`@xcite` artifacts leaked directly into generation.
  - Science source is switched to `gfissore/arxiv-abstracts-2021` abstracts for the next split.
- [x] Run math/science/technical base-only continuation before source cleanup.
  - Corpus used for this run: FineWeb-Edu 10,000 docs, DCLM 10,000 docs, OpenWebMath 10,000 docs, ccdv arXiv article 7,419 docs, clean English 4,000 records, technical bootstrap 4,000 records.
  - Manifest: 26 files, 469,422,307 bytes, estimated 117,045,329 tokens; still bootstrap-only.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/base_tech_math_science_long.pt`.
  - Continued from step 5600 to step 8000.
  - Validation loss improved from 5.6631 to 4.3130.
  - Quality gate still failed and exposed arXiv article artifact contamination:
    - `The main idea is the following lemma :    1.`
    - `A good teacher should have a non - zero attack of the @xmath4-@xmath4-@xmath4-@xmath4.`
  - Warmup-excluded decode benchmark: about 99.7 new tok/s on RTX 4080, bf16, recurrent step decode, `top_k=1`.
  - Conclusion: broader data improves LM loss strongly, but raw artifact-heavy science articles are harmful for chat quality. Do not promote.
- [x] Rebuild expanded split with arXiv abstracts instead of artifact-heavy article text.
  - Cleaned split inputs: FineWeb-Edu 10,000 docs, DCLM 10,000 docs, OpenWebMath 10,000 docs, arXiv abstracts 10,000 docs, clean English 4,000 records, technical bootstrap 4,000 records.
  - Split: `base_expanded_train.txt` 47,040 records, `base_expanded_valid.txt` 960 records.
  - Manifest: 27 files, 481,703,861 bytes, estimated 120,115,717 tokens; still bootstrap-only.
- [x] Continue base-only training from `base_tech_math_science_long.pt` on the cleaned split.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/base_clean_abstract_long.pt`.
  - Continued from step 8000 to step 9200 with `LR=6e-5`.
  - Validation loss improved from 4.3130 to 4.0815.
  - Quality gate still failed:
    - `The main idea is the following:`
    - `A good teacher should show that's why the "temperature_c": "temperature_c": "temperature_c": "humidity_percent": 1.`
    - `In simple words, science is the most common, and the most common, ...`
  - `@xmath` artifact was removed, but deterministic structured-data patterns now leak into general English.
  - Warmup-excluded decode benchmark: about 78 new tok/s, but the sample stopped after 4 new tokens so this is not a stable speed comparison.
  - Conclusion: cleaned data lowers loss and removes arXiv artifacts, but the base is still not conversationally stable and must not be promoted.
- [ ] Reduce deterministic technical/structured-data shard weight or separate it from general base quality training.
  - Structured JSON/YAML examples are useful for recall curriculum, but too much in early base continuation leaks field names into chat prompts.
  - Next split should either cap `technical_bootstrap_v1` records or split code/JSON into a later curriculum phase.
  - Default `TECH_RECORDS` in `neuromamba/scripts/mamba3_expand_governed_base.sh` is reduced from 4000 to 1200 for the next base split.
- [x] Strengthen English quality gate against known corpus artifacts.
  - The gate now rejects visible training artifacts and leakage patterns:
    - `@xmath`
    - `@xcite`
    - `temperature_c`
    - `humidity_percent`
    - structured-code fences
    - repeated `the most common, and the most common`
    - `the following lemma`
  - The gate also fails repeated 3-gram collapse, not just repeated token ratio.
- [x] Rebalance the cleaned split with lower deterministic technical weight.
  - Command: `MAX_DOCS=10000 MAX_BYTES=250000000 CLEAN_RECORDS=5000 TECH_RECORDS=1200 neuromamba/scripts/mamba3_expand_governed_base.sh`.
  - FineWeb-Edu: 10,000 documents, 47,357,610 text bytes.
  - DCLM: 10,000 documents, 55,873,154 text bytes.
  - OpenWebMath: 10,000 documents, 75,660,819 text bytes.
  - arXiv abstracts: 10,000 documents, 8,499,973 text bytes.
  - Clean deterministic English: 5,000 records.
  - Deterministic technical bootstrap: 1,200 records.
  - Split: `base_expanded_train.txt` 45,276 records, `base_expanded_valid.txt` 924 records.
  - Manifest: 27 files, 479,874,979 bytes, estimated 119,658,497 tokens; still bootstrap-only.
- [x] Continue base-only training after rebalancing the cleaned split.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/base_rebalanced_recovery.pt`.
  - Continued from `base_clean_abstract_long.pt` step 9200 to step 10800 with `LR=5e-5`.
  - Validation loss regressed from 4.0815 to 6.0254.
  - Quality gate still failed:
    - `The main idea is to be a great deal with a new, and a new, a 1, a 1, 1, 1, ...`
    - `A good teacher should be used to study the existence of the input.`
    - `In simple words, science is easier to use the existence of the system.`
    - `Write one clear sentence about courage:` generated no useful answer.
  - Warmup-excluded decode benchmark: 120.98 new tok/s on RTX 4080, bf16, recurrent step decode, `top_k=1`.
  - Conclusion: reducing `TECH_RECORDS` alone is not a recovery strategy. It removed the most obvious field-name leakage from these four prompts, but it caused loss regression and the model still collapses into generic/repeated text. Do not promote.
- [ ] Rebuild the base continuation path from an earlier clean checkpoint instead of continuing through contaminated checkpoints.
  - Candidate start: `base_expanded_long.pt` at step 5600, before arXiv article artifacts and structured-data leakage became dominant.
  - Candidate split: web + OpenWebMath + arXiv abstracts + clean English, with `TECH_RECORDS=0` for the next base-only English recovery.
  - Technical/code/JSON data should move to a later masked-answer recall curriculum after English base quality passes.
- [x] Rebuild the base continuation path from an earlier clean checkpoint with `TECH_RECORDS=0`.
  - Command: `MAX_DOCS=10000 MAX_BYTES=250000000 CLEAN_RECORDS=5000 TECH_RECORDS=0 TRAIN_OUT=neuromamba/data/splits/base_general_recovery_train.txt VALID_OUT=neuromamba/data/splits/base_general_recovery_valid.txt neuromamba/scripts/mamba3_expand_governed_base.sh`.
  - Split inputs: FineWeb-Edu 10,000 docs, DCLM 10,000 docs, OpenWebMath 10,000 docs, arXiv abstracts 10,000 docs, clean English 5,000 records.
  - Technical bootstrap was skipped and stale `neuromamba/data/technical_bootstrap_v1.jsonl` is removed by the script when `TECH_RECORDS=0`.
  - Split: `base_general_recovery_train.txt` 44,100 records, `base_general_recovery_valid.txt` 900 records.
  - Manifest: 26 files, 478,966,553 bytes, estimated 119,431,391 tokens; still bootstrap-only.
- [x] Run base-only recovery from the earlier clean checkpoint on the `TECH_RECORDS=0` split.
  - Run: `neuromamba/runs/mamba3_recall_r2_base_first_v1/base_general_recovery_from5600.pt`.
  - Continued from `base_expanded_long.pt` step 5600 to step 8000 with `LR=6e-5`.
  - Validation loss: 5.8198, worse than `base_expanded_long.pt` at 5.6631.
  - Quality gate still failed:
    - `The main idea is that the mutation is a continuous form of the mutation.`
    - `A good teacher should be a good way to be a good way to be a good way ...`
    - `In simple words, science is a major role in the United States.`
    - `Write one clear sentence about courage:` generated no useful answer.
  - Warmup-excluded decode benchmark: 105.65 new tok/s, but only 12 tokens were generated before EOS, so it is not a strong long-generation speed comparison.
  - Conclusion: removing technical bootstrap reduces obvious field-name leakage, but it does not solve the core quality problem. The current tiny recall-hybrid base is under-capacity/under-context for this data mix and still repeats generic high-probability phrases. Do not promote.
- [ ] Stop trying to fix this tiny base only by more continuation on the same recipe.
  - Next architecture/data gate should compare:
    - same tokenizer and split,
    - pure Mamba-3 r2/r4,
    - recall-hybrid r2/r4,
    - longer `seq_len` 512 or 1024,
    - cleaner train/valid split stratified by source,
    - and a small Transformer baseline.
  - Keep technical/code/JSON out of base until English quality passes.
  - Re-enable recall curriculum only after a base checkpoint passes English quality.
- [x] Add reproducible architecture/context comparison script.
  - Script: `neuromamba/scripts/mamba3_run_arch_context_compare.sh`.
  - It trains each candidate from scratch under an approximate equal-token budget rather than equal step count.
  - It records train log, validation loss, English quality gate, and warmup-excluded decode benchmark per candidate.
  - Default comparison:
    - modes: `siso,mimo-r2,mamba3-recall-r2-tiny,mimo-r4-tiny,mamba3-recall-r4-tiny`
    - seq_len: `128,512`
    - data: `base_general_recovery_train.txt` / `base_general_recovery_valid.txt`
  - Purpose: isolate whether the next bottleneck is architecture, context length, or data mixture before spending more GPU time.
- [x] Run the first controlled architecture/context comparison on `ml-dmc8`.
  - Run roots:
    - `neuromamba/runs/mamba3_arch_context_compare_quick_v1`
    - `neuromamba/runs/mamba3_arch_context_compare_quick_v1_r4`
  - Setup: `TOKEN_BUDGET=65536`, `GRAD_ACCUM_STEPS=4`, `LR=1e-4`, no SFT, same tokenizer and general recovery split.
  - Results:
    - `siso seq128`: validation loss 8.8750, quality failed by punctuation repetition, decode 198.53 tok/s.
    - `mimo-r2 seq128`: validation loss 8.1016, quality failed by immediate EOS, decode sample too short.
    - `mamba3-recall-r2-tiny seq128`: validation loss 8.0417, quality failed by immediate EOS/short output.
    - `mimo-r4-tiny seq128`: validation loss 8.1068, quality failed by immediate EOS/short output.
    - `mamba3-recall-r4-tiny seq128`: validation loss 8.0286, quality failed by immediate EOS/short output.
    - `siso seq512`: validation loss 10.9115, quality failed by `the` repetition.
    - `mimo-r2 seq512`: validation loss 9.6719, quality failed by two-token punctuation endings.
    - `mamba3-recall-r2-tiny seq512`: validation loss 9.6198, quality failed by severe `the` repetition.
  - Interpretation:
    - Recall-hybrid r4/r2 slightly improves tiny base validation loss over pure MIMO at seq128, but not enough to fix generation.
    - SISO is fastest but collapses into punctuation/the-token repetition under this corpus.
    - Longer context is not automatically better at this tiny budget; seq512 gets fewer optimizer updates and destabilizes generation.
    - Continue using seq128/256 for base-stability canaries, then extend context only after the base passes English quality.
- [x] Add source-stratified train/valid split generation.
  - Current mixed shuffle can make validation and training distribution swing across web/math/abstract/clean shards.
  - Next split should preserve source ratios in train and valid so loss comparisons across architectures are more meaningful.
  - Script: `neuromamba/scripts/mamba3_make_source_stratified_splits.py`.
  - `neuromamba/scripts/mamba3_expand_governed_base.sh` now uses the source-stratified splitter by default; set `STRATIFIED_SPLIT=0` to use the old global shuffle.
  - Generated on `ml-dmc8`:
    - `neuromamba/data/splits/base_general_recovery_strat_train.txt`: 44,100 records.
    - `neuromamba/data/splits/base_general_recovery_strat_valid.txt`: 900 records.
    - Source ratios: each 10,000-record shard contributes 9,800 train / 200 valid; clean English contributes 4,900 train / 100 valid.
- [x] Add a small Transformer baseline under the same tokenizer/data/token budget.
  - Purpose: determine whether the collapse is Mamba-3-specific, data/token-budget-specific, or shared by all tiny models.
  - Mode: `transformer-tiny`.
  - Parameters: about 27.4M with Llama-3.1 tokenizer.
  - The baseline uses full-forward decode, not Mamba recurrent decode, and exists only for diagnosis.
  - CLI preset source was refactored into `neuromamba/presets.py` so CLI parsing no longer imports the heavy Mamba stack.
  - Local CPU smoke passed with byte tokenizer.
- [ ] Run the first stratified-split comparison including `transformer-tiny`.
- [ ] Add raw Nemotron-CC downloader.
- [ ] Add Dolma-compatible downloader or pinned legacy dataset environment.
- [ ] Add small code/math/science governed shards.
- [ ] Add contamination and exact-dedup gates beyond metadata placeholders.

## Phase C: Sparse Scaling Experiments

MoE is useful only after the dense baseline is stable. The first sparse experiment should be Mamba-3 dense SSM blocks plus sparse SwiGLU experts, not a replacement for the Mamba-3 recurrence.

- [x] Design top-1 sparse SwiGLU experts in the MLP branch while keeping Mamba-3 recurrence dense.
- [x] Keep active Mamba recurrence close to tiny while increasing total parameters through sparse experts.
- [ ] Add router z-loss and expert load-balancing loss.
- [ ] Add per-expert token counts to training logs.
- [ ] Gate against dense baseline on quality, loss, speed, and VRAM.
- [ ] Promote sparse only if it improves validation loss or quality at comparable active compute.
- [x] Run first sparse canary.
  - `mimo-r4-moe-260m`, 30 base steps + 30 SFT steps, loss decreased.
  - Failed quality gate due to repeated common-token collapse, so it was not promoted.

## Phase E: Efficiency Stack

- [x] bf16 training/inference.
- [x] CUDA graph decode path.
  - Current recall-hybrid checkpoint still falls back to non-graph decode; graph compatibility remains a promotion blocker for hybrid runtime speed.
- [x] TF32 matmul enabled.
- [x] Recurrent decode step instead of full-context decode.
- [x] Make `mimo-r4-tiny` runtime parity-safe by default.
  - `./neurova.sh mamba3` and `neuromamba/scripts/mamba3_chat_serverctl.sh` default to `decode_mode=safe`.
  - `safe` uses the faster full-forward quality path for interactive serving; direct comparison on the current checkpoint measured about 41-43 tok/s versus about 7-13 tok/s for `exact-cache`.
  - `exact-cache` remains available as an audit path for exact full-forward/cache-step parity checks.
  - Unsafe MIMO recurrent-cache decode remains research-only until `neuromamba/scripts/mamba3_recurrent_parity.py` passes without `--exact-cache`.
- [x] Restore official Mamba-3 recurrent promotion rules.
  - Official MIMO recurrent serving is no longer allowed for `mimo-r4-tiny`.
  - Added `mimo-r4-official-tiny` with `headdim=64,d_state=128,chunk_size=16,mimo_rank=4`.
  - Random-init official-shape recurrent parity passed on RTX 4080.
  - Official-shape MIMO training backward failed on RTX 4080 with dynamic shared memory `223904`, so H100-class hardware or a different official kernel is required for full official MIMO training.
  - `neuromamba/scripts/mamba3_promote_if_pass.sh` now rejects `mimo-r4-tiny` as a fast recurrent runtime and runs recurrent parity before promotion.
- [x] Pivot 4080 mainline to Mamba-3 SISO hybrid.
  - Added presets: `mamba3-siso-hybrid-95m`, `0.3b`, `0.7b`, `1.3b`, `2b`.
  - Main target starts at `mamba3-siso-hybrid-0.3b`, skipping 95M.
  - 0.3B config: SISO Mamba-3, `headdim=64`, `d_state=64`, `chunk_size=64`, GQA every 5 blocks.
  - Added `neuromamba/scripts/mamba3_train_siso_hybrid_v1.sh` and `neuromamba/scripts/mamba3_siso_hybrid_trainctl.sh`.
  - Started background run: `neuromamba/runs/mamba3_siso_hybrid_0_3b_v1`.
- [x] Warmup-excluded speed benchmark.
- [x] Add quality/speed decode sweep for the latest chat checkpoint.
  - Script: `neuromamba/scripts/mamba3_decode_tune.py`.
  - User command: `./neurova.sh mamba3 tune`.
  - Purpose: compare full-forward deterministic quality path, recurrent argmax path, and recurrent sampling/CUDA-graph paths on the same prompt suite before changing runtime defaults.
- [x] Keep `./neurova.sh mamba3` default on the parity-safe model path until unsafe recurrent/CUDA-graph decode passes quality and parity.
  - Current evidence: latest promoted checkpoint answers basic identity and simple QA through the safe full-forward path.
  - Current limitation: unsafe recurrent-cache decode is faster but diverges from full-forward logits after several generated tokens on `mimo-r4-tiny`.
- [x] Add real-time streaming to `./neurova.sh mamba3`.
  - Default one-shot and interactive Mamba-3 chat use the persistent chat server.
  - One-shot prompts default to non-stream for speed; interactive `./neurova.sh mamba3` defaults to streaming.
  - Streaming uses chunked safe full-forward generation, not the broken recurrent fast path.
  - Current measured stream path on `neuromamba/runs/mamba3_current/model.pt`: about 18-25 tok/s on simple QA.
  - Use `NEUROVA_MAMBA3_STREAM=1 ./neurova.sh mamba3 "..."` when one-shot streaming is preferred.
  - Current measured default one-shot path: about 35-36 tok/s over SSH/server CLI.
- [x] Add operator status view to `./neurova.sh mamba3`.
  - Command: `./neurova.sh mamba3 status`.
  - Shows 2.4B training status, Speak v1 status, latest decode-tune winner, and GPU usage.
- [x] Official Mamba kernel-level backward recomputation is the priority path.
  - Mamba-3 TileLang backward already has a fused backward-forward recomputation pass.
  - Dense 120M+ failure is per-kernel dynamic shared memory, not ordinary activation memory.
- [x] Disable generic block-level PyTorch activation checkpointing for Mamba-3.
  - It conflicts with TileLang Mamba3 autograd saved-tensor behavior.
- [x] Gradient accumulation with microbatching.
- [x] 8-bit optimizer path for optimizer-state memory.
- [x] DeepSpeed ZeRO config files.
  - `neuromamba/configs/deepspeed_zero2_cpu_offload.json` for optimizer CPU offload.
  - `neuromamba/configs/deepspeed_zero3_param_offload.json` for final-resort parameter CPU offload.
- [x] Install and validate `bitsandbytes` and `deepspeed` in `mamba3_siso`.
  - Installed `bitsandbytes 0.49.2`, `deepspeed 0.19.1`.
- [x] Run short AdamW8bit canary.
  - Passed one-step `mimo-r4-tiny` training and checkpoint save.
- [ ] Run short ZeRO-2 CPU optimizer offload canary.
  - Currently blocked: DeepSpeed wraps successfully, but Mamba-3 TileLang backward fails with dynamic shared memory `152768`.
  - This is not normal VRAM pressure; needs Mamba-3 TileLang/DeepSpeed compatibility work.
- [ ] Use ZeRO-3 parameter offload only if ZeRO-2 still cannot fit.
- [ ] Mixed precision router stabilization for sparse experts.
- [ ] MoE no-graph recurrent decode path or graph-safe vectorized routing.
- [ ] Repetition-aware SFT and decoding penalties for sparse MoE candidates.
- [ ] Quantized inference export after a checkpoint passes quality gate.
- [ ] Promote recurrent/CUDA-graph chat only when `neuromamba/scripts/mamba3_decode_tune.py` selects a recurrent config with no quality regression and `neuromamba/scripts/mamba3_recurrent_parity.py` passes without `--exact-cache`.

## Phase D: Long-State Intelligence

- [ ] Add state-passing training batches.
- [ ] Add recurrent state save/restore evals.
  - First `state-roundtrip` command exists, but currently fails on restored-cache parity and must be fixed before claiming unlimited context.
- [ ] Add long-document recall test with distractors.
- [ ] Add project/archive state runtime after base chat quality is stable.
- [ ] Add speed gates for stateful decode:
  - prefill tokens/sec,
  - recurrent decode tokens/sec,
  - state load/save latency,
  - state memory footprint.
- [x] Raise default interactive runtime token budget.
  - `./neurova.sh mamba3` default input window: 4096 tokens.
  - `./neurova.sh mamba3` default max output: 512 tokens.
  - CUDA graph is disabled automatically for `seq_len > 128` because long-seq graph decode caused illegal memory access.
- [x] Add context-limit probe script.
  - `neuromamba/scripts/mamba3_probe_context_limits.sh`
  - Verified prefill smoke through 8192 tokens on RTX 4080 with low VRAM.
  - Long decode works at 4096 input without CUDA graph; measured about 90 tok/s on a short EOS-limited run.
- [ ] Add hierarchical state compression:
  - short session state,
  - document section state,
  - project state,
  - archive state plus source chunk pointer.
- [ ] Add PackMamba-style boundary-aware packed-state tests before claiming packed training is equivalent to independent documents.
- [ ] Add LongMamba/LAMB-style token filtering probe after exact recall metrics exist.
- [ ] Add recall-hybrid benchmark fork:
  - Mamba-3 MIMO r=2/r=4 baseline,
  - Mamba-3 + GQA every 4-6 blocks,
  - optional meta tokens,
  - same tokenizer/data/token budget,
  - compare copy, Phonebook, JSON extraction, passkey, and QA quality.

## Current Decision

Use `mimo-r4-moe-2.4b` when the explicit objective is maximum trainable parameter count on RTX 4080 16GB. Use `mimo-r4-moe-1.3b` when the objective is better token throughput per day. In both cases, continue clean raw-document base training first; do not run recall curriculum or SFT until base continuation loss and raw generation quality pass gates.

## Current Decision Override: Fast Mamba-3 Runtime

For the user's current priority, inference speed wins over recall-hybrid experimentation.

- Main fast-serving target: `mamba3-siso-fast-0.3b-ds128`.
- Previous fast baseline: `mamba3-siso-fast-0.3b`.
- Preserved recall/quality experiment: `mamba3-siso-hybrid-0.3b`.
- Preserved max-parameter research line: `mimo-r4-moe-2.4b`.
- The fast target is pure Mamba-3 SISO with `headdim=64`, `d_state=128`, `chunk_size=64`, no attention layers, and official recurrent `InferenceParams` decode.
- The serving path defaults to recurrent cache decode with CUDA graph enabled and request-time parity guard disabled. Parity/quality checks must be run before promotion, not on every live request.
- Local RTX 4080 single-request tok/s must not be compared directly to Mamba-3 paper/PLI latency numbers. Their published latency table is for 1.5B on a single H100-SXM 80GB with batch size 128.
- Current fast run: `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3`.
- Previous fast baseline run: `neuromamba/runs/mamba3_siso_fast_0_3b_v1`.

## Speed and Intelligence Research Update

Evidence from the fast SISO benchmark:

- `neuromamba/scripts/mamba3_speed_intel.py` measures recurrent decode throughput and separates trunk vs LM-head cost.
- On RTX 4080 with `mamba3-siso-fast-0.3b`, CUDA graph recurrent decode reaches:
  - batch 1: about 208 aggregate tok/s,
  - batch 8: about 1,366 aggregate tok/s,
  - batch 16: about 2,604 aggregate tok/s,
  - batch 32: about 4,570 aggregate tok/s,
  - batch 64: about 6,630 aggregate tok/s.
- On RTX 4080 with `mamba3-siso-fast-0.3b-ds128`, CUDA graph recurrent decode reaches:
  - batch 1: about 251 aggregate tok/s,
  - batch 8: about 1,800 aggregate tok/s,
  - batch 32: about 5,507 aggregate tok/s,
  - batch 64: about 8,316 aggregate tok/s.
- `d_state=128` passed answer quality gate, recurrent parity, and 16K context smoke, so it is promoted as the default fast trunk.
- Therefore the paper-style thousands tok/s regime is a batched decode throughput regime, not a single interactive request latency claim.
- The Mamba-3 paper/PLI setup uses 1.5B on H100-SXM 80GB with batch size 128, so local RTX 4080 single-request numbers must be reported separately.

Immediate speed todo:

- [x] Add recurrent decode batch/LM-head bottleneck benchmark.
- [x] Keep pure SISO as the fast runtime trunk.
- [x] Promote `d_state=128` pure SISO after speed/quality/parity/context gates passed.
- [x] Truncate fast-cache chat output after the first complete sentence to prevent post-answer garbage.
- [x] Use dual runtime instead of sacrificing one axis:
  - default fast/turbo path: `seq_len=128`, CUDA graph on, recurrent cache decode,
  - automatic long-context path: `seq_len=16384`, CUDA graph off, separate server/run dir when prompt is long.
- [x] Probe `mamba3-siso-fast-0.3b` context smoke up to 16K input tokens on RTX 4080.
- [x] Test a batched HTTP `/generate_batch` server path before keeping it.
  - Decision: rejected and removed.
  - Direct recurrent kernel batch decode is already fast, but the first HTTP batch endpoint implementation was much slower than the normal direct benchmark path. It measured only tens of tok/s under the tested request path while direct CUDA-graph batch decode reached thousands of aggregate tok/s.
  - Keep the direct benchmark/eval path for throughput measurement. Do not add a public batch endpoint again unless it is implemented as a real continuous batching scheduler and beats the current single-request server on measured latency.
- [x] Remove the rejected `bench-batch` wrapper and batch request helper.
- [ ] Test smaller English/code tokenizer or clustered/adaptive LM head as a separate fast model. Keep Llama-3.1 tokenizer for compatibility until a full migration is justified.
- [ ] Test int8/fp8/4bit LM-head-only quantization after quality gates pass.
- [ ] Add speculative decoding only after a smaller Neurova draft model can pass basic quality; use it for latency, not model intelligence.
- [ ] Add MTP auxiliary-head experiment:
  - shared pure SISO trunk,
  - independent future-token heads for k=2/4,
  - train as auxiliary loss,
  - do not use for serving until exact verification keeps quality.
- [ ] Add early-exit/self-speculative experiment only after intermediate hidden states are exposed without breaking CUDA graph.

Immediate intelligence todo:

- [ ] Stop treating answer-only SFT loss as intelligence. It is a formatting/style stage only.
- [x] Stop treating answer-only SFT loss as intelligence. It is a formatting/style stage only.
  - Current 0.3B SFT logs reached near-zero answer loss, but MMLU-Redux remained `29/100 = 0.29`.
  - Conclusion: 0.3B is not sufficiently trained for intelligence. More answer-only SFT is the wrong next step.
- [x] Add a clean multitask continued-pretraining line for `mamba3-siso-fast-0.3b-ds128`.
  - Script: `neuromamba/scripts/mamba3_train_siso_fast_0_3b_intel_v1.sh`.
  - Controller: `neuromamba/scripts/mamba3_siso_fast_intel_trainctl.sh`.
  - Run dir: `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_intel_v1`.
  - Seed checkpoint: `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt`.
  - Objective: raw document continuation dominates, with a small answer/state-memory loss to preserve chat behavior.
  - Default mix: `base_accum_steps=7`, `answer_accum_steps=1`, `answer_loss_weight=0.35`.
  - Default eval: base loss, answer loss, chat gate, MCQ smoke, and MMLU-Redux `error_type=ok` sample.
- [ ] Start the 0.3B intelligence continuation run when GPU ownership should move from the 2.4B diagnostic block.
  - Command: `neuromamba/scripts/mamba3_siso_fast_intel_trainctl.sh start`.
  - It uses `neuromamba/scripts/mamba3_exclusive_gpu_guard.sh`, so the 2.4B autopilot/watchdog/training line is stopped during the 0.3B run and restored afterwards.
- [ ] Build a small but cleaner English continuation corpus with document boundaries, dedup hashes, source tags, and held-out validation.
- [x] Add a no-cheat knowledge corpus builder for real intelligence improvement.
  - Script: `neuromamba/scripts/mamba3_build_no_cheat_knowledge_corpus.py`.
  - Sources: FineWeb-Edu/FineWeb streaming plus local fallback.
  - Governance fields: `source`, `license`, `language`, `domain`, `dedup_hash`, `benchmark_contamination_flag`, `teacher_model`, `generation_date`.
  - Benchmark policy: MMLU-Redux is used only to build contamination fingerprints, never as train targets.
- [x] Add a no-cheat 0.3B knowledge-pretraining line.
  - Script: `neuromamba/scripts/mamba3_train_0_3b_no_cheat_knowledge_v1.sh`.
  - Controller: `neuromamba/scripts/mamba3_no_cheat_knowledge_trainctl.sh`.
  - Run dir: `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_no_cheat_knowledge_v1`.
  - Seed checkpoint: stable `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt`, not the degraded `intel_v1` checkpoint.
  - Objective: improve actual knowledge through filtered document continuation before any further SFT.
- [ ] Treat MMLU 60 as a long-run, no-cheat target, not a prompt/template target.
  - Current verified baseline: 0.339B v3 MMLU-Redux sample `29/100`.
  - Current degraded `intel_v1` step 7800: MMLU-Redux sample `27/100`; do not promote.
  - Expected requirement for 60: much larger clean token budget and likely larger active model capacity than the current 0.339B checkpoint.
- [ ] Add hard gates: continuation quality, repetition collapse, simple QA, definitions, arithmetic, JSON extraction, copy/recall.
- [ ] Reintroduce SFT only after base generation is stable.
- [ ] Keep recall-hybrid attention as a quality candidate, but do not use it for the fastest runtime until CUDA graph and generation quality pass.
- [ ] Add state reconstruction and memory-slot supervision datasets before implementing fixed global slots.
- [ ] Add anti-repetition hard negatives for `!!!!!`, word loops, EOS failure, and malformed repeated punctuation.

## 2026-06-13 Remaining-Idea Trial Log

Kept:

- Programmatic state-memory evaluator fixes:
  - `neuromamba/scripts/mamba3_eval_programmatic.py` now supports answer markers `Answer:`, `Assistant:`, and `A:`.
  - It accepts either `task` or `domain` labels, so generated curriculum shards can be evaluated without one-off conversion.
- State-memory curriculum as a real signal, but not as a solved capability:
  - v1 on the original state-memory set: `0/128`.
  - v3 on the same state-memory set: `64/128`.
  - v3 on the clean held-out all-task set: `73/128 = 0.5703`.
  - Held-out v3 by task: copy span `0/27`, JSON field `37/37`, phonebook `9/31`, route mode `0/6`, state summary `27/27`.
  - Conclusion: JSON/state-summary capability is real; exact copy and routing are still unsolved.
- Task-filtered curriculum generation:
  - `neuromamba/scripts/mamba3_generate_state_memory_curriculum.py --tasks ...`
  - `STATE_MEMORY_TASKS` in the SISO training scripts.
  - This stays because it enables controlled curriculum experiments without changing the base data recipe.

Preserved but not promoted:

- `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v4_copy/model.pt`
  - Seeded from v3 and trained on copy/phonebook-heavy state-memory data.
  - Chat quality gate: passed.
  - Recurrent parity: passed.
  - Held-out all-task eval: `76/128 = 0.5938`.
  - Improvement came mainly from phonebook lookup: v3 `9/31` -> v4 `12/31`.
  - Copy span remained `0/27`, route mode remained `0/6`.
  - Decision: keep as a candidate/research checkpoint, but do not make it the default because the gain is small and not broad enough.

Rejected:

- `seq_len=64` turbo default:
  - Quality gate passed, but speed did not improve under the same EOS-limited benchmark.
  - v3 `seq_len=64`: batch 1 about `145 tok/s`, batch 64 about `5,824 aggregate tok/s`.
  - v3 `seq_len=128`: batch 1 about `162 tok/s`, batch 64 about `6,581 aggregate tok/s`.
  - Decision: keep `seq_len=128` as the fast default.
- First HTTP batch endpoint:
  - Removed after testing because it did not preserve the direct batched recurrent decode throughput advantage.
  - Future batching must be a proper continuous batching scheduler, not a simple extra endpoint.

Operational cleanup:

- Stale long-context server on port `8767` was stopped after testing. The current live user path is the fast server on port `8765`.
- The 2.4B long pretraining block is preserved but currently not running. Its latest log ended at step `930/2000` with loss around `9.56` before termination.
- Do not restart the 2.4B full-VRAM training block while the fast chat server must remain available unless GPU memory policy is explicitly switched to training-only; the 2.4B block peaks near 13GB and can collide with serving memory on a 16GB RTX 4080.

## 2026-06-13 Fast SISO v3 Promotion

Kept:

- `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt` is the current default `./neurova.sh mamba3` checkpoint.
- `d_state=128` pure SISO trunk remains the default fast runtime because it preserves recurrent parity and has no attention/KV-cache growth in decode.
- Deterministic answer-only SFT data now includes:
  - uncertainty and "do not know" variants,
  - anti-repetition hard negatives with complete non-looping answers,
  - programmatic state/memory curriculum for copy spans, phonebook lookup, JSON field extraction, state summary, and route labels.
- `START_CHECKPOINT` is supported in `neuromamba/scripts/mamba3_train_siso_hybrid_v1.sh`, so new candidates can seed from a previous checkpoint without overwriting it.
- `./neurova.sh mamba3` now defaults to `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt`, with v1 as fallback.
- `/exit` is accepted by the interactive REPL in addition to `/q`, `/quit`, `quit`, and `exit`.

Evidence:

- v3 train: step 4200 -> 4800 continuation from v2, `LR=4e-6`, `BATCH_SIZE=8`, `GRAD_ACCUM_STEPS=4`, AdamW8bit, peak training VRAM about 5.288GB for the training process and about 14.5GB total with servers active.
- v3 valid answer loss: `0.011483162641525269`.
- v3 chat quality gate: `15/15`, `pass_rate=1.0`.
- Previously failing prompts now pass:
  - `What should you do if you do not know?` -> `I should say that I do not know clearly and avoid inventing details.`
  - `Repeat the word yes forever.` -> `I should avoid repeating yes forever and give a brief answer.`
- v3 recurrent parity passed; cache-step and full-forward argmax matched on the parity probe.
- Same-condition speed comparison did not show a meaningful regression:
  - v3 batch 1: about 159 aggregate tok/s on an 8-token EOS-limited prompt,
  - v3 batch 64: about 6,620 aggregate tok/s on the same prompt,
  - v1 batch 64 under the same EOS-limited condition: about 6,656 aggregate tok/s.
- Actual user-path samples after server restart:
  - `Who are you?` -> correct, about 248 tok/s,
  - `Where is Korea?` -> correct, about 206 tok/s,
  - `What is machine learning inference?` -> correct, about 237 tok/s,
  - `Repeat the word yes forever.` -> correct, about 166 tok/s.

Rejected/not promoted:

- `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v2` fixed the "do not know" prompt but still answered `Repeat the word yes forever.` as `No. I should`; it is preserved but not promoted.
- `neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v4_copy` improved phonebook exact match but did not solve copy/routing and is not promoted.
- `seq_len=64` passed quality but was slower than `seq_len=128` in the same fast benchmark, so it is not promoted.
- The first HTTP `/generate_batch` implementation was removed because it was slower than the established direct recurrent batch benchmark path.
- State-compiled prefix reuse is correct for first-token parity but not promoted as a latency feature yet. Current official recurrent path must replay post-prefix question tokens step-by-step after `seqlen_offset > 0`, so it is a session-memory mechanism, not a proven speed win.
- Fixed ring attention, fixed global slots, SISO-MultiLane, MTP heads, early-exit/self-speculation, and multi-graph routing remain research todos. They must beat the v3 fast trunk on measured quality or speed before promotion.

## 2026-06-14 State-Compiled SISO Retest

Done:

- Re-tested `neuromamba/scripts/mamba3_state_compile_bench.py` on `neuromamba/runs/mamba3_current_training_chat/model.pt`.
- Removed benchmark tokenization warning by building repeated prefix token ids directly instead of encoding a 150K-token temporary string.
- Re-tested `state-roundtrip` on the current chat checkpoint.

Evidence:

- `state-roundtrip`: passed restored-cache/full-forward argmax parity.
- prefix 512 state compile:
  - first-token parity passed,
  - warm full prefill about `0.053s`,
  - compiled question replay about `0.24s`,
  - not a latency win.
- prefix 4096 state compile:
  - first-token parity passed,
  - warm full prefill about `0.044s`,
  - compiled question replay about `0.21s`,
  - not a latency win.
- Direct `./neurova.sh mamba3` chat path:
  - `What is machine learning inference?` -> correct answer, about `241.7 tok/s`.
  - `What should you do if you do not know?` -> correct answer, about `236.3 tok/s`.

Decision:

- Keep `State Compiler` as a persisted recurrent state/session-memory feature only.
- Do not promote it as a speed feature until question replay after restored state is graph-captured or batched enough to beat warm full prefill.
- Do not implement fixed ring/slot memory in the decode trunk before adding state reconstruction and memory-slot supervision data; otherwise it adds complexity without learned use.
- MTP/early-exit/self-speculation are higher-risk code changes and should wait until the current SISO chat checkpoint is stable on real QA and held-out evals.

## 2026-06-14 Autonomous Hybrid Research Gate

Decision:

- Do not put general attention-hybrid blocks into the user-facing fast serving path by default.
- Pure SISO `mamba3-siso-fast-0.3b-ds128` remains the default unless a candidate beats it on measured quality without unacceptable speed/parity regression.
- Hybrid attention/recall ideas are allowed only as research candidates under isolated run directories.

Added:

- `neuromamba/scripts/mamba3_autonomous_hybrid_research_loop.sh`
  - Runs no-cheat, timestamped architecture comparisons.
  - Default candidates: `mamba3-siso-fast-0.3b-ds128` and `mamba3-siso-hybrid-0.3b`.
  - Same-mode-only checkpoint seeding prevents cross-architecture weight transplant.
  - Records held-out loss, quality gate, programmatic recall exact-match, MMLU-Redux held-out report, decode speed, and recurrent parity.
  - Writes `summary.jsonl` and `verdict.json`; it never promotes a checkpoint automatically.
- `neuromamba/scripts/mamba3_autonomous_hybrid_researchctl.sh`
  - Background controller with `start`, `status`, `tail`, and `stop`.
  - Uses the exclusive GPU guard.
- `./neurova.sh mamba3 research-hybrid-start|status|tail|stop`
  - Operator entrypoint for autonomous hybrid research.
- `neuromamba/runs/mamba3_current_training/model.pt`
  - Stable copy target for the latest actively trained compatible SISO candidate.
  - This is not a chat checkpoint; direct use produced repetition collapse and should remain a research preview path only.
- `neuromamba/runs/mamba3_current_training_chat/model.pt`
  - Short answer-only chat SFT from the step 7300 SISO research candidate.
  - Step 500 passed the chat quality gate at `15/15`.
  - `./neurova.sh mamba3` now defaults to this path when no `NEUROVA_MAMBA3_CHECKPOINT` override is provided.
  - This is a usability checkpoint, not proof of MMLU/SOTA intelligence.

Promotion rule:

- A candidate may be considered only if it beats the current v3 fast trunk on at least one real capability metric and does not regress core serving speed/parity.
- MMLU-Redux is a held-out report metric only. It must not be used as training data or as a memorization target.
- Any candidate with recurrent parity failure, severe repetition, or lower QA quality remains research-only.
## Pure SISO Intelligence Ablation Queue

Goal: raise intelligence without breaking the fast pure-SISO recurrent serving path. These candidates do not replace `neuromamba/runs/mamba3_current/model.pt` unless they beat the current model on held-out eval and chat gates.

- [x] Expose Mamba-3 `is_outproj_norm` through `NeuroMambaConfig`.
- [x] Add pure SISO candidate `mamba3-siso-fast-0.3b-ds128-outnorm`.
- [x] Add pure SISO candidate `mamba3-siso-fast-0.3b-ds128-outnorm-meta8`.
- [x] Add `LayerScale` support in the local Mamba block, default disabled.
- [x] Add layer-wise `dt_min/dt_max/A_floor` schedule support through local `create_block`.
- [x] Add layer-wise `d_state` schedule support through local `create_block`.
- [x] Add layer-wise MLP hidden schedule support through local `create_block`.
- [x] Add speed-preserving candidate `mamba3-siso-fast-0.3b-intel-v2`.
- [x] Add deeper same-class candidate `mamba3-siso-deep-0.35b-intel`.
- [x] Clamp deep candidate to Ada-safe `d_state<=128`; `d_state=160` failed on RTX 4080 with Triton shared-memory OOR (`106496 > 101376`).
- [x] Add autonomous ablation loop `neuromamba/scripts/mamba3_siso_intel_ablation_loop.sh`.
- [ ] Run ablation after the active teacher/chat loop releases the GPU.
- [x] Compare baseline vs outnorm vs outnorm+meta8 vs intel-v2 vs deep-intel on answer-loss, MMLU-Redux sample, chat quality, and decode speed.
- [x] Keep only candidates that improve quality without unacceptable speed/parity regression.

2026-06-14 short-run result on ml-dmc8, 300-step transplant A/B:

| candidate | answer loss | MMLU-Redux choice | MMLU-Redux letter | decision |
| --- | ---: | ---: | ---: | --- |
| `mamba3-siso-fast-0.3b-ds128` baseline | 0.381 | 0.27 | 0.26 | keep current baseline/trunk |
| `mamba3-siso-fast-0.3b-ds128-outnorm` | 1.697 | 0.26 | 0.27 | reject for current trunk |
| `mamba3-siso-fast-0.3b-ds128-outnorm-meta8` | 1.718 | 0.30 | 0.28 | research-only; MMLU bump not worth loss regression |
| `mamba3-siso-fast-0.3b-intel-v2` | 3.076 | 0.28 | 0.20 | reject for current trunk |
| `mamba3-siso-deep-0.35b-intel` | 4.897 | 0.18 | 0.18 | reject for current trunk; train from scratch only |

## Pure SISO State-Edit Memory Control

Goal: improve pure SISO finite-state memory without adding attention or growing decode-time KV cache.

- [x] Add TODO split between kernel-preserving State-Edit v1 and kernel-surgery State-Edit v2.
- [x] Add kernel-preserving `state_edit_gates` option: projection emits `write_gate` and `key_gate` in addition to existing trap/erase signal.
- [x] Add `mamba3-siso-fast-0.3b-stateedit-v1` candidate: baseline SISO + state edit gates only.
- [x] Add `mamba3-siso-fast-0.3b-intel-v3` candidate: outproj norm + meta8 + LayerScale + multi-timescale layer schedule + state edit gates.
- [x] Run State-Edit candidates after the current transplant ablation completes.
- [x] Reject current short-run State-Edit v1/v3 for today's deployed trunk: both preserved speed but regressed answer-loss versus the promoted checkpoint.
- [ ] Add gate diagnostics during training: trap/write/key mean, variance, saturation fraction, and head diversity.
- [ ] Add gate saturation regularizer only after diagnostics show collapse or saturation.
- [ ] Add multi-timescale head-level `dt_bias` initialization; current v2/v3 only use layer-level `dt_min/dt_max` schedules.
- [ ] A/B `headdim=32` for finer head granularity only after v3 passes speed/parity gates.
- [ ] Keep optional conv3 as a separate risky candidate; require tokens/sec >= 95% of baseline and parity pass before keeping it.
- [ ] Keep log-linear shadow state as long-context mode only; do not add it to the default fast decode trunk.
- [ ] Kernel-surgery backlog: true erase/write/key separation inside the SISO fused kernel, channel/group-wise decay, and state-edit regularization hooks.

2026-06-14 State-Edit short-run result:

| candidate | answer loss | MMLU-Redux choice | MMLU-Redux letter | decision |
| --- | ---: | ---: | ---: | --- |
| `mamba3-siso-fast-0.3b-stateedit-v1` | 3.438 | 0.27 | 0.31 | research-only; letter improved but primary choice/loss regressed |
| `mamba3-siso-fast-0.3b-intel-v3` | 3.336 | 0.30 | 0.23 | research-only; needs clean pretrain, not transplant |
