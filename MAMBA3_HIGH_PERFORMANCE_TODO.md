# Mamba-3 High-Performance TODO

This is the working execution plan for the pure Mamba-3 path. It follows the original Mamba-3 paper and the official `state-spaces/mamba` implementation, not generic transformer advice.

## Non-Negotiable Architecture

- [x] Use only Mamba-3 layers for the target path.
- [x] Use MIMO rank 4 for the quality target.
- [x] Keep SwiGLU/GatedMLP enabled in every block.
- [x] Keep pre-norm block structure from the official `Block`.
- [x] Keep BC/QK Norm.
- [x] Keep learnable B/C biases initialized positive.
- [x] Keep short convolution removed.
- [x] Use Llama-3.1 tokenizer.
- [x] Keep recurrent step decode for interactive use.

Verification:

```bash
python -m mamba3_kr.cli check-contract --mode mimo-r4-1.5b --tokenizer llama31
python -m mamba3_kr.cli model-info --mode mimo-r4-1.5b --tokenizer llama31
```

## Paper-Matched Presets

- [x] `mimo-r4-paper-180m`: d_model 768, layers 12, MLP 1264, d_state 128, headdim 64.
- [x] `mimo-r4-440m`: d_model 1024, layers 16, MLP 1792, d_state 128, headdim 64.
- [x] `mimo-r4-880m`: d_model 1536, layers 20, MLP 2800, d_state 128, headdim 64.
- [x] `mimo-r4-1.5b`: d_model 2048, layers 24, MLP 3824, d_state 128, headdim 64.
- [x] Disable accidental GatedMLP hidden-dim rounding for paper presets with `mlp_multiple_of=1`.

Success gate:

```bash
for mode in mimo-r4-paper-180m mimo-r4-440m mimo-r4-880m mimo-r4-1.5b; do
  python -m mamba3_kr.cli check-contract --mode "$mode" --tokenizer llama31
done
```

## RTX 4080 Runtime Path

- [x] Fix Mamba-3 decode step `tile_D` to follow `headdim`.
- [x] Keep `mimo-r4-tiny` as the RTX 4080 trainable kernel-validation model.
- [x] Provide interactive streaming chat via `./neurova.sh`.
- [x] Show response speed at the end of each streamed answer.
- [x] Provide warmup-excluded decode benchmark.
- [x] Provide kernel capability probe.

Commands:

```bash
./neurova.sh
./neurova.sh mamba3 bench "The main idea is"
./neurova.sh mamba3 probe
scripts/mamba3_prepare_corpora.sh
scripts/mamba3_train_governed_tiny.sh
scripts/mamba3_run_gates.sh
```

Known hardware boundary:

- `mimo-r4-paper-180m` decode works on RTX 4080.
- `mimo-r4-paper-180m` backward currently fails on RTX 4080 because the TileLang MIMO backward kernel requests more dynamic shared memory than the card allows.
- Therefore RTX 4080 is for tiny validation and interactive inference experiments, not full paper-scale MIMO-R4 pretraining.

## Data And Training

- [x] Bootstrap instruction/completion data exists for smoke training.
- [x] Packed-token training path exists.
- [x] RLVR verifier bootstrap exists without external LLM teacher/judge.
- [x] Corpus manifest and token estimate gate exists.
- [ ] Build real English pretraining stream: FineWeb-Edu-style education web, code/docs, math/science, long-document QA, verifiable reasoning.
- [ ] Minimum serious target: 100B tokens.
- [ ] Preferred target: 300B tokens.
- [ ] Stretch target: 1T tokens.
- [ ] Train context curriculum: 2K first, then 8K, 32K, and state-passing long replay.

Do not claim intelligence from the tiny bootstrap checkpoint. It proves kernel/runtime behavior only.

Current local corpus status:

```bash
scripts/mamba3_corpus_manifest.py data/*.txt data/*.jsonl
```

Latest local estimate: about 2.23M tokens, status `bootstrap_only`.

Reproducible corpus/training pipeline:

```bash
# Download enabled governed shards from configs/mamba3_corpus_sources.json.
MAX_DOCS=2000 MAX_BYTES=100000000 scripts/mamba3_prepare_corpora.sh

# Staged training: base pretrain -> validation -> instruction SFT -> eval.
BASE_STEPS=200 SFT_STEPS=200 scripts/mamba3_train_scientific_tiny.sh
```

## Evaluation Gates

- [x] `diagnose-decode`: cache step must be finite and match full-forward argmax.
- [x] `bench-decode`: report warmup and measured token/s separately.
- [x] `eval-english`: show exact input/output for basic English prompts.
- [ ] Add LM loss eval over a held-out English validation file.
- [ ] Add exact-answer RLVR eval.
- [ ] Add code unit-test eval.
- [ ] Add long-state recall eval.
- [ ] Add LAMBADA/HellaSwag/PIQA/ARC/OpenBookQA via LM Evaluation Harness after real pretraining.

## Next Execution Order

1. Keep the interactive Mamba-3 REPL stable and clean.
2. Probe every architecture after code changes.
3. Continue tiny MIMO-R4 training only as a kernel/runtime canary.
4. Build the real corpus pipeline.
5. Move paper-scale training to a GPU with enough shared-memory support for MIMO-R4 backward, or patch/rewrite the TileLang backward kernel for RTX 4080.
6. Only after real pretraining, run SFT for instruction format.
7. Only use verifier-accepted RLVR/self-improvement data; no external LLM teacher or judge.
